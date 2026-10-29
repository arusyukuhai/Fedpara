"""Opt-in disk-layer/optimizer and heterogeneous expert shape regressions."""
from __future__ import annotations
import torch
from torch import nn
import pytest
from patchlib.core import UniversalAdapter
from patchlib.layer_offload import install_disk_linear_offload
from patchlib.disk_adamw import DiskAdamW
from patchlib.device_planner import detect_device_budget, estimate_adapter_memory, suggest_rank_and_experts
from patchlib.moe_planner import inspect_expert_shapes, plan_expert_placement, ExpertAdapterRegistry, ExpertGroup


@pytest.mark.parametrize('kind',['loha','lora'])
def test_disk_offload_base_and_factor_gradients(tmp_path,kind):
    torch.manual_seed(42)
    original=nn.Sequential(nn.Linear(10,9),nn.ReLU(),nn.Linear(9,7))
    adapter=UniversalAdapter(original,'toy','v1',kind=kind,rank=4,target_scope='auto')
    x=torch.randn(4,10,requires_grad=True)
    ref=original(x);ref.sum().backward()
    grads={id(p):p.grad.detach().clone() for p in adapter.trainable_parameters()}
    grad_input=x.grad.detach().clone()
    original.zero_grad(set_to_none=True);x.grad=None
    result=install_disk_linear_offload(original,adapter,tmp_path)
    assert result['layers']==2
    assert original[0].parametrizations.weight.original.device.type=='cpu'
    assert not original[0].parametrizations.weight.original.requires_grad
    off=original(x)
    assert torch.allclose(off,ref,atol=2e-5,rtol=1e-5)
    off.sum().backward()
    assert torch.allclose(x.grad,grad_input,atol=2e-5,rtol=1e-5)
    for p in adapter.trainable_parameters():
        assert torch.allclose(p.grad,grads[id(p)],atol=2e-5,rtol=1e-4)


def test_disk_adamw_matches_torch_and_resume(tmp_path):
    torch.manual_seed(20)
    a=nn.Parameter(torch.randn(4,6));b=nn.Parameter(a.detach().clone())
    reference=torch.optim.AdamW([a],lr=.003,weight_decay=.07)
    off=DiskAdamW([('a',b)],tmp_path,lr=.003,weight_decay=.07)
    for i in range(5):
        grad=torch.randn_like(a)
        a.grad=grad.clone();b.grad=grad.clone()
        reference.step();off.step()
        assert torch.allclose(a,b,atol=1e-6,rtol=1e-5)
    off.flush()
    before=b.detach().clone()
    off2=DiskAdamW([('a',b)],tmp_path,lr=.003,weight_decay=.07,resume=True)
    assert off2.state[b]['step']==5
    grad=torch.randn_like(a)
    a.grad=grad.clone();b.grad=grad.clone()
    reference.step();off2.step()
    assert torch.allclose(a,b,atol=1e-6,rtol=1e-5)


def test_disk_adamw_rejects_stale_layout(tmp_path):
    p=nn.Parameter(torch.randn(3))
    DiskAdamW([('x',p)],tmp_path)
    with pytest.raises(ValueError):
        DiskAdamW([('x',nn.Parameter(torch.randn(4)))],tmp_path,resume=True)


class WeirdExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts=nn.ModuleList([nn.Linear(5,11),nn.Linear(5,7),nn.Linear(5,13)])
        self.router=nn.Linear(5,3)

class PackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        class Packed(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1=nn.Parameter(torch.randn(4,6,5))
                self.w2=nn.Parameter(torch.randn(4,5,6))
        self.experts=Packed()


def test_heterogeneous_expert_inventory_and_placement():
    m=nn.Module();m.a=WeirdExperts();m.b=PackedExperts()
    groups=inspect_expert_shapes(m)
    assert {g.path for g in groups}=={'a.experts','b.experts'}
    unpack=next(g for g in groups if g.path=='a.experts')
    assert unpack.count==3 and len(set(unpack.bytes_per_expert))==3
    assert unpack.executable
    packed=next(g for g in groups if g.path=='b.experts')
    assert packed.count==4 and not packed.executable
    plan=plan_expert_placement(groups,'room-1',{'p1':1,'p2':3},replicas=2)
    assert len(plan['experts'])==7
    assert all(len(row['owners'])==2 for row in plan['experts'].values())
    assert plan==plan_expert_placement(groups,'room-1',{'p1':1,'p2':3},replicas=2)


def test_registry_custom_shape_and_device_heuristics(tmp_path):
    registry=ExpertAdapterRegistry()
    custom=ExpertGroup('x','custom',3,(500,1000,200),False,'custom')
    registry.register('custom-v2',lambda model:[custom])
    assert inspect_expert_shapes(nn.Linear(4,4),registry=registry,model_type='custom-v2')==[custom]
    model=nn.Sequential(nn.Linear(12,8),nn.Linear(8,5))
    budget=detect_device_budget('cpu',tmp_path)
    assert budget.free_disk_bytes>0
    estimate=estimate_adapter_memory(model,kind='loha',rank=4)
    assert estimate['layer_count']==2
    suggestion=suggest_rank_and_experts(model,device='cpu',disk_dir=tmp_path,
                                         expert_bytes=[200,500,1000])
    assert suggestion['rank_default'] in (None,4,8,16,32,64)
    assert suggestion['expert_count_default'] in (0,1,2,3)


def test_disk_expert_recompute_handles_multiple_outstanding_calls(tmp_path):
    from patchlib.expert_disk import DiskExpertProxy
    torch.manual_seed(28)
    module=nn.Sequential(nn.Linear(8,14),nn.SiLU(),nn.Linear(14,7))
    for p in module.parameters():p.requires_grad_(False)
    x1=torch.randn(4,8,requires_grad=True);x2=torch.randn(2,8,requires_grad=True)
    ref1=module(x1);ref2=module(x2)
    (ref1.square().sum()+ref2.square().sum()).backward()
    want1=x1.grad.detach().clone();want2=x2.grad.detach().clone()
    x1.grad=x2.grad=None
    disk=DiskExpertProxy(module,tmp_path,identifier='expert-a')
    y1=disk(x1);y2=disk(x2)
    assert torch.allclose(y1,ref1,atol=1e-6)
    assert torch.allclose(y2,ref2,atol=1e-6)
    (y1.square().sum()+y2.square().sum()).backward()
    assert torch.allclose(x1.grad,want1,atol=1e-6)
    assert torch.allclose(x2.grad,want2,atol=1e-6)


def test_sharded_snapshot_with_disk_linear(tmp_path):
    from patchlib.shards import export_state_sharded, apply_sharded_state
    torch.manual_seed(100)
    model=nn.Sequential(nn.Linear(12,8),nn.Linear(8,4))
    adapter=UniversalAdapter(model,'a-model','r1',kind='loha',rank=4,target_scope='auto')
    # Establish a nonzero effective factor delta.
    with torch.no_grad():
        for group in adapter.groups:
            for part in group.overlay.local.parts:
                part.add_(.008)
    source={k:v.detach().clone() for k,v in model.state_dict().items()}
    expected=[model[0].weight.detach().clone(),model[1].weight.detach().clone()]
    install_disk_linear_offload(model,adapter,tmp_path/'linear')
    export_state_sharded(model,tmp_path/'objects',tmp_path/'state.fmanifest',
                         model_id='a-model',revision='r1',peer_id='peer-0',seq=1,
                         adapter=adapter,shard_bytes=65536,max_state_bytes=1024*1024)
    destination=nn.Sequential(nn.Linear(12,8),nn.Linear(8,4))
    apply_sharded_state(destination,tmp_path/'state.fmanifest',tmp_path/'objects',
                        model_id='a-model',revision='r1',max_state_bytes=1024*1024)
    assert torch.allclose(destination[0].weight,expected[0],atol=1e-6)
    assert torch.allclose(destination[1].weight,expected[1],atol=1e-6)


def test_packed_swiglu_adapter_matches_manual_and_gradients():
    import torch.nn.functional as F
    from patchlib.moe_planner import packed_swiglu_expert_views
    class Packed(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1=nn.Parameter(torch.randn(6,11,7))
            self.w2=nn.Parameter(torch.randn(6,7,11))
            self.w3=nn.Parameter(torch.randn(6,11,7))
    packed=Packed()
    views=packed_swiglu_expert_views(packed)
    assert len(views)==6
    x=torch.randn(5,7,requires_grad=True)
    out=views['4'](x)
    expected=F.linear(F.silu(F.linear(x,packed.w1[4]))*F.linear(x,packed.w3[4]),packed.w2[4])
    assert torch.allclose(out,expected)
    expected.sum().backward(retain_graph=True)
    x_grad=x.grad.detach().clone();x.grad.zero_()
    out.sum().backward()
    assert torch.allclose(x.grad,x_grad,atol=1e-5)


def test_capacity_bounded_expert_allocation_and_defaults():
    from patchlib.device_planner import automatic_expert_counts
    a=ExpertGroup('a','module',3,(200,300,400),True,'linear')
    b=ExpertGroup('b','module',5,(100,100,100,100,100),True,'linear')
    assert automatic_expert_counts([a,b],500)=={'a':1,'b':1}
    count=automatic_expert_counts([a,b],1200)
    assert count['a']>=1 and count['b']>=1
    with pytest.raises(MemoryError):automatic_expert_counts([a,b],499)
    with pytest.raises(MemoryError):
        plan_expert_placement([a],'r',{'p1':1,'p2':1},replicas=2,
                              peer_byte_budgets={'p1':500,'p2':500})
    fits=plan_expert_placement([a],'r',{'p1':1,'p2':1},replicas=1,
                               peer_byte_budgets={'p1':500,'p2':500})
    assert len(fits['experts'])==3
    assert all(remain>=0 for remain in fits['peer_remaining_bytes'].values())


def test_packed_expert_rpc_forward_and_backward():
    from patchlib.expert_parallel import ExpertOwner, _tensor_msg, _read_tensor
    from patchlib.moe_planner import packed_swiglu_expert_views
    class Packed(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1=nn.Parameter(torch.randn(2,6,4))
            self.w2=nn.Parameter(torch.randn(2,4,6))
            self.w3=nn.Parameter(torch.randn(2,6,4))
    views=packed_swiglu_expert_views(Packed())
    owner=ExpertOwner(views,'generation-1')
    rid='0'*32
    x=torch.randn(3,4)
    resp=owner.handle('peer-0',{'version':1,'generation':'generation-1','expert':'1',
                                'op':'forward','id':rid,'tensor':_tensor_msg(x)})
    y=_read_tensor(resp['tensor'])
    assert y.shape==(3,4)
    bwd=owner.handle('peer-0',{'version':1,'generation':'generation-1','expert':'1',
                               'op':'backward','id':rid,'tensor':_tensor_msg(torch.ones_like(y))})
    assert _read_tensor(bwd['tensor']).shape==x.shape


def test_plan_expert_roster_cli_and_autorank_recommendation(tmp_path):
    import json
    from scripts.plan_expert_roster import main
    from patchlib.device_planner import automatic_expert_counts
    data={'expert_shapes':[
        {'path':'layers.0.experts','representation':'module','count':2,
         'bytes_per_expert':[200,350],'executable':True,'router':'linear'}]}
    report=tmp_path/'report.json';report.write_text(json.dumps(data))
    roster=tmp_path/'roster.json';roster.write_text(json.dumps({
        'a':{'weight':2,'expert_budget_bytes':2000},
        'b':{'weight':1,'expert_budget_bytes':2000}}))
    output=tmp_path/'owners.json'
    main(['--report',str(report),'--roster',str(roster),
          '--room','room','--generation','genesis','--replicas','2',
          '--output',str(output)])
    result=json.loads(output.read_text())
    assert len(result['experts'])==2
    assert all(len(e['owners'])==2 for e in result['experts'].values())
    count=automatic_expert_counts([ExpertGroup(**data['expert_shapes'][0])],1200)
    assert count['layers.0.experts']==2


def test_disk_offload_training_gradients_and_optimizer(tmp_path):
    torch.manual_seed(88)
    model=nn.Sequential(nn.Linear(5,7),nn.Tanh(),nn.Linear(7,3))
    adapter=UniversalAdapter(model,'toy','v1',kind='lora',rank=3,target_scope='auto')
    install_disk_linear_offload(model,adapter,tmp_path/'weights')
    from patchlib.disk_adamw import adapter_named_parameters
    optim=DiskAdamW(adapter_named_parameters(adapter),tmp_path/'moments',lr=.02)
    x=torch.randn(16,5);target=torch.randn(16,3)
    first=None;last=None
    for step in range(20):
        optim.zero_grad(set_to_none=True)
        loss=(model(x)-target).square().mean()
        if first is None:first=float(loss.item())
        loss.backward();optim.step()
    last=float((model(x)-target).square().mean().item())
    assert last<first
    optim.flush()


def test_input_device_does_not_select_mapped_frozen_weight(tmp_path):
    from patchlib.cli import input_device
    m=nn.Sequential(nn.Linear(5,6),nn.ReLU(),nn.Linear(6,3))
    a=UniversalAdapter(m,'t','r',kind='lora',rank=3,target_scope='auto')
    install_disk_linear_offload(m,a,tmp_path)
    assert input_device(m,'cpu').type=='cpu'
