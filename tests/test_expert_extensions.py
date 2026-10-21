import io
import json
import zipfile
import asyncio
from pathlib import Path
import pytest
import torch
from torch import nn
from safetensors.torch import save_file
from patchlib.expert_parallel import (ExpertOwner,LocalEndpoint,RemoteExpert,expert_owner,
                                      discover_moe,install_remote_experts,ExpertRPCError)
from patchlib.expert_memory import (pack_int4,unpack_int4,QuantizedLoHaLinear,
                                   quantize_expert_linears,ExpertPager)
from patchlib.expert_safetensors import (iter_selected_tensors,expert_tensor_key,load_into_meta_model)
from patchlib.code_registry import validate_source,register_code
from patchlib.expert_channel import encode_rpc,decode_rpc

class ToyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate=nn.Linear(4,3)
        self.experts=nn.ModuleList([nn.Linear(4,4) for i in range(3)])
    def forward(self,x):
        route=self.gate(x).topk(1).indices.view(-1)
        return torch.stack([self.experts[int(i)](x[k]) for k,i in enumerate(route)])


def test_router_discovery_and_owner_stable():
    m=ToyMoE()
    assert discover_moe(m)[0]['experts']=='experts'
    peers={'A':3,'B':1,'C':2}
    assert expert_owner('room','experts.0',peers)==expert_owner('room','experts.0',peers)
    assert len(expert_owner('room','experts.0',peers,replicas=2))==2


def test_remote_autograd_same_as_local():
    a=nn.Linear(4,4)
    b=nn.Linear(4,4);b.load_state_dict(a.state_dict())
    owner=ExpertOwner({'experts.2':b},'hash0')
    proxy=RemoteExpert(LocalEndpoint(owner),'experts.2','hash0')
    x=torch.randn(6,4,requires_grad=True)
    y=proxy(x)
    loss=(y**2).mean();loss.backward()
    xt=x.detach().clone().requires_grad_(True)
    (a(xt)**2).mean().backward()
    assert torch.allclose(y.detach(),a(xt).detach())
    assert torch.allclose(x.grad,xt.grad)
    assert torch.allclose(b.weight.grad,a.weight.grad)
    assert torch.allclose(b.bias.grad,a.bias.grad)


def test_replace_remote_expert_and_failure():
    m=ToyMoE();owner=ExpertOwner({'experts.1':m.experts[1]},'v')
    install_remote_experts(m,'experts',{1:LocalEndpoint(owner)},'v')
    assert isinstance(m.experts[1],RemoteExpert)
    inp=torch.randn(2,4,requires_grad=True)
    y=m.experts[1](inp);y.sum().backward()
    assert inp.grad is not None
    with pytest.raises(ExpertRPCError):
        RemoteExpert(LocalEndpoint(owner),'experts.1','wrong')(torch.randn(4))


def test_int4_base_and_adapter_gradient():
    original=nn.Linear(32,16)
    q=QuantizedLoHaLinear(original,rank=4)
    recovered=unpack_int4(q.packed_base,q.scales,q.shape,q.group_size)
    assert (original.weight-recovered).abs().mean()<0.05
    assert q.storage_bytes < original.weight.numel()*2
    x=torch.randn(4,32)
    q(x).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in q.adapter.parts)


def test_quantize_expert_submodules():
    m=nn.Sequential(nn.Linear(8,16),nn.GELU(),nn.Linear(16,8))
    assert quantize_expert_linears(m,rank=4)==2
    assert m(torch.randn(2,8)).shape==(2,8)


def test_cpu_nvme_paging(tmp_path):
    ex=nn.Linear(8,8)
    old=ex.weight.detach().clone()
    pager=ExpertPager({'x':ex},tmp_path)
    pager.page_out('x','nvme')
    assert ex.weight.device.type=='meta'
    with pager.lease('x') as live:
        assert torch.allclose(live.weight,old)
        with pytest.raises(RuntimeError):pager.page_out('x','nvme')
    pager.page_out('x','cpu')


def test_safetensors_select_only(tmp_path):
    m=ToyMoE()
    fname=tmp_path/'model.safetensors'
    save_file({k:v.detach().clone() for k,v in m.state_dict().items()},str(fname))
    selected={'experts':{1}}
    keys={k for k,_ in iter_selected_tensors(fname,selected)}
    assert 'gate.weight' in keys and 'experts.1.weight' in keys
    assert 'experts.0.weight' not in keys and 'experts.2.weight' not in keys
    assert expert_tensor_key('experts.1.weight')==('experts',1)
    with torch.device('meta'):m2=ToyMoE()
    m2.experts[0]=nn.Identity();m2.experts[2]=nn.Identity()
    loaded=load_into_meta_model(m2,fname,selected)
    assert loaded==4
    assert m2.gate.weight.device.type=='cpu'
    assert torch.equal(m2.experts[1].weight,m.experts[1].weight)


def test_code_registry_reject_traversal(tmp_path):
    got=register_code(tmp_path/'codes','build.py',b'def build_model(): pass\n')
    assert Path(got['path']).is_file()
    bio=io.BytesIO()
    with zipfile.ZipFile(bio,'w') as z:z.writestr('../bad.py','x')
    with pytest.raises(ValueError):validate_source(bio.getvalue(),'source.zip')
    with pytest.raises(ValueError):validate_source(b'bad','bad.pkl')


def test_rpc_envelope_roundtrip():
    msg={'version':1,'id':'a'*32,'tensor':b'foo','op':'reply'}
    assert decode_rpc(encode_rpc(msg))==msg

@pytest.mark.asyncio
async def test_expert_datachannel_bridge_over_loop():
    from patchlib.expert_channel import ExpertDataChannelBridge
    from patchlib.expert_parallel import ExpertOwner,RemoteExpert
    owner_model=nn.Linear(4,4)
    owner=ExpertOwner({'experts.0':owner_model},'same-generation')
    loop=asyncio.get_running_loop()
    class FakeNode:
        def __init__(self,peer):
            self.remote=peer;self.connected=asyncio.Event();self.connected.set()
            self.verifier=object();self.expert_bridge=None
            self._channel=None
    class FakeChannel:
        readyState='open';bufferedAmount=0
        def __init__(self,other):self.other=other
        def send(self,frame):self.other.accept(frame)
    client=ExpertDataChannelBridge(loop,timeout=5)
    server=ExpertDataChannelBridge(loop,owner=owner,timeout=5)
    a=FakeNode('server');b=FakeNode('client')
    client.attach(a);server.attach(b)
    a._channel=FakeChannel(server);b._channel=FakeChannel(client)
    def do_training():
        proxy=RemoteExpert(client,'experts.0','same-generation')
        x=torch.randn(5,4,requires_grad=True)
        y=proxy(x)
        y.square().sum().backward()
        return x.grad is not None
    assert await asyncio.to_thread(do_training)
    assert owner_model.weight.grad is not None


def test_deepseek_style_gate_discovery():
    class DeepseekGate(nn.Module):
        def __init__(self):
            super().__init__();self.weight=nn.Parameter(torch.randn(3,4))
        def forward(self,x):
            scores=torch.nn.functional.linear(x,self.weight)
            return scores.topk(2).values,scores.topk(2).indices
    class DeepseekBlock(nn.Module):
        def __init__(self):
            super().__init__();self.gate=DeepseekGate();self.experts=nn.ModuleList([nn.Linear(4,4) for _ in range(3)])
    info=discover_moe(DeepseekBlock())
    assert info[0]['router']=='deepseek-weight-topk' and info[0]['supported']


def test_switch_router_dict_discovery_and_proxy():
    class SwitchRouter(nn.Module):
        def __init__(self):
            super().__init__();self.classifier=nn.Linear(4,2)
    class SwitchBlock(nn.Module):
        def __init__(self):
            super().__init__();self.router=SwitchRouter()
            self.experts=nn.ModuleDict({'expert_0':nn.Linear(4,4),'expert_1':nn.Linear(4,4)})
    model=SwitchBlock()
    assert discover_moe(model)[0]['router']=='switch-classifier'
    ex=model.experts['expert_1'];owner=ExpertOwner({'experts.expert_1':ex},'v')
    install_remote_experts(model,'experts',{1:LocalEndpoint(owner)},'v')
    x=torch.randn(3,4,requires_grad=True)
    model.experts['expert_1'](x).sum().backward()
    assert ex.weight.grad is not None
