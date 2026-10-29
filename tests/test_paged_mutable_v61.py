"""v61: sparse page COW, signed updates, frozen MoE input gradients."""
import hashlib
from pathlib import Path
import pytest
import torch
from torch import nn
from safetensors.torch import save_file
from patchlib.core import UniversalAdapter
from patchlib.continual import ContinualElasticEngine
from patchlib.elastic import EventJournal
from patchlib.streaming_base import materialize_meta_from_safetensors
from patchlib.layer_offload import install_disk_linear_offload
from patchlib.paged_mutable_v61 import attach_paged_mutable, SparsePageStore, LazyWeightError
from patchlib.expert_disk import StreamedFrozenExpert, install_streamed_experts


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.one=nn.Linear(10,8)
        self.two=nn.Linear(8,5)
    def forward(self,x):return self.two(torch.tanh(self.one(x)))


def _setup(tmp_path):
    torch.manual_seed(91)
    reference=Toy()
    root=tmp_path/'source';root.mkdir()
    save_file({k:v.detach().clone().contiguous() for k,v in reference.state_dict().items()},
              str(root/'model.safetensors'))
    with torch.device('meta'):model=Toy()
    materialize_meta_from_safetensors(model,root,writable=False)
    adapter=UniversalAdapter(model,'p','base',kind='loha',rank=3,target_scope='auto')
    install_disk_linear_offload(model,adapter,tmp_path/'offload',min_weight_mib=0.)
    manager=attach_paged_mutable(model,adapter,scratch_dir=tmp_path/'scratch',page_kib=4)
    return root,model,adapter,manager


def test_sparse_page_unchanged_reads_and_sparse_edits(tmp_path):
    base=torch.arange(10_000,dtype=torch.float32).view(100,100)
    store=SparsePageStore(base,tmp_path/'pages',page_bytes=4096)
    assert torch.equal(store[1:3,5:9],base[1:3,5:9])
    assert store.stats()['allocated_page_bytes']==0
    rows=store.rows(1,2)
    rows[0,7]+=9
    store.write_rows(1,rows)
    assert store.stats()['changed_pages']==1
    assert store[1:2,7:8][0,0]==base[1,7]+9
    assert store[90:91,90:91][0,0]==base[90,90]
    assert base[1,7]==107  # source unchanged
    assert torch.equal(store.T[5:9,1:3],store[1:3,5:9].T)
    assert torch.equal(store.gather_rows(torch.tensor([1,4,1]))[0],store.rows(1,2)[0])
    store.close()


def test_signed_paged_updates_forward_and_restart(tmp_path):
    root,m,a,pages=_setup(tmp_path)
    virgin=(root/'model.safetensors').read_bytes()
    j=EventJournal(tmp_path/'a.sqlite','a'*64)
    engine=ContinualElasticEngine(a,'alice','a'*64,j,unsafe=True)
    x=torch.randn(4,10)
    optim=torch.optim.AdamW(list(a.trainable_parameters()),lr=.008)
    m(x).square().mean().backward();optim.step();optim.zero_grad(set_to_none=True)
    wire=engine.seal_local(1)
    assert pages.stats()['allocated_page_bytes']>0
    assert (root/'model.safetensors').read_bytes()==virgin
    after=m(x).detach().clone()
    assert not engine.ingest(wire)
    assert torch.allclose(m(x),after,atol=1e-6)
    # reconstruct clean from immutable genesis and signed journal
    tmp2=tmp_path/'restart';tmp2.mkdir()
    root2=tmp2/'source';root2.mkdir()
    (root2/'model.safetensors').write_bytes(virgin)
    with torch.device('meta'):m2=Toy()
    materialize_meta_from_safetensors(m2,root2,writable=False)
    a2=UniversalAdapter(m2,'p','base',kind='loha',rank=3,target_scope='auto')
    install_disk_linear_offload(m2,a2,tmp2/'offload',min_weight_mib=0.)
    p2=attach_paged_mutable(m2,a2,scratch_dir=tmp2/'scratch',page_kib=4)
    j2=EventJournal(tmp2/'b.sqlite','a'*64)
    eng2=ContinualElasticEngine(a2,'bob','a'*64,j2,unsafe=True)
    assert eng2.ingest(wire)
    assert torch.allclose(m(x),m2(x),atol=2e-5)
    # whole-state export materializes the actual effective pages, not stale base
    old=m2(x).detach().clone()
    p2.materialize_all(a2)
    assert torch.allclose(m2(x),old,atol=2e-5)
    j.close();j2.close()


def test_paged_limit_rejects_before_write(tmp_path):
    root,m,a,pages=_setup(tmp_path)
    pages.limit_bytes=2
    j=EventJournal(tmp_path/'j.sqlite','b'*64)
    eng=ContinualElasticEngine(a,'me','b'*64,j,unsafe=True)
    x=torch.randn(2,10)
    opt=torch.optim.AdamW(list(a.trainable_parameters()),lr=.001)
    m(x).sum().backward();opt.step()
    with pytest.raises(LazyWeightError,match='quota|capacity'):
        eng.seal_local(1)
    assert pages.stats()['allocated_page_bytes']==0
    j.close()


def test_streamed_frozen_experts_identical_input_gradients(tmp_path):
    torch.manual_seed(234)
    class Switch(nn.Module):
        def __init__(self):
            super().__init__();self.experts=nn.ModuleList([nn.Sequential(nn.Linear(7,11),nn.GELU(),nn.Linear(11,7)) for _ in range(4)])
            self.router=nn.Linear(7,4)
        def forward(self,x):
            w=self.router(x).softmax(-1)
            return sum(self.experts[i](x)*w[:,i,None] for i in range(4))
    baseline=Switch();paged=Switch();paged.load_state_dict(baseline.state_dict())
    for model in (baseline,paged):
        for expert in model.experts:
            for p in expert.parameters():p.requires_grad_(False)
    assert install_streamed_experts(paged,'experts')==4
    x=torch.randn(3,7,requires_grad=True)
    y=x.detach().clone().requires_grad_(True)
    baseline(x).square().mean().backward();paged(y).square().mean().backward()
    assert torch.allclose(baseline(x),paged(y),atol=1e-6)
    assert torch.allclose(x.grad,y.grad,atol=1e-5)
    assert torch.allclose(baseline.router.weight.grad,paged.router.weight.grad,atol=1e-5)
    assert isinstance(paged.experts[2],StreamedFrozenExpert)
    paged.experts[3].expert[0].weight.requires_grad_(True)
    with pytest.raises(ValueError,match='trainable'):
        install_streamed_experts(paged,'experts',indices=[3])


def test_paged_export_refuses_stale_base_and_explicit_materialization(tmp_path):
    root,m,a,pages=_setup(tmp_path)
    group=a.groups[0]
    assert pages.materialize_group(group)
    store=pages.stores[group.canonical]
    changed=store.rows(0,1);changed+=1
    store.write_rows(0,changed)
    from patchlib.shards import export_state_sharded
    from patchlib.core import PatchError
    with pytest.raises(LazyWeightError,match='state_dict'):
        m.state_dict()
    with pytest.raises(PatchError,match='explicit'):
        export_state_sharded(m,tmp_path/'shards',tmp_path/'out.fmanifest',
            model_id='toy',revision='v1',peer_id='test',seq=1,adapter=a)
    pages.materialize_all(a)
    assert not pages.stores
    assert torch.equal(m.get_submodule(group.aliases[0].rpartition('.')[0]).parametrizations.weight.original[:1],changed)


def test_poisoned_sparse_store_rejects_later_reads(tmp_path,monkeypatch):
    import os
    base=torch.zeros((2000,20),dtype=torch.float32)
    store=SparsePageStore(base,tmp_path/'pages',page_bytes=4096)
    def bad_write(*args):raise OSError('injected I/O fault')
    monkeypatch.setattr(os,'pwrite',bad_write)
    with pytest.raises(OSError,match='injected'):
        store.write_rows(0,torch.ones((1,20)))
    with pytest.raises(LazyWeightError,match='poisoned'):
        store.rows(0,1)
    store.close()


def test_streamed_expert_rejects_stochastic_training():
    expert=nn.Sequential(nn.Linear(5,7),nn.Dropout(.5),nn.Linear(7,3))
    for p in expert.parameters():p.requires_grad_(False)
    proxy=StreamedFrozenExpert(expert)
    with pytest.raises(RuntimeError,match='stochastic'):
        proxy(torch.randn(2,5,requires_grad=True))
    proxy.eval()
    result=proxy(torch.randn(2,5,requires_grad=True))
    assert result.shape==(2,3)


def test_conv1d_paged_signed_replay_and_input_grads(tmp_path):
    from test_lazy_mutable_v60 import Conv1D
    torch.manual_seed(55)
    original=Conv1D(7,9)
    root=tmp_path/'source';root.mkdir()
    save_file({k:v.detach().clone().contiguous() for k,v in original.state_dict().items()},
              str(root/'model.safetensors'))
    with torch.device('meta'):model=Conv1D(7,9)
    materialize_meta_from_safetensors(model,root,writable=False)
    adapter=UniversalAdapter(model,'conv','v1',kind='loha',rank=3,target_scope='auto')
    install_disk_linear_offload(model,adapter,tmp_path/'offload',min_weight_mib=0.)
    manager=attach_paged_mutable(model,adapter,scratch_dir=tmp_path/'cow',page_kib=4)
    j=EventJournal(tmp_path/'j.sqlite','d'*64)
    eng=ContinualElasticEngine(adapter,'a','d'*64,j,unsafe=True)
    x=torch.randn(4,9,requires_grad=True)
    model(x).square().mean().backward()
    opt=torch.optim.AdamW(list(adapter.trainable_parameters()),lr=.01)
    opt.step();opt.zero_grad(set_to_none=True)
    eng.seal_local(1)
    assert manager.stats()['allocated_page_bytes']>0
    z=torch.randn(4,9,requires_grad=True)
    out=model(z).square().mean()
    out.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    j.close()


def test_cli_paged_mode_on_real_elastic_loader_path(tmp_path,monkeypatch):
    from types import SimpleNamespace,ModuleType
    import sys
    from patchlib.cli import load_model
    torch.manual_seed(34)
    root=tmp_path/'immutable';root.mkdir()
    base=Toy()
    save_file({k:v.detach().contiguous() for k,v in base.state_dict().items()},
              str(root/'model.safetensors'))
    class Config:
        _commit_hash=None
        use_cache=True
    class FakeConfig:
        @staticmethod
        def from_pretrained(*a,**kw):return Config()
    class FakeAuto:
        @staticmethod
        def from_config(config,**kw):
            m=Toy();m.config=config;return m
    fake=ModuleType('transformers')
    fake.AutoConfig=FakeConfig;fake.AutoModelForCausalLM=FakeAuto
    monkeypatch.setitem(sys.modules,'transformers',fake)
    args=SimpleNamespace(cmd='train-elastic',mode='elastic',initialization='pretrained',
        model=str(root),revision='main',factory_file=None,trust_remote_code=False,
        streaming_base=True,streaming_lazy_write=True,streaming_page_cow=True,
        stream_page_kib=4,streaming_scratch_dir=tmp_path/'nvme',stream_chunk_mib=1,
        stream_lazy_limit_gb=0,device='cpu',auto_class=None,hf_task='causal-lm',
        model_id=None,kind='loha',rank=3,alpha=None,max_peers=2,target_scope='auto',
        state_base=None,sharded_base_manifest=None,sharded_base_store=None,
        disk_linear_dir=None,disk_linear_min_mib=.00001,
        auto_rank=False,resource_report=None,gradient_checkpointing=False)
    m,a=load_model(args,use_adapter=True)
    mgr=m._fedpara_lazy_mutable_weights
    assert mgr.stats()['allocated_page_bytes']==0
    j=EventJournal(tmp_path/'journal.sqlite','e'*64)
    eng=ContinualElasticEngine(a,'one','e'*64,j,unsafe=True)
    opt=torch.optim.AdamW(list(a.trainable_parameters()),lr=.002)
    m(torch.randn(2,10)).sum().backward();opt.step()
    eng.seal_local(1)
    assert mgr.stats()['allocated_page_bytes']>0
    j.close()
