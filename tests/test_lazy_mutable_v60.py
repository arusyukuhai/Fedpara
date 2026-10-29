"""v60: on-demand writable out-of-core training and failure handling."""
import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from patchlib.core import UniversalAdapter
from patchlib.continual import ContinualElasticEngine
from patchlib.elastic import EventJournal
from patchlib.lazy_mutable_weights_v60 import attach_lazy_mutable, LazyWeightError
from patchlib.layer_offload import install_disk_linear_offload
from patchlib.streaming_base import materialize_meta_from_safetensors


class Little(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=nn.Linear(10,8)
        self.second=nn.Linear(8,6)
        self.tail=nn.LayerNorm(6)

    def forward(self,x):
        return self.tail(self.second(torch.relu(self.first(x))))


def _make(root, scratch):
    with torch.device('meta'):
        model=Little()
    materialize_meta_from_safetensors(model,root,writable=False,chunk_mib=1)
    adapter=UniversalAdapter(model,'toy','v1',kind='loha',rank=3,target_scope='auto')
    info=install_disk_linear_offload(model,adapter,scratch/'kernel',tile_mib=1)
    manager=attach_lazy_mutable(model,adapter,scratch_dir=scratch,copy_mib=1)
    return model,adapter,manager


def _reference(root):
    torch.manual_seed(12)
    base=Little()
    save_file({k:v.detach().contiguous() for k,v in base.state_dict().items()},str(root/'model.safetensors'))
    return base


def _seal_event(adapter, engine):
    optim=torch.optim.AdamW(list(adapter.trainable_parameters()),lr=.002)
    x=torch.randn(3,10)
    engine.adapter.model(x).square().mean().backward()
    optim.step();optim.zero_grad(set_to_none=True)
    return engine.seal_local(1)


def test_lazy_unsigned_read_and_signed_update_dedup_replay(tmp_path):
    root=tmp_path/'immutable';root.mkdir()
    _reference(root)
    model,adapter,manager=_make(root,tmp_path/'active')
    genesis=hashlib.sha256(b'signed-cow-genesis').hexdigest()
    journal=EventJournal(tmp_path/'a.sqlite',genesis)
    engine=ContinualElasticEngine(adapter,'alice',genesis,journal,unsafe=True)
    assert manager.used_bytes==0 and not manager.materialized
    immutable=(root/'model.safetensors').read_bytes()
    initial=model(torch.randn(2,10))
    assert initial.isfinite().all()
    wire=_seal_event(adapter,engine)
    assert manager.used_bytes>0
    assert len(manager.materialized)==2
    assert (root/'model.safetensors').read_bytes()==immutable
    # The disk kernel must refer to the writable per-model file, not a stale
    # cached safetensors tensor after a signed update.
    for name in ('first','second'):
        module=getattr(model,name)
        assert isinstance(module._fedpara_disk_source,str)
        assert Path(module._fedpara_disk_source).exists()
    x=torch.randn(3,10)
    expected=model(x).detach().clone()
    assert engine.ingest(wire) is False
    assert torch.allclose(model(x),expected,atol=1e-6)
    journal.close()
    reboot,adapt2,mgr2=_make(root,tmp_path/'reboot')
    j2=EventJournal(tmp_path/'b.sqlite',genesis)
    follower=ContinualElasticEngine(adapt2,'bob',genesis,j2,unsafe=True)
    assert follower.ingest(wire)
    # Distinct random local LoHA offsets cancel their anchors after first seal;
    # apply exact additive effective base update to the same immutable genesis.
    for ga,gb in zip(adapter.groups,adapt2.groups):
        ma=adapter.model.get_submodule(ga.aliases[0].rpartition('.')[0]).parametrizations.weight.original
        mb=adapt2.model.get_submodule(gb.aliases[0].rpartition('.')[0]).parametrizations.weight.original
        assert torch.allclose(ma,mb,atol=2e-6)
    assert mgr2.used_bytes==manager.used_bytes
    j2.close()


def test_lazy_disk_quota_fail_before_base_change(tmp_path):
    root=tmp_path/'immutable';root.mkdir()
    _reference(root)
    model,adapter,manager=_make(root,tmp_path/'scratch')
    manager.limit_bytes=1
    genesis=hashlib.sha256(b'quota').hexdigest()
    journal=EventJournal(tmp_path/'j.sqlite',genesis)
    engine=ContinualElasticEngine(adapter,'alice',genesis,journal,unsafe=True)
    original=model.first.parametrizations.weight.original.detach().clone()
    with pytest.raises(LazyWeightError,match='quota'):
        _seal_event(adapter,engine)
    assert manager.used_bytes==0
    assert torch.equal(model.first.parametrizations.weight.original,original)
    journal.close()


def test_lazy_unmodified_groups_stay_shared_and_epochs_get_private_dirs(tmp_path):
    root=tmp_path/'immutable';root.mkdir()
    _reference(root)
    m,a,one=_make(root,tmp_path/'scratch')
    n,b,two=_make(root,tmp_path/'scratch')
    assert one.directory!=two.directory
    one.materialize_group(a.groups[0])
    assert one.used_bytes>0 and two.used_bytes==0
    assert not one.materialize_group(a.groups[0])
    first_source=a.model.get_submodule(a.groups[0].aliases[0].rpartition('.')[0]).parametrizations.weight.original
    second_source=b.model.get_submodule(b.groups[0].aliases[0].rpartition('.')[0]).parametrizations.weight.original
    with torch.no_grad():first_source.add_(2)
    assert not torch.allclose(first_source,second_source)
    assert (root/'model.safetensors').is_file()


def test_writable_stream_loader_can_tile_with_row_wider_than_chunk(tmp_path):
    class Wide(nn.Module):
        def __init__(self):
            super().__init__()
            self.w=nn.Parameter(torch.empty((2,300_000)))
    data=torch.arange(600_000,dtype=torch.float32).reshape(2,300_000)
    save_file({'w':data},str(tmp_path/'model.safetensors'))
    with torch.device('meta'):
        meta=Wide()
    materialize_meta_from_safetensors(meta,tmp_path,writable=True,chunk_mib=1)
    assert torch.equal(meta.w,data)
    meta.w.data[1,-1]=42
    assert data[1,-1]!=42


def test_loader_flag_rejects_unsafe_configuration():
    from types import SimpleNamespace
    from patchlib.cli import load_model
    args=SimpleNamespace(streaming_lazy_write=True,streaming_base=False,
                         cmd='train-elastic',initialization='pretrained',factory_file=None)
    with pytest.raises(ValueError,match='streaming-lazy-write'):
        load_model(args,use_adapter=True)

class Conv1D(nn.Module):
    """GPT-2-style Conv1D orientation without a transformers dependency."""
    def __init__(self,features,input_features):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(input_features,features)*.15)
        self.bias=nn.Parameter(torch.randn(features)*.1)

    def forward(self,x):
        return torch.addmm(self.bias,x.reshape(-1,x.shape[-1]),self.weight).view(*x.shape[:-1],-1)


def test_gpt2_style_conv1d_training_gradient_and_mutable_update(tmp_path):
    torch.manual_seed(17)
    baseline=Conv1D(11,9)
    model=Conv1D(11,9)
    model.load_state_dict(baseline.state_dict())
    a=UniversalAdapter(model,'c','v1',kind='loha',rank=3,target_scope='auto')
    b=UniversalAdapter(baseline,'c','v1',kind='loha',rank=3,target_scope='auto')
    for g,ref in zip(a.groups,b.groups):
        for x,y in zip(g.overlay.local.parts,ref.overlay.local.parts):
            y.data.copy_(x.data)
    installed=install_disk_linear_offload(model,a,tmp_path/'paged',tile_mib=1)
    assert installed['layers']==1
    # All gradients, including the input and LoHA factors, must agree.
    x=torch.randn(3,2,9,requires_grad=True)
    y=x.detach().clone().requires_grad_(True)
    m_out=model(x);b_out=baseline(y)
    assert torch.allclose(m_out,b_out,atol=1e-5)
    m_out.square().mean().backward();b_out.square().mean().backward()
    assert torch.allclose(x.grad,y.grad,atol=1e-5)
    for p,q in zip(a.trainable_parameters(),b.trainable_parameters()):
        assert torch.allclose(p.grad,q.grad,atol=1e-5)


def test_gpt2_conv1d_lazy_stream_signed_apply(tmp_path):
    model_base=Conv1D(7,9)
    immutable=tmp_path/'immutable';immutable.mkdir()
    save_file({k:v.detach().contiguous() for k,v in model_base.state_dict().items()},
              str(immutable/'model.safetensors'))
    with torch.device('meta'):
        model=Conv1D(7,9)
    materialize_meta_from_safetensors(model,immutable,writable=False)
    adapter=UniversalAdapter(model,'toy','genesis',kind='lora',rank=3,target_scope='auto')
    install_disk_linear_offload(model,adapter,tmp_path/'page',tile_mib=1)
    mgr=attach_lazy_mutable(model,adapter,scratch_dir=tmp_path/'scratch')
    j=EventJournal(tmp_path/'journal.sqlite','c'*64)
    eng=ContinualElasticEngine(adapter,'alice','c'*64,j,unsafe=True)
    initial=(immutable/'model.safetensors').read_bytes()
    optim=torch.optim.AdamW(list(adapter.trainable_parameters()),lr=.02)
    x=torch.randn(3,9)
    model(x).pow(2).mean().backward();optim.step()
    eng.seal_local(1)
    assert mgr.used_bytes==9*7*4
    assert isinstance(model._fedpara_disk_source,str)
    assert (immutable/'model.safetensors').read_bytes()==initial
    assert model(x).isfinite().all()
    j.close()


def test_cli_train_elastic_meta_loader_is_lazily_writable(tmp_path,monkeypatch):
    # Exercise actual load_model() wiring without downloading a HF artifact.
    # This substitutes only the HF model factory, not streaming or the adapter.
    import sys
    from types import ModuleType, SimpleNamespace
    from patchlib.cli import load_model
    root=tmp_path/'immutable';root.mkdir()
    _reference(root)
    class Config:
        _commit_hash=None
        use_cache=True
    class FakeConfig:
        @staticmethod
        def from_pretrained(*args,**kwargs):return Config()
    class FakeAuto:
        @staticmethod
        def from_config(config,**kwargs):
            result=Little()
            result.config=config
            return result
    transformers=ModuleType('transformers')
    transformers.AutoConfig=FakeConfig
    transformers.AutoModelForCausalLM=FakeAuto
    monkeypatch.setitem(sys.modules,'transformers',transformers)
    args=SimpleNamespace(cmd='train-elastic',initialization='pretrained',
        model=str(root),revision='main',factory_file=None,trust_remote_code=False,
        streaming_base=True,streaming_lazy_write=True,
        streaming_scratch_dir=tmp_path/'scratch',stream_chunk_mib=1,
        stream_lazy_limit_gb=0,device='cpu',auto_class=None,hf_task='causal-lm',
        model_id=None,kind='loha',rank=3,alpha=None,max_peers=2,target_scope='auto',
        state_base=None,sharded_base_manifest=None,sharded_base_store=None,
        disk_linear_dir=None,auto_rank=False,resource_report=None,
        gradient_checkpointing=False)
    mdl,ada=load_model(args,use_adapter=True)
    manager=mdl._fedpara_lazy_mutable_weights
    assert manager.used_bytes==0
    assert not list((tmp_path/'scratch'/'kagami-lazy-v60').rglob('*.raw'))
    assert mdl._fedpara_streaming_base_info['mode']=='read-only-mmap'
    assert mdl(torch.rand(2,10)).shape==(2,6)
    j=EventJournal(tmp_path/'e.sqlite','e'*64)
    eng=ContinualElasticEngine(ada,'peer','e'*64,j,unsafe=True)
    _seal_event(ada,eng)
    assert manager.used_bytes>0
    j.close()


def test_disk_full_before_publication_does_not_mutate_genesis(tmp_path,monkeypatch):
    import errno
    import os
    root=tmp_path/'immutable';root.mkdir()
    _reference(root)
    model,adapter,manager=_make(root,tmp_path/'scratch')
    import hashlib
    g=adapter.groups[0]
    owner=g.aliases[0].rpartition('.')[0]
    original=adapter.model.get_submodule(owner).parametrizations.weight.original
    old=original.detach().clone()
    def no_space(*args):raise OSError(errno.ENOSPC,'no space left')
    if not hasattr(os,'posix_fallocate'):
        pytest.skip('requires posix_fallocate failure injection')
    monkeypatch.setattr(os,'posix_fallocate',no_space)
    with pytest.raises(OSError,match='no space'):
        manager.materialize_group(g)
    assert manager.used_bytes==0
    assert not list(manager.directory.glob('*.raw'))
    assert torch.equal(original,old)
    assert not isinstance(getattr(adapter.model.get_submodule(owner),'_fedpara_disk_source',None),str)
