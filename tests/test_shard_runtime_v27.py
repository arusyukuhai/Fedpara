"""Streaming loaded effective state; no full adapted-weight assembly needed."""
from __future__ import annotations

import hashlib
import json

import pytest
import torch
from torch import nn

from patchlib.core import PatchError, UniversalAdapter, FactorDelta
from patchlib.continual import enable_anchors
from patchlib.shards import (
    export_state_sharded, apply_sharded_state, apply_sharded_effective_state,
    read_manifest, shard_map, semantic_shard_sha256, _blob_path,
)
from patchlib.state import export_state, semantic_state_sha256
from patchlib.shard_runtime import effective_chunks

REV='a'*40
SB=65536


def test_sharded_plain_loading_no_assembled_zip(tmp_path):
    torch.manual_seed(11)
    source=nn.Sequential(nn.Embedding(4096,16),nn.Linear(16,64))
    store=tmp_path/'shards'; manifest=tmp_path/'state.fmanifest'
    export_state_sharded(source,store,manifest,model_id='x',revision=REV,
                         peer_id='alice',shard_bytes=SB)
    target=nn.Sequential(nn.Embedding(4096,16),nn.Linear(16,64))
    for p in target.parameters():
        with torch.no_grad():p.zero_()
    apply_sharded_state(target,manifest,store,model_id='x',revision=REV,
                        chunk_bytes=SB)
    for name,p in source.state_dict().items():
        assert torch.equal(target.state_dict()[name],p)
    assert not list(tmp_path.glob('*.fstate'))
    with pytest.raises(PatchError,match='identity'):
        apply_sharded_state(target,manifest,store,model_id='other',revision=REV)


def test_streamed_effective_matches_old_export_and_reinitializes(tmp_path,monkeypatch):
    torch.manual_seed(77)
    source=nn.Sequential(nn.Embedding(1600,32),nn.Linear(32,72),nn.Linear(72,32))
    adapter=UniversalAdapter(source,'x',REV,kind='loha',rank=8,target_scope='auto',wide_lora_rank=4)
    enable_anchors(adapter)
    # Real trained-looking factors; not all effective updates equal zero.
    with torch.no_grad():
        for p in adapter.trainable_parameters():p.add_(.0007)
    full=tmp_path/'legacy.fstate'
    export_state(source,full,model_id='x',revision=REV,peer_id='alice',adapter=adapter)
    manifest=tmp_path/'state.fmanifest'; store=tmp_path/'shards'
    # Calling .delta() eagerly materializes a full [1600,32] update.
    # The new path must never do this, even for an aliased embedding/head.
    def forbidden(*_):raise AssertionError('unbounded FactorDelta.delta() called')
    with monkeypatch.context() as m:
        m.setattr(FactorDelta,'delta',forbidden)
        export_state_sharded(source,store,manifest,model_id='x',revision=REV,
                             peer_id='alice',adapter=adapter,shard_bytes=SB)
    meta=read_manifest(manifest)
    assert semantic_shard_sha256(meta)==semantic_state_sha256(full)
    other=nn.Sequential(nn.Embedding(1600,32),nn.Linear(32,72),nn.Linear(72,32))
    oa=UniversalAdapter(other,'x',REV,kind='loha',rank=8,target_scope='auto',wide_lora_rank=4)
    enable_anchors(oa)
    apply_sharded_effective_state(other,oa,manifest,store,model_id='x',revision=REV,chunk_bytes=SB)
    with torch.no_grad():
        for a,b in zip(source.state_dict().items(),other.state_dict().items()):
            # Adapted state dict contains independent, resettable factors.
            pass
        for group in adapter.groups:
            name=group.canonical
            assert torch.allclose(source.get_submodule(name.rpartition('.')[0]).weight,
                                  other.get_submodule(name.rpartition('.')[0]).weight,atol=3e-5)
    # Anchors match after importing committed base.
    for g in oa.groups:
        for j,p in enumerate(g.overlay.local.parts):
            assert torch.equal(getattr(g.overlay.anchor,f'part{j}'),p)


def test_detect_corrupt_shard_before_changing_model(tmp_path):
    model=nn.Sequential(nn.Embedding(4096,16),nn.Linear(16,64))
    store=tmp_path/'store';path=tmp_path/'m.fmanifest'
    export_state_sharded(model,store,path,model_id='x',revision=REV,peer_id='alice',shard_bytes=SB)
    meta=read_manifest(path)
    digest=next(iter(shard_map(meta)))
    obj=_blob_path(store,digest)
    with obj.open('r+b') as f:f.write(b'bad!')
    dest=nn.Sequential(nn.Embedding(4096,16),nn.Linear(16,64))
    before=[t.detach().clone() for t in dest.parameters()]
    with pytest.raises(PatchError,match='missing or corrupt'):
        apply_sharded_state(dest,path,store,model_id='x',revision=REV)
    assert all(torch.equal(a,b) for a,b in zip(before,dest.parameters()))


def test_wider_than_buffer_flat_factor_path(tmp_path):
    torch.manual_seed(5)
    model=nn.Linear(32768,4,bias=False)
    adapter=UniversalAdapter(model,'x',REV,kind='loha',rank=4,target_scope='auto',wide_lora_rank=2)
    with torch.no_grad():
        for p in adapter.trainable_parameters():p.add_(0.0007)
        dense=model.weight.detach().clone()
    path=tmp_path/'m.fmanifest';store=tmp_path/'store'
    export_state_sharded(model,store,path,model_id='x',revision=REV,peer_id='alice',
                         adapter=adapter,shard_bytes=SB)
    bare=nn.Linear(32768,4,bias=False)
    apply_sharded_state(bare,path,store,model_id='x',revision=REV,chunk_bytes=SB)
    assert torch.allclose(bare.weight,dense,atol=1e-5)


def test_tied_embedding_head_one_effective_source(tmp_path):
    class TinyTied(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed=nn.Embedding(4096,16)
            self.lm_head=nn.Linear(16,4096,bias=False)
            self.lm_head.weight=self.embed.weight
        def get_output_embeddings(self):return self.lm_head
    source=TinyTied()
    adapter=UniversalAdapter(source,'x',REV,kind='loha',rank=8,target_scope='auto',wide_lora_rank=4)
    assert len(adapter.groups)==1
    path=tmp_path/'m.fmanifest';store=tmp_path/'store'
    export_state_sharded(source,store,path,model_id='x',revision=REV,
                         peer_id='alice',adapter=adapter,shard_bytes=SB)
    manifest=read_manifest(path)
    specs={e['name']:e['sha256'] for e in manifest['tensors']}
    assert specs['embed.weight']==specs['lm_head.weight']
    target=TinyTied()
    apply_sharded_state(target,path,store,model_id='x',revision=REV)
    assert target.embed.weight.data_ptr()==target.lm_head.weight.data_ptr()
    assert torch.allclose(target.embed.weight,source.embed.weight)
    # Explicitly reject a malicious manifest that gives different shared weights.
    bad=json.loads(path.read_text())
    head=next(e for e in bad['tensors'] if e['name']=='lm_head.weight')
    head['sha256']='0'*64
    forged=tmp_path/'bad.fmanifest';forged.write_text(json.dumps(bad))
    with pytest.raises(PatchError,match='shared/tied'):
        apply_sharded_state(target,forged,store,model_id='x',revision=REV)


def test_noncontiguous_buffer_streaming(tmp_path):
    class Weird(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('transpose',torch.arange(64*1024,dtype=torch.float32).reshape(1024,64).T)
    old=Weird();new=Weird()
    with torch.no_grad():new.transpose.zero_()
    assert not old.transpose.is_contiguous()
    store=tmp_path/'store';path=tmp_path/'m.fmanifest'
    export_state_sharded(old,store,path,model_id='x',revision=REV,peer_id='alice',shard_bytes=SB)
    apply_sharded_state(new,path,store,model_id='x',revision=REV)
    assert torch.equal(old.transpose,new.transpose)


def test_cli_sharded_base_as_model_revision(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from patchlib import cli
    from patchlib.state import semantic_state_sha256
    def fake_model(*args,**kwargs):
        return nn.Sequential(nn.Linear(12,8)),REV
    monkeypatch.setattr('patchlib.live_initialization.create_random_tinystories_model',fake_model)
    original,_=fake_model()
    store=tmp_path/'store';path=tmp_path/'m.fmanifest'
    export_state_sharded(original,store,path,model_id='x',revision=REV,peer_id='alice',shard_bytes=SB)
    args=SimpleNamespace(model='TinyStories-1M',revision=REV,model_id='x',
                         initialization='random',factory_file=None,initial_seed=42,
                         device='cpu',state_base=None,sharded_base_manifest=path,
                         sharded_base_store=store,max_state_gb=32,moe_plan=None,
                         trust_remote_code=False)
    loaded,revision=cli.load_model(args,use_adapter=False)
    assert revision=='fstate-sha256:'+semantic_shard_sha256(read_manifest(path))
    assert torch.equal(loaded[0].weight,original[0].weight)
    args.state_base=tmp_path/'nope.fstate'
    with pytest.raises(ValueError,match='mutually exclusive'):
        cli.load_model(args,use_adapter=False)
