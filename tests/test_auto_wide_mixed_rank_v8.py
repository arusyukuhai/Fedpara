"""Architecture-independent wide LoRA and mixed-rank snapshot/reconnect checks."""
from copy import deepcopy
import hashlib

import pytest
import torch
from torch import nn

from patchlib.core import UniversalAdapter, PatchError
from patchlib.continual import enable_anchors
from patchlib.elastic import ElasticEngine, EventJournal
from patchlib.livefed import FedAvgRound
from patchlib.state import apply_effective_state, export_state


class OtherArchitecture(nn.Module):
    """Deliberately NOT GPT-Neo: a typical custom/tied language model."""
    def __init__(self):
        super().__init__()
        self.backbone=nn.Module()
        self.backbone.embed_tokens=nn.Embedding(128,32)
        self.backbone.layers=nn.ModuleList([nn.Module()])
        layer=self.backbone.layers[0]
        layer.self_attn=nn.Module()
        layer.self_attn.q_proj=nn.Linear(32,32)
        layer.mlp=nn.Module()
        layer.mlp.up_proj=nn.Linear(32,256,bias=False)
        layer.mlp.down_proj=nn.Linear(256,32,bias=False)
        self.final_norm=nn.LayerNorm(32)
        # The output module name deliberately does not say 'head'.
        self.decoder=nn.Linear(32,128,bias=False)
        self.decoder.weight=self.backbone.embed_tokens.weight

    def get_output_embeddings(self):
        return self.decoder

    def forward(self, token_ids):
        h=self.backbone.embed_tokens(token_ids)
        x=self.backbone.layers[0]
        h=h+x.self_attn.q_proj(h)
        h=h+x.mlp.down_proj(torch.tanh(x.mlp.up_proj(h)))
        return self.decoder(self.final_norm(h))


class UntiedHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(80,32)
        self.lm_head=nn.Linear(32,80,bias=False)
        self.fc=nn.Linear(32,32)
    def forward(self, x):
        h=self.fc(self.emb(x))
        return self.lm_head(h)


@pytest.fixture
def mixed():
    torch.manual_seed(72)
    return UniversalAdapter(OtherArchitecture(),'model-X','rev',kind='loha',rank=16,
                            target_scope='auto',wide_lora_rank=4)


def test_model_agnostic_tied_head_and_rectangular_layers(mixed):
    by_name={g.canonical:g for g in mixed.groups}
    assert 'backbone.embed_tokens.weight' in by_name
    tied=by_name['backbone.embed_tokens.weight']
    assert set(tied.aliases)=={'backbone.embed_tokens.weight','decoder.weight'}
    assert tied.overlay.local.kind=='lora' and tied.overlay.local.rank==4
    assert mixed.model.decoder.parametrizations.weight[0] is mixed.model.backbone.embed_tokens.parametrizations.weight[0]
    assert by_name['backbone.layers.0.self_attn.q_proj.weight'].overlay.local.kind=='loha'
    assert by_name['backbone.layers.0.self_attn.q_proj.weight'].overlay.local.rank==16
    assert by_name['backbone.layers.0.mlp.up_proj.weight'].overlay.local.kind=='lora'
    assert by_name['backbone.layers.0.mlp.up_proj.weight'].overlay.local.rank==4
    assert by_name['backbone.layers.0.mlp.down_proj.weight'].overlay.local.kind=='lora'
    assert 'final_norm.weight' not in by_name
    assert not mixed.model.final_norm.weight.requires_grad
    assert not any(p.requires_grad for p in mixed.model.parameters()
                   if id(p) not in {id(t) for t in mixed.trainable_parameters()})
    sample=torch.tensor([[1,2,3]])
    loss=mixed.model(sample).square().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in mixed.trainable_parameters())


def test_unshared_output_head_also_low_rank():
    adapter=UniversalAdapter(UntiedHeads(),'m','r',kind='loha',rank=16,target_scope='auto')
    groups={g.canonical:g.overlay.local for g in adapter.groups}
    assert groups['emb.weight'].kind=='lora' and groups['emb.weight'].rank==4
    assert groups['lm_head.weight'].kind=='lora' and groups['lm_head.weight'].rank==4
    assert groups['fc.weight'].kind=='loha' and groups['fc.weight'].rank==16
    assert len({id(g.overlay) for g in adapter.groups})==len(adapter.groups)


def test_mixed_effective_snapshot_restores_without_rank_mismatch_and_anchor_stale(tmp_path):
    torch.manual_seed(17)
    source=OtherArchitecture()
    receiving=deepcopy(source)
    src=UniversalAdapter(source,'m','r',kind='loha',rank=16,target_scope='auto')
    dest=UniversalAdapter(receiving,'m','r',kind='loha',rank=16,target_scope='auto')
    enable_anchors(dest)
    # Ensure the pre-snapshot local and anchor differ, so stale anchor changes
    # model outputs after a snapshot even if the rank mismatch were fixed.
    with torch.no_grad():
        for g in src.groups:
            g.overlay.local.parts[-1].add_(0.025)
        for g in dest.groups:
            g.overlay.local.parts[-1].add_(0.04)
    x=torch.tensor([[3,4,5]])
    before=source(x).detach().clone()
    path=tmp_path/'mixed.fstate'
    export_state(source,path,model_id='m',revision='r',peer_id='alice',adapter=src)
    apply_effective_state(receiving,dest,path,model_id='m',revision='r')
    torch.testing.assert_close(receiving(x),before,atol=1e-5,rtol=1e-5)
    for group in dest.groups:
        if hasattr(group.overlay,'anchor'):
            for i,part in enumerate(group.overlay.local.parts):
                assert torch.equal(part,getattr(group.overlay.anchor,f'part{i}'))
    assert any(g.overlay.local.rank==4 for g in dest.groups)
    assert any(g.overlay.local.rank==16 for g in dest.groups)
    torch.testing.assert_close(receiving.backbone.embed_tokens.weight,
                               receiving.decoder.weight,rtol=0,atol=0)


def test_elastic_mixed_adapter_reset_does_not_global_rank_copy(tmp_path):
    torch.manual_seed(51)
    a=UniversalAdapter(OtherArchitecture(),'m','r',kind='loha',rank=16,target_scope='auto')
    b=UniversalAdapter(OtherArchitecture(),'m','r',kind='loha',rank=16,target_scope='auto')
    genesis=hashlib.sha256(b'elastic mixed').hexdigest()
    journal=EventJournal(tmp_path/'legacy.db',genesis)
    engine=ElasticEngine(b,'bob',genesis,journal,unsafe=True)
    # Legacy elastic receive path previously tried to reset rank-4 LoRA
    # factors using rank-16 LoHa global adapter settings.
    j_a=EventJournal(tmp_path/'alice.db',genesis)
    engine_a=ElasticEngine(a,'alice',genesis,j_a,unsafe=True)
    with torch.no_grad():
        for g in a.groups:g.overlay.local.parts[-1].add_(0.0001)
    wire=engine_a.seal_local(denominator=1,dtype='float32')
    assert engine.ingest(wire)
    engine.reset_factors()
    for group in b.groups:
        assert [tuple(p.shape) for p in group.overlay.local.parts] == group.overlay.local.expected_shapes()
    journal.close();j_a.close()


def test_synchronous_mixed_rank_commit_does_not_global_rank_copy():
    torch.manual_seed(99)
    m=OtherArchitecture()
    a=UniversalAdapter(deepcopy(m),'m','r',kind='loha',rank=16,target_scope='auto')
    b=UniversalAdapter(deepcopy(m),'m','r',kind='loha',rank=16,target_scope='auto')
    anchor='a'*64
    ra=FedAvgRound(a,'alice',['alice','bob'],anchor)
    rb=FedAvgRound(b,'bob',['alice','bob'],anchor)
    wa=ra.seal_local(wire_dtype='float32')
    wb=rb.seal_local(wire_dtype='float32')
    assert ra.receive('bob',wb)
    assert rb.receive('alice',wa)
    ra.commit();rb.commit()
    assert any(g.overlay.local.rank==4 for g in a.groups)
    assert any(g.overlay.local.rank==16 for g in a.groups)


def test_layout_rejects_mismatched_adapter_policy():
    base=UntiedHeads()
    auto=UniversalAdapter(deepcopy(base),'m','r',kind='loha',rank=16,target_scope='auto')
    all_layers=UniversalAdapter(deepcopy(base),'m','r',kind='loha',rank=16,target_scope='all')
    assert auto.layout!=all_layers.layout
    with pytest.raises(PatchError,match='incompatible: layout'):
        all_layers.import_remote(auto.export('alice',1))
