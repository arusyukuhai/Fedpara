"""Offline comparison end-to-end via a tiny deterministic model (no HF download)."""
import base64
import json
import math
import sys
import types

import torch
from torch import nn
import torch.nn.functional as F

from patchlib.core import UniversalAdapter
from patchlib.elastic import EventJournal, pack_event
from patchlib.elastic_v2 import compare_legacy_swarm_trace
from patchlib.roomid import auto_identity
from patchlib.state import export_state, semantic_state_sha256


class TinyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        g=torch.Generator().manual_seed(8821)
        self.emb=nn.Embedding(64,8)
        self.head=nn.Linear(8,64,bias=False)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn((64,8),generator=g)*.01)
            self.head.weight.copy_(torch.randn((64,8),generator=g)*.01)

    @classmethod
    def from_pretrained(cls,*args,**kwargs):return cls()

    def forward(self,input_ids,labels):
        logits=self.head(self.emb(input_ids))
        return types.SimpleNamespace(loss=F.cross_entropy(logits.reshape(-1,64),labels.reshape(-1)))


def test_signed_trace_counterfactual_workflow(tmp_path,monkeypatch):
    m=types.ModuleType('transformers');m.GPT2LMHeadModel=TinyCausalModel
    monkeypatch.setitem(sys.modules,'transformers',m)
    root=tmp_path/'run';(root/'fixture/model').mkdir(parents=True)
    (root/'peer-00').mkdir(parents=True)
    (root/'scenario.json').write_text(json.dumps({'kind':'loha','rank':2}))
    model=TinyCausalModel()
    export_state(model,root/'fixture/genesis.fstate',model_id='fixture/model',revision='main',peer_id='p'+'1'*32)
    rev='fstate-sha256:'+semantic_state_sha256(root/'fixture/genesis.fstate')
    adapter=UniversalAdapter(model,'fixture/model',rev,kind='loha',rank=2)
    digest='3'*64
    journal=EventJournal(root/'peer-00/events.sqlite3',digest)
    for i in range(2):
        origin,sign,pub,_=auto_identity(root/f'p{i}.key')
        for seq in (1,2):
            with torch.no_grad():
                for group in adapter.groups:
                    # Both branches nonzero so the resulting update isn't zero.
                    for factor in group.overlay.local.parts:factor.fill_(.01*(seq+i+1))
            patch=adapter.export(origin,seq=seq,dtype='float32',extra_meta={
                'elastic_protocol':'fedpara-elastic-v1','genesis':digest,'denominator':2})
            wire=pack_event(patch,origin,signer=sign,public_key=pub)
            journal.append(origin,seq,wire)
    journal.close()
    export_state(model,root/'peer-00/final.fstate',model_id='fixture/model',revision=rev,
                 peer_id='p'+'1'*32,adapter=adapter)
    a=compare_legacy_swarm_trace(root,cap_per_peer=2)
    b=compare_legacy_swarm_trace(root,cap_per_peer=2)
    assert a['participants']==2 and a['selected_events']==4
    assert a['digest']==b['digest']
    assert math.isfinite(a['heldout_loss']) and math.isfinite(a['v1_final_heldout_loss'])
    assert a['status']=='COUNTERFACTUAL_ONLY'


def test_signed_v2_proposal_codec(tmp_path):
    import pytest
    from patchlib.elastic_v2 import EpochBuffer, pack_epoch_proposal, ingest_signed_epoch_proposal
    from patchlib.core import PatchError
    m=TinyCausalModel()
    adapter=UniversalAdapter(m,'tiny','main',kind='lora',rank=2)
    epoch='a'*64
    peer,sign,pub,_=auto_identity(tmp_path/'sign.pem')
    shapes={g.canonical:g.shape for g in adapter.groups}
    buffer=EpochBuffer(epoch,shapes)
    wire=pack_epoch_proposal(adapter,peer,1,epoch,32,sign,pub,dtype='float32')
    assert ingest_signed_epoch_proposal(buffer,adapter,wire)
    assert not ingest_signed_epoch_proposal(buffer,adapter,wire)
    assert len(buffer.events)==1
    assert buffer.materialize({g.canonical:m.get_submodule(g.aliases[0].rpartition('.')[0]).parametrizations[g.aliases[0].rpartition('.')[2]].original.detach().cpu().clone() for g in adapter.groups})
    wrong=EpochBuffer('b'*64,shapes)
    with pytest.raises(PatchError,match='stale'):ingest_signed_epoch_proposal(wrong,adapter,wire)
