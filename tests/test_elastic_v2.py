import hashlib
from dataclasses import replace
import pytest
import torch

from patchlib.elastic_v2 import EpochBuffer, EpochProposal
from patchlib.core import PatchError

EPOCH='a'*64

def proposal(origin,seq,value,samples=1):
    return EpochProposal(origin,seq,EPOCH,samples,{'weight':torch.full((2,2),value,dtype=torch.float32)},hashlib.sha256(f'{origin}:{seq}:{value}'.encode()).hexdigest())

def buffer(cap=3):
    return EpochBuffer(EPOCH,{'weight':(2,2)},max_updates_per_peer=cap)

def calc(buffer_):
    return buffer_.materialize({'weight':torch.ones(2,2)},eta=1.,seal=True)['weight']

def test_order_invariance_8_peers():
    patches=[proposal(f'peer{i}',seq,(i+1)*0.01*seq,samples=seq) for i in range(8) for seq in (1,2,3)]
    a,b=buffer(),buffer()
    for p in patches:a.add(p)
    for p in reversed(patches):b.add(p)
    assert torch.equal(calc(a),calc(b))
    assert a.digest()==b.digest()


def test_mean_per_peer_prevents_fast_peer_dominance():
    b=buffer(cap=50)
    for i in range(40):b.add(proposal('fast',i+1,2.))
    b.add(proposal('slow',1,0.))
    assert torch.allclose(calc(b),torch.full((2,2),2.))  # base 1 + (2+0)/2


def test_weighted_samples_and_ridge():
    b=buffer()
    b.add(proposal('a',1,2,samples=1))
    b.add(proposal('a',2,6,samples=3))
    b.add(proposal('b',1,0,samples=1))
    assert torch.allclose(b.materialize({'weight':torch.ones(2,2)},ridge=1)['weight'],torch.full((2,2),1.+5./3.))


def test_cap_is_sequence_deterministic():
    b=buffer(cap=2)
    b.add(proposal('a',5,10));b.add(proposal('a',2,2));b.add(proposal('a',1,1))
    assert [p.seq for p in b.selected()['a']]==[1,2]
    assert torch.allclose(calc(b),torch.full((2,2),2.5))


def test_duplicate_equivocation_and_stale_epoch():
    b=buffer();p=proposal('a',1,1)
    assert b.add(p);assert b.add(p) is False
    with pytest.raises(PatchError,match='equivocation'):b.add(replace(p,digest='f'*64))
    with pytest.raises(PatchError,match='epoch'):b.add(replace(p,epoch='b'*64))


def test_rejects_malicious_delta_and_invalid_shapes():
    b=buffer()
    with pytest.raises(PatchError,match='nonfinite'):b.add(replace(proposal('a',1,1),deltas={'weight':torch.full((2,2),float('nan'))}))
    with pytest.raises(PatchError,match='shape'):b.add(replace(proposal('a',1,1),deltas={'weight':torch.ones(3,3)}))
    assert not b.events


def test_seal_and_min_quorum():
    b=buffer();b.add(proposal('a',1,1))
    with pytest.raises(PatchError,match='participants'):b.materialize({'weight':torch.ones(2,2)},min_participants=2)
    calc(b)
    with pytest.raises(PatchError,match='sealed'):b.add(proposal('b',1,2))


def test_rejects_unbounded_model_and_event_count():
    with pytest.raises(ValueError,match='size cap'):EpochBuffer(EPOCH,{'large':(2000001,)})
    b=EpochBuffer(EPOCH,{'weight':(2,2)},max_events=1)
    b.add(proposal('a',1,1))
    with pytest.raises(PatchError,match='event cap'):b.add(proposal('b',1,1))
