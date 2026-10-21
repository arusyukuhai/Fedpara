import copy
import hashlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from patchlib.core import PatchError, UniversalAdapter, decode_patch, encode_patch
from patchlib.livefed import FedAvgRound, _delta_block, parse_links, parse_peer_keys, round_anchor


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(9, 6)
        self.proj = nn.Linear(6, 9, bias=True)
        self.proj.weight = self.embed.weight
        self.conv = nn.Conv2d(2, 4, (3, 3))
        self.high_order = nn.Parameter(torch.randn(4, 5, 6))
        self.scalar = nn.Parameter(torch.tensor(1.0))

    def forward(self):
        x = self.embed(torch.tensor([[1, 2]]))
        return (self.proj(x).square().mean() + self.conv(torch.randn(1,2,5,5)).square().mean()
                + self.high_order.square().mean() + self.scalar.square())


def make_engines(n=2, kind='loha'):
    torch.manual_seed(1)
    m = Tiny()
    anchor = hashlib.sha256(b'initial identical state').hexdigest()
    peers = [f'p{i}' for i in range(n)]
    engines = []
    for p in peers:
        model = copy.deepcopy(m)
        adapter = UniversalAdapter(model,'shared','fstate-sha256:fixed',kind=kind,rank=3)
        engines.append(FedAvgRound(adapter,p,peers,anchor,max_delta_rms=100))
    return engines


@pytest.mark.parametrize('kind', ['lora','loha'])
@pytest.mark.parametrize('num_peers', [2,3])
def test_all_tensor_shapes_tied_and_fedavg(kind,num_peers):
    engines = make_engines(num_peers,kind)
    before = {}
    patches = {}
    deltas = {}
    for i,eng in enumerate(engines):
        for group in eng.adapter.groups:
            before[group.canonical] = group.overlay.local.delta().detach().clone() * 0
            with torch.no_grad():
                # Nonzero LoRA and LoHA factors, all tensor ranks and scalar.
                for j,part in enumerate(group.overlay.local.parts):
                    part.fill_(0.06 + 0.01*i + 0.002*j)
        patches[eng.self_id] = eng.seal_local()
        deltas[eng.self_id] = {g.canonical:g.overlay.local.delta().detach().clone()
                               for g in eng.adapter.groups}
    for i,eng in enumerate(engines):
        orig = {}
        for g in eng.adapter.groups:
            path,_,name=g.aliases[0].rpartition('.')
            orig[g.canonical]=eng.adapter.model.get_submodule(path).parametrizations[name].original.detach().clone()
        for peer in reversed([e.self_id for e in engines]):
            assert eng.receive(peer,patches[peer]) == (peer != eng.self_id)
            assert eng.receive(peer,patches[peer]) is False
        assert eng.complete
        result = eng.commit()
        assert result.round == 0
        assert eng.round == 1
        assert not eng.patches
        for g in eng.adapter.groups:
            path,_,name=g.aliases[0].rpartition('.')
            actual=eng.adapter.model.get_submodule(path).parametrizations[name].original
            expected=orig[g.canonical]+sum(deltas[p][g.canonical] for p in patches)/num_peers
            assert torch.allclose(actual,expected,atol=2e-6,rtol=2e-6),g.canonical
            assert torch.allclose(g.overlay.local.delta(),torch.zeros_like(g.overlay.local.delta()),atol=1e-7)
        if i:
            assert result.anchor == engines[0].anchor
        # Same tied Parameter identity preserved through in-place commit.
        assert eng.adapter.model.embed.parametrizations.weight.original is eng.adapter.model.proj.parametrizations.weight.original
    # Next round can continue immediately with new factors and new anchor.
    for eng in engines:
        assert eng.seal_local() and eng.round == 1


def test_stale_update_wrong_base_wrong_roster_and_equivocation():
    a,b=make_engines()
    payload=a.seal_local()
    assert b.receive('p0',payload)
    assert not b.receive('p0',payload)
    meta,tensors=decode_patch(payload)
    meta['base_anchor']='0'*64
    with pytest.raises(PatchError,match='base/participant'):
        b.receive('p0',encode_patch(meta,tensors))
    meta['base_anchor']=a.anchor
    meta['round']=3
    with pytest.raises(PatchError,match='round'):
        b.receive('p0',encode_patch(meta,tensors))
    meta['round']=0
    tkey=next(iter(tensors)); tensors[tkey]=tensors[tkey]+0.01
    with pytest.raises(PatchError,match='conflicting'):
        b.receive('p0',encode_patch(meta,tensors))
    with pytest.raises(PatchError,match='sender'):
        b.receive('p1',payload)
    with pytest.raises(PatchError,match='one contribution'):
        b.commit()


def test_large_update_preflight_no_mutation():
    a,b=make_engines()
    with torch.no_grad():
        for group in a.adapter.groups:
            for part in group.overlay.local.parts:
                part.fill_(500.0)
    pa=a.seal_local()
    b.seal_local()
    b.receive('p0',pa)
    before={g.canonical:g.overlay.local.parts[0].detach().clone() for g in b.adapter.groups}
    original={}
    for g in b.adapter.groups:
        path,_,name=g.aliases[0].rpartition('.')
        original[g.canonical]=b.adapter.model.get_submodule(path).parametrizations[name].original.detach().clone()
    with pytest.raises(PatchError,match='excessive|nonfinite'):
        b.commit()
    assert b.round==0
    for g in b.adapter.groups:
        path,_,name=g.aliases[0].rpartition('.')
        base=b.adapter.model.get_submodule(path).parametrizations[name].original
        assert torch.equal(base,original[g.canonical])
        assert torch.equal(before[g.canonical],g.overlay.local.parts[0])


def test_nonfinite_factors_rejected_on_receive():
    a,b=make_engines()
    with torch.no_grad():
        a.adapter.groups[0].overlay.local.parts[0].fill_(float('inf'))
    with pytest.raises(PatchError,match='nonfinite'):
        b.receive('p0',a.seal_local())


def test_one_peer_per_round_and_validator():
    engines=make_engines(3)
    c=engines[0]
    c.seal_local()
    with pytest.raises(PatchError,match='one contribution'):
        c.commit()
    with pytest.raises(ValueError):
        parse_links(['p0=room0000000000000'],'p0')
    with pytest.raises(ValueError):
        parse_links(['p1=too-short'],'p0')
    assert parse_links(['p1=pair_room_123456789'],'p0') == {'p1':'pair_room_123456789'}
    assert parse_peer_keys(['p1=one.pem'])['p1'].name=='one.pem'
