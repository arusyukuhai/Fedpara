"""v78 root-pinned, certified-epoch Critic stream failure/replay cases."""
from types import SimpleNamespace
import torch
import pytest
from patchlib.core import PatchError
from patchlib.distributed_critic_v78 import CriticGossip, pack, unpack
from patchlib.mixed_learning_v65 import LocalValueHead
from patchlib.critic_epochs_v78 import verified_epochs
from patchlib.online_generation import _digest
from patchlib.live_epoch_switch_v48 import certified_genesis

ROOT='a'*64
EPOCH1='b'*64
EPOCH2='c'*64

def make_peer(tmp_path,name,epochs=None,actor_epoch=ROOT):
    torch.manual_seed(44)
    head=LocalValueHead('linear',16)
    m=SimpleNamespace(critic=head,critic_optim=torch.optim.AdamW(head.parameters(),lr=.001),
        args=SimpleNamespace(critic_kind='linear',critic_hidden=16),
        _critic_identity='room-v78',cached=None,reuses=0)
    def sign(n,h):return 'sig:'+h
    def verify(info):
        if info['sig']!='sig:'+info['sha256']:raise PatchError('bad signature')
    return CriticGossip(tmp_path/f'{name}.sqlite3',ROOT,m,name,signer=sign,
        trusted={x:verify for x in ('alpha','beta')},
        allowed_actor_epochs=epochs or {ROOT},actor_epoch=actor_epoch)

def step(peer,amount):
    before=[p.detach().clone() for p in peer.params]
    with torch.no_grad():
        for p in peer.params:p.add_(amount)
    return peer.record_local(before,actor_tip=[0,None])


def test_epoch_signed_root_journal_continues_after_certified_switch(tmp_path):
    a=make_peer(tmp_path,'alpha');b=make_peer(tmp_path,'beta')
    try:
        first=step(a,.02)
        assert b.ingest(first)
        saved=a.genesis
        a.register_certified_epoch(EPOCH1,ROOT)
        assert a.genesis==saved
        second=step(a,.03)
        # Even a valid signature cannot introduce uncertified Actor generation.
        old=[x.detach().clone() for x in b.params]
        with pytest.raises(PatchError,match='uncertified'):
            b.ingest(second)
        assert b.journal.count()==1
        assert all(torch.equal(p,q) for p,q in zip(b.params,old))
        b.register_certified_epoch(EPOCH1,ROOT)
        assert b.ingest(second)
        assert not b.ingest(second)
        for x,y in zip(a.params,b.params):torch.testing.assert_close(x,y)
        assert a.journal.count()==b.journal.count()==2
        b.register_certified_epoch(EPOCH2,EPOCH1)
        third=step(b,.01)
        a.register_certified_epoch(EPOCH2,EPOCH1)
        assert a.ingest(third)
        for x,y in zip(a.params,b.params):torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-5)
    finally:
        a.close();b.close()
    a2=make_peer(tmp_path,'alpha',epochs={ROOT,EPOCH1,EPOCH2},actor_epoch=EPOCH2)
    try:
        assert a2.journal.count()==3
        for x,y in zip(a2.params,old):assert not torch.equal(x,y)
        assert a2.actor_epoch==EPOCH2
    finally:a2.close()


def test_missing_epoch_blocks_replay_instead_of_partial_restore(tmp_path):
    a=make_peer(tmp_path,'alpha')
    try:
        a.register_certified_epoch(EPOCH1,ROOT)
        step(a,.01)
    finally:a.close()
    with pytest.raises(PatchError,match='uncertified'):
        make_peer(tmp_path,'alpha')


def test_signed_actor_epoch_cannot_be_tampered(tmp_path):
    a=make_peer(tmp_path,'alpha')
    try:
        wire=step(a,.01)
        bad=wire.replace(b'"actor_epoch":"'+ROOT.encode()+b'"',
            b'"actor_epoch":"'+EPOCH1.encode()+b'"')
        assert bad!=wire
        with pytest.raises(PatchError):unpack(bad,a.genesis,a.params,a.trusted,
            allowed_actor_epochs={ROOT,EPOCH1})
        with pytest.raises(PatchError,match='requires a certified'):
            pack('alpha',3,a.genesis,[torch.zeros_like(x) for x in a.params],a.signer)
    finally:a.close()


def test_unauthorized_epoch_registration_and_tampered_head(tmp_path):
    a=make_peer(tmp_path,'alpha')
    try:
        with pytest.raises(PatchError):a.register_certified_epoch(EPOCH2,EPOCH1)
        assert a.actor_epoch==ROOT
        with pytest.raises(PatchError):a.register_certified_epoch('nothex',ROOT)
    finally:a.close()


def test_certified_epoch_ancestry_requires_no_gap_or_fork():
    # Use structural fake proposals; the helper checks certificate integrity
    # through ledger.certified_epoch (real ledger signature verification is
    # independently tested in existing generation-certificate suites).
    from patchlib.online_generation import make_proposal
    proposals=[];parent=ROOT
    for epoch in (1,2):
        p=make_proposal(room='x',epoch=epoch,parent=parent,manifest='d'*64,
            semantic='e'*64,cut='f'*64,cutoffs={},replicas=['alpha','beta'])
        proposals.append(p);parent=_digest(p)
    ledger=SimpleNamespace(certified_epoch=lambda i:{'proposal':proposals[i-1]})
    target=certified_genesis(proposals[-1])
    known=verified_epochs(ledger,ROOT,2,target)
    assert known=={ROOT,certified_genesis(proposals[0]),target}
    bad=SimpleNamespace(certified_epoch=lambda i:None if i==1 else {'proposal':proposals[i-1]})
    with pytest.raises(PatchError):verified_epochs(bad,ROOT,2,target)
    proposals[1]=dict(proposals[1],parent='0'*64)
    with pytest.raises(PatchError):verified_epochs(ledger,ROOT,2,target)


def test_v78_policy_and_network_are_version_isolated():
    from pathlib import Path
    import patchlib.elasticnet as net
    source=Path(net.__file__).read_text()
    assert "critic_v78=True" in source
    assert "b'KVC78\\0'" in source
    assert ".critic-v78.sqlite3" in source
    assert "register_certified_epoch(new.genesis,archive.source_genesis)" in source
    assert "args._v65_mixer_adapter=adapter" in source
    assert 'critic_v77=True' not in source

def test_invalid_local_state_rejected_before_durable_log_append(tmp_path):
    a=make_peer(tmp_path,'alpha');b=make_peer(tmp_path,'beta')
    try:
        delta=[torch.full_like(p,.1,dtype=torch.float32) for p in b.params]
        wire=pack('alpha',1,b.genesis,delta,a.signer,actor_epoch=ROOT)
        with torch.no_grad():b.params[0].fill_(float('nan'))
        with pytest.raises(PatchError,match='overflow'):
            b.ingest(wire)
        assert b.journal.count()==0
        # Explicitly verify float32 math can overflow on target conversion,
        # independently of the stricter <=10 delta bound on signed events.
        for p in b.params:p.data=p.data.half()
        with torch.no_grad():
            for p in b.params:p.fill_(65504.)
        with pytest.raises(PatchError,match='target dtype'):
            b._propose([torch.full_like(p,100,dtype=torch.float32) for p in b.params])
    finally:a.close();b.close()
