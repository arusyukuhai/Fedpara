from pathlib import Path
import copy
import pytest

from patchlib.core import PatchError, _canonical_json
from patchlib.frontier_commitment_v44 import (
    build_frontier_pages, verify_frontier_page, verify_full_frontier, header_digest)
from patchlib.identity import generate_keypair, signer_from_pem, verifier_from_pem
from patchlib.online_generation import GenerationLedger, AutomaticGenerationControl, make_proposal
from patchlib.live_generation_transport import GenerationWebRTCBridge


class DummyNode:
    verified_at=1
    remote_supports_generation=True
    def __init__(self):self.wires=[];self.ready=True
    def control(self, kind, **fields):
        if not self.ready:return False
        self.wires.append((kind, fields));return True


def bridge(tmp_path):
    members=['a','b','c']; verifiers={}; keys={}
    for k in members:
        prv=tmp_path/f'{k}.pem';pub=tmp_path/f'{k}.pub'
        generate_keypair(prv,pub)
        keys[k]=signer_from_pem(prv)
        verifiers[k]=verifier_from_pem(pub)
    ledger=GenerationLedger(tmp_path/'a.sqlite3',room='room',peer='a',
        committee=members,signer=keys['a'],trusted=verifiers,genesis='0'*64)
    ctl=AutomaticGenerationControl(ledger,lambda _: True,lambda _: True)
    return GenerationWebRTCBridge(ctl,interval=.2),ledger,keys


def test_idle_gossip_is_sparse_and_reconnect_syncs_immediately(tmp_path):
    b,ledger,keys=bridge(tmp_path)
    n=DummyNode()
    for tick in range(200):
        b.tick([('neighbor',n)],now=tick*.2+1)
    # With old 0.6-sec ticker this would emit about 67 messages.
    heads=[kind for kind,_ in n.wires if kind=='generation-head']
    assert len(heads)==2
    b.disconnect('neighbor')
    n2=DummyNode()
    b.tick([('neighbor',n2)],now=41.2)
    assert n2.wires[0][0]=='generation-head'
    ledger.close()


def test_changed_certificate_triggers_immediate_head_not_long_poll(tmp_path):
    b,ledger,keys=bridge(tmp_path)
    n=DummyNode();b.tick([('neighbor',n)],now=1)
    p=make_proposal(room='room',epoch=1,parent='0'*64,manifest='a'*64,
        semantic='b'*64,cut='c'*64,cutoffs={'alice':1},replicas=['a','b','c'])
    votes=[b.control.ledger.vote(p,verified_snapshot=True,verified_cut=True)]
    # Synthetic second vote from pinned b through an independent ledger.
    bledger=GenerationLedger(tmp_path/'b.sqlite3',room='room',peer='b',
        committee=['a','b','c'],signer=keys['b'],trusted=ledger.trusted,genesis='0'*64)
    votes.append(bledger.vote(p,verified_snapshot=True,verified_cut=True))
    ledger.accept(ledger.certificate(p,votes))
    before=len(n.wires)
    b.tick([('neighbor',n)],now=1.1)
    assert any(kind=='generation-head' and fields['epoch']==1 for kind,fields in n.wires[before:])
    ledger.close();bledger.close()


def test_snapshot_presence_checked_once_per_tick_for_multiple_neighbors(tmp_path):
    b,ledger,keys=bridge(tmp_path)
    calls=[]
    b.snapshot_available=lambda digest: calls.append(digest) or True
    p=make_proposal(room='room',epoch=1,parent='0'*64,manifest='a'*64,
        semantic='b'*64,cut='c'*64,cutoffs={'alice':1},replicas=['a','b','c'])
    lb=GenerationLedger(tmp_path/'b.sqlite3',room='room',peer='b',
        committee=['a','b','c'],signer=keys['b'],trusted=ledger.trusted,genesis='0'*64)
    votes=[l.vote(p,verified_snapshot=True,verified_cut=True) for l in (ledger,lb)]
    ledger.accept(ledger.certificate(p,votes))
    nodes=[('p'+str(i),DummyNode()) for i in range(50)]
    b.tick(nodes,now=1)
    assert calls==['a'*64]
    ledger.close();lb.close()


def test_merkle_frontier_thousands_of_peers_constant_certificate():
    frontier={'peer%06d'%i:i*7 for i in range(4096)}
    header,pages=build_frontier_pages(frontier)
    assert header['total']==4096
    assert len(_canonical_json(header))<256
    assert len(pages)==128
    assert len(_canonical_json(pages[10]))<10000
    assert len(header_digest(header))==64
    assert verify_full_frontier(header,pages)==frontier
    assert verify_frontier_page(header,pages[10])==pages[10]['entries']
    bad=copy.deepcopy(pages)
    bad[40]['entries'][0][1]+=1
    with pytest.raises(PatchError):verify_full_frontier(header,bad)
    with pytest.raises(PatchError):verify_full_frontier(header,pages[:-1])
    bad=copy.deepcopy(pages)
    bad[0]['siblings'][0]='0'*64
    with pytest.raises(PatchError):verify_full_frontier(header,bad)


def test_empty_frontier_and_bad_sequence():
    h,p=build_frontier_pages({})
    assert verify_full_frontier(h,p)=={}
    with pytest.raises(PatchError):build_frontier_pages({'peer':-1})
    with pytest.raises(PatchError):build_frontier_pages({'peer':True})


def test_eight_neighbors_thirty_seconds_no_idle_flood(tmp_path):
    b,ledger,_=bridge(tmp_path)
    neighbors=[('peer-'+str(i),DummyNode()) for i in range(8)]
    for tick in range(151):
        b.tick(neighbors, now=1.+tick*.2)
    heads=sum(kind=='generation-head' for _,node in neighbors for kind,_ in node.wires)
    # ~8 initial + 8 at second 30, rather than old ~400 head frames.
    assert 8 <= heads <= 16
    ledger.close()


def test_missing_snapshot_retries_backoff(tmp_path):
    b,ledger,keys=bridge(tmp_path)
    p=make_proposal(room='room',epoch=1,parent='0'*64,manifest='a'*64,
        semantic='b'*64,cut='c'*64,cutoffs={'alice':1},replicas=['a','b','c'])
    lb=GenerationLedger(tmp_path/'b.sqlite3',room='room',peer='b',
        committee=['a','b','c'],signer=keys['b'],trusted=ledger.trusted,genesis='0'*64)
    votes=[l.vote(p,verified_snapshot=True,verified_cut=True) for l in (ledger,lb)]
    ledger.accept(ledger.certificate(p,votes))
    n=DummyNode()
    for tick in range(300):b.tick([('b',n)],now=1+tick*.2)
    requests=[x for x in n.wires if x[0]=='generation-snapshot-request']
    # Old code retried every 3s indefinitely (~20 requests / min).
    assert 4 <= len(requests) <= 6
    ledger.close();lb.close()
