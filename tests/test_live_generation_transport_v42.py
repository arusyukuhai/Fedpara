import hashlib
import json
from pathlib import Path
import sqlite3
import pytest

from patchlib.core import PatchError, _canonical_json
from patchlib.identity import generate_keypair, signer_from_pem, verifier_from_pem
from patchlib.online_generation import GenerationLedger,AutomaticGenerationControl,make_proposal,mark_local_generation_loaded
from patchlib.live_generation_transport import GenerationWebRTCBridge
from patchlib.live_generation_adoption import OnlineAdoptionStore


def fixture(tmp_path):
    keys={};verifiers={};members=['committee-a','committee-b','committee-c']
    for k in members:
        private=tmp_path/f'{k}.pem';public=tmp_path/f'{k}.pub'
        generate_keypair(private,public)
        keys[k]=signer_from_pem(private);verifiers[k]=verifier_from_pem(public)
    ledgers={k:GenerationLedger(tmp_path/f'{k}.db',room='room',peer=k,
      committee=members,signer=keys[k],trusted=verifiers,genesis='0'*64) for k in members}
    control={k:AutomaticGenerationControl(ledgers[k],lambda _:True,lambda _:True) for k in members}
    return ledgers,control


class Node:
    verified_at=1.0
    remote_supports_generation=True
    def __init__(self):self.wires=[];self.connected=True
    def control(self,kind,**data):
        if not self.connected:return False
        self.wires.append((kind,data));return True


def move(src:GenerationWebRTCBridge,dst:GenerationWebRTCBridge, src_peer,dst_peer,now):
    node=Node();src.tick([(dst_peer,node)],now=now)
    for typ,wire in node.wires:dst.receive(src_peer,typ,wire)
    return node.wires


def test_certificate_p2p_relay_and_offline_rejoin(tmp_path):
    ledgers,controllers=fixture(tmp_path)
    bridges={k:GenerationWebRTCBridge(controllers[k],interval=.2) for k in ledgers}
    p=make_proposal(room='room',epoch=1,parent='0'*64,manifest='a'*64,semantic='b'*64,
                    cut='c'*64,cutoffs={'alice':10},replicas=list(ledgers))
    # p2p proposal travels over signed outer link. Inner signed votes are checked.
    controllers['committee-a'].propose(p)
    for step in range(3):
        move(bridges['committee-a'],bridges['committee-b'],'committee-a','committee-b',now=step+1.)
        move(bridges['committee-a'],bridges['committee-c'],'committee-a','committee-c',now=step+1.)
        move(bridges['committee-b'],bridges['committee-a'],'committee-b','committee-a',now=step+1.)
        move(bridges['committee-c'],bridges['committee-a'],'committee-c','committee-a',now=step+1.)
    assert ledgers['committee-a'].latest() is not None
    # A disconnected ordinary node does not need to acknowledge the rollover.
    late=GenerationLedger(tmp_path/'late.db',room='room',peer='offline',committee=list(ledgers),
                          signer=None,trusted=ledgers['committee-a'].trusted,genesis='0'*64)
    late_bridge=GenerationWebRTCBridge(AutomaticGenerationControl(late,lambda _:False,lambda _:False),interval=.2)
    move(bridges['committee-a'],late_bridge,'committee-a','offline',now=5.)
    move(late_bridge,bridges['committee-a'],'offline','committee-a',now=6.)
    move(bridges['committee-a'],late_bridge,'committee-a','offline',now=7.)
    assert late.latest()['proposal']==p
    # An equal-height fork advertisement cannot silently select a competing base.
    with pytest.raises(PatchError,match='fork'):
        late_bridge.receive('attack','generation-head',{'epoch':1,'digest':'f'*64})
    for ledger in [*ledgers.values(),late]:ledger.close()


def test_offline_requests_not_lost_on_backpressure(tmp_path):
    ledgers,ctl=fixture(tmp_path)
    br=GenerationWebRTCBridge(ctl['committee-a'],interval=.2)
    br.receive('late','generation-head',{'epoch':1,'digest':'a'*64})
    node=Node();node.connected=False
    assert br.tick([('late',node)],now=1.)==0
    node.connected=True
    assert br.tick([('late',node)],now=2.)>0
    assert any(k=='generation-need' for k,_ in node.wires)
    for l in ledgers.values():l.close()


def test_online_adoption_carries_suffix_and_refuses_unsafe_gc(tmp_path):
    import torch
    from patchlib.elastic import EventJournal
    from patchlib.shards import export_state_sharded,read_manifest,semantic_shard_sha256
    ledgers,ctl=fixture(tmp_path)
    old=tmp_path/'old.sqlite3'
    j=EventJournal(old,'d'*64,max_bytes=0)
    j.append('alice',1,b'first');j.append('alice',2,b'second')
    # A certified frontier covers ONLY the first event; second must be rebased.
    cut=hashlib.sha256()
    r=j.db.execute("SELECT origin,seq,digest FROM events WHERE origin='alice' AND seq=1").fetchone()
    cut.update(_canonical_json(list(r))+b'\n')
    store=tmp_path/'store';manifest=tmp_path/'snapshot.fmanifest'
    export_state_sharded(torch.nn.Linear(2,3),store,manifest,model_id='test/model',revision='1',
                         peer_id='alice',seq=1,shard_bytes=65536)
    semantic=semantic_shard_sha256(read_manifest(manifest))
    proposal=make_proposal(room='room',epoch=1,parent='0'*64,
            manifest=hashlib.sha256(manifest.read_bytes()).hexdigest(),semantic=semantic,
            cut=cut.hexdigest(),cutoffs={'alice':1},replicas=list(ledgers))
    votes=[ledgers[p].vote(proposal,verified_snapshot=True,verified_cut=True) for p in list(ledgers)[:2]]
    ledgers['committee-a'].accept(ledgers['committee-a'].certificate(proposal,votes))
    adoption=OnlineAdoptionStore(tmp_path/'handoff.db',ledgers['committee-a'])
    assert adoption.prepare(old)['pending']==1
    # Crash/restart keeps the pending event index.
    adoption.close();adoption=OnlineAdoptionStore(tmp_path/'handoff.db',ledgers['committee-a'])
    assert adoption.status(1)['pending']==1
    with pytest.raises(PatchError):adoption.request_gc(1,old)
    j.close()
    new=tmp_path/'new.sqlite3'
    nj=EventJournal(new,'e'*64,max_bytes=0);nj.close()
    mark_local_generation_loaded(ledger=ledgers['committee-a'],new_journal=new,
           manifest=manifest,store=store,weights_loaded_and_verified=True)
    adoption.mark_active(epoch=1,manifest=manifest,new_journal=new,
                         load_confirmed=True,new_journal_confirmed=True)
    assert not adoption.gc_eligible(1,old)
    sha=j if False else hashlib.sha256(b'second').hexdigest()
    receipt=hashlib.sha256(b'new authenticated event receipt').hexdigest()
    adoption.record_carry(epoch=1,origin='alice',seq=2,digest=sha,new_event_receipt=receipt)
    assert not adoption.gc_eligible(1,old)  # receipt lacks cryptographic rebase proof
    # A late-arriving third update invalidates the old scan even if 2 was carried.
    j=EventJournal(old,'d'*64,max_bytes=0);j.append('alice',3,b'third');j.close()
    assert not adoption.gc_eligible(1,old)
    adoption.prepare(old)
    sha3=hashlib.sha256(b'third').hexdigest()
    adoption.record_carry(epoch=1,origin='alice',seq=3,digest=sha3,
                          new_event_receipt=hashlib.sha256(b'new3').hexdigest())
    with pytest.raises(PatchError):adoption.request_gc(1,old)
    assert old.exists()
    assert new.exists() and manifest.exists()
    adoption.close()
    for l in ledgers.values():l.close()


def test_no_control_frame_can_contain_unbounded_payload(tmp_path):
    from patchlib.live_generation_transport import validate_control
    with pytest.raises(PatchError):validate_control('generation-wire',{'payload':{'kind':'proposal','data':{'blob':'x'*20000}}})
    with pytest.raises(PatchError):validate_control('generation-need',{'after':-1})


def test_generation_control_real_elastic_node_authenticated_and_negotiated(tmp_path):
    import asyncio
    from patchlib.elasticnet import ElasticNode,encode_control
    from patchlib.roomid import auto_identity
    from tests.test_snapshot_request_delivery import FakeChannel
    async def run():
        a,sa,pa,_=auto_identity(tmp_path/'aa.pem')
        b,sb,pb,_=auto_identity(tmp_path/'bb.pem')
        node=ElasticNode('ws://localhost/ws','x'*32,origin=a,remote=b,
                 genesis='4'*64,public_key=pa,signer=sa,enable_generation=True)
        ch=FakeChannel();node._attach_channel(ch);ch.handlers['open']()
        ch.handlers['message'](encode_control({'type':'hello','peer':b,'genesis':'4'*64,
            'counter':1,'pub':pb,'generation_v1':True},sb))
        assert node.remote_supports_generation
        ch.handlers['message'](encode_control({'type':'generation-head','peer':b,'genesis':'4'*64,
              'counter':2,'epoch':3,'digest':'a'*64},sb))
        assert node.generation_controls.get_nowait()==('generation-head',{'epoch':3,'digest':'a'*64})
        assert node.control('generation-need',after=0)
        assert any('generation-need' in str(x) for x in ch.sent)
        ch.close()
        await asyncio.sleep(.01)
    asyncio.run(run())


def test_checkpoint_request_uses_certificate_digest_only(tmp_path):
    ls,controls=fixture(tmp_path)
    p=make_proposal(room='room',epoch=1,parent='0'*64,manifest='a'*64,semantic='b'*64,
                    cut='c'*64,cutoffs={'alice':1},replicas=list(ls))
    votes=[ls[k].vote(p,verified_snapshot=True,verified_cut=True) for k in list(ls)[:2]]
    ls['committee-a'].accept(ls['committee-a'].certificate(p,votes))
    bridge=GenerationWebRTCBridge(controls['committee-a'],interval=.2,
                                 snapshot_available=lambda digest:False)
    node=Node();bridge.tick([('committee-b',node)],now=15)
    assert ('generation-snapshot-request',{'epoch':1,'manifest':'a'*64}) in node.wires
    for l in ls.values():l.close()


def test_many_disconnected_peers_do_not_accumulate_unbounded_ram(tmp_path):
    ls,controls=fixture(tmp_path)
    bridge=GenerationWebRTCBridge(controls['committee-a'])
    for i in range(2000):
        bridge.receive('peer-'+str(i),'generation-head',
                       {'epoch':0,'digest':'0'*64})
    assert len(bridge.neighbors)<=256
    for l in ls.values():l.close()


def test_automatic_gc_only_after_activated_empty_suffix(tmp_path):
    import torch
    from patchlib.elastic import EventJournal
    from patchlib.shards import export_state_sharded,read_manifest,semantic_shard_sha256
    ls,_=fixture(tmp_path)
    old=tmp_path/'old.db';j=EventJournal(old,'d'*64,max_bytes=0)
    j.append('alice',1,b'covered')
    rec=j.db.execute('SELECT origin,seq,digest FROM events').fetchone()
    cut=hashlib.sha256(_canonical_json(list(rec))+b'\n').hexdigest()
    j.close()
    manifest=tmp_path/'base.fmanifest';store=tmp_path/'obj'
    export_state_sharded(torch.nn.Linear(2,3),store,manifest,model_id='demo',
                         revision='1',peer_id='alice',seq=1,shard_bytes=65536)
    p=make_proposal(room='room',epoch=1,parent='0'*64,
        manifest=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        semantic=semantic_shard_sha256(read_manifest(manifest)),cut=cut,
        cutoffs={'alice':1},replicas=list(ls))
    votes=[ls[k].vote(p,verified_snapshot=True,verified_cut=True) for k in list(ls)[:2]]
    ls['committee-a'].accept(ls['committee-a'].certificate(p,votes))
    adoption=OnlineAdoptionStore(tmp_path/'adoption.db',ls['committee-a'])
    adoption.prepare(old)
    assert adoption.collect_ready({1:old})==[]
    new=tmp_path/'new.db';j=EventJournal(new,'e'*64,max_bytes=0);j.close()
    mark_local_generation_loaded(ledger=ls['committee-a'],new_journal=new,manifest=manifest,
                                 store=store,weights_loaded_and_verified=True)
    adoption.mark_active(epoch=1,manifest=manifest,new_journal=new,
                         load_confirmed=True,new_journal_confirmed=True)
    assert adoption.collect_ready({1:old})==[1]
    assert not old.exists() and new.exists()
    adoption.close()
    for l in ls.values():l.close()
