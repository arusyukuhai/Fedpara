"""v43 durable handoff and automatic *eligible* local log GC tests."""
from pathlib import Path
import hashlib
import sqlite3
import pytest
import torch

from patchlib.core import PatchError,_canonical_json
from patchlib.identity import generate_keypair, signer_from_pem, verifier_from_pem
from patchlib.elastic import EventJournal
from patchlib.shards import export_state_sharded,read_manifest,semantic_shard_sha256
from patchlib.online_generation import GenerationLedger,make_proposal,_digest
from patchlib.epoch_runtime_v43 import EpochRuntimeV43


def prepare(tmp_path, *, suffix=False):
    peers=['committee-a','committee-b','committee-c']
    keys={};trusted={}
    for p in peers:
        priv=tmp_path/(p+'.key');pub=tmp_path/(p+'.pub')
        generate_keypair(priv,pub)
        keys[p]=signer_from_pem(priv);trusted[p]=verifier_from_pem(pub)
    ledgers=[GenerationLedger(tmp_path/(p+'.gen.sqlite'),room='room',peer=p,
                              committee=peers,signer=keys[p],trusted=trusted,genesis='0'*64) for p in peers]
    old=tmp_path/'old.sqlite3'
    j=EventJournal(old,'d'*64,max_bytes=0)
    j.append('alice',1,b'first event')
    if suffix:j.append('alice',2,b'excluded suffix')
    cut=hashlib.sha256()
    origin,seq,digest=j.db.execute("SELECT origin,seq,digest FROM events WHERE origin='alice' AND seq=1").fetchone()
    cut.update(_canonical_json([origin,seq,digest])+b'\n')
    manifest=tmp_path/'state.fmanifest';store=tmp_path/'shards'
    export_state_sharded(torch.nn.Linear(2,3),store,manifest,model_id='example',revision='rev',peer_id='committee-a',seq=1)
    snap=read_manifest(manifest)
    sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
    p=make_proposal(room='room',epoch=1,parent='0'*64,manifest=sha,
                    semantic=semantic_shard_sha256(snap),cut=cut.hexdigest(),
                    cutoffs={'alice':1},replicas=peers)
    votes=[ledgers[i].vote(p,verified_snapshot=True,verified_cut=True) for i in range(2)]
    cert=ledgers[2].certificate(p,votes)
    for l in ledgers:l.accept(cert)
    # New trainer already owns an independently initialized new journal.
    replacement=tmp_path/'new.sqlite3'
    newer=EventJournal(replacement,'e'*64,max_bytes=0)
    with newer.db:
        for k,v in (('generation_certified_epoch','1'),('generation_certified_manifest',sha),
                    ('generation_certified_proposal',_digest(p))):
            newer.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(k,v))
    newer.close()
    return j,old,replacement,manifest,store,ledgers


def test_verified_epoch_can_retire_complete_previous_history(tmp_path):
    old_owner,old,new,manifest,store,ledgers=prepare(tmp_path)
    manager=EpochRuntimeV43(tmp_path/'runtime',ledgers[0])
    assert manager.stage(old)['state']=='quarantined'
    assert manager.stage(old)['state']=='quarantined'
    assert manager.poll()==[]
    with pytest.raises(PatchError,match='confirm'):
        manager.acknowledge_install(epoch=1,manifest=manifest,store=store,new_journal=new,loader_confirmation=False)
    assert manager.acknowledge_install(epoch=1,manifest=manifest,store=store,new_journal=new,loader_confirmation=True)['state']=='installed'
    assert manager.poll()==[]   # Old trainer still owns the exclusive lock.
    assert old.exists()
    old_owner.close()
    assert manager.poll()==[1]
    assert not old.exists()
    assert not Path(str(old)+'.fplog').exists()
    assert new.exists()
    assert manager.poll()==[]
    manager.close()
    for l in ledgers:l.close()


def test_uncovered_update_is_retained_even_when_new_worker_is_active(tmp_path):
    old_owner,old,new,manifest,store,ledgers=prepare(tmp_path,suffix=True)
    manager=EpochRuntimeV43(tmp_path/'runtime',ledgers[0])
    status=manager.stage(old)
    assert status['pending']==1
    manager.acknowledge_install(epoch=1,manifest=manifest,store=store,new_journal=new,loader_confirmation=True)
    old_owner.close()
    assert manager.poll()==[]
    assert old.exists()
    assert manager.status(1)['pending']==1
    manager.close()
    for l in ledgers:l.close()


def test_unverified_checkpoint_never_allows_retirement(tmp_path):
    old_owner,old,new,manifest,store,ledgers=prepare(tmp_path)
    manager=EpochRuntimeV43(tmp_path/'runtime',ledgers[0])
    manager.stage(old)
    manifest.write_bytes(manifest.read_bytes()+b'changed')
    with pytest.raises(PatchError,match='manifest'):
        manager.acknowledge_install(epoch=1,manifest=manifest,store=store,new_journal=new,loader_confirmation=True)
    old_owner.close()
    assert manager.poll()==[]
    assert old.exists()
    manager.close()
    for l in ledgers:l.close()


def test_runtime_state_survives_reopen_without_unsafe_gc(tmp_path):
    old_owner,old,new,manifest,store,ledgers=prepare(tmp_path,suffix=True)
    manager=EpochRuntimeV43(tmp_path/'runtime',ledgers[0]);manager.stage(old);manager.close()
    manager=EpochRuntimeV43(tmp_path/'runtime',ledgers[0]);
    assert manager.status(1)['pending']==1
    old_owner.close()
    assert manager.poll()==[]
    manager.close()
    for l in ledgers:l.close()


def test_any_certified_holder_can_advertise_and_relay_snapshot(tmp_path):
    from patchlib.online_generation import AutomaticGenerationControl
    from patchlib.live_generation_transport import GenerationWebRTCBridge
    class Node:
        verified_at=1.0
        remote_supports_generation=True
        def __init__(self):self.wires=[]
        def control(self,kind,**data):
            self.wires.append((kind,data))
            return True
    old_owner,old,new,manifest,store,ledgers=prepare(tmp_path)
    target=ledgers[0].latest()['proposal']['manifest']
    # This peer is NOT a designated snapshot replica; it has verified a copy.
    original_replicas=ledgers[0].latest()['proposal']['replicas']
    holder=GenerationWebRTCBridge(AutomaticGenerationControl(ledgers[0],lambda _:False,lambda _:False),
                                   snapshot_available=lambda sha:sha==target,interval=.2)
    receiver=GenerationWebRTCBridge(AutomaticGenerationControl(ledgers[1],lambda _:False,lambda _:False),
                                     snapshot_available=lambda sha:False,interval=.2)
    node=Node();holder.tick([('some-other-peer',node)],now=10.)
    hints=[d for k,d in node.wires if k=='generation-have']
    assert len(hints)==1 and hints[0]['manifest']==target
    receiver.receive('some-other-peer','generation-have',hints[0])
    download=Node();receiver.tick([('some-other-peer',download)],now=10.)
    assert any(k=='generation-snapshot-request' and d['manifest']==target for k,d in download.wires)
    # Someone advertising another digest must not trigger an arbitrary-file fetch.
    receiver.receive('fake','generation-have',{'epoch':1,'manifest':'f'*64})
    fake=Node();receiver.tick([('fake',fake)],now=11.)
    assert not any(k=='generation-snapshot-request' for k,_ in fake.wires)
    old_owner.close()
    for l in ledgers:l.close()


def test_gc_tombstone_prevents_resurrection(tmp_path):
    old_owner,old,new,manifest,store,ledgers=prepare(tmp_path)
    manager=EpochRuntimeV43(tmp_path/'runtime',ledgers[0]);manager.stage(old)
    manager.acknowledge_install(epoch=1,manifest=manifest,store=store,new_journal=new,loader_confirmation=True)
    old_owner.close()
    assert manager.collect(epoch=1)
    assert Path(str(old)+'.retired.json').exists()
    with pytest.raises(PatchError,match='retired'):
        EventJournal(old,'d'*64,max_bytes=0)
    manager.close()
    for l in ledgers:l.close()


def test_gc_can_finish_after_restart_between_deletions(tmp_path,monkeypatch):
    old_owner,old,new,manifest,store,ledgers=prepare(tmp_path)
    manager=EpochRuntimeV43(tmp_path/'runtime',ledgers[0]);manager.stage(old)
    manager.acknowledge_install(epoch=1,manifest=manifest,store=store,new_journal=new,loader_confirmation=True)
    old_owner.close()
    original_unlink=Path.unlink
    blocked={str(old)}
    def crash_once(self, *args, **kwargs):
        if str(self) in blocked:
            blocked.remove(str(self));raise OSError('simulated power loss after durable GC intent')
        return original_unlink(self,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',crash_once)
    with pytest.raises(OSError,match='power loss'):
        manager.collect(epoch=1)
    assert manager.adoption.status(1)['state']=='collecting'
    manager.close()
    manager=EpochRuntimeV43(tmp_path/'runtime',ledgers[0])
    assert manager.poll()==[1]
    assert manager.status(1)['state']=='retired'
    assert not old.exists()
    manager.close()
    for l in ledgers:l.close()
