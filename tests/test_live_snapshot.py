"""Full-state P2P transfer regression tests, no ICE/TURN dependency."""
import asyncio
import hashlib
from pathlib import Path

import torch
from torch import nn

from patchlib.core import UniversalAdapter
from patchlib.snapshot_live import LiveSnapshotLink, journal_tip
from patchlib.state import export_state,apply_effective_state
from patchlib.shards import export_state_sharded,read_manifest
from patchlib.identity import _message
from patchlib.roomid import auto_identity
from patchlib.elastic import EventJournal


class FakeChannel:
    readyState='open'
    bufferedAmount=0
    def __init__(self):
        self.handlers={}
        self.other=None
    def on(self, event):
        def wrapped(callback):
            self.handlers[event]=callback
            return callback
        return wrapped
    def send(self, data):
        asyncio.get_running_loop().call_soon(self.other.handlers['message'], data)
    def close(self):
        self.readyState='closed'
        if 'close' in self.handlers:self.handlers['close']()


def test_live_shards_signed_and_verified(tmp_path):
    async def run():
        sender_id,signer,_,verify=auto_identity(tmp_path/'id.pem')
        model=nn.Sequential(nn.Linear(5,7),nn.Linear(7,3))
        adapter=UniversalAdapter(model,'test','r1',kind='lora',rank=2)
        store=tmp_path/'store'
        manifest=tmp_path/'snap.fmanifest'
        export_state_sharded(model,store,manifest,model_id='test',revision='r1',
            peer_id=sender_id,adapter=adapter,shard_bytes=64*1024)
        alice=LiveSnapshotLink(store,tmp_path/'a',32*1024**2,lambda:verify)
        bob=LiveSnapshotLink(tmp_path/'bobstore',tmp_path/'b',32*1024**2,lambda:verify)
        a,b=FakeChannel(),FakeChannel()
        a.other,b.other=b,a
        alice.attach(a);bob.attach(b)
        alice.on_authenticated(signer)
        bob.on_authenticated(signer)
        a.handlers['open']();b.handlers['open']()
        try:
            digest=alice.offer(manifest,signer)
            finished=await asyncio.wait_for(bob.completed.get(),timeout=10)
            assert hashlib.sha256(finished.read_bytes()).hexdigest()==digest
            assert read_manifest(finished)['model_id']=='test'
            assert bob.session.received_objects>=1
            await asyncio.sleep(.05)
        finally:
            a.close();b.close()
            await asyncio.sleep(.01)
    asyncio.run(run())


def test_effective_snapshot_replaces_adapted_base(tmp_path):
    torch.manual_seed(4)
    source=nn.Sequential(nn.Linear(5,6),nn.Linear(6,2))
    receiving=nn.Sequential(nn.Linear(5,6),nn.Linear(6,2))
    other_id='p'+'a'*32
    src_ad=UniversalAdapter(source,'test','rev',kind='loha',rank=2)
    dest_ad=UniversalAdapter(receiving,'test','rev',kind='loha',rank=2)
    with torch.no_grad():
        for g in src_ad.groups:
            g.overlay.local.parts[0].add_(0.1)
    exported=tmp_path/'s.fstate'
    export_state(source,exported,model_id='test',revision='rev',peer_id=other_id,adapter=src_ad)
    before=source(torch.randn(2,5))
    apply_effective_state(receiving,dest_ad,exported,model_id='test',revision='rev')
    check=source(torch.randn(2,5))
    # Test pointwise equality for the same input (including adapter effective values).
    x=torch.randn(2,5)
    assert torch.allclose(source(x),receiving(x),atol=1e-5)
    assert all(torch.isfinite(p).all() for p in dest_ad.trainable_parameters())


def test_inventory_digest_order_independent(tmp_path):
    j1=EventJournal(tmp_path/'j1.db','a'*64)
    j2=EventJournal(tmp_path/'j2.db','a'*64)
    try:
        events=[('p'+'a'*32,1,b'a'),('p'+'b'*32,1,b'b')]
        for peer,seq,wire in events:j1.append(peer,seq,wire)
        for peer,seq,wire in reversed(events):j2.append(peer,seq,wire)
        assert journal_tip(j1)==journal_tip(j2)
    finally:
        j1.close();j2.close()


def test_snapshot_vector_cutoff_handles_later_events(tmp_path):
    from patchlib.snapshot_live import snapshot_inventory
    peer='p'+'c'*32
    journal=EventJournal(tmp_path/'j.db','e'*64)
    try:
        journal.append(peer,1,b'one')
        digest,count,cutoffs=snapshot_inventory(journal)
        assert count==1 and cutoffs=={peer:1}
        journal.append(peer,2,b'two')
        assert snapshot_inventory(journal,cutoffs)[:2]==(digest,count)
        assert snapshot_inventory(journal)[1]==2
        journal.append('p'+'d'*32,1,b'three')
        assert snapshot_inventory(journal,cutoffs)[:2]==(digest,count)
    finally:
        journal.close()


def test_snapshot_invalid_vector_rejected(tmp_path):
    import pytest
    from patchlib.core import PatchError
    from patchlib.snapshot_live import snapshot_inventory
    journal=EventJournal(tmp_path/'j.db','e'*64)
    try:
        with pytest.raises(PatchError):snapshot_inventory(journal,{'../evil':3})
        with pytest.raises(PatchError):snapshot_inventory(journal,{'p'+'a'*32:0})
    finally:
        journal.close()
