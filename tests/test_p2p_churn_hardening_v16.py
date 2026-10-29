"""Invariant tests for async GUI pause/resume and recycled RTC channels."""
import asyncio
import hashlib
import json

import pytest

from patchlib.core import PatchError
from patchlib.elasticnet import (
    ElasticNode, encode_control, inventory_missing_bounded,
    FRONTIER_PENDING_LIMIT,
)
from patchlib.p2p import PatchNode

A = 'p'+'a'*32
B = 'p'+'b'*32
GENESIS='a'*64


class Channel:
    bufferedAmount=0
    label='fedpara-patch'
    def __init__(self, state='open'):
        self.readyState=state
        self.handlers={}
        self.sent=[]
    def on(self,name):
        def register(fn):self.handlers[name]=fn;return fn
        return register
    def send(self,wire):
        if self.readyState!='open': raise ConnectionError('closed')
        self.sent.append(wire)
    def close(self):
        if self.readyState == 'closed': return
        self.readyState='closed'
        cb=self.handlers.get('close')
        if cb:cb()


def deliver(node,ch,kind,counter,**fields):
    ch.handlers['message'](encode_control(dict(type=kind,peer=node.remote,
        genesis=GENESIS,counter=counter,**fields)))


def node():
    return ElasticNode('ws://127.0.0.1:8080/ws','c'*32,origin=A,remote=B,
                       genesis=GENESIS,unsafe=True)


def test_stale_patch_callbacks_do_not_cross_channel_generations():
    async def scenario():
        n=PatchNode('ws://127.0.0.1:8080/ws','d'*32)
        c1=Channel();n._attach_channel(c1)
        assert n.connected.is_set()
        c2=Channel();n._attach_channel(c2)
        assert n.connected.is_set()
        # A deferred message from the old channel must have no effect.
        c1.handlers['message']('not json')
        assert n.incoming.empty()
        assert n._channel is c2 and n.connected.is_set()
        c1.close()
        assert n.connected.is_set() and not n.link_closed.is_set()
        c2.close(); await asyncio.sleep(0)
    asyncio.run(scenario())


def test_abandoned_partial_snapshot_file_removed_at_rebind(tmp_path):
    async def scenario():
        n=PatchNode('ws://127.0.0.1:8080/ws','d'*32,state_dir=tmp_path)
        c1=Channel();n._attach_channel(c1)
        raw=b'partial-state'
        c1.handlers['message'](json.dumps(dict(v=1,type='start',id='abcd',
            size=len(raw),sha256=hashlib.sha256(raw).hexdigest())))
        c1.handlers['message'](raw[:5])
        assert list(tmp_path.glob('*.part'))
        c2=Channel();n._attach_channel(c2)
        assert not list(tmp_path.glob('*.part'))
        c1.handlers['message'](raw[5:])
        assert n.incoming.empty() and n.connected.is_set()
        c1.close();assert n.connected.is_set()
        c2.close();await asyncio.sleep(0)
    asyncio.run(scenario())


def test_signed_snapshot_status_is_latched_during_pause_and_queue_saturation():
    async def scenario():
        n=node();ch=Channel();n._attach_channel(ch)
        deliver(n,ch,'hello',1)
        while not n.controls.full():
            n.controls.put_nowait({'type':'heartbeat'})
        n.snapshot_request_id='1234567890abcdef'
        deliver(n,ch,'snapshot-status',2,request_id=n.snapshot_request_id,
                stage='accepted')
        deliver(n,ch,'snapshot-status',3,request_id=n.snapshot_request_id,
                stage='exporting')
        deliver(n,ch,'snapshot-status',4,request_id='f'*16,
                stage='source-not-ready')
        assert n.snapshot_status_latch['stage']=='exporting'
        assert n.snapshot_status_latch['counter']==3
        assert n.controls.full()
        assert not n.protocol_failure.is_set() and n.connected.is_set()
        ch.close();await asyncio.sleep(0)
    asyncio.run(scenario())


def test_reconnect_discards_old_control_queue_and_status_latch():
    async def scenario():
        n=node(); c1=Channel();n._attach_channel(c1)
        deliver(n,c1,'hello',1)
        n.snapshot_status_latch={'type':'snapshot-status','counter':3}
        n.snapshot_pending=True
        n.snapshot_pending_id='a'*16
        c2=Channel();n._attach_channel(c2)
        assert n.controls.empty()
        assert n.snapshot_status_latch is None
        assert not n.snapshot_pending
        assert n.remote_counter == -1
        c1.handlers['message']('invalid')
        assert not n.protocol_failure.is_set() and n.connected.is_set()
        deliver(n,c2,'hello',1)
        assert n.hello_received
        c1.close();assert n.connected.is_set()
        c2.close();await asyncio.sleep(0)
    asyncio.run(scenario())


def test_advertised_inventory_cannot_grow_pending_unbounded():
    n=node()
    class NoJournal:
        def has(self,*key):return None
    journal=NoJournal()
    n.pending={(B,i) for i in range(1, FRONTIER_PENDING_LIMIT)}
    n.ready=True
    added=inventory_missing_bounded(n,journal,[[B,FRONTIER_PENDING_LIMIT,'0'*64],
                                            [B,FRONTIER_PENDING_LIMIT+1,'0'*64]])
    assert added == [[B,FRONTIER_PENDING_LIMIT]]
    assert len(n.pending)==FRONTIER_PENDING_LIMIT
    assert not n.ready and n.control_overflow_drops==1
    with pytest.raises(PatchError):
        inventory_missing_bounded(n,journal,[[B,True,'0'*64]])


def test_failed_requested_gossip_send_restores_priority_id(monkeypatch):
    import patchlib.elasticnet as en
    async def broken(*args,**kwargs):raise ConnectionError('forced disconnect')
    monkeypatch.setattr(en,'transmit',broken)
    async def scenario():
        n=node();ch=Channel();n._channel=ch;n.hello_received=True
        n.queue_requested_events([(B,17)])
        n.lookup_requested_event=lambda peer,seq:b'valid-patch-for-test'
        task=asyncio.create_task(n._pump())
        await asyncio.wait_for(n.protocol_failure.wait(),2)
        await task
        assert (B,17) in n.requested_events
        assert n.link_closed.is_set()
    asyncio.run(scenario())
