"""Protocol correctness and speed regressions for heavy WebRTC/Gossip traffic."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from patchlib.async_ingest import AsyncInboundPipeline
from patchlib.elasticnet import ElasticNode
from patchlib.p2p import CHUNK, FrameReceiver, transmit


def test_receive_digest_is_computed_incrementally(monkeypatch):
    """No multi-MiB full-buffer re-hash in aiortc's synchronous callback."""
    import patchlib.p2p as p2p
    raw = b'0123456789abcdef' * (1024 * 96)
    digest = hashlib.sha256(raw).hexdigest()
    real_hash = p2p.hashlib.sha256
    hashed_sizes = []

    def traced_hash(data=b''):
        hashed_sizes.append(len(data))
        assert len(data) <= CHUNK, 'end-of-transfer must not hash the full receive buffer'
        return real_hash(data)

    monkeypatch.setattr(p2p.hashlib, 'sha256', traced_hash)
    receiver = FrameReceiver(max_bytes=len(raw))
    receiver.feed(json.dumps({'v': 1, 'type': 'start', 'id': 'test',
                              'size': len(raw), 'sha256': digest}))
    for offset in range(0, len(raw), CHUNK):
        assert receiver.feed(raw[offset:offset + CHUNK]) is None
    assert receiver.feed(json.dumps({'v': 1, 'type': 'end', 'id': 'test'})) == raw
    assert max(hashed_sizes) <= CHUNK
    assert receiver.data == bytearray()


def test_large_transmit_hash_is_off_event_loop(monkeypatch):
    import patchlib.p2p as p2p
    real_to_thread = p2p.asyncio.to_thread
    calls = []

    async def monitored_to_thread(fn, *args, **kw):
        calls.append(True)
        return await real_to_thread(fn, *args, **kw)

    class Channel:
        readyState = 'open'
        bufferedAmount = 0
        def __init__(self): self.messages = []
        def send(self, data): self.messages.append(data)

    async def check():
        monkeypatch.setattr(p2p.asyncio, 'to_thread', monitored_to_thread)
        channel = Channel()
        data = b'signed-gradient-123' * 102400
        await transmit(channel, data, max_bytes=4 * 1024 * 1024)
        assert calls, 'large digest calculation must be offloaded'
        decoder = FrameReceiver(max_bytes=4 * 1024 * 1024)
        completed = None
        for frame in channel.messages:
            result = decoder.feed(frame)
            if result is not None: completed = result
        assert completed == data

    asyncio.run(check())


def test_inbound_payload_byte_budget_and_single_oversize_item(tmp_path):
    """32 large updates must not queue gigabytes just because slots remain."""
    from test_fast_inbound_v10 import pair, remote_wire
    alice,bob=pair(tmp_path)
    try:
        wire=remote_wire(alice)
        async def check():
            received=[]
            async def committed(info,raw,sender,new):
                received.append(new)
            # Allow only one wire in flight by byte budget; retain the
            # ability to handle a single wire larger than a soft budget.
            pipeline=AsyncInboundPipeline(bob,committed,max_queue=4,
                                          max_queue_bytes=1024)
            assert pipeline.submit('alice',wire)
            assert pipeline.pending_bytes==len(wire)
            assert not pipeline.submit('alice',wire)
            assert pipeline.received==1
            assert pipeline.max_observed_bytes==len(wire)
            pipeline.start()
            await pipeline.drain()
            assert pipeline.pending_bytes==0
            assert received==[True]
            assert pipeline.submit('alice',wire)
            await pipeline.drain()
            assert pipeline.pending_bytes==0
            assert received==[True,False]
            await pipeline.stop()
        asyncio.run(check())
    finally:
        alice.journal.close();bob.journal.close()


def test_backlog_never_starves_deferred_live_gossip(monkeypatch):
    """Repair gets priority, but after eight repairs *deferred* push gets a slot."""
    import patchlib.elasticnet as elasticnet
    node=ElasticNode('ws://127.0.0.1/ws','a'*32,origin='a'*32,
                     remote='b'*32,genesis='0'*64,unsafe=True)
    class Channel:
        readyState='open';bufferedAmount=0
    node._channel=Channel()
    node.remote_supports_aggregate=True
    node.hello_received=True
    node.lookup_requested_event=lambda _origin,seq: b'FPEL1\0' + seq.to_bytes(4,'big') + b'X'*64
    for i in range(1,65):
        node.requested_events[('c'*32,i)]=None
    node.outgoing.put_nowait(b'FPEL1\0' + b'A'*64)
    node.outgoing.put_nowait(b'ordinary-push')
    emitted=[]
    sent_live=asyncio.Event()

    async def fake_transmit(_channel,payload,**_kwargs):
        emitted.append(payload)
        if payload==b'ordinary-push':
            sent_live.set()
        await asyncio.sleep(0)

    async def check():
        monkeypatch.setattr(elasticnet,'transmit',fake_transmit)
        task=asyncio.create_task(node._pump())
        try:
            await asyncio.wait_for(sent_live.wait(),timeout=2)
            # The live push may follow eight NEED replies, but it must not
            # have to wait for all 64 outstanding repairs to drain.
            assert emitted.index(b'ordinary-push') < 25
            assert node.requested_events, 'repair backlog should still be nonempty'
        finally:
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)

    asyncio.run(check())


def test_snapshot_history_recheck_only_after_durable_change():
    from patchlib.elasticnet import snapshot_history_still_unchanged
    kwargs = dict(manifest_sha='a'*64, inventory_sha='b'*64,
                  event_count=100000, journal_count=98882)
    mismatch = (kwargs['manifest_sha'],kwargs['inventory_sha'],
                kwargs['event_count'],kwargs['journal_count'])
    assert snapshot_history_still_unchanged(mismatch, **kwargs)
    assert not snapshot_history_still_unchanged(None, **kwargs)
    assert not snapshot_history_still_unchanged(mismatch, **(kwargs | {'journal_count':98883}))
    assert not snapshot_history_still_unchanged(mismatch, **(kwargs | {'inventory_sha':'c'*64}))


def test_transfer_control_rejects_unbounded_json_and_bool_length():
    from patchlib.core import PatchError
    fr=FrameReceiver(max_bytes=16*1024)
    with pytest.raises(PatchError,match='oversized'):
        fr.feed(' ' * 4097)
    with pytest.raises(PatchError,match='patch size'):
        fr.feed(json.dumps({'v':1,'type':'start','id':'bad',
                            'size':True,'sha256':'a'*64}))
