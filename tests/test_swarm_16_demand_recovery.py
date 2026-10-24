"""Focused regressions for a real 16-peer run that stalled with 343 missing updates.

These are mocked transports; the macOS aiortc 16-process run remains the
source of truth for actual WebRTC recovery and latency.
"""
import asyncio

from patchlib.elasticnet import ElasticNode, REQUEST_BACKLOG_LIMIT


def node():
    return ElasticNode('ws://127.0.0.1:8080/ws','a'*32,
                       origin='p'+'a'*32, remote='p'+'b'*32,
                       genesis='0'*64,unsafe=True)


def test_need_replies_survive_full_broadcast_queue():
    n=node()
    for i in range(3):
        n.outgoing.put_nowait(b'broadcast'+str(i).encode())
    identities=[('p'+'b'*32,i) for i in range(1,33)]
    n.queue_requested_events(identities)
    assert list(n.requested_events)==identities
    assert n.outgoing.qsize()==3
    assert n.request_backlog_overflow==0
    n.queue_requested_events(identities)
    assert len(n.requested_events)==32


def test_bounded_demand_ids_and_retryable_overflow():
    n=node()
    n.queue_requested_events([('p'+'b'*32,i) for i in range(1,REQUEST_BACKLOG_LIMIT+101)])
    assert len(n.requested_events)==REQUEST_BACKLOG_LIMIT
    assert n.request_backlog_overflow==1


def test_requested_events_are_served_before_broadcast_with_fairness(monkeypatch):
    sent=[]
    async def fake_transmit(ch, wire, max_bytes=None, signer=None):
        sent.append(wire)
        await asyncio.sleep(0)
    monkeypatch.setattr('patchlib.elasticnet.transmit',fake_transmit)
    n=node()
    n._channel=type('Channel',(),{'readyState':'open','bufferedAmount':0})()
    n.lookup_requested_event=lambda origin,seq: b'urgent:'+str(seq).encode()
    n.queue_requested_events([('p'+'b'*32,i) for i in range(1,21)])
    n.outgoing.put_nowait(b'ordinary')
    async def main():
        task=asyncio.create_task(n._pump())
        for _ in range(300):
            if len(sent)>=21:break
            await asyncio.sleep(.001)
        task.cancel()
        try:await task
        except asyncio.CancelledError:pass
    asyncio.run(main())
    assert sent[0]==b'urgent:1'
    assert sent[8]==b'ordinary'  # do not starve ordinary gossip
    assert sent[9]==b'urgent:9'
    assert sent.count(b'ordinary')==1
    assert n.request_responses_sent==20
    assert len(n.requested_events)==0


def test_disconnected_transmitter_marks_failed_for_reconnect(monkeypatch):
    async def fail(*args,**kwargs):
        raise ConnectionError('SCTP closed mid-event')
    monkeypatch.setattr('patchlib.elasticnet.transmit',fail)
    n=node()
    n._channel=type('Channel',(),{'readyState':'open','bufferedAmount':0})()
    n.lookup_requested_event=lambda origin,seq: b'signed'
    n.queue_requested_events([('p'+'b'*32,1)])
    async def main():
        await asyncio.wait_for(n._pump(),2)
    asyncio.run(main())
    assert n.protocol_failure.is_set()
    assert n.link_closed.is_set()
    assert isinstance(n.last_error,ConnectionError)
