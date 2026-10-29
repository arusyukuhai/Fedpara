"""Cross-origin signed FedSum, partial overlap, wire bounds and adaptive scale."""
from __future__ import annotations

import asyncio
import hashlib
from copy import deepcopy

import pytest
import torch
from torch import nn

from patchlib.aggregate_wire import (adaptive_publish_steps, is_aggregate,
                                     pack_aggregate, unpack_aggregate, HEADER)
from patchlib.async_ingest import AsyncInboundPipeline
from patchlib.continual import ContinualElasticEngine
from patchlib.core import PatchError, UniversalAdapter
from patchlib.elastic import EventJournal
from patchlib.elasticnet import ElasticNode


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=nn.Linear(16,16,bias=False)
        self.b=nn.Linear(16,16,bias=False)


def engines(tmp_path):
    torch.manual_seed(79011)
    initial=SmallModel()
    genesis=hashlib.sha256(b'cross-origin-v19-genesis').hexdigest()
    out=[]
    for pid in ('alice','bob','cathy','receiver','sequential','partial'):
        adapter=UniversalAdapter(deepcopy(initial),'small','1',rank=4,
                                 kind='loha',target_scope='auto')
        # Test signers pin independent author credentials (not transport signer).
        def signer(size,digest,pid=pid):
            return hashlib.sha256(f'{pid}:{size}:{digest}'.encode()).hexdigest()
        def verifier(origin):
            def check(auth):
                expected=hashlib.sha256(f'{origin}:{auth["size"]}:{auth["sha256"]}'.encode()).hexdigest()
                if auth['sig']!=expected:
                    raise PatchError('invalid independent event signature')
            return check
        trust={x:verifier(x) for x in ('alice','bob','cathy','receiver','sequential','partial')}
        journal=EventJournal(tmp_path/(pid+'.sqlite3'),genesis,sync='NORMAL')
        out.append(ContinualElasticEngine(adapter,pid,genesis,journal,
                signer=signer,trusted=trust,unsafe=False,reinit_every=20))
    return out


def make_wire(source,amount):
    with torch.no_grad():
        for group in source.adapter.groups:
            group.overlay.local.parts[0].add_(amount)
    return source.seal_local(1)


def originals(engine):
    result=[]
    for g in engine.adapter.groups:
        mod,_,attr=g.aliases[0].rpartition('.')
        result.append(engine.adapter.model.get_submodule(mod).parametrizations[attr].original.detach().clone())
    return result


def test_cross_origin_aggregate_equals_individual_signatures_and_sum(tmp_path):
    es=engines(tmp_path)
    try:
        authors=es[:3]
        wires=[make_wire(e,k*.007) for k,e in enumerate(authors,1)]
        packet=pack_aggregate(wires,max_bytes=2*1024*1024)
        assert is_aggregate(packet)
        assert unpack_aggregate(packet,max_bytes=2*1024*1024)==wires
        async def run():
            seen=[]
            async def committed(info,wire,src,fresh):
                seen.append((info.origin,info.seq,fresh))
            pipeline=AsyncInboundPipeline(es[3],committed)
            pipeline.start()
            assert pipeline.submit('relay',packet)
            await pipeline.drain()
            assert pipeline.aggregate_batches==1
            assert pipeline.applied==3
            assert {x[0] for x in seen}=={'alice','bob','cathy'}
            assert all(fresh for _,_,fresh in seen)
            await pipeline.stop()
        asyncio.run(run())
        for wire in wires:
            es[4].ingest(wire)
        for a,b in zip(originals(es[3]),originals(es[4])):
            torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-6)
        assert es[3].journal.count()==3
        # No accepted event is only represented by a relay certificate: each
        # signed constituent is independently replayable after a process crash.
        assert list(es[3].journal.events())==wires
    finally:
        for e in es:e.journal.close()


def test_signed_aggregate_can_replay_from_durable_journal(tmp_path):
    engines_now=engines(tmp_path)
    try:
        base=deepcopy(engines_now[3].adapter.model)
        wire_a=make_wire(engines_now[0],.009)
        wire_b=make_wire(engines_now[1],.013)
        payload=pack_aggregate([wire_a,wire_b],max_bytes=2*1024*1024)
        async def callback(*args):pass
        async def run():
            p=AsyncInboundPipeline(engines_now[3],callback)
            p.start()
            assert p.submit('relay',payload)
            await p.drain()
            await p.stop()
        asyncio.run(run())
        committed=originals(engines_now[3])
        genesis=engines_now[3].genesis
        for e in engines_now:
            e.journal.close()
        # Restore from original genesis weights (not trained base); the
        # archive stores constituent events, not unverifiable relay sums.
        torch.manual_seed(79011)
        initial=SmallModel()
        adapter=UniversalAdapter(initial,'small','1',rank=4,kind='loha',target_scope='auto')
        archive=EventJournal(tmp_path/'receiver.sqlite3',genesis,sync='NORMAL')
        try:
            replayed=ContinualElasticEngine(adapter,'receiver',genesis,archive,
                                              unsafe=True,reinit_every=20)
            replayed.replay()
            for expected,found in zip(committed,originals(replayed)):
                torch.testing.assert_close(found,expected,atol=2e-6,rtol=1e-6)
            assert archive.count()==2
        finally:
            archive.close()
    finally:
        for e in engines_now:
            try:e.journal.close()
            except Exception:pass


def test_partial_overlap_is_not_added_twice(tmp_path):
    es=engines(tmp_path)
    try:
        wires=[make_wire(e,i*.004) for i,e in enumerate(es[:3],1)]
        es[5].ingest(wires[0])
        before=es[5].journal.count()
        seen=[]
        async def callback(info,wire,sender,fresh):
            seen.append((info.origin,fresh))
        async def run():
            pipeline=AsyncInboundPipeline(es[5],callback)
            pipeline.start()
            packet=pack_aggregate(wires,max_bytes=1024*1024)
            assert pipeline.submit('some-relay',packet)
            await pipeline.drain()
            assert pipeline.applied==2
            assert pipeline.duplicates==1
            assert pipeline.submit('some-relay',packet)
            await pipeline.drain()
            assert pipeline.applied==2
            assert pipeline.duplicates==4
            await pipeline.stop()
        asyncio.run(run())
        assert es[5].journal.count()==before+2
        for wire in wires:
            es[4].ingest(wire)
        for a,b in zip(originals(es[5]),originals(es[4])):
            torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-6)
    finally:
        for e in es:e.journal.close()


def test_relay_cannot_mutate_signed_contribution(tmp_path):
    es=engines(tmp_path)
    try:
        wires=[make_wire(e,.005) for e in es[:2]]
        # Tamper the inner author signature while leaving relay container
        # checksum valid: independent source verification must still reject.
        changed=wires[0].replace(b'"signature":"',b'"signature":"bad',1)
        packet=pack_aggregate([changed,wires[1]],max_bytes=2*1024*1024)
        async def callback(*args):pass
        async def run():
            receiver=AsyncInboundPipeline(es[3],callback)
            receiver.start()
            assert receiver.submit('malicious-relay',packet)
            with pytest.raises(PatchError):
                await receiver.drain()
            await receiver.stop()
        asyncio.run(run())
        assert es[3].journal.count()==0
    finally:
        for e in es:e.journal.close()


def test_compressed_bomb_and_corruption_are_rejected():
    w1=b'FPEL1\0' + b'X'*400
    w2=b'FPEL1\0' + b'Y'*400
    valid=pack_aggregate([w1,w2],max_bytes=4096)
    assert unpack_aggregate(valid,max_bytes=4096)==[w1,w2]
    with pytest.raises(PatchError):unpack_aggregate(valid[:-4]+b'xxxx',max_bytes=4096)
    with pytest.raises(PatchError):unpack_aggregate(valid,max_bytes=100)
    evil=bytearray(valid)
    evil[8:12]=(100000000).to_bytes(4,'big')
    with pytest.raises(PatchError):unpack_aggregate(bytes(evil),max_bytes=4096)
    with pytest.raises(PatchError):pack_aggregate([w1,w1],max_bytes=4096)


@pytest.mark.parametrize('peers,expected',[(1,4),(5,4),(19,4),(20,8),
                                            (79,8),(80,16),(319,16),
                                            (320,32),(1280,64),(4096,64)])
def test_adaptive_peer_count(peers,expected):
    assert adaptive_publish_steps(4,peers)==expected
    assert adaptive_publish_steps(4,peers,max_steps=16)<=16


def test_outbound_aggregates_original_signed_frames(monkeypatch):
    imports=[]
    async def transmit(ch,wire,*,max_bytes=None,signer=None):
        imports.append(wire)
    monkeypatch.setattr('patchlib.elasticnet.transmit',transmit)
    node=ElasticNode('ws://127.0.0.1:8080/ws','a'*32,
       origin='p'+'a'*32,remote='p'+'b'*32,genesis='0'*64,unsafe=True)
    node.remote_supports_aggregate=True
    node.hello_received=True  # Simulate authenticated HELLO before the send pump.
    node._channel=type('C',(),{'readyState':'open','bufferedAmount':0})()
    sources=[b'FPEL1\0'+bytes([i])*1000 for i in range(1,4)]
    for wire in sources:
        node.outgoing.put_nowait(wire)
    async def run():
        task=asyncio.create_task(node._pump())
        for _ in range(500):
            if imports:break
            await asyncio.sleep(.001)
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass
    asyncio.run(run())
    assert imports
    assert is_aggregate(imports[0])
    assert unpack_aggregate(imports[0],max_bytes=node.max_bytes)==sources
    assert node.push_sent_events==3
    assert node.aggregate_sent_batches==1
