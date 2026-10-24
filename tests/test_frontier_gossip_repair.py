"""Regression tests for missing signed events despite 300s of quiesce.

These tests exercise the real SQLite reconciliation and authenticated-channel
parsing but not aiortc's ICE/SCTP implementation.
"""
import asyncio
from collections import Counter
import json

import pytest

from patchlib.elastic import EventJournal
from patchlib.elasticnet import (
    ElasticNode, encode_control, validated_frontiers, validated_need,
)
from patchlib.core import PatchError

SELF='p'+'a'*32
REMOTE='p'+'b'*32
GENESIS='0'*64


def new_node():
    return ElasticNode('ws://127.0.0.1:8080/ws','a'*32,
                       origin=SELF,remote=REMOTE,genesis=GENESIS,unsafe=True)


class Channel:
    readyState='open'
    bufferedAmount=0
    def __init__(self):
        self.callbacks={}
        self.messages=[]
    def on(self,name):
        def register(fn):self.callbacks[name]=fn;return fn
        return register
    def send(self, data):self.messages.append(data)
    def close(self):
        self.readyState='closed'
        self.callbacks['close']()


def message(node,ch,kind,counter,**data):
    wire=encode_control({'type':kind,'peer':node.remote,'genesis':GENESIS,
                         'counter':counter,**data})
    ch.callbacks['message'](wire)


def test_authenticated_need_is_latched_even_when_control_queue_is_full():
    async def run():
        node=new_node()
        ch=Channel()
        node._attach_channel(ch)
        message(node,ch,'hello',1)
        while not node.controls.full():
            node.controls.put_nowait({'type':'heartbeat'})
        message(node,ch,'need',2,events=[[REMOTE,i] for i in range(1,33)])
        assert node.need_received==1
        assert list(node.requested_events)==[(REMOTE,i) for i in range(1,33)]
        assert not node.protocol_failure.is_set()
        assert node.controls.full()  # bypassed, not discarded
        ch.close()
        await asyncio.sleep(0)
    asyncio.run(run())


def test_invalid_signed_need_is_rejected_atomically():
    node=new_node()
    for invalid in ([[REMOTE,1],['@invalid',2]], [[REMOTE,1],[REMOTE,1.5]]):
        with pytest.raises(PatchError): validated_need(invalid)
    assert not node.requested_events


def test_frontier_envelope_has_bounded_unique_authenticated_origins():
    with pytest.raises(PatchError):
        validated_frontiers([[REMOTE,2],[REMOTE,3]])
    with pytest.raises(PatchError):
        validated_frontiers([[REMOTE,True]])
    with pytest.raises(PatchError):
        validated_frontiers([[REMOTE,0]])


def test_sqlite_frontier_finds_holes_below_same_high_watermark(tmp_path):
    j=EventJournal(tmp_path/'events.db',GENESIS)
    try:
        for seq in (1,3,5,6): j.append(SELF,seq,f'wire-{seq}'.encode())
        assert j.frontiers()==[[SELF,6]]
        assert j.missing_range(SELF,1,6)==[(SELF,2),(SELF,4)]
        n=new_node()
        n.remote_frontiers={SELF:6}
        assert n.reconcile_frontiers(j)==2
        assert n.pending=={(SELF,2),(SELF,4)}
        # An indexed inventory cannot hide holes merely because the latest
        # local event has the same sequence number as the sender.
    finally:j.close()


def test_frontier_scans_all_sixteen_origins_not_just_first(tmp_path):
    j=EventJournal(tmp_path/'events.db',GENESIS)
    try:
        origins=[f'p{i:032x}' for i in range(16)]
        n=new_node()
        n.remote_frontiers={origin:256 for origin in origins}
        for _ in range(20): n.reconcile_frontiers(j, budget=256)
        assert n.pending=={(origin,seq) for origin in origins for seq in range(1,257)}
        assert n.frontier_scanned>=4096
    finally:j.close()


def test_signed_frontier_bypasses_saturated_control_queue():
    async def run():
        n=new_node(); ch=Channel(); n._attach_channel(ch)
        message(n,ch,'hello',1)
        while not n.controls.full():n.controls.put_nowait({'type':'heartbeat'})
        message(n,ch,'frontier',2,heads=[[REMOTE,256],[SELF,4]])
        assert n.remote_frontiers=={REMOTE:256,SELF:4}
        assert not n.protocol_failure.is_set()
        ch.close();await asyncio.sleep(0)
    asyncio.run(run())


def test_frontier_reconciliation_recovers_all_with_no_inventories(tmp_path):
    """Simulate a 16-node, degree-8 mesh with dropped advertisements.

    Exercises partial update ownership, gaps and multi-hop relays.  We
    deliberately never deliver full inventory pages to emulate the reported
    failure mode where page-control traffic is consistently dropped.
    """
    from patchlib.topology import neighbors
    peers=[f'p{i:032x}' for i in range(16)]
    events={(origin,seq):f'{origin}/{seq}'.encode()
            for origin in peers for seq in range(1,17)}
    inventories=[]
    for i in range(16):
        owned={key:wire for key,wire in events.items()
               if key[0]==peers[i] or ((i+int(key[0][1:],16)+key[1]) % 9 not in (0,1))}
        inventories.append(owned)
    # All events exist at some node, but nobody starts with every event.
    assert all(len(inv)<len(events) for inv in inventories)
    adjacency={p:neighbors(p,peers,max_degree=8) for p in peers}
    assert all(len(v)==8 for v in adjacency.values())
    nodes={(i,j):new_node() for i in range(16) for j in range(16)
           if peers[j] in adjacency[peers[i]]}

    class FakeJournal:
        def __init__(self,data): self.data=data
        def missing_range(self,origin,first,last):
            return [(origin,k) for k in range(first,last+1) if (origin,k) not in self.data]

    for tick in range(160):
        for (receiver,source),node in nodes.items():
            # One in seven frontier frames is dropped deterministically.
            if (tick+receiver+source)%7==0: continue
            hi={}
            for origin,seq in inventories[source]:
                hi[origin]=max(hi.get(origin,0),seq)
            node.remote_frontiers.update(hi)
            node.reconcile_frontiers(FakeJournal(inventories[receiver]),budget=256)
            # Simulate an authenticated NEED and signed durable response.
            for key in list(sorted(node.pending))[:32]:
                if key in inventories[source]:
                    inventories[receiver][key]=inventories[source][key]
                    node.pending.remove(key)
        if all(set(inv)==set(events) for inv in inventories):break
    assert all(inv==events for inv in inventories),[len(events)-len(inv) for inv in inventories]
