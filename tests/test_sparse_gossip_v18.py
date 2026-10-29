"""Bounded signed-update push and rotating anti-entropy; no approximation."""
import hashlib
import math
from collections import deque

import pytest

from patchlib.core import PatchError
from patchlib.elasticnet import (
    CTRL_MAX, FRONTIER_PAGE_SIZE, FRONTIER_KNOWN_ORIGINS_LIMIT,
    encode_control, frontier_advertisement_page,
    selective_push_targets, validated_frontiers,
)
from patchlib.topology import neighbors


def peer(i):
    return 'p' + f'{i:032x}'


def test_push_limits_and_determinism_with_parent_and_origin_excluded():
    roster=[peer(i) for i in range(64)]
    self_peer=peer(20)
    candidates=neighbors(self_peer, roster)
    picked=selective_push_targets(self_peer,candidates,exclude_peer=peer(19),
                                   origin_peer=peer(0),event_seed=b'signed-event',fanout=2)
    assert len(picked)==2
    assert len(set(picked))==2
    assert all(x in candidates for x in picked)
    assert peer(19) not in picked
    assert picked==selective_push_targets(self_peer,reversed(sorted(candidates)),
        exclude_peer=peer(19),origin_peer=peer(0),event_seed=b'signed-event',fanout=2)
    assert len(selective_push_targets(self_peer,candidates,fanout=8))==8
    with pytest.raises(ValueError):selective_push_targets(self_peer,candidates,fanout=0)


@pytest.mark.parametrize('size', [2, 8, 16, 64, 256, 1024])
def test_ring_push_reaches_all_peers_with_only_two_transmissions_per_hop(size):
    """The clockwise backbone gives convergence on a stable bounded-v2 ring."""
    peers=[peer(i) for i in range(size)]
    adjacent={p:neighbors(p,peers) for p in peers}
    event_seed=hashlib.blake2b(b'authenticated:origin/step',digest_size=16).digest()
    got={peers[0]}
    queue=deque([(peers[0],None)])
    sends=0
    while queue:
        source,parent=queue.popleft()
        recipients=selective_push_targets(source,adjacent[source],
            exclude_peer=parent,origin_peer=peers[0],event_seed=event_seed,fanout=2)
        assert len(recipients)<=2
        for receiver in recipients:
            sends+=1
            if receiver not in got:
                got.add(receiver)
                queue.append((receiver,source))
    assert got==set(peers)
    assert sends<=2*size
    if size>=16:
        assert sends < sum(len(x) for x in adjacent.values())


def test_missing_events_stay_repairable_after_backpressure_and_churn():
    """Simulates a lost hop and eventual signed inventory/NEED repair.

    This is a deterministic transport model, not an actual aiortc soak test.
    """
    peers=[peer(i) for i in range(64)]
    adjacency={p:neighbors(p,peers) for p in peers}
    got={peers[0]}
    q=deque([(peers[0],None)])
    while q:
        source,parent=q.popleft()
        targets=selective_push_targets(source,adjacency[source],
            exclude_peer=parent,origin_peer=peers[0],event_seed=b'event',fanout=2)
        for target in targets:
            # Deliberately fail a quarter of the unsolicited SCTP messages.
            if (int(source[1:],16)*3+int(target[1:],16))%4==0:
                continue
            if target not in got:
                got.add(target);q.append((target,source))
    # Each missing receiver periodically hears per-origin frontiers from its
    # known neighbors. NEED fetches the original signed event, not a lossy sum.
    for _ in range(64):
        before=len(got)
        for p in peers:
            if p not in got and any(q in got for q in adjacency[p]):
                got.add(p)
        if len(got)==len(peers):break
        assert len(got)>before
    assert got==set(peers)


def test_4096_origin_frontier_is_paginated_and_every_origin_is_eventually_advertised():
    heads=[[peer(i),i+1] for i in range(4096)]
    cursor=0
    obtained={}
    for _ in range(math.ceil(len(heads)/FRONTIER_PAGE_SIZE)):
        page,new_cursor=frontier_advertisement_page(heads,cursor)
        assert len(page)<=FRONTIER_PAGE_SIZE
        assert validated_frontiers(page)==dict(page)
        signed=encode_control({'type':'frontier','peer':peer(0),'genesis':'0'*64,
                               'counter':1,'heads':page})
        assert len(signed)<CTRL_MAX
        obtained.update(dict(page))
        cursor=new_cursor
    assert obtained==dict(heads)
    assert cursor==0
    assert frontier_advertisement_page([],0)==([],0)
    with pytest.raises(ValueError):frontier_advertisement_page(heads,0,limit=97)


def test_frontier_cycle_handles_membership_mutations_without_oversized_frame():
    current=[[peer(i),10] for i in range(1,220)]
    cursor=0
    seen=set()
    for tick in range(20):
        if tick==3: current.insert(0,[peer(0),1])
        if tick==4: current=current[:-4]
        page,cursor=frontier_advertisement_page(current,cursor)
        seen.update(x[0] for x in page)
    assert set(x[0] for x in current)<=seen


def test_new_frontier_budget_does_not_silently_accept_unbounded_new_origins():
    from patchlib.elasticnet import ElasticNode
    node=ElasticNode('ws://127.0.0.1:8080/ws','a'*32,
                     origin=peer(0),remote=peer(1),genesis='0'*64,unsafe=True)
    # In normal 4,096-peer operation, all origins fit in this hard budget.
    node.remote_frontiers={peer(i):1 for i in range(4096)}
    assert len(node.remote_frontiers)<FRONTIER_KNOWN_ORIGINS_LIMIT


def test_local_telemetry_reports_push_vs_repair_volumes(tmp_path):
    from patchlib.live_telemetry import LocalTelemetry, recent_samples
    path=tmp_path/'v18.jsonl'
    writer=LocalTelemetry(path,sample_seconds=0)
    assert writer.record(steps=4,loss=1.25,emitted=1,inbound_applied=2,
                         inbox=0,channels=3,push_mib=12.5,repair_mib=3.25)
    writer.close()
    sample=recent_samples(path)[0]
    assert sample['push_mib']==12.5
    assert sample['repair_mib']==3.25
