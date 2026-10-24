"""An initial genesis peer is not a permanent manager or snapshot bottleneck."""
from types import SimpleNamespace
import asyncio

from patchlib.join_guard import BootstrapState
from patchlib.elasticnet import SnapshotProviderPool, can_supply_snapshot
from patchlib.topology import neighbors


def peer(i):
    return f'p{i:032x}'


def slot(i, *, verified=True, online=True, serves=True):
    connected = asyncio.Event()
    if online:
        connected.set()
    return SimpleNamespace(peer=peer(i), retries=0, disabled=False,
        node=SimpleNamespace(connected=connected, verified_at=1. if verified else 0.,
                             remote_can_serve_snapshot=serves))


def test_genesis_publisher_has_no_permanent_authority():
    founder=BootstrapState.restored(False)
    assert not can_supply_snapshot(room_only=True,bootstrap=founder)
    founder.confirm_genesis_founder(decision=None,lobby_online=True,
                                   lobby_members=set(),first_roster_was_empty=True)
    assert can_supply_snapshot(room_only=True,bootstrap=founder)
    # The second participant is eligible as soon as it has installed an
    # authenticated full-state snapshot, irrespective of the original creator.
    second=BootstrapState.restored(False)
    assert not can_supply_snapshot(room_only=True,bootstrap=second)
    second.confirm_verified_snapshot()
    assert can_supply_snapshot(room_only=True,bootstrap=second)
    assert can_supply_snapshot(room_only=True,bootstrap=BootstrapState.restored(True))


def test_creator_leaves_and_verified_peer_automatically_inherits_service():
    # C has joined before A exits; B already completed full-state verification.
    # Once A goes away, C can use B without a special owner transfer message.
    a,b,c=peer(0),peer(1),peer(2)
    slots={a:slot(0),b:slot(1)}
    pool=SnapshotProviderPool()
    primary=pool.select(slots,c,0)
    assert primary is not None
    survivor=b if primary.peer==a else a
    slots[primary.peer].node.connected.clear()
    assert pool.select(slots,c,1).peer==survivor
    assert pool.current==survivor
    assert set(neighbors(c,{survivor,c}))=={survivor}


def test_failed_source_rotates_without_appointing_a_new_admin():
    slots={peer(i):slot(i) for i in range(4)}
    p=SnapshotProviderPool()
    first=p.select(slots,peer(99),10)
    assert first is not None
    p.failed(first.peer,10,delay=30)
    second=p.select(slots,peer(99),10.1)
    assert second is not None and second.peer!=first.peer
    # All other sources may also fail. Never fall back to a new, unsynced peer.
    for i in range(4):
        if slots[peer(i)].node.remote_can_serve_snapshot:
            p.failed(peer(i),10.1)
    assert p.select(slots,peer(99),11) is None
    newcomer=slot(50,verified=True,serves=False)
    slots[peer(50)]=newcomer
    assert p.select(slots,peer(99),11) is None
    # A verified holder returning from the failure cooldown is eligible again.
    assert p.select(slots,peer(99),50) is not None


def test_creator_leaves_during_first_transfer_does_not_trust_unready_peer():
    a,b,c=peer(0),peer(1),peer(2)
    slots={a:slot(0),b:slot(1,serves=False)}
    p=SnapshotProviderPool()
    assert p.select(slots,c,0).peer==a
    slots[a].disabled=True
    assert p.select(slots,c,1) is None
    # Only after verified full-state installation may B become successor.
    recipient=BootstrapState.restored(False)
    recipient.confirm_verified_snapshot()
    slots[b].node.remote_can_serve_snapshot=can_supply_snapshot(
        room_only=True,bootstrap=recipient)
    assert p.select(slots,c,2).peer==b


def test_creator_leaves_and_small_world_overlay_stays_connected():
    ids=[peer(i) for i in range(16)]
    for gone in ids:
        remaining=set(ids)-{gone}
        graph={p:neighbors(p,remaining) for p in remaining}
        reached={next(iter(remaining))}
        while True:
            new=reached.union(*(graph[x] for x in reached))
            if new==reached:break
            reached=new
        assert reached==remaining
        assert max(map(len,graph.values()))<=8
