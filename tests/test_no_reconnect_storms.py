"""Failure injection for the two-window GUI's presence and P2P transport.

No network or GPU needed; behavior is deterministic, including fake time.
"""
import asyncio
import json
import time

import pytest

from patchlib.core import PatchError
from patchlib.elasticnet import PresenceHysteresis
from patchlib.p2p import wait_send_window
from patchlib.roomid import RoomDiscovery
from patchlib.shardp2p import _wait_buffer
from patchlib.topology import neighbors


def peer(i):
    return f'p{i:032x}'


def test_presence_flap_does_not_destroy_good_two_gui_pair():
    history = PresenceHysteresis(grace_seconds=20)
    p = peer(1)
    assert history.observe({p}, 0, online=True) == {p}
    # A few `group-peer-left`, `group-peer-joined` events from a failing
    # signaling service should NOT make both apps destroy their live RTC.
    assert history.observe(set(), 2, online=True) == {p}
    assert history.observe({p}, 3, online=True) == {p}
    assert history.observe(set(), 8, online=True) == {p}
    assert history.observe(set(), 18, online=True) == {p}
    # Offline signaling cannot revoke an existing direct link.
    assert history.observe(set(), 100, online=False) == {p}
    # After an authoritative roster stays empty, the departed peer is gone.
    assert history.observe(set(), 101, online=True) == set()


def test_membership_changes_preserve_8_neighbour_cap_and_symmetry():
    ids = [peer(i) for i in range(16)]
    histories = {x: PresenceHysteresis(grace_seconds=20) for x in ids}
    all_ids = set(ids)
    for at, roster in ((0, all_ids), (3, all_ids - {peer(11)}),
                       (4, all_ids), (5, all_ids - {peer(11)}),
                       (8, all_ids), (15, all_ids - {peer(11)}),
                       (39, all_ids - {peer(11)})):
        # Authoritative roster events reach all clients and contain all peers.
        graph = {}
        for node in roster:
            view = histories[node].observe(roster - {node}, at, online=True) | {node}
            graph[node] = neighbors(node, view)
            assert len(graph[node]) <= 8
        if at <= 15:  # departed peer remains in the stable topology during grace
            assert all(peer(11) in histories[node].last_seen
                       for node in roster if node != peer(11))
    assert all(peer(11) not in histories[x].last_seen for x in all_ids - {peer(11)})


class WedgedChannel:
    readyState = 'open'
    bufferedAmount = 1_000_000


@pytest.mark.parametrize('check', [wait_send_window, _wait_buffer])
def test_open_but_stalled_sctp_fails_instead_of_hanging_forever(check):
    async def scenario():
        start=time.monotonic()
        with pytest.raises(TimeoutError, match='(stall|progress)'):
            await asyncio.wait_for(check(WedgedChannel(), stall_seconds=.04), .25)
        assert time.monotonic()-start < .24
    asyncio.run(scenario())


@pytest.mark.parametrize('check', [wait_send_window, _wait_buffer])
def test_sctp_window_wait_keeps_event_loop_alive(check):
    async def scenario():
        channel=WedgedChannel()
        ticks=[]
        async def drain():
            await asyncio.sleep(.03)
            ticks.append(1)
            channel.bufferedAmount=1
        await asyncio.gather(check(channel, stall_seconds=.1), drain())
        assert len(ticks)==1
    asyncio.run(scenario())


def test_invalid_roster_stops_reconnect_rather_than_spinning(monkeypatch):
    import sys
    import types
    attempted=[]
    class FakeWS:
        async def __aenter__(self):return self
        async def __aexit__(self, *args):pass
        async def send(self, message):pass
        async def recv(self):
            return json.dumps({'type':'group-members','peers':['bad-id']})
    def fake_connect(*args, **kwargs):
        attempted.append(1)
        return FakeWS()
    import websockets
    monkeypatch.setattr(websockets, 'connect', fake_connect)
    lobby=RoomDiscovery('ws://localhost:8080/ws','x'*32,peer(0))
    async def scenario():
        await asyncio.wait_for(lobby.run(asyncio.Event()), .2)
    asyncio.run(scenario())
    assert len(attempted)==1
    assert isinstance(lobby.fatal_error,PatchError)
    assert not lobby.online


def test_signed_snapshot_tip_survives_saturated_control_queue(tmp_path):
    from patchlib.elasticnet import ElasticNode, encode_control
    from patchlib.roomid import auto_identity

    alice, alice_sign, alice_pub, _ = auto_identity(tmp_path/'alice.pem')
    bob, _, _, bob_verify = auto_identity(tmp_path/'bob.pem')
    class FakeChannel:
        readyState='connecting'
        bufferedAmount=0
        def __init__(self):self.callbacks={};self.sent=[]
        def on(self,name):
            def decorator(cb):self.callbacks[name]=cb;return cb
            return decorator
        def send(self,data):self.sent.append(data)
        def close(self):
            self.readyState='closed'
            if 'close' in self.callbacks:self.callbacks['close']()

    async def scenario():
        node=ElasticNode('ws://localhost/ws','y'*32,
                         origin=bob,remote=alice,genesis='a'*64,
                         signer=None, verifier=auto_identity(tmp_path/'alice.pem')[3],
                         unsafe=False)
        channel=FakeChannel()
        node._attach_channel(channel)
        channel.readyState='open'
        channel.callbacks['open']()
        count=0
        def send(kind, **data):
            nonlocal count
            count+=1
            channel.callbacks['message'](encode_control(
                dict(type=kind,peer=alice,genesis='a'*64,counter=count,**data),alice_sign))
        send('hello',pub=alice_pub)
        # Fill the normal bounded queue with redundant controls.
        for _ in range(150):send('heartbeat')
        assert node.controls.full()
        send('snapshot-tip',manifest_sha256='a'*64, inventory_sha256='b'*64,
             events=0,cutoffs={})
        assert not node.protocol_failure.is_set()
        assert node.snapshot_tip_latch['type']=='snapshot-tip'
        channel.close()
        await asyncio.sleep(.01)
    asyncio.run(scenario())


def test_transient_duplicate_presence_is_backed_off_not_permanently_blocked(monkeypatch):
    import websockets
    attempts=[]
    class BusyWS:
        async def __aenter__(self):return self
        async def __aexit__(self, *args):pass
        async def send(self, message):pass
        async def recv(self):
            return json.dumps({'type':'error',
                 'message':'Group full, identity already online, or server busy'})
    def connect(*args,**kwargs):
        attempts.append(time.monotonic())
        return BusyWS()
    monkeypatch.setattr(websockets, 'connect', connect)
    lobby=RoomDiscovery('ws://localhost:8080/ws','x'*32,peer(0))
    async def scenario():
        stop=asyncio.Event()
        task=asyncio.create_task(lobby.run(stop))
        await asyncio.sleep(.10)
        stop.set()
        await asyncio.wait_for(task,.4)
    asyncio.run(scenario())
    assert len(attempts)==1, 'rapid reconnect storm despite busy signaling'
    assert lobby.fatal_error is None, 'temporary duplicate sign-in must be retried'
