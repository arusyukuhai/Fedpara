"""Regression tests for lost-room discovery and missing historic updates."""
import asyncio
import json

from patchlib.join_guard import initial_join_decision
from patchlib.roomid import RoomDiscovery, create_room_id


def test_new_peer_never_trains_with_unknown_room_state():
    decision = initial_join_decision(initial_complete=False, lobby_online=False,
        lobby_members=set(), neighbors={}, durable_history=False, signaling_seen=False)
    assert not decision.can_train
    assert 'room-signaling' in decision.reason


def test_first_creator_can_train_only_after_confirmed_empty_room():
    assert initial_join_decision(initial_complete=False, lobby_online=True,
        lobby_members=set(), neighbors={}, durable_history=False,
        signaling_seen=True).can_train


def test_second_window_waits_for_signed_history_not_just_presence():
    a='p'+'a'*32
    not_ready=initial_join_decision(initial_complete=False,lobby_online=True,
        lobby_members={a}, neighbors={a: False},durable_history=False,
        signaling_seen=True)
    assert not not_ready.can_train
    now_ready=initial_join_decision(initial_complete=False,lobby_online=True,
        lobby_members={a}, neighbors={a: True},durable_history=False,
        signaling_seen=True)
    assert now_ready.can_train


def test_signaling_outage_allows_existing_participant_not_newcomer():
    common=dict(initial_complete=False,lobby_online=False,
                lobby_members=set(),neighbors={})
    assert not initial_join_decision(**common,durable_history=True,signaling_seen=False).can_train
    assert initial_join_decision(**common,durable_history=True,signaling_seen=True).can_train


def test_both_peers_must_be_caught_up():
    a,b='p'+'a'*32,'p'+'b'*32
    assert not initial_join_decision(initial_complete=False,lobby_online=True,
        lobby_members={a,b},neighbors={a:True,b:False},
        durable_history=False,signaling_seen=True).can_train


def test_real_websocket_presence_updates_with_multiple_clients():
    """Small protocol-compatible signaller; does not replace a Nim test."""
    import websockets
    ids=['p'+'1'*32,'p'+'2'*32]
    room=create_room_id()
    roster={}
    async def handler(ws):
        msg=json.loads(await ws.recv())
        assert msg['type']=='group-join'
        peer=msg['peer']
        old=list(roster.items())
        roster[peer]=ws
        await ws.send(json.dumps({'type':'group-members','peers':[p for p,_ in old]}))
        for _,client in old:
            await client.send(json.dumps({'type':'group-peer-joined','peer':peer}))
        try:
            async for _ in ws: pass
        finally:
            roster.pop(peer,None)
    async def run():
        async with websockets.serve(handler,'127.0.0.1',0) as server:
            url=f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/ws'
            a=RoomDiscovery(url,room,ids[0]);b=RoomDiscovery(url,room,ids[1])
            stops=[asyncio.Event(),asyncio.Event()]
            tasks=[asyncio.create_task(a.run(stops[0])),asyncio.create_task(b.run(stops[1]))]
            for _ in range(200):
                if a.online and b.online and a.members=={ids[1]} and b.members=={ids[0]}:break
                await asyncio.sleep(.01)
            assert a.online and b.online
            assert a.members=={ids[1]} and b.members=={ids[0]}
            for event in stops:event.set()
            await asyncio.gather(*tasks)
    asyncio.run(run())


def test_missing_event_requests_retry_after_backpressure():
    from patchlib.elasticnet import ElasticNode, decode_control, WINDOW
    from patchlib.roomid import auto_identity
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as folder:
        origin,signer,_,verify=auto_identity(Path(folder)/'sender.pem')
        remote='p'+'e'*32
        node=ElasticNode('ws://127.0.0.1:8080/ws','r'*32,
                         origin=origin,remote=remote,genesis='a'*64,signer=signer)
        class FakeChannel:
            readyState='open'
            bufferedAmount=WINDOW+1
            def __init__(self): self.frames=[]
            def send(self,value): self.frames.append(value)
        ch=FakeChannel()
        node._channel=ch
        node.connected.set()
        node.pending={(remote,2),(remote,1)}
        assert not node.retry_missing(10.0)
        assert node.pending=={(remote,1),(remote,2)}
        assert ch.frames==[]
        ch.bufferedAmount=0
        assert node.retry_missing(10.3)
        # A recovered connection must authenticate with signed hello FIRST.
        assert decode_control(ch.frames[0],verify)['type']=='hello'
        frame=decode_control(ch.frames[1],verify)
        assert frame['type']=='need'
        assert frame['events']==[[remote,1],[remote,2]]
        assert len(node.pending)==2
        assert not node.retry_missing(10.5)
        assert node.retry_missing(11.5)
        assert len(ch.frames)==3  # hello, need, retried need


def test_founder_keeps_training_after_a_second_window_joins():
    """Exact regression: old instance previously became a cold join again."""
    from patchlib.join_guard import BootstrapState
    gate=BootstrapState.restored(False)
    decision=initial_join_decision(initial_complete=False,lobby_online=True,
        lobby_members=set(),neighbors={},durable_history=False,signaling_seen=True)
    assert gate.confirm_genesis_founder(decision=decision,lobby_online=True,
        lobby_members=set(),first_roster_was_empty=True)
    assert not gate.fresh_join
    assert gate.snapshot_installed
    newcomer='p'+'a'*32
    assert not gate.confirm_genesis_founder(decision=decision,lobby_online=True,
        lobby_members={newcomer},first_roster_was_empty=True)
    assert gate.snapshot_installed and not gate.fresh_join
    decision=initial_join_decision(initial_complete=True,lobby_online=True,
        lobby_members={newcomer}, neighbors={newcomer:False},
        durable_history=True,signaling_seen=True)
    assert decision.can_train  # existing trainer is NOT a snapshot receiver


def test_first_roster_founder_survives_quick_second_member():
    from patchlib.join_guard import BootstrapState,JoinDecision
    from patchlib.roomid import RoomDiscovery,create_room_id
    founder='p'+'a'*32
    newcomer='p'+'b'*32
    lobby=RoomDiscovery('ws://127.0.0.1:8080/ws',create_room_id(),founder)
    lobby.accept({'type':'group-members','peers':[]})
    lobby.accept({'type':'group-peer-joined','peer':newcomer})
    assert lobby.members=={newcomer}
    assert lobby.first_roster_was_empty is True
    gate=BootstrapState.restored(False)
    initial=initial_join_decision(initial_complete=False,lobby_online=True,
        lobby_members=lobby.members,neighbors={newcomer:False},
        durable_history=False,signaling_seen=True)
    assert not initial.can_train
    assert gate.confirm_genesis_founder(decision=initial,lobby_online=True,
        lobby_members=lobby.members,first_roster_was_empty=lobby.first_roster_was_empty)
    assert not gate.fresh_join
    assert gate.snapshot_installed


def test_newcomer_cannot_skip_verified_full_snapshot():
    from patchlib.join_guard import BootstrapState
    from patchlib.roomid import RoomDiscovery,create_room_id
    founder='p'+'a'*32
    newcomer='p'+'b'*32
    lobby=RoomDiscovery('ws://127.0.0.1:8080/ws',create_room_id(),newcomer)
    lobby.accept({'type':'group-members','peers':[founder]})
    gate=BootstrapState.restored(False)
    decision=initial_join_decision(initial_complete=False,lobby_online=True,
        lobby_members=lobby.members,neighbors={founder:True},
        durable_history=False,signaling_seen=True)
    assert decision.can_train  # inventory-only is not enough
    assert lobby.first_roster_was_empty is False
    assert not gate.confirm_genesis_founder(decision=decision,lobby_online=True,
        lobby_members=lobby.members,first_roster_was_empty=lobby.first_roster_was_empty)
    assert gate.fresh_join and not gate.snapshot_installed
    gate.confirm_verified_snapshot()
    assert gate.snapshot_installed and not gate.fresh_join


def test_bootstrap_recovered_founder_does_not_fetch_peer_snapshot():
    from patchlib.join_guard import BootstrapState
    gate=BootstrapState.restored(True)
    assert gate.snapshot_installed and not gate.fresh_join
    assert not gate.confirm_genesis_founder(
        decision=initial_join_decision(initial_complete=True,lobby_online=True,
            lobby_members=set(),neighbors={},durable_history=True,signaling_seen=True),
        lobby_online=True,lobby_members=set(),first_roster_was_empty=True)
    import pytest
    with pytest.raises(RuntimeError):
        gate.confirm_verified_snapshot()


def test_first_roster_cannot_change_after_reconnect():
    from patchlib.roomid import RoomDiscovery,create_room_id
    founder='p'+'a'*32
    newcomer='p'+'b'*32
    lobby=RoomDiscovery('ws://127.0.0.1:8080/ws',create_room_id(),newcomer)
    lobby.accept({'type':'group-members','peers':[founder]})
    assert lobby.first_roster_was_empty is False
    lobby.accept({'type':'group-members','peers':[]})
    assert lobby.first_roster_was_empty is False


def test_newcomer_cannot_claim_genesis_when_original_peer_leaves():
    from patchlib.join_guard import BootstrapState
    from patchlib.roomid import RoomDiscovery, create_room_id
    first='p'+'a'*32
    newcomer='p'+'b'*32
    lobby=RoomDiscovery('ws://127.0.0.1:8080/ws',create_room_id(),newcomer)
    lobby.accept({'type':'group-members','peers':[first]})
    lobby.accept({'type':'group-peer-left','peer':first})
    assert lobby.members==set()
    assert lobby.first_roster_was_empty is False
    # A current empty roster MUST NOT promote an unsynchronized newcomer.
    gate=BootstrapState.restored(False)
    decision=initial_join_decision(initial_complete=False,lobby_online=True,
        lobby_members=set(),neighbors={},durable_history=False,signaling_seen=True)
    assert decision.can_train  # generic guard alone is insufficient
    assert not gate.confirm_genesis_founder(decision=decision,
        lobby_online=True,lobby_members=set(),first_roster_was_empty=False)
    assert gate.fresh_join and not gate.snapshot_installed
