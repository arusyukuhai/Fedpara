"""Large-room control plane: bounded adjacencies, races and incremental repair."""
import random
import pytest
from patchlib.core import PatchError
from patchlib.roomid import RoomDiscovery
from patchlib.topology import neighbors


def identities(n):
    return [f'p{i:032x}' for i in range(n)]


def lobby(size=4096, degree=8):
    peer=identities(1)[0]
    return RoomDiscovery('wss://example.test/ws','X'*32,peer,max_degree=degree)


def welcome(d, member_set, revision, *, existing=True):
    d.accept({'type':'group-members','topology':'bounded-v2','degree':d.max_degree,
              'revision':revision,'peers':sorted(member_set),
              'online':max(1, len(member_set)+1),'existing_room':existing})


def notice(revision, peers, count):
    return {'type':'group-neighbors', 'topology':'bounded-v2',
            'revision':revision,'peers':sorted(peers),'online':count}


@pytest.mark.parametrize('degree',[2,4,6,8])
def test_room_4096_limit_is_local_degree_not_roster(degree):
    ids=identities(4096)
    client=RoomDiscovery('wss://example.test/ws','X'*32,ids[0],max_degree=degree)
    expected=neighbors(ids[0],ids,max_degree=degree)
    welcome(client,expected,7)
    assert client.topology_v2 and client.online_count==1+len(expected)
    assert client.members==expected
    assert len(client.members)<=degree
    assert client.roster_revision==7


def test_join_leave_reorder_and_outdated_control_frames():
    ids=identities(4096)
    me=ids[2000]
    d=RoomDiscovery('wss://example.test/ws','X'*32,me)
    welcome(d,neighbors(me,ids),10)
    assert len(d.members)<=8
    after=ids[1999:2009]
    new={x for x in after if x!=me}
    new=set(sorted(new)[:8])
    d.accept(notice(12,new,4000))
    d.accept(notice(11,{'p'+f'{123:032x}'},4005))
    assert d.members==new
    assert d.online_count==4000
    assert d.roster_revision==12
    # Equal-revision duplicate is idempotent.
    d.accept(notice(12,[],1))
    assert d.members==new
    d.accept(notice(13,[],3999))
    assert not d.members
    assert d.online_count==3999


def test_reject_bad_compact_rosters_and_protocol_mismatch():
    d=lobby()
    with pytest.raises(PatchError,match='too many'):
        welcome(d,identities(9),1)
    with pytest.raises(PatchError,match='degree mismatch'):
        d.accept({'type':'group-members','topology':'bounded-v2',
                  'peers':identities(5),'online':6,'revision':1,'degree':4})
    welcome(d,set(identities(5)[1:]),2)
    with pytest.raises(PatchError,match='invalid'):
        d.accept(notice(3,identities(9),6))
    with pytest.raises(PatchError,match='invalid'):
        d.accept(notice(3,['bad'],6))
    with pytest.raises(PatchError,match='invalid'):
        d.accept(notice(3,['p'+f'{8:032x}'],4097))
    with pytest.raises(PatchError,match='invalid'):
        d.accept(notice(3,['p'+f'{8:032x}'],True))
    with pytest.raises(PatchError,match='invalid'):
        d.accept(notice(3,['p'+f'{8:032x}'],6.5))


def _affected_ids(before, after, changed):
    """Independent all-roster diff oracle, not an obsolete +/-10 guess."""
    common = set(before) & set(after)
    return {p for p in common if neighbors(p,before)!=neighbors(p,after)}


@pytest.mark.parametrize('degree',[2,4,6,8])
def test_changed_neighborhood_notifications_complete_over_churn(degree):
    # Mirrors bounded recipient enumeration in Nim signaling server.
    rng=random.Random(1917)
    roster=identities(96)
    for n in range(96, 155):
        peer=f'p{n:032x}'
        before=sorted(roster)
        roster.append(peer)
        after=sorted(roster)
        affected=_affected_ids(before,after,peer)
        for current in before:
            if neighbors(current,before,max_degree=degree)!=neighbors(current,after,max_degree=degree):
                assert current in affected
    for _ in range(45):
        peer=rng.choice(roster)
        before=sorted(roster)
        roster.remove(peer)
        after=sorted(roster)
        affected=_affected_ids(before,after,peer)
        for current in after:
            if neighbors(current,before,max_degree=degree)!=neighbors(current,after,max_degree=degree):
                assert current in affected


def test_high_degree_large_roster_maximum_graph_degree():
    # Verify graph symmetry and connectivity with a 4096 participant roster.
    ids=identities(4096)
    # Sparse spot checks; global graph is validated with O(N) expected ring edges.
    for offset in [0,1,7,16,96,499,2047,4088,4095]:
        adjacent=neighbors(ids[offset],ids)
        assert len(adjacent)==8
        for neighbor in adjacent:
            assert ids[offset] in neighbors(neighbor,ids)


def test_legacy_lobby_is_rejected_instead_of_silently_using_partial_roster():
    d=lobby()
    with pytest.raises(PatchError,match='obsolete'):
        d.accept({'type':'group-members','peers':identities(96)[1:],
                  'existing_room':True})
    welcome(d,{identities(96)[1]},1)
    with pytest.raises(PatchError,match='obsolete'):
        d.accept({'type':'group-peer-left','peer':identities(96)[1]})


def test_founding_room_sticky_when_new_peers_join():
    d=lobby()
    welcome(d,set(),1,existing=False)
    d.accept(notice(2,set(identities(2)[1:]),2))
    assert d.first_roster_was_empty is True and d.room_existing is False


def test_count_heartbeat_does_not_prevent_late_neighbor_repair():
    d=lobby()
    peers=identities(4)[1:]
    welcome(d,peers,1)
    assert d.online_count==4
    d.accept({'type':'group-count','revision':3,'online':3000})
    assert d.online_count==3000 and d.roster_revision==1
    d.accept(notice(2,{peers[0]},7))
    assert d.members=={peers[0]}  # revision 2 still advances topology
    assert d.online_count==3000 # never regress aggregate count
    d.accept(notice(4,set(peers),3001))
    assert d.online_count==3001
    d.accept({'type':'group-count','revision':3,'online':50})
    assert d.online_count==3001


def test_4096_roster_middle_insertion_changes_only_a_bounded_neighborhood():
    before=[f'p{i*2:032x}' for i in range(4095)]
    insert=f'p{3999:032x}'
    after=sorted(before+[insert])
    affected=_affected_ids(before,after,insert)
    changed=set()
    for peer in before:
        if neighbors(peer,before)!=neighbors(peer,after):changed.add(peer)
    assert changed
    assert changed<=affected
    # One inserted participant can rewire at most two existing endpoints per
    # ring, plus the inserted node. No global long-chord reshuffle.
    assert len(affected)<=16
    assert len(changed)<=16
    assert len(neighbors(insert,after))<=8


def test_real_signaling_socket_out_of_order_welcome_and_status():
    """Exercise RoomDiscovery.run against real WS frames without requiring Nim."""
    import asyncio
    import json
    import websockets

    async def scenario():
        peer=identities(4)[0]
        other=identities(4)[1]
        stop=asyncio.Event()
        initial=[]

        async def ws_handler(ws):
            payload=json.loads(await ws.recv())
            assert payload['topology_v2'] is True
            assert payload['topology_degree']==8
            initial.append(payload)
            await ws.send(json.dumps(notice(2,{other},9)))
            await ws.send(json.dumps({'type':'group-members','peers':[],
                  'online':1,'revision':1,'degree':8,
                  'topology':'bounded-v2','existing_room':False}))
            await ws.send(json.dumps({'type':'group-count','online':10,'revision':3}))
            while not stop.is_set():await asyncio.sleep(.05)

        async with websockets.serve(ws_handler,'127.0.0.1',0) as server:
            port=server.sockets[0].getsockname()[1]
            d=RoomDiscovery(f'ws://127.0.0.1:{port}','X'*32,peer)
            task=asyncio.create_task(d.run(stop))
            until=asyncio.get_running_loop().time()+5
            try:
                while asyncio.get_running_loop().time()<until:
                    if d.online and d.members=={other} and d.online_count==10:
                        break
                    await asyncio.sleep(.02)
                assert initial
                assert d.topology_v2
                assert d.members=={other}
                assert d.roster_revision==2
                assert d.online_count==10
                assert d.first_roster_was_empty is True
            finally:
                stop.set()
                await asyncio.wait_for(task,5)

    asyncio.run(scenario())


def test_embedded_signaling_rejects_legacy_group_join():
    """An old client must not mistake a partial neighbor list for a full roster."""
    import asyncio
    import json
    import websockets
    from patchlib.swarm_sim import LocalSignaling

    async def scenario():
        async with LocalSignaling() as server:
            async with websockets.connect(server.url) as ws:
                await ws.send(json.dumps({'type': 'group-join',
                             'room': 'r'*32, 'peer': identities(1)[0]}))
                response=json.loads(await asyncio.wait_for(ws.recv(),3))
                assert response['type']=='error'
                assert 'bounded-v2' in response['message']
    asyncio.run(scenario())


def test_embedded_signaling_rejects_neighbor_degree_mismatch():
    import asyncio
    import json
    import websockets
    from patchlib.swarm_sim import LocalSignaling
    from patchlib.room_manifest import make_manifest
    mani=make_manifest('fixture/model','main','synthetic')
    async def scenario():
        async with LocalSignaling() as server:
            async with websockets.connect(server.url) as alice:
                await alice.send(json.dumps({'type':'group-join',
                    'room':'r'*32,'peer':identities(2)[0],
                    'topology_v2':True,'topology_degree':8,'manifest':mani}))
                response=json.loads(await asyncio.wait_for(alice.recv(),3))
                assert response['type']=='group-members'
                assert response['online']==1
                async with websockets.connect(server.url) as bob:
                    await bob.send(json.dumps({'type':'group-join',
                        'room':'r'*32,'peer':identities(2)[1],
                        'topology_v2':True,'topology_degree':4}))
                    result=json.loads(await asyncio.wait_for(bob.recv(),3))
                    assert result['type']=='error'
                # Joining with a bad degree did not change the live roster.
                assert len(server.groups['r'*32])==1
    asyncio.run(scenario())
