"""96-peer overlay, churn and public-room registry regression tests."""
import io
import json
import random
from unittest.mock import patch

import pytest

from patchlib.topology import ROOM_PEER_LIMIT, neighbors, topology_metrics
from patchlib.roomid import list_public_rooms, public_room_endpoint, RoomDiscovery
from patchlib.webview_controller import WebviewController
from patchlib.webview_gui import FedparaBridge


def ids(count):
    return [f'p{i:032x}' for i in range(count)]


def reachable(graph, start):
    seen={start}
    fringe=[start]
    while fringe:
        node=fringe.pop()
        for p in graph[node]-seen:
            seen.add(p)
            fringe.append(p)
    return seen


@pytest.mark.parametrize('n', [1, 2, 3, 4, 8, 16, 32, 64, 80, 96])
def test_small_world_no_mesh_and_connected(n):
    roster=ids(n)
    graph={p:neighbors(p,roster) for p in roster}
    assert max(map(len,graph.values())) <= 8
    assert all(p in graph[q] for p in graph for q in graph[p])
    assert reachable(graph,roster[0])==set(roster)
    assert topology_metrics(roster)['links']<=4*n


def test_96_nodes_successive_join_leave_eventual_connectivity():
    rng=random.Random(27271)
    roster=[]
    for p in ids(96):
        roster.append(p)
        rng.shuffle(roster)
        graph={q:neighbors(q,roster) for q in roster}
        assert reachable(graph,roster[0])==set(roster)
    for _ in range(60):
        roster.remove(rng.choice(roster))
        graph={q:neighbors(q,roster) for q in roster}
        assert reachable(graph,roster[0])==set(roster)
        assert max(map(len,graph.values()))<=8


def test_peer_caps_and_input_rejection():
    with pytest.raises(ValueError):neighbors(ids(97)[0], ids(97))
    with pytest.raises(ValueError):neighbors(ids(3)[0], ids(3),max_degree=7)
    with pytest.raises(ValueError):neighbors(ids(3)[0], ['p0','p1'])
    with pytest.raises(ValueError):neighbors(ids(3)[0], 'malformed')


def test_room_discovery_96_members_and_explicit_publication():
    first=ids(96)[0]
    d=RoomDiscovery('wss://example.test/ws','A'*32,first)
    d.accept({'type':'group-members','peers':ids(96)[1:]})
    assert len(d.members)==95
    assert d.max_peers==96
    assert not d.publish
    publish=RoomDiscovery('wss://example.test/ws','B'*32,first,publish=True,publish_token='operator')
    assert publish.publish and publish.publish_token=='operator'


class Reply:
    status=200
    def __init__(self,data):
        self.data=data
        self.headers={'Content-Length':str(len(data))}
    def read(self,size=-1):
        return self.data[:size]
    def __enter__(self):return self
    def __exit__(self,*args):return False


def test_public_directory_bounded_and_no_private_disclosure(monkeypatch,tmp_path):
    entry={'room_id':'a'*32,'model':'sshleifer/tiny-gpt2','online':3,
           'capacity':96,'datasets':['roneneldan/TinyStories'], 'revision':'main'}
    received=[]
    def fake(url, timeout):
        received.append((url,timeout))
        return Reply(json.dumps({'rooms':[entry]}).encode())
    monkeypatch.setattr('urllib.request.urlopen',fake)
    controller=WebviewController(tmp_path)
    got=controller.list_rooms({'ws':'ws://127.0.0.1:8080/ws'})
    assert got['rooms']==[entry]
    assert received[0][0]=='http://127.0.0.1:8080/api/rooms'
    assert FedparaBridge(controller).list_rooms({'ws':'ws://127.0.0.1:8080/ws'})['rooms']==[entry]
    with pytest.raises(PermissionError):public_room_endpoint('ws://external.example/ws')
    assert public_room_endpoint('wss://fedpara.example/ws')=='https://fedpara.example/api/rooms'


def test_directory_rejects_oversize_payload(monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:Reply(b'X'*270000))
    with pytest.raises(ValueError,match='large'):
        list_public_rooms('ws://127.0.0.1:8080/ws')


def test_snapshot_source_selection_is_signed_and_bounded():
    import asyncio
    from types import SimpleNamespace
    from patchlib.elasticnet import eligible_snapshot_sources
    def slot(i, ready, retries=0, connected=True):
        event=asyncio.Event()
        if connected:event.set()
        return SimpleNamespace(peer=ids(20)[i],retries=retries,disabled=False,
             node=SimpleNamespace(connected=event, verified_at=55., remote_can_serve_snapshot=ready))
    slots={str(i):slot(i,i%2==0) for i in range(15)}
    # New peers without verified snapshots must not serve the genesis state.
    candidates=eligible_snapshot_sources(slots,ids(20)[19])
    assert candidates and all(s.node.remote_can_serve_snapshot for s in candidates)
    assert len(candidates)==8  # eight eligible; caller requests only candidates[:1]
    selected=candidates[0]
    selected.retries=4
    assert eligible_snapshot_sources(slots,ids(20)[19])[0] is not selected
    selected.disabled=True
    assert selected not in eligible_snapshot_sources(slots,ids(20)[19])


def test_registration_does_not_join_as_fake_training_peer(monkeypatch):
    import asyncio
    import websockets
    from patchlib.roomid import register_room_manifest
    from patchlib.room_manifest import make_hf_manifest
    manifest=make_hf_manifest('sshleifer/tiny-gpt2','main',[{
        'repo':'roneneldan/TinyStories','revision':'main','config':None,
        'split':'train','columns':{'text':'text'},'weight':1.0}])
    room='P'*32
    messages=[]
    async def handler(ws):
        msg=json.loads(await ws.recv())
        messages.append(msg)
        await ws.send(json.dumps({'type':'room-registered','room':msg['room']}))
    async def scenario():
        async with websockets.serve(handler,'127.0.0.1',0) as server:
            ws=f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/ws'
            await register_room_manifest(ws,room,manifest,publish=False)
    asyncio.run(scenario())
    assert len(messages)==1
    assert messages[0]['type']=='room-register'  # not group-join
    assert messages[0]['manifest']==manifest
    assert messages[0]['publish'] is False


def test_gui_room_registration_uses_local_manifest(monkeypatch,tmp_path):
    from patchlib.room_manifest import make_hf_manifest
    calls=[]
    async def register(*args,**kwargs):calls.append((args,kwargs))
    monkeypatch.setattr('patchlib.webview_controller.register_room_manifest',register)
    controller=WebviewController(tmp_path)
    m=controller.settings()
    m['room']='P'*32
    controller.save(m)
    result=controller.register_room({'room':m['room'],'ws':'ws://127.0.0.1:8080/ws','room_public':True})
    assert result['registered'] and result['public']
    assert calls[0][1]['publish'] is True
    assert calls[0][0][1]==m['room']
