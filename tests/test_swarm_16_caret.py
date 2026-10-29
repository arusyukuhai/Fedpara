"""16-node bounded overlay, real TCP signaling, and NOIR-EMERALD-01 text-caret regressions."""
import asyncio
import io
import json
from types import SimpleNamespace

from patchlib.room_manifest import make_manifest
from patchlib.swarm_sim import LocalSignaling, SwarmRunner
from patchlib.topology import neighbors, topology_metrics
from patchlib.webview_gui import compose_html


def _members(n=16):
    return [f'p{i:032x}' for i in range(n)]


def test_sixteen_peer_overlay_is_eight_regular_symmetric_and_connected():
    roster=_members()
    adj={p:neighbors(p,roster,max_degree=8) for p in roster}
    assert all(len(v)==8 for v in adj.values())
    assert all(p in adj[q] for p in adj for q in adj[p])
    seen={roster[0]}
    frontier=[roster[0]]
    while frontier:
        for item in adj[frontier.pop()]-seen:
            seen.add(item)
            frontier.append(item)
    assert seen==set(roster)
    assert topology_metrics(roster,max_degree=8)['links']==64
    # Two arbitrary departures and the subsequent restoration must not
    # create a one-way edge or disconnect the residual gossip graph.
    for removed in ((roster[3],), (roster[3],roster[11]), ()):
        live=[p for p in roster if p not in removed]
        sub={p:neighbors(p,live,max_degree=8) for p in live}
        assert all(len(v)<=8 for v in sub.values())
        assert all(p in sub[q] for p in sub for q in sub[p])
        reached={live[0]}; queue=[live[0]]
        while queue:
            for peer in sub[queue.pop()]-reached:
                reached.add(peer);queue.append(peer)
        assert len(reached)==len(live)


async def _real_tcp_sixteen_peer_join_rejoin():
    import websockets
    mani=make_manifest('fixture/model','main','synthetic')
    room='D'*40
    peers=_members()
    async with LocalSignaling(limit=16) as server:
        clients=[]
        try:
            for i,peer in enumerate(peers):
                ws=await websockets.connect(server.url)
                clients.append(ws)
                msg={'type':'group-join','room':room,'peer':peer,
                     'topology_v2':True,'topology_degree':8}
                if i==0:msg['manifest']=mani
                await ws.send(json.dumps(msg))
                response=json.loads(await asyncio.wait_for(ws.recv(),3))
                assert response['type']=='group-members'
                assert response['topology']=='bounded-v2'
                assert set(response['peers'])==neighbors(peer,peers[:i+1])
                for old in clients[:-1]:
                    old_peer=peers[clients.index(old)]
                    if neighbors(old_peer,peers[:i])!=neighbors(old_peer,peers[:i+1]):
                        notice=json.loads(await asyncio.wait_for(old.recv(),3))
                        assert notice['type']=='group-neighbors'
                        assert set(notice['peers'])==neighbors(old_peer,peers[:i+1])
            await clients[6].close()
            for i,old in enumerate(clients):
                if i!=6:
                    if neighbors(peers[i],peers)!=neighbors(peers[i],[p for j,p in enumerate(peers) if j!=6]):
                        notice=json.loads(await asyncio.wait_for(old.recv(),3))
                        assert notice['type']=='group-neighbors'
                        assert set(notice['peers'])==neighbors(peers[i],[p for j,p in enumerate(peers) if j!=6])
            clients[6]=await websockets.connect(server.url)
            await clients[6].send(json.dumps({'type':'group-join','room':room,'peer':peers[6],
                      'topology_v2':True,'topology_degree':8}))
            response=json.loads(await asyncio.wait_for(clients[6].recv(),3))
            assert set(response['peers'])==neighbors(peers[6],peers)
            assert server.stats['group_joins']==17
        finally:
            for conn in clients:
                await conn.close()


def test_16_real_tcp_group_membership_crash_rejoin():
    asyncio.run(_real_tcp_sixteen_peer_join_rejoin())


def test_gui_native_emerald_caret_and_focus():
    html=compose_html()
    assert 'caret-color: #34d399 !important' in html
    assert '-webkit-text-fill-color: #ecfff3' in html
    assert 'input:focus,textarea:focus,select:focus' in html.replace(' ', '')
    assert 'cursor: text' in html
    assert 'textarea::selection' in html
    assert 'Myriad' in html and 'Noto Sans JP' in html


async def _worker_status_parser(tmp_path):
    args=SimpleNamespace(output=tmp_path/'s', peers=16, seed=22)
    s=SwarmRunner(args)
    w=s.workers[0]
    reader=asyncio.StreamReader()
    for msg in ('elastic local event=5, living=16, active=9, bytes=20\n',
                'JOIN WAIT: active=8 synced=7 living=16\n',
                'elastic active=9 living=16\n'):
        reader.feed_data(msg.encode())
    reader.feed_eof()
    class MockProcess:
        stdout=reader
        async def wait(self):return 0
    out=io.StringIO()
    await s.monitor(w, MockProcess(), out)
    assert w.max_active==9 and w.max_living==16


def test_sixteen_connection_cap_parser_both_metric_orders(tmp_path):
    asyncio.run(_worker_status_parser(tmp_path))

async def _real_tcp_all_64_pair_signals():
    """Exercise every *actual* TCP signaling route in the 16-peer topology.

    This does not emulate aiortc: SDP negotiation and UDP/DTLS/SCTP remain
    the responsibility of the real user-machine 16-process Swarm test.
    """
    import websockets
    from patchlib.roomid import pair_room
    peers=_members()
    adj={p:neighbors(p,peers) for p in peers}
    edges=sorted({tuple(sorted((a,b))) for a in peers for b in adj[a]})
    assert len(edges)==64
    room='F'*40
    async with LocalSignaling(limit=16) as server:
        async def check_link(a,b):
            # Each symmetric graph edge gets its own genuine TCP WS pair.
            # Pair rooms match the live Fedpara pair_room scheme.
            ra=pair_room(room,a,b)
            alice=await websockets.connect(server.url)
            bob=await websockets.connect(server.url)
            try:
                join=json.dumps({'type':'join','mode':'p2p','room':ra})
                await alice.send(join)
                assert json.loads(await asyncio.wait_for(alice.recv(),3))['role']=='initiator'
                await bob.send(join)
                assert json.loads(await asyncio.wait_for(bob.recv(),3))['role']=='answerer'
                assert json.loads(await asyncio.wait_for(alice.recv(),3))['type']=='peer-joined'
                await alice.send(json.dumps({'type':'offer','sdp':'v=0\r\n'}))
                assert json.loads(await asyncio.wait_for(bob.recv(),3))['type']=='offer'
                await bob.send(json.dumps({'type':'answer','sdp':'v=0\r\n'}))
                assert json.loads(await asyncio.wait_for(alice.recv(),3))['type']=='answer'
            finally:
                await bob.close()
                await alice.close()
        # Batched to bound client and server sockets, not a fake transport.
        for offset in range(0,len(edges),8):
            await asyncio.gather(*(check_link(*edge) for edge in edges[offset:offset+8]))
        assert server.stats['pair_joins']==128
        assert server.stats['signals_forwarded']==128


def test_sixteen_overlay_all_64_real_tcp_signaling_routes():
    asyncio.run(_real_tcp_all_64_pair_signals())


async def _spawn_arguments_include_neighbor_cap(tmp_path, monkeypatch):
    args=SimpleNamespace(output=tmp_path/'swarm',peers=16,seed=1,
                         kind='loha',rank=2,local_steps=4,max_neighbors=8,
                         max_events_per_peer=0,epoch_v2_probe=False)
    sim=SwarmRunner(args)
    sim.room_id='A'*40
    sim.signaling='ws://127.0.0.1:8080/ws'
    argv_collected=[]
    class MockProc:
        def __init__(self):
            self.pid=177
            self.returncode=0
            self.stdout=asyncio.StreamReader()
            self.stdout.feed_eof()
        async def wait(self):return 0
    async def fake_spawn(*argv,**kwargs):
        argv_collected.extend(argv)
        return MockProc()
    monkeypatch.setattr('patchlib.swarm_sim.asyncio.create_subprocess_exec',fake_spawn)
    await sim.spawn(0)
    await sim.workers[0].monitor
    assert argv_collected[argv_collected.index('--max-neighbors')+1]=='8'
    assert argv_collected[argv_collected.index('--mode')+1]=='elastic'
    assert 'train-live' in argv_collected


def test_swarm_16_launcher_passes_degree_cap_to_real_training_cli(tmp_path,monkeypatch):
    asyncio.run(_spawn_arguments_include_neighbor_cap(tmp_path,monkeypatch))
