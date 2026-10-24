"""Swarm harness regressions; real TCP WebSocket control plane, not fake callbacks."""
import asyncio
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from patchlib.swarm_sim import LocalSignaling, SwarmRunner, inventory_report, compare_fstates
from patchlib.room_manifest import make_manifest
from patchlib.state import export_state


def run(coro):return asyncio.run(coro)


async def signaled_group_roundtrip():
    import websockets
    mani=make_manifest('fixture/model','main','synthetic')
    room='A'*40
    identities=['p'+f'{i:032x}' for i in range(8)]
    async with LocalSignaling() as server:
        ws=[]
        try:
            for i in range(8):
                c=await websockets.connect(server.url)
                ws.append(c)
                m={'type':'group-join','room':room,'peer':identities[i]}
                if i==0:m['manifest']=mani
                await c.send(json.dumps(m))
                reply=json.loads(await asyncio.wait_for(c.recv(),2))
                assert reply['type']=='group-members'
                assert reply['peers']==identities[:i]
                assert reply['manifest']==mani
                assert reply['existing_room']==(i>0)
                if i>0:
                    for previous in ws[:-1]:
                        notification=json.loads(await asyncio.wait_for(previous.recv(),2))
                        assert notification=={'type':'group-peer-joined','peer':identities[i]}
            # Genuine TCP close, not mocked membership event.
            await ws[4].close()
            for idx, previous in enumerate(ws):
                if idx==4:continue
                notification=json.loads(await asyncio.wait_for(previous.recv(),2))
                assert notification=={'type':'group-peer-left','peer':identities[4]}
            ws[4]=await websockets.connect(server.url)
            await ws[4].send(json.dumps({'type':'group-join','room':room,'peer':identities[4]}))
            reply=json.loads(await asyncio.wait_for(ws[4].recv(),2))
            assert set(reply['peers'])==set(identities)-{identities[4]}
            assert server.stats['group_joins']==9
        finally:
            for c in ws:await c.close()


def test_eight_real_ws_join_leave_rejoin():
    run(signaled_group_roundtrip())


async def signaled_pair_roundtrip():
    import websockets
    async with LocalSignaling() as server:
        a=await websockets.connect(server.url)
        b=await websockets.connect(server.url)
        try:
            room='B'*40
            payload=json.dumps({'type':'join','mode':'p2p','room':room})
            await a.send(payload)
            assert json.loads(await a.recv())['role']=='initiator'
            await b.send(payload)
            assert json.loads(await b.recv())['role']=='answerer'
            assert json.loads(await a.recv())['type']=='peer-joined'
            await a.send(json.dumps({'type':'offer','sdp':'v=0\r\n'}))
            assert json.loads(await b.recv())=={'type':'offer','sdp':'v=0\r\n'}
            await b.send(json.dumps({'type':'answer','sdp':'v=0\r\n'}))
            assert json.loads(await a.recv())=={'type':'answer','sdp':'v=0\r\n'}
            await b.close()
            assert json.loads(await a.recv())['type']=='peer-left'
        finally:
            await a.close();await b.close()


def test_signaling_offer_answer_real_tcp():
    run(signaled_pair_roundtrip())


def _journal(path,entries):
    db=sqlite3.connect(path)
    db.execute('CREATE TABLE events(origin TEXT,seq INTEGER,digest TEXT)')
    db.executemany('INSERT INTO events VALUES(?,?,?)',entries)
    db.commit();db.close()


def test_inventory_detects_missing_and_conflicting(tmp_path):
    a,b=tmp_path/'a.sqlite',tmp_path/'b.sqlite'
    _journal(a,[('p'+'0'*32,1,'A')])
    _journal(b,[('p'+'0'*32,1,'A'),('p'+'1'*32,1,'B')])
    r=inventory_report([a,b]);assert r['per_peer_missing']==[1,0]
    assert not r['all_equal']
    a.unlink()
    _journal(a,[('p'+'0'*32,1,'Z'),('p'+'1'*32,1,'B')])
    r=inventory_report([a,b]);assert not r['all_equal']
    assert sum(r['per_peer_conflicting'])>=1


def test_numeric_model_comparison_measures_actual_weights(tmp_path):
    import torch
    m1=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.LayerNorm(3))
    m2=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.LayerNorm(3))
    m2.load_state_dict(m1.state_dict())
    p1,p2=tmp_path/'a.fstate',tmp_path/'b.fstate'
    kw={'model_id':'fixture/model','revision':'main'}
    export_state(m1,p1,peer_id='p'+'0'*32,**kw)
    export_state(m2,p2,peer_id='p'+'1'*32,**kw)
    r=compare_fstates([p1,p2]);assert r['ok'] and r['max_abs']==0
    with torch.no_grad():m2[0].weight.add_(0.2)
    export_state(m2,p2,peer_id='p'+'1'*32,**kw)
    r=compare_fstates([p1,p2]);assert not r['ok'] and r['max_abs']>0.1


def test_separate_peer_dirs_and_no_clobber(tmp_path):
    args=SimpleNamespace(output=tmp_path/'swarm',peers=8,seed=5)
    sim=SwarmRunner(args)
    assert len({p.journal for p in sim.workers})==8
    assert all(x.root.parent==sim.root for x in sim.workers)
    assert len({p.full_state for p in sim.workers})==8
