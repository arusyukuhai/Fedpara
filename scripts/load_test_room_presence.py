#!/usr/bin/env python3
"""Signaling-only scalable presence soak test. Does not start WebRTC or train.

Use a disposable room and authorized server. Rate-limit ramp-up to avoid a
large simultaneous WebSocket handshake burst. Uses bounded-v2 only.
"""
import argparse
import asyncio
import json
import os
import secrets
import time

from patchlib.roomid import create_room_id
from patchlib.room_manifest import make_hf_manifest, manifest_digest
from patchlib.topology import neighbors


async def run(ws_url, count, publish=False, timeout=90, *, degree=8, depart=0):
    import websockets
    if not 2 <= count <= 4096:
        raise ValueError('participant count exceeds signaling room admission cap')
    if depart<0 or depart>=count:raise ValueError('depart must be in [0,peers)')
    if degree not in (2,4,6,8):raise ValueError('degree must be 2,4,6,8')
    room=create_room_id()
    model=make_hf_manifest('sshleifer/tiny-gpt2','main',[{
        'repo':'roneneldan/TinyStories','revision':'main','config':None,
        'split':'train','columns':{'text':'text'},'weight':1.0
    }])
    clients=[]
    listeners=[]
    peers=['p'+secrets.token_hex(16) for _ in range(count)]
    knowledge=[set() for _ in range(count)]
    revisions=[-1]*count
    started=time.monotonic()
    try:
        for i,peer in enumerate(peers):
            ws=await websockets.connect(ws_url,max_size=65536,ping_interval=30,ping_timeout=60)
            clients.append(ws)
            payload={'type':'group-join','room':room,'peer':peer,
                     'topology_v2':True,'topology_degree':degree}
            if i==0:
                payload['manifest']=model
                if publish:
                    payload['publish']=True
                    if os.environ.get('ROOM_PUBLISH_TOKEN'):
                        payload['publish_token']=os.environ['ROOM_PUBLISH_TOKEN']
            await ws.send(json.dumps(payload))
            pending=[]
            msg=json.loads(await asyncio.wait_for(ws.recv(),10))
            while msg.get('type') == 'group-neighbors':
                pending.append(msg)
                if len(pending)>128:raise RuntimeError('excessive pending notifications')
                msg=json.loads(await asyncio.wait_for(ws.recv(),10))
            if msg.get('type')!='group-members':
                raise RuntimeError(f'join {i} failed: {msg}')
            if manifest_digest(msg['manifest'])!=manifest_digest(model):
                raise RuntimeError(f'manifest mismatch on peer {i}')
            if msg.get('topology')!='bounded-v2' or msg.get('degree')!=degree:
                raise RuntimeError('server does not support compact signaling topology v2')
            knowledge[i]=set(msg['peers'])
            revisions[i]=msg.get('revision',-1)
            if len(knowledge[i])>degree:
                raise RuntimeError('server sent overfull compact neighbor view')

            def accept(idx,evt):
                kind=evt.get('type')
                if kind=='group-neighbors':
                    rev=evt.get('revision',-1)
                    if rev>revisions[idx]:
                        revisions[idx]=rev
                        knowledge[idx]=set(evt['peers'])

            for evt in pending:accept(i,evt)
            async def watch(client,idx):
                async for data in client:
                    accept(idx,json.loads(data))
            listeners.append(asyncio.create_task(watch(ws,i)))
            if i % 16 == 0: await asyncio.sleep(.03)

        async def verify(active):
            until=time.monotonic()+timeout
            while time.monotonic()<until:
                roster=sorted(peers[i] for i in active)
                if all(knowledge[i]==neighbors(peers[i],roster,max_degree=degree)
                       for i in active):
                    return
                await asyncio.sleep(.1)
            inconsistent=[i for i in active if knowledge[i] !=
                          neighbors(peers[i],sorted(peers[j] for j in active),max_degree=degree)]
            raise TimeoutError(f'presence did not converge for {len(inconsistent)} peers: {inconsistent[:16]}')

        active=list(range(count))
        await verify(active)
        if depart:
            leaving=active[-depart:]
            for i in leaving:await clients[i].close()
            active=active[:-depart]
            await verify(active)
        print(f'PASS {count} participants, departed={depart}, degree={degree}, '
              f'elapsed={time.monotonic()-started:.2f}s')
        return True
    finally:
        await asyncio.gather(*(ws.close() for ws in clients),return_exceptions=True)
        for task in listeners:task.cancel()
        await asyncio.gather(*listeners,return_exceptions=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ws',default='ws://127.0.0.1:8080/ws')
    parser.add_argument('--peers',type=int,default=96)
    parser.add_argument('--depart',type=int,default=0,
                        help='close the last N participants after the initial join convergence')
    parser.add_argument('--degree',type=int,default=8,choices=(2,4,6,8))
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    asyncio.run(run(args.ws,args.peers,args.publish,
                    degree=args.degree,depart=args.depart))
