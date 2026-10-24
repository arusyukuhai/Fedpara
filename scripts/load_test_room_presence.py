#!/usr/bin/env python3
"""End-to-end Mummy signaling/presence smoke test (NOT a GPU/WebRTC test).

Creates a new, otherwise-unused room. Never modifies existing model/journal.
Use --publish only against a local or operator-authorized server.
"""
import argparse
import asyncio
import json
import os
import secrets
import time
from pathlib import Path

from patchlib.roomid import create_room_id
from patchlib.room_manifest import make_hf_manifest, manifest_digest


async def run(ws_url, count, publish=False, timeout=45):
    import websockets
    if not 2 <= count <= 96:
        raise ValueError('count must be between 2 and 96')
    room=create_room_id()
    model=make_hf_manifest('sshleifer/tiny-gpt2','main',[{
        'repo':'roneneldan/TinyStories','revision':'main','config':None,
        'split':'train','columns':{'text':'text'},'weight':1.0
    }])
    clients=[]
    listeners=[]
    peers=['p'+secrets.token_hex(16) for _ in range(count)]
    knowledge=[set() for _ in range(count)]
    started=time.monotonic()
    try:
        # Load spread gently rather than 96 simultaneous websocket upgrades.
        for i,peer in enumerate(peers):
            ws=await websockets.connect(ws_url,max_size=65536,ping_interval=30,ping_timeout=60)
            clients.append(ws)
            payload={'type':'group-join','room':room,'peer':peer}
            if i==0:
                payload['manifest']=model
                if publish:
                    payload['publish']=True
                    if os.environ.get('ROOM_PUBLISH_TOKEN'):
                        payload['publish_token']=os.environ['ROOM_PUBLISH_TOKEN']
            await ws.send(json.dumps(payload))
            msg=json.loads(await asyncio.wait_for(ws.recv(),10))
            if msg.get('type')!='group-members':
                raise RuntimeError(f'join {i} failed: {msg}')
            if manifest_digest(msg['manifest'])!=manifest_digest(model):
                raise RuntimeError(f'manifest mismatch on peer {i}')
            knowledge[i]=set(msg['peers'])
            async def watch(client,idx):
                async for data in client:
                    evt=json.loads(data)
                    if evt.get('type')=='group-peer-joined':knowledge[idx].add(evt['peer'])
                    if evt.get('type')=='group-peer-left':knowledge[idx].discard(evt['peer'])
            listeners.append(asyncio.create_task(watch(ws,i)))
            if i % 12 == 0: await asyncio.sleep(.03)
        until=time.monotonic()+timeout
        while time.monotonic()<until:
            if all(knowledge[i]==set(peers)-{peers[i]} for i in range(count)):
                print(f'PASS {count} signaling participants; verified full rosters; '+
                      f'elapsed={time.monotonic()-started:.2f}s')
                return True
            await asyncio.sleep(.05)
        missing=[(i,len(set(peers)-{peers[i]}-knowledge[i])) for i in range(count)]
        raise TimeoutError(f'signaling rosters did not converge: {missing[:16]}')
    finally:
        for ws in clients:
            await ws.close()
        for task in listeners:task.cancel()
        await asyncio.gather(*listeners,return_exceptions=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ws',default='ws://127.0.0.1:8080/ws')
    parser.add_argument('--peers',type=int,default=48)
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    asyncio.run(run(args.ws,args.peers,args.publish))
