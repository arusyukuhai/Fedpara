#!/usr/bin/env python3
"""Verify real Mummy multi-peer membership without starting an ML process.

Uses a fresh random room, so an existing training room is never touched.
Needs only `websockets` and `patchlib` from the project.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from patchlib.roomid import create_room_id


async def check(url, token):
    import websockets
    room=create_room_id()
    a='p'+'a'*32
    b='p'+'b'*32
    opts={'open_timeout':5,'max_size':65536}
    async with websockets.connect(url,**opts) as left:
        await left.send(json.dumps({'type':'group-join','room':room,'peer':a,'token':token}))
        first=json.loads(await asyncio.wait_for(left.recv(),5))
        if first.get('type')!='group-members' or first.get('peers')!=[]:
            raise RuntimeError(f'FIRST group-join rejected: {first}. Is the running Mummy server up to date?')
        async with websockets.connect(url,**opts) as right:
            await right.send(json.dumps({'type':'group-join','room':room,'peer':b,'token':token}))
            second=json.loads(await asyncio.wait_for(right.recv(),5))
            notice=json.loads(await asyncio.wait_for(left.recv(),5))
            if (second.get('type')!='group-members' or second.get('peers')!=[a] or
                notice.get('type')!='group-peer-joined' or notice.get('peer')!=b):
                raise RuntimeError(f'membership is broken: second={second}, notice={notice}')
    print('PASS: Mummy /ws supports two distinct group peers and broadcasts join events')
    print('This checks signaling only. Actual WebRTC/ICE and model synchronization still need validation.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ws',default='ws://127.0.0.1:8080/ws')
    p.add_argument('--token',default='')
    args=p.parse_args()
    try:
        asyncio.run(check(args.ws,args.token))
    except Exception as exc:
        print(f'FAIL: {type(exc).__name__}: {exc}')
        raise SystemExit(1)

if __name__=='__main__':main()
