#!/usr/bin/env python3
"""Actual WebRTC transport diagnostic against the running Mummy signaling server.

Launches TWO distinct signed peers in one Python process. It does not load a
model or modify any Fedpara checkpoint. Run after updating and starting Mummy.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from patchlib.roomid import auto_identity, pair_room, create_room_id
from patchlib.elasticnet import ElasticNode

async def check(url,token,timeout):
    with tempfile.TemporaryDirectory(prefix='fedpara-peer-diagnostic-') as directory:
        a,sa,pa,_=auto_identity(Path(directory)/'a.pem')
        b,sb,pb,_=auto_identity(Path(directory)/'b.pem')
        pair=pair_room(create_room_id(),a,b)
        genesis='0'*64
        one=ElasticNode(url,pair,token,origin=a,remote=b,genesis=genesis,
                        signer=sa,public_key=pa)
        two=ElasticNode(url,pair,token,origin=b,remote=a,genesis=genesis,
                        signer=sb,public_key=pb)
        tasks=[asyncio.create_task(one.run()),asyncio.create_task(two.run())]
        try:
            end=asyncio.get_running_loop().time()+timeout
            while asyncio.get_running_loop().time()<end:
                if all(n.connected.is_set() and n.verified_at for n in (one,two)):
                    print('PASS: two distinct peers, DataChannel open, signatures verified')
                    print('peer A:',a,'peer B:',b)
                    return
                if any(t.done() for t in tasks):break
                await asyncio.sleep(.15)
            print('FAIL: Mummy->ICE/DTLS/SCTP->Ed25519 handshake incomplete')
            for tag,n,t in zip('AB',(one,two),tasks):
                print(tag,'connected=',n.connected.is_set(),
                      'signed_hello=',bool(n.verified_at),
                      'task_done=',t.done(),
                      'error=',repr(n.last_error or (t.exception() if t.done() and not t.cancelled() else None)))
            raise SystemExit(2)
        finally:
            for t in tasks:t.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--ws',default='ws://127.0.0.1:8080/ws')
    p.add_argument('--token',default='')
    p.add_argument('--timeout',type=float,default=25.)
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(name)s %(message)s')
    asyncio.run(check(args.ws,args.token,args.timeout))
