import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
from copy import deepcopy

import pytest
import torch

from patchlib.core import PatchError, UniversalAdapter
from patchlib.elastic import ElasticEngine, EventJournal, unpack_event
from patchlib.elasticnet import ElasticNode, decode_control
from patchlib.roomid import (auto_identity, create_room_id, pair_room,
                             validate_room, identity_from_public_b64, RoomDiscovery)
from test_livefed import Tiny


def test_room_id_secret_and_deterministic_pairs():
    a=create_room_id()
    validate_room(a)
    assert len(a)>=24
    with pytest.raises(ValueError):validate_room('abc')
    with pytest.raises(ValueError):validate_room('a'*100)
    p='p'+'0'*32;q='p'+'1'*32
    assert pair_room(a,p,q)==pair_room(a,q,p)
    assert pair_room(a,p,q)!=pair_room(create_room_id(),p,q)


def test_key_autocreate_restart_and_privacy(tmp_path):
    path=tmp_path/'node.pem'
    id1,sign,pub,v=auto_identity(path)
    assert path.is_file()
    assert (path.stat().st_mode & 0o777)==0o600
    id2,sign2,pub2,v2=auto_identity(path)
    assert (id1,pub)==(id2,pub2)
    data=b'a';digest=hashlib.sha256(data).hexdigest()
    v({'size':len(data),'sha256':digest,'sig':sign(len(data),digest)})
    v2({'size':len(data),'sha256':digest,'sig':sign2(len(data),digest)})
    with pytest.raises(PatchError):identity_from_public_b64(pub,'p'+'f'*32)


def test_auto_origin_signed_event_gossip_no_manual_verifiers(tmp_path):
    src=UniversalAdapter(Tiny(),'m','v',rank=3)
    dst=UniversalAdapter(Tiny(),'m','v',rank=3)
    for group0,group1 in zip(src.groups,dst.groups):
        path,_,attr=group0.aliases[0].rpartition('.')
        x=src.model.get_submodule(path).parametrizations[attr].original.detach()
        dst.model.get_submodule(path).parametrizations[attr].original.data.copy_(x)
    genesis=hashlib.sha256(b'shared genesis').hexdigest()
    alice,alice_sign,alice_pub,_=auto_identity(tmp_path/'alice.pem')
    bob,bob_sign,bob_pub,_=auto_identity(tmp_path/'bob.pem')
    a=ElasticEngine(src,alice,genesis,EventJournal(tmp_path/'a.db',genesis),signer=alice_sign,event_public=alice_pub)
    b=ElasticEngine(dst,bob,genesis,EventJournal(tmp_path/'b.db',genesis),signer=bob_sign,event_public=bob_pub)
    for g in a.adapter.groups:
        with torch.no_grad():
            for i,t in enumerate(g.overlay.local.parts):t.fill_(.025+i*.001)
    event=a.seal_local(2)
    assert b.ingest(event)
    assert not b.ingest(event)
    # Any third peer can validate this historic event, even after Alice leaves.
    meta=unpack_event(event,{})
    assert meta[0]==alice
    forged=bytearray(event);forged[-5]^=7
    with pytest.raises(PatchError):unpack_event(bytes(forged),{})


def test_self_certified_hello_peer_validation(tmp_path):
    alice,sa,pa,_=auto_identity(tmp_path/'a.pem')
    bob,sb,pb,_=auto_identity(tmp_path/'b.pem')
    g='a'*64
    class Channel:
        readyState='connecting';bufferedAmount=0
        def __init__(self): self.events={};self.other=None
        def on(self,name):
            def register(func): self.events[name]=func;return func
            return register
        def send(self,payload):self.other.events['message'](payload)
        def close(self):
            self.readyState='closed'
            if 'close' in self.events:self.events['close']()
    async def run():
        a=ElasticNode('ws://localhost/ws','b'*32,origin=alice,remote=bob,genesis=g,signer=sa,public_key=pa)
        b=ElasticNode('ws://localhost/ws','b'*32,origin=bob,remote=alice,genesis=g,signer=sb,public_key=pb)
        ac=Channel();bc=Channel();ac.other=bc;bc.other=ac
        a._attach_channel(ac);b._attach_channel(bc)
        ac.readyState=bc.readyState='open'
        ac.events['open']();bc.events['open']()
        await asyncio.sleep(.02)
        assert a.verifier is not None and b.verifier is not None
        assert a.controls.get_nowait()['type']=='hello'
        assert b.controls.get_nowait()['type']=='hello'
        a.control('heartbeat');assert b.controls.get_nowait()['type']=='heartbeat'
        ac.close();bc.close()
        await asyncio.sleep(.01)
    asyncio.run(run())


def test_room_discovery_membership_and_limit():
    id1='p'+'0'*32;id2='p'+'1'*32
    d=RoomDiscovery('wss://example.test/ws','A'*32,id1)
    d.accept({'type':'group-members','topology':'bounded-v2','degree':8,
              'revision':1,'online':2,'peers':[id2,id1]})
    assert d.members=={id2}
    d.accept({'type':'group-neighbors','topology':'bounded-v2',
              'revision':2,'online':1,'peers':[]})
    assert not d.members
    d.accept({'type':'group-neighbors','topology':'bounded-v2',
              'revision':3,'online':2,'peers':[id2]})
    assert d.members=={id2}
    with pytest.raises(PatchError):d.accept({'type':'group-members','topology':'bounded-v2',
              'degree':8,'revision':4,'online':3,'peers':[id2,id2]})
    with pytest.raises(PatchError,match='obsolete'):
        d.accept({'type':'group-peer-joined','peer':id2})
    with pytest.raises(PatchError,match='bounded-v2'):
        d.accept({'type':'group-members','peers':[id2]})


def test_room_only_elastic_training_with_no_peers(tmp_path,monkeypatch):
    from patchlib import elasticnet
    from types import SimpleNamespace
    from patchlib.core import _canonical_json
    # Solo training is allowed only AFTER signaling has confirmed that the
    # room really has no other participants. A failed lobby must not be
    # silently interpreted as an empty room.
    async def confirmed_empty_lobby(self,stop):
        self.members.clear()
        self.online=True
        self.updated.set()
        await stop.wait()
    import patchlib.roomid as roomid
    monkeypatch.setattr(roomid.RoomDiscovery,'run',confirmed_empty_lobby)
    base=tmp_path/'genesis.fstate'
    model=Tiny()
    from patchlib.state import export_state
    export_state(model,base,model_id='m',revision='v',peer_id='bootstrap')
    def load(_):return model,UniversalAdapter(model,'m','v',rank=3)
    def batches(_):
        while True:yield None
    def step(m,a,opt,batch,args):
        with a.lock:
            loss=m.forward();loss.backward();return loss.item()
    args=SimpleNamespace(
        room_id='x'*32,trust_remote_code=False,factory_file=None,allow_legacy_bin=False,
        resume=None,state_base=str(base),grad_accum=1,local_steps=2,rounds=2,
        signing_key=None,unsafe_unsigned=False,ws='ws://127.0.0.1:8080/ws',link=[],peer_id=None,
        trusted_peer_key=[],roster_file=None,model_id='m',journal=tmp_path/'events.db',
        self_public_key=None,max_journal_gb=1,max_delta_rms=1000.,lr=.0001,
        weight_decay=0.,device='cpu',wire_dtype='float16',token='',max_patch_mb=5,
        peer_timeout=.2,heartbeat_interval=.05,boot_grace=0.,sync_timeout=.1,
        inventory_interval=.05,inventory_cycle=.1,shard_state_manifest=None,
        shard_store=None,full_state_output=None,log_steps=100,peers=None,
        checkpoint_every=0,checkpoint_dir=tmp_path/'ck',shard_mb=4,max_state_gb=1)
    asyncio.run(elasticnet.train_elastic(args,load_model=load,training_batches=batches,run_step=step))
    assert args.peer_id.startswith('p') and len(args.peer_id)==33
    assert (tmp_path/'events.db.identity.pem').is_file()
    from patchlib.elastic import EventJournal
    import sqlite3
    with sqlite3.connect(args.journal) as db:
        genesis=db.execute("SELECT value FROM meta WHERE name='genesis'").fetchone()[0]
    journal=EventJournal(args.journal,genesis)
    try:
        events=list(journal.events())
        assert len(events)==2
        for ev in events:assert unpack_event(ev,{})[0]==args.peer_id
    finally:
        journal.close()


def test_room_discovery_over_real_websockets(tmp_path):
    # Uses a tiny fake lobby to integration-test client protocol, not Mummy.
    import websockets
    async def run():
        room='A'*32
        a='p'+'0'*32
        b='p'+'1'*32
        connections={}
        version=0
        async def handler(ws):
            nonlocal version
            msg=json.loads(await ws.recv())
            assert msg['type']=='group-join' and msg['room']==room
            assert msg['topology_v2'] is True and msg['topology_degree']==8
            pid=msg['peer']
            peers=list(connections)
            connections[pid]=ws
            version+=1
            await ws.send(json.dumps({'type':'group-members','topology':'bounded-v2',
                'degree':8,'revision':version,'online':len(connections), 'peers':peers}))
            for other in peers:
                await connections[other].send(json.dumps({'type':'group-neighbors',
                   'topology':'bounded-v2','revision':version,
                   'online':len(connections),'peers':[pid]}))
            try:
                async for _ in ws:pass
            finally:
                connections.pop(pid,None)
                version+=1
                for other in connections.values():
                    await other.send(json.dumps({'type':'group-neighbors',
                        'topology':'bounded-v2','revision':version,
                        'online':len(connections),'peers':[]}))
        async with websockets.serve(handler,'127.0.0.1',0) as server:
            port=server.sockets[0].getsockname()[1]
            stopa=asyncio.Event();stopb=asyncio.Event()
            la=RoomDiscovery(f'ws://127.0.0.1:{port}',room,a)
            lb=RoomDiscovery(f'ws://127.0.0.1:{port}',room,b)
            ta=asyncio.create_task(la.run(stopa))
            await asyncio.wait_for(la.updated.wait(),3);la.updated.clear()
            tb=asyncio.create_task(lb.run(stopb))
            for _ in range(100):
                if a in lb.members and b in la.members:break
                await asyncio.sleep(.01)
            assert a in lb.members and b in la.members
            stopb.set();tb.cancel();await asyncio.gather(tb,return_exceptions=True)
            for _ in range(100):
                if b not in la.members:break
                await asyncio.sleep(.01)
            assert b not in la.members
            stopa.set();ta.cancel();await asyncio.gather(ta,return_exceptions=True)
    asyncio.run(run())
