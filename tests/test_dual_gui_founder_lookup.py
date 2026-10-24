"""Two GUI instances must not poison the first trainer's founder role."""
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from patchlib.roomid import fetch_room_manifest, create_room_id
from patchlib.webview_controller import WebviewController
from patchlib.room_manifest import make_hf_manifest, load_manifest

SOURCE = [{'repo': 'roneneldan/TinyStories', 'revision': 'main', 'config': None,
           'split': 'train', 'columns': {'text': 'text'}, 'weight': 1.0}]


def params(room, *, seed=42):
    return {'room': room, 'model': 'sshleifer/tiny-gpt2', 'revision': 'main',
            'task': 'causal-lm', 'seed': seed, 'sources': SOURCE}


def test_room_lookup_is_read_only_websocket(tmp_path):
    import websockets
    room=create_room_id()
    manifest=make_hf_manifest('sshleifer/tiny-gpt2','main',SOURCE,
                              architecture='causal-lm',seed=42)
    observed=[]
    async def run():
        async def accept(ws):
            request=json.loads(await ws.recv())
            observed.append(request)
            await ws.send(json.dumps({'type':'room-description','room':room,
                                      'manifest':manifest}))
        async with websockets.serve(accept, '127.0.0.1', 0) as server:
            port=server.sockets[0].getsockname()[1]
            return await fetch_room_manifest(f'ws://127.0.0.1:{port}/ws',room)
    result=asyncio.run(run())
    assert result==manifest
    assert observed==[{'type':'group-describe','room':room,'token':''}]
    # No ephemeral peer ID: this call cannot trigger group-peer-joined or everJoined.
    assert all('peer' not in request for request in observed)


def test_old_signaling_server_fails_early_instead_of_ghost_join():
    import websockets
    room=create_room_id()
    observed=[]
    async def run():
        async def accept(ws):
            observed.append(json.loads(await ws.recv()))
            await ws.send(json.dumps({'type':'error','message':'Unknown message type'}))
        async with websockets.serve(accept,'127.0.0.1',0) as server:
            port=server.sockets[0].getsockname()[1]
            await fetch_room_manifest(f'ws://127.0.0.1:{port}/ws',room)
    with pytest.raises(RuntimeError,match='outdated'):
        asyncio.run(run())
    assert observed[0]['type']=='group-describe'


def test_room_definition_cannot_be_overwritten_by_second_gui(tmp_path):
    room=create_room_id()
    a=WebviewController(tmp_path)
    b=WebviewController(tmp_path)
    first=a.save(params(room))
    assert b.save(params(room))['sha256']==first['sha256']
    with pytest.raises(ValueError,match='固定'):
        b.save(params(room,seed=13))
    assert load_manifest(a._room_path(room))==first['manifest']


def test_simultaneous_conflicting_gui_save_has_one_winner(tmp_path):
    room=create_room_id()
    controllers=[WebviewController(tmp_path),WebviewController(tmp_path)]
    gate=threading.Barrier(2)
    def attempt(i):
        gate.wait()
        try:
            return controllers[i].save(params(room, seed=40+i))
        except ValueError as exc:
            return str(exc)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(attempt,(0,1)))
    assert sum(isinstance(x,dict) for x in results)==1
    winner=next(x for x in results if isinstance(x,dict))
    assert load_manifest(controllers[0]._room_path(room))==winner['manifest']


def test_two_windows_allocate_distinct_peer_profiles(tmp_path):
    from patchlib.profiles import allocate_profile
    a=allocate_profile(tmp_path)
    b=allocate_profile(tmp_path)
    try:
        assert a.name!=b.name
        assert a.journal!=b.journal
        assert a._lock_fd is not None and b._lock_fd is not None
    finally:
        a.close()
        b.close()


def test_parallel_automatic_genesis_is_serialized_not_rejected(tmp_path,monkeypatch):
    import patchlib.bootstrap as boot
    path=tmp_path/'same-room'/'genesis.fstate'
    called=[]
    def fake_export(model,output,**kw):
        time.sleep(.1)
        Path(output).write_bytes(b'genesis')
        called.append(True)
    def fake_verify(p,**kw):
        assert Path(p).read_bytes()==b'genesis'
        return {'model_id':'sshleifer/tiny-gpt2'}
    monkeypatch.setattr(boot,'export_state',fake_export)
    monkeypatch.setattr(boot,'verify_state',fake_verify)
    monkeypatch.setattr(boot,'semantic_state_sha256',lambda *a,**kw:'1'*64)
    def run():
        args=SimpleNamespace(auto_state=True, factory_file=None, trust_remote_code=False,
              allow_legacy_bin=False, room_manifest=None, state_base=str(path),
              model='sshleifer/tiny-gpt2', model_id=None,revision='main',device='cpu',
              moe_plan=None,max_state_gb=1)
        boot.ensure_state(args,load_model=lambda *a,**kw:(object(),'main'))
        return args.state_base
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:run(),range(2)))
    assert results==[str(path)]*2
    assert len(called)==1
    assert path.read_bytes()==b'genesis'


def test_nim_group_describe_does_not_modify_room_membership():
    root=Path(__file__).resolve().parent.parent
    server=(root/'src'/'main.nim').read_text()
    block=server.split('proc groupDescribe(')[1].split('proc groupJoin(')[0]
    statements='\n'.join(line for line in block.splitlines() if not line.strip().startswith('#'))
    assert 'roomEverJoined' not in statements
    assert 'groups.mgetOrPut' not in block
    assert 'group-peer-joined' not in statements
    assert '"room-description"' in block
    assert 'of "group-describe": groupDescribe(ws, doc)' in server
