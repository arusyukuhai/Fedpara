"""Audit regressions: no event-loop stalls during heavy model verification."""
import asyncio
import time
import torch
from torch import nn
from patchlib import snapshot_live
from patchlib.snapshot_live import LiveSnapshotLink
from patchlib.shards import export_state_sharded, read_manifest, shard_map
from patchlib.roomid import auto_identity


class Channel:
    def __init__(self):
        self.readyState='open'
        self.bufferedAmount=0
        self.handlers={}
    def on(self, event):
        def decorate(callback):
            self.handlers[event]=callback
            return callback
        return decorate
    def send(self, data):
        if self.other.readyState=='open':
            asyncio.get_running_loop().call_soon(self.other.handlers['message'],data)
    def close(self):
        if self.readyState=='closed':return
        self.readyState='closed'
        self.handlers['close']()


def test_multishard_verification_keeps_event_loop_responsive(tmp_path,monkeypatch):
    async def scenario():
        peer, signer, _, verify=auto_identity(tmp_path/'identity.pem')
        model=nn.Linear(2048,512)
        manifest=tmp_path/'model.fmanifest'
        export_state_sharded(model,tmp_path/'source',manifest,
            model_id='audit',revision='v1',peer_id=peer,shard_bytes=64*1024)
        assert len(shard_map(read_manifest(manifest))) > 4
        source=LiveSnapshotLink(tmp_path/'source',tmp_path/'source_in',8*1024**2,lambda:verify)
        target=LiveSnapshotLink(tmp_path/'target',tmp_path/'target_in',8*1024**2,lambda:verify)
        a,b=Channel(),Channel()
        a.other,b.other=b,a
        source.attach(a);target.attach(b)
        source.on_authenticated(signer)
        a.handlers['open']();b.handlers['open']()
        original=snapshot_live.verify_shards
        started=asyncio.Event()
        # Called from a worker; use a threading.Event rather than asyncio.Event
        import threading
        started_thread=threading.Event()
        def slow_verify(meta,store):
            started_thread.set()
            time.sleep(.3)
            return original(meta,store)
        monkeypatch.setattr(snapshot_live,'verify_shards',slow_verify)
        source.offer(manifest,signer)
        completed=asyncio.create_task(target.completed.get())
        deadline=time.monotonic()+8
        while not started_thread.is_set() and time.monotonic()<deadline:
            await asyncio.sleep(.01)
        assert started_thread.is_set(), f'verification never started: err={target.error} remote={target.session.remote_meta is not None} outstanding={len(target.session.outstanding)} iterator={target.session.remote_iterator is not None}'
        begin=time.monotonic()
        await asyncio.sleep(.05)
        elapsed=time.monotonic()-begin
        assert elapsed < .2, f'event loop stalled {elapsed:.3f}s in snapshot hash verification'
        path=await asyncio.wait_for(completed,8)
        assert path.is_file()
        await asyncio.sleep(.05)  # allow ACK delivery
        assert source.session.acked_local
        a.close();b.close()
        await asyncio.sleep(.05)
    asyncio.run(scenario())
