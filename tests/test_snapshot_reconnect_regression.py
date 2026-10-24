"""Regression tests for stalled cold joins, reconnects and NoneType errors."""
import asyncio
import time

from patchlib.snapshot_live import LiveSnapshotLink
from patchlib.shardp2p import ShardSession


class StubChannel:
    def __init__(self):
        self.readyState='open'
        self.handlers={}
        self.bufferedAmount=0
    def on(self,name):
        def decorate(func):
            self.handlers[name]=func
            return func
        return decorate
    def send(self,_data):
        pass
    def close(self):
        self.readyState='closed'
        cb=self.handlers.get('close')
        if cb:cb()


def test_reconnect_replaces_partially_initialized_snapshot_session(tmp_path):
    async def scenario():
        x=LiveSnapshotLink(tmp_path/'objects',tmp_path/'incoming',2**20,lambda:None)
        c1=StubChannel()
        x.attach(c1)
        c1.handlers['open']()
        x.session.remote_meta={'partial':True}
        x.session.outstanding['0'*64]=42
        x.session.local_data=b'old transport local manifest'
        x.error=ConnectionError('previous transport')
        c2=StubChannel()
        x.attach(c2)
        c2.handlers['open']()
        assert x.session.remote_meta is None
        assert x.session.outstanding=={}
        assert x.session.local_data is None
        assert x.error is None
        assert x.connected.is_set()
        # Old connection's close notification must NOT clear the new one.
        c1.handlers['close']()
        assert x.connected.is_set()
        c2.close()
        await asyncio.sleep(0)
    asyncio.run(scenario())


def test_unexpected_none_type_error_is_reported_and_closes_channel(tmp_path):
    async def scenario():
        x=LiveSnapshotLink(tmp_path/'objects',tmp_path/'incoming',2**20,lambda:None)
        c=StubChannel()
        x.attach(c);c.handlers['open']()
        def fails(_):
            missing=None
            return missing.readyState
        x.session.feed=fails
        c.handlers['message']('any')
        assert isinstance(x.error,AttributeError)
        assert 'NoneType' in str(x.error)
        assert not x.connected.is_set()
        assert c.readyState=='closed'
        await asyncio.sleep(0)
    asyncio.run(scenario())


def test_receiver_stall_is_detected(tmp_path):
    async def scenario():
        x=LiveSnapshotLink(tmp_path/'objects',tmp_path/'incoming',2**20,lambda:None)
        c=StubChannel();x.attach(c);c.handlers['open']()
        x.last_activity=time.monotonic()-100
        assert x.stalled(60)
        x.last_activity=time.monotonic()
        assert not x.stalled(60)
        c.close()
        await asyncio.sleep(0)
    asyncio.run(scenario())

def test_actual_signed_manifest_restarts_on_second_channel(tmp_path):
    import torch
    from torch import nn
    from patchlib.core import UniversalAdapter
    from patchlib.shards import export_state_sharded
    from patchlib.roomid import auto_identity

    class LinkedChannel(StubChannel):
        def send(self,data):
            asyncio.get_running_loop().call_soon(self.other.handlers['message'],data)

    async def scenario():
        peer, signer, _, verify=auto_identity(tmp_path/'identity.pem')
        net=nn.Sequential(nn.Linear(8,9),nn.Linear(9,2))
        adapter=UniversalAdapter(net,'tiny','rev1',kind='loha',rank=2)
        manifest=tmp_path/'state.fmanifest'
        export_state_sharded(net,tmp_path/'source-objects',manifest,
            model_id='tiny',revision='rev1',peer_id=peer,adapter=adapter,
            shard_bytes=64*1024)
        source=LiveSnapshotLink(tmp_path/'source-objects',tmp_path/'a',2**20,lambda:verify)
        target=LiveSnapshotLink(tmp_path/'target-objects',tmp_path/'b',2**20,lambda:verify)
        for round in (1,2):
            a,b=LinkedChannel(),LinkedChannel()
            a.other,b.other=b,a
            source.attach(a);target.attach(b)
            source.on_authenticated(signer)
            a.handlers['open']();b.handlers['open']()
            if round==1:
                # Simulate a previous transfer that died mid-manifest.
                target.session.remote_meta={'partial':True}
                target.session.outstanding['f'*64]=123
                a.close();b.close()
            else:
                sent=source.offer(manifest,signer)
                got=await asyncio.wait_for(target.completed.get(),8)
                assert got.is_file()
                import hashlib
                assert hashlib.sha256(got.read_bytes()).hexdigest()==sent
                a.close();b.close()
            await asyncio.sleep(.05)
    asyncio.run(scenario())
