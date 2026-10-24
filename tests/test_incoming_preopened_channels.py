"""Regression: aiortc fires peer datachannel with readyState='open' already.

See RTCSctpTransport._data_channel_receive: _setReadyState('open') before
RTCSctpTransport.emit('datachannel', channel). No subsequent 'open' fires.
"""
import asyncio
import hashlib
from pathlib import Path

from patchlib.p2p import PatchNode
from patchlib.elasticnet import ElasticNode
from patchlib.roomid import auto_identity
from patchlib.snapshot_live import LiveSnapshotLink
from patchlib.shards import export_state_sharded


class Channel:
    def __init__(self, state='open', label='fedpara-state'):
        self.readyState=state
        self.label=label
        self.id=1
        self.handlers={}
        self.sent=[]
        self.other=None
        self.bufferedAmount=0
    def on(self, name):
        def deco(cb):
            self.handlers[name]=cb
            return cb
        return deco
    def send(self, message):
        if self.readyState!='open':
            raise ConnectionError('closed')
        self.sent.append(message)
        if self.other is not None:
            asyncio.get_running_loop().call_soon(self.other.handlers['message'],message)
    def close(self):
        if self.readyState=='closed':return
        self.readyState='closed'
        if 'close' in self.handlers:
            self.handlers['close']()
    def transition_to_open(self):
        self.readyState='open'
        self.handlers['open']()


def test_remote_state_channel_preopened_needs_no_open_event(tmp_path):
    async def test():
        link=LiveSnapshotLink(tmp_path/'store',tmp_path/'recv',2**20,lambda:None)
        preopened=Channel()
        link.attach(preopened)
        assert link.connected.is_set(), 'attach must notice already-open channel'
        task=link.worker
        assert task is not None and not task.done()
        preopened.handlers['open']()   # repeated event must be idempotent
        assert link.worker is task
        preopened.close()
        assert not link.connected.is_set()
        await asyncio.sleep(0)
    asyncio.run(test())


def test_preopened_remote_state_channel_receives_full_manifest(tmp_path):
    import torch
    from torch import nn
    from patchlib.core import UniversalAdapter

    async def test():
        peer,signer,_,verify=auto_identity(tmp_path/'key.pem')
        model=nn.Sequential(nn.Linear(8,10),nn.Linear(10,3))
        adapter=UniversalAdapter(model,'test','r1',kind='loha',rank=2)
        manifest=tmp_path/'state.fmanifest'
        source_store=tmp_path/'source-objects'
        recv_store=tmp_path/'recv-objects'
        export_state_sharded(model,source_store,manifest,model_id='test',revision='r1',
                             peer_id=peer,adapter=adapter,shard_bytes=64*1024)
        source=LiveSnapshotLink(source_store,tmp_path/'source-in',2**20,lambda:verify)
        receiver=LiveSnapshotLink(recv_store,tmp_path/'receiver-in',2**20,lambda:verify)
        a=Channel('connecting');b=Channel('open')
        a.other,b.other=b,a
        source.attach(a)
        receiver.attach(b)
        assert receiver.connected.is_set()
        assert not source.connected.is_set()
        a.transition_to_open()
        assert source.connected.is_set()
        source.offer(manifest,signer)
        got=await asyncio.wait_for(receiver.completed.get(),timeout=8)
        assert got.is_file()
        assert receiver.session.remote_complete
        a.close();b.close()
        await asyncio.sleep(0)
    asyncio.run(test())


def test_preopened_incoming_elastic_control_immediately_sends_signed_hello(tmp_path):
    async def test():
        local,signer,pub,_=auto_identity(tmp_path/'a.pem')
        remote,_,_,_=auto_identity(tmp_path/'b.pem')
        n=ElasticNode('ws://localhost:8080/ws','x'*32,origin=local,remote=remote,
                      genesis='a'*64,public_key=pub,signer=signer)
        channel=Channel('open',label='fedpara-patch')
        n._attach_channel(channel)
        assert n.connected.is_set()
        assert n.hello_sent
        assert channel.sent and '"hello"' in channel.sent[0]
        initial=channel.sent[:]
        channel.handlers['open']()
        assert channel.sent==initial
        channel.close()
        await asyncio.sleep(0)
    asyncio.run(test())


def test_preopened_patch_channel_opens_and_has_single_pump():
    async def test():
        n=PatchNode('ws://localhost:8080/ws','x'*32)
        c=Channel('open',label='fedpara-patch')
        n._attach_channel(c)
        assert n.connected.is_set()
        first=n._send_task
        c.handlers['open']()
        assert first is n._send_task
        c.close();await asyncio.sleep(0)
    asyncio.run(test())


def test_preopened_sender_channel_drains_queued_manifest(tmp_path):
    from torch import nn
    from patchlib.core import UniversalAdapter

    async def scenario():
        pid,signer,_,verify=auto_identity(tmp_path/'sender-key.pem')
        model=nn.Sequential(nn.Linear(4,4))
        adapter=UniversalAdapter(model,'demo','r1',kind='loha',rank=2)
        source_objects=tmp_path/'source'
        mf=tmp_path/'source.fmanifest'
        export_state_sharded(model,source_objects,mf,model_id='demo',revision='r1',
                             peer_id=pid,adapter=adapter,shard_bytes=64*1024)
        sender=LiveSnapshotLink(source_objects,tmp_path/'send-cache',2**20,lambda:verify)
        receiver=LiveSnapshotLink(tmp_path/'recv-objects',tmp_path/'recv-cache',2**20,lambda:verify)
        sending_channel=Channel('open')  # answerer receives already-open channel
        receiving_channel=Channel('connecting')
        sending_channel.other=receiving_channel
        receiving_channel.other=sending_channel
        sender.attach(sending_channel)
        receiver.attach(receiving_channel)
        assert sender.connected.is_set()  # old implementation failed this
        assert not receiver.connected.is_set()
        receiving_channel.transition_to_open()
        sender.offer(mf,signer)
        assert (await asyncio.wait_for(receiver.completed.get(),8)).exists()
        sending_channel.close();receiving_channel.close()
        await asyncio.sleep(0)

    asyncio.run(scenario())
