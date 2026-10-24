"""Regression coverage: peer-specific shuffling and SCTP-close recovery."""
import asyncio
from types import SimpleNamespace

from patchlib.data_sampling import peer_sampling_seed, shuffled_cycle
from patchlib.p2p import PatchNode
from patchlib.elasticnet import ElasticNode
from patchlib.room_manifest import make_hf_manifest
from patchlib.hf_datasets import rows_from_room
from patchlib.cli import training_batches


class Channel:
    def __init__(self):
        self.readyState='open'
        self.bufferedAmount=0
        self.callbacks={}
    def on(self, event):
        def register(cb):
            self.callbacks[event]=cb
            return cb
        return register
    def send(self, data):
        assert self.readyState=='open'
    def close(self):
        self.readyState='closed'
        self.callbacks['close']()


def test_peer_sampling_unique_stable_and_epoch_shuffle():
    first=peer_sampling_seed('p'+'a'*32,42)
    second=peer_sampling_seed('p'+'b'*32,42)
    assert first != second
    assert first==peer_sampling_seed('p'+'a'*32,42)
    assert first != peer_sampling_seed('p'+'a'*32,43)
    records=list(range(64))
    a=list(__import__('itertools').islice(shuffled_cycle(records,first),128))
    b=list(__import__('itertools').islice(shuffled_cycle(records,second),128))
    assert a != b
    assert a==list(__import__('itertools').islice(shuffled_cycle(records,first),128))
    assert set(a[:64])==set(records)==set(a[64:128])


def test_hf_loader_gets_distinct_shuffle_seeds():
    manifest=make_hf_manifest('org/model','main',[
        {'repo':'org/data','revision':'main','config':None,'split':'train',
         'columns':{'text':'text'},'weight':1}],seed=20)
    calls=[]
    class Stream:
        def shuffle(self, *, seed, buffer_size):
            calls.append(seed)
            return self
        def __iter__(self):
            return iter([{'text':'sample'}])
    def loader(**kwargs): return Stream()
    next(rows_from_room(manifest,peer_id='p'+'a'*32,loader=loader))
    next(rows_from_room(manifest,peer_id='p'+'b'*32,loader=loader))
    assert len(calls)==2 and calls[0]!=calls[1]


def test_local_safetensor_files_use_independent_order(tmp_path):
    # Fake load_file suffices to inspect which file each peer chose.
    from unittest.mock import patch
    files=[tmp_path/f'batch-{n:03}.safetensors' for n in range(25)]
    for file in files:file.touch()
    def get(peer):
        args=SimpleNamespace(room_manifest=None,batches_dir=tmp_path,text_file=None,jsonl=None,
                             peer_id=peer,hf_task='causal-lm')
        with patch('patchlib.cli.load_file',side_effect=lambda f: f.name):
            iterator=training_batches(args)
            return [next(iterator) for _ in range(25)]
    x=get('p'+'a'*32);y=get('p'+'b'*32)
    assert x != y
    assert x == get('p'+'a'*32)
    assert len(set(x))==25


def test_closing_patch_datachannel_is_reported_instead_of_stalling():
    async def scenario():
        node=PatchNode('ws://localhost:8080/ws','a'*32)
        ch=Channel();node._attach_channel(ch)
        assert node.connected.is_set() and not node.link_closed.is_set()
        ch.close()
        assert node.link_closed.is_set() and not node.connected.is_set()
        assert 'closed' in str(node.last_error)
        await asyncio.sleep(0)
    asyncio.run(scenario())


def test_closing_signed_datachannel_wakes_reconnection():
    async def scenario():
        node=ElasticNode('ws://localhost:8080/ws','a'*32,
                         origin='p'+'a'*32,remote='p'+'b'*32,
                         genesis='a'*64,unsafe=True)
        ch=Channel();node._attach_channel(ch)
        assert node.connected.is_set()
        ch.close()
        assert node.link_closed.is_set() and not node.connected.is_set()
        assert not node.hello_received
        await asyncio.sleep(0)
    asyncio.run(scenario())


def test_websocket_open_does_not_trap_closed_datachannel(monkeypatch):
    """Regression: stale ws.recv must not mask a closed DataChannel."""
    import json
    import sys
    import types

    class FakePC:
        instance=None
        def __init__(self, configuration):
            self.events={}
            self.localDescription=None
            self.connectionState='connected'
            self.channel=Channel()
            FakePC.instance=self
        def on(self,name):
            def wrap(f):self.events[name]=f;return f
            return wrap
        def createDataChannel(self,*args,**kwargs):return self.channel
        async def createOffer(self):return SimpleNamespace(sdp='fake-offer')
        async def setLocalDescription(self,sdp):self.localDescription=sdp
        async def close(self):pass

    class FakeSignal:
        def __init__(self):
            self.queue=asyncio.Queue()
            self.queue.put_nowait(json.dumps({'type':'joined','role':'initiator','iceServers':[]}))
            self.queue.put_nowait(json.dumps({'type':'peer-joined'}))
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def send(self,message):
            if json.loads(message).get('type')=='offer':
                asyncio.get_running_loop().call_later(0.02,FakePC.instance.channel.close)
        async def recv(self):return await self.queue.get()
        async def close(self,**kwargs):pass

    fake_aiortc=types.ModuleType('aiortc')
    fake_aiortc.RTCPeerConnection=FakePC
    fake_aiortc.RTCSessionDescription=lambda **kw:SimpleNamespace(**kw)
    fake_aiortc.RTCConfiguration=lambda **kw:SimpleNamespace(**kw)
    fake_aiortc.RTCIceServer=lambda **kw:SimpleNamespace(**kw)
    fake_sdp=types.ModuleType('aiortc.sdp')
    fake_sdp.candidate_from_sdp=lambda value:value
    monkeypatch.setitem(sys.modules,'aiortc',fake_aiortc)
    monkeypatch.setitem(sys.modules,'aiortc.sdp',fake_sdp)
    import websockets
    monkeypatch.setattr(websockets,'connect',lambda *args,**kwargs:FakeSignal())

    async def scenario():
        node=PatchNode('ws://localhost:8080/ws','a'*32)
        try:
            await asyncio.wait_for(node.run(),timeout=1.5)
        except ConnectionError as exc:
            assert 'DataChannel closed' in str(exc)
        else:
            assert False,'stale open signaling must not prevent reconnect'
        assert node.stopped.is_set()
    asyncio.run(scenario())


def test_full_control_queue_does_not_drop_signed_connection(tmp_path):
    from patchlib.roomid import auto_identity
    async def scenario():
        alice,asign,apub,_=auto_identity(tmp_path/'alice.pem')
        bob,bsign,bpub,_=auto_identity(tmp_path/'bob.pem')
        receiving=ElasticNode('ws://localhost:8080/ws','b'*32,origin=alice,
                              remote=bob,genesis='a'*64,signer=asign,public_key=apub)
        sending=ElasticNode('ws://localhost:8080/ws','b'*32,origin=bob,
                            remote=alice,genesis='a'*64,signer=bsign,public_key=bpub)
        # The channels use the existing fake pre-open semantics, with actual
        # signed frames to exercise replay protection and queue backpressure.
        class Pipe(Channel):
            peer=None
            pending=None
            def send(self, payload):
                super().send(payload)
                if 'message' in self.peer.callbacks:
                    self.peer.callbacks['message'](payload)
                else:
                    if self.pending is None: self.pending=[]
                    self.pending.append(payload)
            def flush(self):
                for frame in self.pending or []:
                    self.peer.callbacks['message'](frame)
                self.pending=[]
        a=Pipe();b=Pipe();a.peer=b;b.peer=a
        receiving._attach_channel(a)
        sending._attach_channel(b)
        a.flush();b.flush()
        assert receiving.hello_received and sending.hello_received
        while not receiving.controls.full():
            receiving.controls.put_nowait({'type':'inventory'})
        before=receiving.verified_at
        assert sending.control('heartbeat')
        assert receiving.verified_at >= before
        assert receiving.connected.is_set() and not receiving.protocol_failure.is_set()
        a.close();b.close()
        await asyncio.sleep(0)
    asyncio.run(scenario())
