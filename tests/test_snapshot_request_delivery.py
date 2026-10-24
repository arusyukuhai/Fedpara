"""Cold join regressions with authenticated patch channel preceding state channel."""
import asyncio
from types import SimpleNamespace

from patchlib.elasticnet import ElasticNode, PeerSlot, encode_control, service_snapshot_requests
from patchlib.roomid import auto_identity


class FakeChannel:
    readyState='open'
    bufferedAmount=0
    def __init__(self):
        self.handlers={}
        self.sent=[]
    def on(self,name):
        def deco(fn):
            self.handlers[name]=fn
            return fn
        return deco
    def send(self,payload):
        self.sent.append(payload)
    def close(self):
        self.readyState='closed'
        if 'close' in self.handlers:
            self.handlers['close']()


def test_snapshot_request_survives_closed_secondary_channel(tmp_path):
    async def run():
        founder,founder_signer,founder_pub,_=auto_identity(tmp_path/'founder.pem')
        newcomer,join_signer,join_pub,_=auto_identity(tmp_path/'joiner.pem')
        node=ElasticNode('ws://localhost/ws','r'*32,origin=founder,remote=newcomer,
                         genesis='9'*64,public_key=founder_pub,signer=founder_signer,
                         snapshot_store=tmp_path/'objects',snapshot_recv=tmp_path/'receive')
        control=FakeChannel()
        node._attach_channel(control)
        control.handlers['open']()
        wire=encode_control({'type':'hello','peer':newcomer,'genesis':'9'*64,
                             'counter':1,'pub':join_pub},join_signer)
        control.handlers['message'](wire)
        request_id='1234567890abcdef'
        control.handlers['message'](encode_control({
            'type':'snapshot-request','peer':newcomer,'genesis':'9'*64,
            'counter':2,'request_id':request_id},join_signer))
        assert node.controls.get_nowait()['type']=='hello'
        request=node.controls.get_nowait()
        assert request['type']=='snapshot-request'
        # The actual incoming callback now records and ACKs the request immediately.
        assert node.snapshot_pending
        assert node.snapshot_pending_id==request['request_id']
        slot=PeerSlot(newcomer,'r'*32,None,node=node)
        served=[]
        async def serve(s):
            served.append(s.peer)
            return True
        await service_snapshot_requests({newcomer:slot},source_allowed=True,send_snapshot=serve,now=10)
        assert node.snapshot_pending and not served
        state=FakeChannel()
        node.snapshot_link.attach(state)
        state.handlers['open']()
        await service_snapshot_requests({newcomer:slot},source_allowed=True,send_snapshot=serve,now=11)
        assert served==[newcomer]
        assert not node.snapshot_pending
        state.close();control.close()
        await asyncio.sleep(.01)
    asyncio.run(run())


def test_snapshot_request_not_lost_during_temporary_backpressure(tmp_path):
    async def run():
        founder,signer,pub,_=auto_identity(tmp_path/'founder.pem')
        new,_,_,_=auto_identity(tmp_path/'new.pem')
        node=ElasticNode('ws://localhost/ws','r'*32,origin=founder,remote=new,
                         genesis='9'*64,public_key=pub,signer=signer,
                         snapshot_store=tmp_path/'s',snapshot_recv=tmp_path/'r')
        ctrl=FakeChannel()
        node._attach_channel(ctrl);ctrl.handlers['open']()
        state=FakeChannel()
        node.snapshot_link.attach(state);state.handlers['open']()
        node.snapshot_pending=True
        node.snapshot_pending_id='0123456789abcdef'
        node.verified_at=1
        slot=PeerSlot(new,'r'*32,None,node=node)
        calls=[]
        async def serve(s):
            calls.append(s.peer)
            return len(calls)>1
        await service_snapshot_requests({new:slot},source_allowed=True,send_snapshot=serve,now=10)
        assert node.snapshot_pending
        await service_snapshot_requests({new:slot},source_allowed=True,send_snapshot=serve,now=11)
        assert not node.snapshot_pending
        assert len(calls)==2
        state.close();ctrl.close()
        await asyncio.sleep(.01)
    asyncio.run(run())


def test_newcomer_cannot_serve_unverified_genesis(tmp_path):
    async def run():
        founder,signer,pub,_=auto_identity(tmp_path/'founder.pem')
        new,_,_,_=auto_identity(tmp_path/'new.pem')
        node=ElasticNode('ws://localhost/ws','r'*32,origin=founder,remote=new,
                         genesis='9'*64,public_key=pub,signer=signer,
                         snapshot_store=tmp_path/'s',snapshot_recv=tmp_path/'r')
        ctrl=FakeChannel();node._attach_channel(ctrl);ctrl.handlers['open']()
        state=FakeChannel();node.snapshot_link.attach(state);state.handlers['open']()
        node.snapshot_pending=True;node.snapshot_pending_id='0123456789abcdef';node.verified_at=1
        slot=PeerSlot(new,'r'*32,None,node=node)
        served=[]
        async def serve(s):served.append(True);return True
        await service_snapshot_requests({new:slot},source_allowed=False,send_snapshot=serve,now=10)
        assert node.snapshot_pending and not served
        state.close();ctrl.close()
        await asyncio.sleep(.01)
    asyncio.run(run())


def test_two_authenticated_peers_snapshot_after_late_state_channel(tmp_path):
    """Paired signed control + state links, with request preceding state open."""
    import hashlib
    import torch
    from torch import nn
    from patchlib.core import UniversalAdapter
    from patchlib.shards import export_state_sharded,read_manifest
    from patchlib.snapshot_live import snapshot_inventory
    from patchlib.elastic import EventJournal

    class PairChannel(FakeChannel):
        def __init__(self):
            super().__init__()
            self.other=None
            self.readyState='connecting'
        def send(self,payload):
            assert self.readyState=='open'
            asyncio.get_running_loop().call_soon(self.other.handlers['message'],payload)

    async def run():
        alice,alice_signer,alice_pub,_=auto_identity(tmp_path/'alice.pem')
        bob,bob_signer,bob_pub,_=auto_identity(tmp_path/'bob.pem')
        a=ElasticNode('ws://localhost/ws','r'*32,origin=alice,remote=bob,genesis='a'*64,
                      public_key=alice_pub,signer=alice_signer,
                      snapshot_store=tmp_path/'alice-store',snapshot_recv=tmp_path/'alice-in')
        b=ElasticNode('ws://localhost/ws','r'*32,origin=bob,remote=alice,genesis='a'*64,
                      public_key=bob_pub,signer=bob_signer,
                      snapshot_store=tmp_path/'bob-store',snapshot_recv=tmp_path/'bob-in')
        ca,cb=PairChannel(),PairChannel()
        ca.other=cb;cb.other=ca
        a._attach_channel(ca);b._attach_channel(cb)
        ca.readyState=cb.readyState='open'
        ca.handlers['open']();cb.handlers['open']()
        await asyncio.sleep(.05)
        assert a.verified_at and b.verified_at
        # Remove signed hello from the queues. This is what the training loop does.
        assert a.controls.get_nowait()['type']=='hello'
        assert b.controls.get_nowait()['type']=='hello'
        rid='baddcafedeadbeef'
        assert b.control('snapshot-request',request_id=rid)
        await asyncio.sleep(.05)
        msg=a.controls.get_nowait()
        assert msg['type']=='snapshot-request'
        assert a.snapshot_pending and a.snapshot_pending_id==msg['request_id']
        slot=PeerSlot(bob,'r'*32,None,node=a)
        served=[]
        async def serve(slot):
            model=nn.Linear(3,4)
            adapter=UniversalAdapter(model,'toy','rev1',kind='lora',rank=2)
            journal=EventJournal(tmp_path/'empty.sqlite','a'*64)
            try:
                tip,count,cutoffs=snapshot_inventory(journal)
            finally:
                journal.close()
            manifest=tmp_path/'snap.fmanifest'
            export_state_sharded(model,a.snapshot_link.store,manifest,model_id='toy',
                revision='rev1',peer_id=alice,adapter=adapter,shard_bytes=64*1024)
            digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
            assert a.control('snapshot-tip',manifest_sha256=digest,events=count,
                             inventory_sha256=tip,cutoffs=cutoffs)
            a.snapshot_link.offer(manifest,alice_signer)
            served.append(True)
            return True
        await service_snapshot_requests({bob:slot},source_allowed=True,send_snapshot=serve,now=10)
        assert not served and a.snapshot_pending
        sa,sb=PairChannel(),PairChannel()
        sa.other=sb;sb.other=sa
        a._attach_extra_channel(sa);b._attach_extra_channel(sb)
        sa.readyState=sb.readyState='open'
        sa.handlers['open']();sb.handlers['open']()
        await service_snapshot_requests({bob:slot},source_allowed=True,send_snapshot=serve,now=11)
        assert served and not a.snapshot_pending
        received=await asyncio.wait_for(b.snapshot_link.completed.get(),timeout=10)
        assert read_manifest(received)['peer_id']==alice
        await asyncio.sleep(.05)
        frames=[]
        while not b.controls.empty():frames.append(b.controls.get_nowait())
        assert any(m['type']=='snapshot-tip' for m in frames)
        assert hashlib.sha256(received.read_bytes()).hexdigest()==next(
            m['manifest_sha256'] for m in frames if m['type']=='snapshot-tip')
        sa.close();sb.close();ca.close();cb.close()
        await asyncio.sleep(.01)
    asyncio.run(run())
