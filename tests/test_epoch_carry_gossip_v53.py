"""Adjacent-epoch wire repair, explicit limits and real signed LoHA replay."""
import asyncio
import base64
import hashlib

import pytest
import torch

from patchlib.core import PatchError,UniversalAdapter
from patchlib.continual import ContinualElasticEngine
from patchlib.elastic import EventJournal
from patchlib.epoch_carry_v46 import CrossEpochCarryArchive
from patchlib.epoch_carry_gossip_v53 import (
    AdjacentEpochCarryGossip,validate_carry_control,CHUNK)
from patchlib.identity import generate_keypair,signer_from_pem,verifier_from_pem
from patchlib.async_ingest import AsyncInboundPipeline

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(4,4,bias=False)
    def forward(self):return self.linear(torch.ones(4)).square().sum()

class Link:
    hello_received=True
    remote_supports_generation=True
    bridge_only=True
    def __init__(self,other,peer):
        self.other=other;self.peer=peer;self.messages=[];self.accept=True
    def control(self,kind,**kw):
        if not self.accept:return False
        self.messages.append((kind,kw))
        self.other.receive(self.peer,kind,kw)
        return True

def context(tmp_path):
    torch.manual_seed(31)
    key=tmp_path/'key';pub=tmp_path/'pub'
    generate_keypair(key,pub)
    signer=signer_from_pem(key)
    trusted={'alice':verifier_from_pem(pub)}
    def engine(path,genesis,peer,signer_=None):
        model=Tiny()
        adapter=UniversalAdapter(model,'test','v1',kind='loha',rank=2)
        journal=EventJournal(tmp_path/path,genesis,max_bytes=0)
        return ContinualElasticEngine(adapter,peer,genesis,journal,trusted=trusted,
                                      unsafe=False,signer=signer_)
    src=engine('old.db','a'*64,'alice',signer)
    dst=engine('new.db','b'*64,'bob')
    archive=CrossEpochCarryArchive(tmp_path/'carry.db',source_genesis=src.genesis,
        target_genesis=dst.genesis,certified_manifest='c'*64,epoch=1,cutoffs={})
    archive.replay(dst)
    source=AdjacentEpochCarryGossip(source_genesis=src.genesis,
        source_journal=src.journal,spool=tmp_path/'sender')
    recv=AdjacentEpochCarryGossip(source_genesis=src.genesis,
        source_journal=src.journal,spool=tmp_path/'receiver',carry_archive=archive)
    return src,dst,archive,source,recv

@pytest.mark.asyncio
async def test_signed_cross_epoch_transfer_and_durable_dedup(tmp_path):
    src,dst,archive,sender,receiver=context(tmp_path)
    try:
        opt=torch.optim.AdamW(src.adapter.trainable_parameters(),lr=.01)
        for _ in range(2):
            src.adapter.model().backward();opt.step();opt.zero_grad()
            src.seal_local(1)
        assert src.journal.count()==2
        committed=[]
        async def callback(info,wire,source,applied):
            committed.append((info.origin,info.seq,applied))
        pipeline=AsyncInboundPipeline(dst,callback,carry_archive=archive)
        pipeline.start()
        sender_link=Link(receiver,'old')
        receiver_link=Link(sender,'new')
        for i in range(500):
            sender.tick([('new',sender_link)],now=float(i+1))
            receiver.tick([('old',receiver_link)],now=float(i+1))
            await sender.flush_completed(lambda _peer,_wire: False)
            await receiver.flush_completed(pipeline.submit)
            if archive.journal.count()>=2:break
            await asyncio.sleep(.003)
        await pipeline.drain()
        assert archive.journal.count()==2
        assert sorted(committed)==[('alice',1,True),('alice',2,True)]
        assert receiver.events_received==2
        assert all(len(f['data'])<=4*((CHUNK+2)//3) for kind,f in sender_link.messages
                   if kind=='generation-carry-chunk')
        await pipeline.stop()
    finally:
        sender.close();receiver.close();archive.close();src.journal.close();dst.journal.close()

@pytest.mark.asyncio
async def test_reconnect_resumes_missing_signed_body(tmp_path):
    src,dst,archive,sender,recv=context(tmp_path)
    try:
        wire=b'FPEL1'+bytes(range(256))*66 # transport only, signed check is the downstream pipeline's job
        src.journal.append('alice',1,wire)
        accepted=[]
        l1=Link(recv,'old');l2=Link(sender,'new')
        sender.tick([('new',l1)],now=1)
        recv.tick([('old',l2)],now=1)
        l1.accept=False
        sender.tick([('new',l1)],now=2)
        assert recv.events_received==0
        sender.disconnect('new');recv.disconnect('old')
        l1.accept=True
        for n in range(3,80):
            sender.tick([('new',l1)],now=float(n))
            recv.tick([('old',l2)],now=float(n))
            await sender.flush_completed(lambda peer,wire:False)
            await recv.flush_completed(lambda peer,wire:(accepted.append(wire) or True))
            if accepted:break
            await asyncio.sleep(.002)
        assert accepted==[wire]
        assert recv.events_received==1
    finally:
        sender.close();recv.close();archive.close();src.journal.close();dst.journal.close()


def test_rejects_invalid_or_malicious_fragments(tmp_path):
    src,dst,archive,sender,recv=context(tmp_path)
    try:
        digest=hashlib.sha256(b'x'*CHUNK).hexdigest()
        with pytest.raises(PatchError):
            validate_carry_control('generation-carry-need',{
                'source':'a'*64,'origin':'alice','seq':True,'digest':digest})
        with pytest.raises(PatchError):
            validate_carry_control('generation-carry-chunk',{
                'source':'a'*64,'origin':'alice','seq':1,'digest':digest,
                'length':CHUNK,'index':1,'data':base64.b64encode(b'x').decode()})
        with pytest.raises(PatchError):
            validate_carry_control('generation-carry-offer',{
                'source':'a'*64,'events':[['alice',1,digest,CHUNK]]*17})
        # Wrong epoch is never coerced into source epoch.
        recv.receive('old','generation-carry-offer',{
            'source':'d'*64,'events':[['alice',1,digest,CHUNK]]})
        assert not recv._peers
    finally:
        sender.close();recv.close();archive.close();src.journal.close();dst.journal.close()

@pytest.mark.asyncio
async def test_same_generation_relay_repairs_late_joiner_after_source_leaves(tmp_path):
    """A new-epoch member can forward an old signed suffix without old peers online."""
    src,dst,archive,source,receiver=context(tmp_path)
    torch.manual_seed(31)
    late_model=Tiny()
    late_adapter=UniversalAdapter(late_model,'test','v1',kind='loha',rank=2)
    late_journal=EventJournal(tmp_path/'late.db',dst.genesis,max_bytes=0)
    late_engine=ContinualElasticEngine(late_adapter,'late',dst.genesis,late_journal,
        trusted=src.trusted,unsafe=False)
    late_archive=CrossEpochCarryArchive(tmp_path/'late-carry.db',source_genesis=src.genesis,
        target_genesis=dst.genesis,certified_manifest='c'*64,epoch=1,cutoffs={})
    late_archive.replay(late_engine)
    class OldSource:
        genesis=src.genesis
        db=archive.journal.db
        get=archive.journal.get
        has=archive.journal.has
    upstream=AdjacentEpochCarryGossip(source_genesis=src.genesis,
        source_journal=OldSource(),spool=tmp_path/'upstream',carry_archive=archive)
    downstream=AdjacentEpochCarryGossip(source_genesis=src.genesis,
        source_journal=OldSource(),spool=tmp_path/'downstream',carry_archive=late_archive)
    try:
        opt=torch.optim.AdamW(src.adapter.trainable_parameters(),lr=.01)
        src.adapter.model().backward();opt.step();opt.zero_grad()
        src.seal_local(1)
        async def committed(*args):pass
        original=AsyncInboundPipeline(dst,committed,carry_archive=archive)
        original.start()
        a=Link(receiver,'old');b=Link(source,'new')
        for i in range(100):
            source.tick([('new',a)],now=float(i+1))
            receiver.tick([('old',b)],now=float(i+1))
            await source.flush_completed(lambda peer,wire:False)
            await receiver.flush_completed(original.submit)
            await original.drain()
            await asyncio.sleep(.002)
            if archive.journal.count():break
        assert archive.journal.count()==1
        await original.stop()
        # Original source is permanently unavailable now. The two participants
        # share the NEW epoch, but must repair OLD signed carry from archived bytes.
        source.close();receiver.close()
        aa=Link(downstream,'member');bb=Link(upstream,'member')
        aa.bridge_only=False;bb.bridge_only=False
        late_pipe=AsyncInboundPipeline(late_engine,committed,
                                       carry_archive=late_archive)
        late_pipe.start()
        for i in range(100):
            upstream.tick([('member',aa)],now=float(i+1))
            downstream.tick([('member',bb)],now=float(i+1))
            await upstream.flush_completed(lambda peer,wire:False)
            await downstream.flush_completed(late_pipe.submit)
            await late_pipe.drain()
            await asyncio.sleep(.002)
            if late_archive.journal.count():break
        assert late_archive.journal.count()==1
        assert late_archive.journal.get('alice',1)==archive.journal.get('alice',1)
        await late_pipe.stop()
    finally:
        upstream.close();downstream.close();source.close();receiver.close()
        archive.close();late_archive.close();src.journal.close()
        dst.journal.close();late_journal.close()


def test_bounded_inventory_prioritizes_recent_without_starving_old_events(tmp_path):
    src,dst,archive,sender,recv=context(tmp_path)
    try:
        for seq in range(1,105):
            src.journal.append('alice',seq,('sealed-'+str(seq)).encode())
        from patchlib.epoch_carry_gossip_v53 import INVENTORY_PAGE
        link=Link(recv,'old')
        for now in range(1,18):
            sender.tick([('new',link)],now=float(now))
        pages=[fields['events'] for kind,fields in link.messages
               if kind=='generation-carry-offer']
        assert pages and all(len(x)<=INVENTORY_PAGE for x in pages)
        # Latest 8 updates propagate on first gossip even with old history.
        assert {e[1] for e in pages[0]} >= set(range(97,105))
        # History is not silently dropped: a round-robin scan covers all 104.
        assert {e[1] for p in pages for e in p}==set(range(1,105))
    finally:
        sender.close();recv.close();archive.close()
        src.journal.close();dst.journal.close()
