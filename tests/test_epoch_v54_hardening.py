"""v54 concurrency, backpressure, off-loop I/O and atomic-cache regressions."""
import asyncio
import base64
import hashlib
from pathlib import Path

import pytest
import torch

from patchlib.core import PatchError
from patchlib.epoch_carry_gossip_v53 import AdjacentEpochCarryGossip, CHUNK
from patchlib.epoch_local_stage_v53 import stage_verified_local_view
from patchlib.epoch_carry_v46 import CrossEpochCarryArchive
from patchlib.elastic import EventJournal
from test_epoch_carry_gossip_v53 import context, Link
from test_epoch_local_stage_v53 import build


@pytest.mark.asyncio
async def test_recv_queue_backpressure_reads_verified_body_only_once(tmp_path):
    src,dst,archive,sender,receiver=context(tmp_path)
    try:
        wire=b'test signed wire'*(CHUNK//16)
        src.journal.append('alice',1,wire)
        left=Link(receiver,'old');right=Link(sender,'new')
        for n in range(80):
            sender.tick([('new',left)],now=float(n+1))
            receiver.tick([('old',right)],now=float(n+1))
            await sender.flush_completed(lambda peer,data:False)
            await receiver.flush_completed(lambda peer,data:False)
            if receiver._completed and next(iter(receiver._completed.values()))._verified:
                break
            await asyncio.sleep(.003)
        assert receiver._completed
        # Even when the signed pipeline refuses admission, the spool body is
        # cached instead of allocating/read()ing it at every network tick.
        for i in range(25):
            await receiver.flush_completed(lambda peer,data:False)
        assert receiver.bytes_read_for_delivery == len(wire)
        assert receiver.events_received == 0
        await asyncio.sleep(.07)
        received=[]
        await receiver.flush_completed(lambda peer,data:(received.append(data) or True))
        assert received==[wire]
        assert receiver.bytes_read_for_delivery==len(wire)
        assert not list((tmp_path/'receiver').glob('*.part'))
    finally:
        sender.close();receiver.close();archive.close();src.journal.close();dst.journal.close()


@pytest.mark.asyncio
async def test_need_does_not_read_large_wire_on_control_callback(tmp_path):
    src,dst,archive,sender,receiver=context(tmp_path)
    try:
        wire=b'z'*CHUNK
        src.journal.append('alice',1,wire)
        event=asyncio.Event()
        class SlowJournal:
            genesis=src.genesis
            db=src.journal.db
            has=src.journal.has
            def get(self,*args):
                raise AssertionError('synchronous journal read forbidden')
            async def get_async(self,origin,seq):
                await event.wait()
                return await src.journal.get_async(origin,seq)
        sender.close()
        sender=AdjacentEpochCarryGossip(source_genesis=src.genesis,
            source_journal=SlowJournal(),spool=tmp_path/'sender2')
        left=Link(receiver,'old');right=Link(sender,'new')
        sender.tick([('new',left)],now=1.)
        receiver.tick([('old',right)],now=1.)
        assert sender._send_requests  # callback returned without disk I/O
        await sender.flush_completed(lambda peer,data:False)
        assert sender._fetch_tasks
        sender.tick([('new',left)],now=2.)
        assert not sender._sending
        event.set()
        for n in range(3,35):
            await sender.flush_completed(lambda peer,data:False)
            sender.tick([('new',left)],now=float(n))
            receiver.tick([('old',right)],now=float(n))
            await receiver.flush_completed(lambda peer,data:True)
            if receiver.events_received:break
            await asyncio.sleep(.003)
        assert receiver.events_received==1
    finally:
        sender.close();receiver.close();archive.close();src.journal.close();dst.journal.close()


@pytest.mark.asyncio
async def test_corrupted_spool_fails_closed_without_submit(tmp_path):
    src,dst,archive,sender,receiver=context(tmp_path)
    try:
        wire=b'k'*(CHUNK+1)
        src.journal.append('alice',1,wire)
        left=Link(receiver,'old');right=Link(sender,'new')
        for n in range(50):
            sender.tick([('new',left)],now=float(n+1))
            receiver.tick([('old',right)],now=float(n+1))
            await sender.flush_completed(lambda p,data:False)
            if receiver._completed:break
            await asyncio.sleep(.003)
        assert receiver._completed
        inbox=next(iter(receiver._completed.values()))
        inbox.path.write_bytes(b'wrong body')
        delivered=[]
        for n in range(30):
            try:
                await receiver.flush_completed(lambda p,data:delivered.append(data))
            except PatchError as exc:
                assert 'SHA-256' in str(exc)
                break
            await asyncio.sleep(.002)
        else:
            pytest.fail('corrupted spool did not fail closed')
        assert delivered==[]
        assert not list((tmp_path/'receiver').glob('*.part'))
    finally:
        sender.close();receiver.close();archive.close();src.journal.close();dst.journal.close()


@pytest.mark.asyncio
async def test_concurrent_signed_carry_is_once_and_o1_index(tmp_path):
    src,dst,archive,sender,recv=context(tmp_path)
    try:
        opt=torch.optim.AdamW(src.adapter.trainable_parameters(),lr=.01)
        src.adapter.model().backward();opt.step();opt.zero_grad()
        wire=src.seal_local(1)
        if wire is None:
            wire=src.journal.get('alice',1)
        info,parts=archive._check(dst,wire)
        results=await asyncio.gather(*(
            archive.accept_prechecked(dst,wire,info,parts) for _ in range(8)))
        assert results.count(True)==1 and results.count(False)==7
        assert archive.journal.count()==1
        assert not hasattr(archive,'_applied')
        assert archive._replayed_count==1
    finally:
        sender.close();recv.close();archive.close();src.journal.close();dst.journal.close()


def test_checkpoint_crash_stage_does_not_publish_partial(tmp_path,monkeypatch):
    store,incoming,proposal=build(tmp_path)
    root=tmp_path/'epoch'
    import patchlib.epoch_local_stage_v53 as mod
    original=mod.os.link
    counter=0
    def crash(src,dst):
        nonlocal counter
        counter+=1
        if counter>1:raise OSError('simulated disk failure after first hardlink')
        return original(src,dst)
    monkeypatch.setattr(mod.os,'link',crash)
    with pytest.raises(OSError):
        stage_verified_local_view(proposal,root=root,incoming=incoming,objects=store)
    target=root/'checkpoints'/'views'/proposal['manifest']
    assert not target.exists()
    assert not list(target.parent.glob('.building-*'))
    monkeypatch.setattr(mod.os,'link',original)
    orphan=target.parent/'.building-orphan-after-powerloss'
    orphan.mkdir()
    (orphan/'junk').write_bytes(b'x')
    view=stage_verified_local_view(proposal,root=root,incoming=incoming,objects=store)
    assert view.manifest.exists() and not orphan.exists()
    assert not list(target.parent.glob('.building-*'))
