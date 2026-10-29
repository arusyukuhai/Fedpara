"""Real persistence/concurrency regressions for disk-bound two-window training."""
from __future__ import annotations
import asyncio
import hashlib
import os
import time
from pathlib import Path
from copy import deepcopy

import pytest
import torch

from patchlib.elastic import EventJournal
from patchlib.core import UniversalAdapter, PatchError
from patchlib.localprogress import AsyncWorkingSaver, restore_working, _tip, snapshot_path
from test_livefed import Tiny


def test_full_durable_wal_background_checkpoint_and_equivocation(tmp_path):
    async def work():
        journal=EventJournal(tmp_path/'signed.sqlite3','a'*64)
        try:
            assert journal.db.execute('PRAGMA synchronous').fetchone()[0]==2
            assert journal.db.execute('PRAGMA wal_autocheckpoint').fetchone()[0]==0
            for seq in range(1,8):
                assert await journal.append_async('alice',seq,bytes([seq])*1024*256)
            assert journal.count()==7
            assert not await journal.append_async('alice',1,bytes([1])*1024*256)
            with pytest.raises(PatchError,match='equivocation'):
                await journal.append_async('alice',1,b'bad')
            result=await asyncio.to_thread(journal.passive_checkpoint)
            assert result[0] == 0
            assert result[1] == result[2]
        finally:
            journal.close()
        reopened=EventJournal(tmp_path/'signed.sqlite3','a'*64)
        try:
            assert reopened.count()==7
            assert reopened.get('alice',7)==bytes([7])*1024*256
        finally:
            reopened.close()
    asyncio.run(work())


def test_slow_fsync_does_not_starve_peer_event_loop(tmp_path, monkeypatch):
    async def work():
        journal=EventJournal(tmp_path/'slow.sqlite3','c'*64)
        original=journal._append_worker
        def delayed(*a):
            time.sleep(.24)
            return original(*a)
        monkeypatch.setattr(journal,'_append_worker',delayed)
        running=True
        ticks=0
        async def heartbeats():
            nonlocal ticks
            while running:
                await asyncio.sleep(.01)
                ticks+=1
        hb=asyncio.create_task(heartbeats())
        try:
            assert await journal.append_async('alice',1,b'wire')
            assert ticks>=10
            assert journal.count()==1
        finally:
            running=False
            await hb
            journal.close()
    asyncio.run(work())


def test_snapshot_serialization_off_trainer_and_restore(tmp_path, monkeypatch):
    from patchlib import localprogress
    async def work():
        torch.manual_seed(77)
        model=Tiny()
        adapter=UniversalAdapter(model,'model','v',rank=3)
        journal=EventJournal(tmp_path/'work.sqlite3',hashlib.sha256(b'gen').hexdigest())
        optimizer=torch.optim.AdamW(adapter.trainable_parameters(),lr=1e-3)
        loss=adapter.model.forward()
        loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        pstate=[p.detach().clone() for p in adapter.trainable_parameters()]
        original=localprogress.save_file
        def slow_writer(*a,**k):
            time.sleep(.3)
            return original(*a,**k)
        monkeypatch.setattr(localprogress,'save_file',slow_writer)
        saver=AsyncWorkingSaver()
        try:
            started=time.monotonic()
            assert await saver.checkpoint(journal,adapter,optimizer,3,journal_tip=_tip(journal))
            # Capture completes without blocking on slow filesystem fsync.
            assert time.monotonic()-started<.22
            assert not await saver.checkpoint(journal,adapter,optimizer,3,journal_tip=_tip(journal))
            assert saver.skipped==1
            await saver.drain()
            assert saver.disk_ms>=280
            restored=UniversalAdapter(deepcopy(Tiny()),'model','v',rank=3)
            opt2=torch.optim.AdamW(restored.trainable_parameters(),lr=1e-3)
            assert restore_working(journal,restored,opt2)==3
            for a,b in zip(pstate,restored.trainable_parameters()):
                torch.testing.assert_close(a,b)
        finally:
            journal.close()
    asyncio.run(work())


def test_stale_snapshot_does_not_override_newly_committed_peer_event(tmp_path):
    async def work():
        journal=EventJournal(tmp_path/'work.sqlite3','f'*64)
        adapter=UniversalAdapter(Tiny(),'model','v',rank=3)
        opt=torch.optim.AdamW(adapter.trainable_parameters())
        writer=AsyncWorkingSaver()
        try:
            assert await writer.checkpoint(journal,adapter,opt,0,journal_tip=_tip(journal))
            await journal.append_async('alice',1,b'committed-wire')
            await writer.drain()
            new_adapter=UniversalAdapter(Tiny(),'model','v',rank=3)
            new_opt=torch.optim.AdamW(new_adapter.trainable_parameters())
            assert restore_working(journal,new_adapter,new_opt)==0
        finally:
            journal.close()
    asyncio.run(work())
