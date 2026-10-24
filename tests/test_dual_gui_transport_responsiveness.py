"""Slow TinyStories/LoHA work must not cancel healthy RTC signaling timers."""
import asyncio
import hashlib
import time
from copy import deepcopy

import pytest
import torch

from patchlib.core import UniversalAdapter
from patchlib.elastic import ElasticEngine, EventJournal
from patchlib.localprogress import _tip, restore_working, save_working
from test_livefed import Tiny


def test_signing_and_applying_heavy_update_stays_off_signaling_loop(tmp_path, monkeypatch):
    """The working journal remains on its creating thread, even when tensors are slow."""
    async def scenario():
        torch.manual_seed(17)
        base=Tiny()
        genesis=hashlib.sha256(b'genesis').hexdigest()
        a_adapter=UniversalAdapter(deepcopy(base),'one-base','fstate-sha256:seed',rank=3)
        b_adapter=UniversalAdapter(deepcopy(base),'one-base','fstate-sha256:seed',rank=3)
        a_journal=EventJournal(tmp_path/'a.db',genesis)
        b_journal=EventJournal(tmp_path/'b.db',genesis)
        a=ElasticEngine(a_adapter,'alice',genesis,a_journal,unsafe=True)
        b=ElasticEngine(b_adapter,'bob',genesis,b_journal,unsafe=True)
        ticks=0
        running=True

        async def net_timers():
            nonlocal ticks
            while running:
                await asyncio.sleep(.015)
                ticks+=1

        try:
            with torch.no_grad():
                for group in a_adapter.groups:
                    for part in group.overlay.local.parts:
                        part.fill_(0.12)
            old_export=a_adapter.export
            def very_slow_export(*args,**kwargs):
                time.sleep(.27)
                return old_export(*args,**kwargs)
            monkeypatch.setattr(a_adapter,'export',very_slow_export)
            timer=asyncio.create_task(net_timers())
            wire=await a.seal_local_responsive(2)
            assert ticks>=8, f'asyncio signaling stalled during model export (ticks={ticks})'
            assert await b.ingest_responsive(wire)
            assert not await b.ingest_responsive(wire)
            assert a_journal.count()==b_journal.count()==1
            for g1,g2 in zip(a_adapter.groups,b_adapter.groups):
                path,_,attr=g1.aliases[0].rpartition('.')
                pa=a_adapter.model.get_submodule(path).parametrizations[attr].original
                pb=b_adapter.model.get_submodule(path).parametrizations[attr].original
                torch.testing.assert_close(pa,pb,atol=1e-5,rtol=1e-5)
        finally:
            running=False
            if 'timer' in locals():
                await timer
            a_journal.close();b_journal.close()
    asyncio.run(scenario())


def test_working_checkpoint_off_loop_preserves_optimizer_state(tmp_path, monkeypatch):
    async def scenario():
        model=Tiny()
        adapter=UniversalAdapter(model,'one-base','fstate-sha256:seed',rank=3)
        journal=EventJournal(tmp_path/'a.db',hashlib.sha256(b'base').hexdigest())
        optimizer=torch.optim.AdamW(list(adapter.trainable_parameters()),lr=0.001)
        before=_tip(journal)  # SQLite must NOT run in a worker thread
        from patchlib import localprogress
        original=localprogress.save_file
        def slow_write(*args,**kwargs):
            time.sleep(.25)
            return original(*args,**kwargs)
        monkeypatch.setattr(localprogress,'save_file',slow_write)
        ticks=0
        running=True
        async def net_timers():
            nonlocal ticks
            while running:
                await asyncio.sleep(.02)
                ticks+=1
        task=asyncio.create_task(net_timers())
        try:
            await asyncio.to_thread(save_working,journal,adapter,optimizer,2,journal_tip=before)
            assert ticks>=7
            assert restore_working(journal,adapter,optimizer)==2
        finally:
            running=False
            await task
            journal.close()
    asyncio.run(scenario())


def test_localhost_signaling_keeps_sctp_alive_without_self_timeout():
    from patchlib.transport_keepalive import signaling_keepalive
    for url in ('ws://127.0.0.1:8080/ws', 'ws://localhost:8080/ws',
                'ws://[::1]:8080/ws'):
        assert signaling_keepalive(url)=={'ping_interval':None,'ping_timeout':None}
    assert signaling_keepalive('wss://example.org/ws')=={
        'ping_interval':30,'ping_timeout':150}
