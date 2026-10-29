"""Regressions for P2P connect-time training stalls and redundant gossip."""
import asyncio
import hashlib
import threading
from copy import deepcopy

import pytest
import torch

from patchlib.async_ingest import AsyncInboundPipeline
from patchlib.continual import ContinualElasticEngine
from patchlib.core import UniversalAdapter
from patchlib.elastic import EventJournal
from test_livefed import Tiny


def pair(tmp_path):
    torch.manual_seed(13001)
    initial=Tiny()
    genesis=hashlib.sha256(b'fast-path-2026').hexdigest()
    engines=[]
    for name in ('alice','bob'):
        adapter=UniversalAdapter(deepcopy(initial),'m','r',rank=3)
        engines.append(ContinualElasticEngine(adapter,name,genesis,
            EventJournal(tmp_path/(name+'.db'),genesis),unsafe=True))
    return engines


def remote_wire(src):
    optimizer=torch.optim.SGD(list(src.adapter.trainable_parameters()),lr=.05)
    loss=src.adapter.model()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return src.seal_local(1)


def test_remote_verification_does_not_acquire_training_lock(tmp_path):
    alice,bob=pair(tmp_path)
    try:
        wire=remote_wire(alice)
        finished=threading.Event()
        error=[]
        def verify():
            try: bob._check(wire)
            except BaseException as exc: error.append(exc)
            finally: finished.set()
        with bob.adapter.lock:
            thread=threading.Thread(target=verify,daemon=True)
            thread.start()
            assert finished.wait(3.),'network verification blocked behind local optimizer lock'
        thread.join(timeout=3)
        assert not error
    finally:
        alice.journal.close();bob.journal.close()


def test_precomputed_delta_used_once_and_matches_replay(tmp_path,monkeypatch):
    import patchlib.continual as continual
    alice,bob=pair(tmp_path)
    try:
        wire=remote_wire(alice)
        original=continual._delta_block
        calls=[0]
        def wrapped(*args,**kw):
            calls[0]+=1
            return original(*args,**kw)
        monkeypatch.setattr(continual,'_delta_block',wrapped)
        info,prepared=bob._check(wire)
        checked=calls[0]
        assert checked>0
        assert all(t is not None for t in prepared.deltas)
        bob.journal.append(info.origin,info.seq,wire)
        bob._apply_checked(info,prepared)
        assert calls[0]==checked, 'remote apply recomputed an already verified delta'
        for a,b in zip(alice.adapter.groups,bob.adapter.groups):
            pa,pb=a.aliases[0],b.aliases[0]
            module,_,attr=pa.rpartition('.')
            got=bob.adapter.model.get_submodule(module).parametrizations[attr].original
            want=alice.adapter.model.get_submodule(module).parametrizations[attr].original
            torch.testing.assert_close(got,want,atol=1e-6,rtol=1e-6)
    finally:
        alice.journal.close();bob.journal.close()


def test_identical_authenticated_gossip_skips_reconstruction(tmp_path,monkeypatch):
    alice,bob=pair(tmp_path)
    try:
        wire=remote_wire(alice)
        checks=[0]
        original=bob._check
        def checked(w):
            checks[0]+=1
            return original(w)
        monkeypatch.setattr(bob,'_check',checked)
        async def exercise():
            seen=[]
            async def callback(info,wire,sender,fresh):
                seen.append(fresh)
            pipeline=AsyncInboundPipeline(bob,callback)
            pipeline.start()
            for _ in range(3):
                assert pipeline.submit('alice',wire)
                await pipeline.drain()
            assert checks[0]==1
            assert seen==[True,False,False]
            assert pipeline.applied==1 and pipeline.duplicates==2
            assert bob.journal.count()==1
            assert pipeline.check_count==1
            await pipeline.stop()
        asyncio.run(exercise())
    finally:
        alice.journal.close();bob.journal.close()
