"""Regressions for v11's bounded batched safety checks and inbound timings."""
from copy import deepcopy
import asyncio
import hashlib
import torch
from torch import nn
import pytest

from patchlib.core import UniversalAdapter, PatchError
from patchlib.elastic import EventJournal
from patchlib.continual import ContinualElasticEngine
from patchlib.async_ingest import AsyncInboundPipeline


class ManyTinyLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(16,16,bias=False) for _ in range(24)])


def make_pair(tmp_path):
    torch.manual_seed(20261024)
    orig=ManyTinyLayers()
    genesis=hashlib.sha256(b'v11-batch-checks').hexdigest()
    out=[]
    for pid in ('a','b'):
        adapter=UniversalAdapter(deepcopy(orig),'many','1',kind='loha',rank=4,target_scope='auto')
        out.append(ContinualElasticEngine(adapter,pid,genesis,
            EventJournal(tmp_path/(pid+'.db'),genesis),unsafe=True))
    return out


def test_batched_preflight_reduces_host_synchronizations(tmp_path,monkeypatch):
    a,b=make_pair(tmp_path)
    try:
        with torch.no_grad():
            for g in a.adapter.groups:
                g.overlay.local.parts[0].add_(0.005)
        wire=a.seal_local(1)
        info,prepared=b._check(wire)
        original=torch.Tensor.item
        count=[0]
        def counted(t):
            count[0]+=1
            return original(t)
        with monkeypatch.context() as m:
            m.setattr(torch.Tensor,'item',counted)
            b._apply_checked(info,prepared)
        assert count[0] <= 2, f'expected one batched synchronization, got {count[0]}'
        assert (info.origin,info.seq) in b.applied
        for x,y in zip(a.adapter.groups,b.adapter.groups):
            pa=x.aliases[0];pb=y.aliases[0]
            xp=x.overlay.local.parts[0]
            ap=a.adapter.model.get_submodule(pa.rpartition('.')[0]).parametrizations.weight.original
            bp=b.adapter.model.get_submodule(pb.rpartition('.')[0]).parametrizations.weight.original
            torch.testing.assert_close(ap,bp,atol=1e-6,rtol=1e-6)
    finally:
        a.journal.close();b.journal.close()


def test_nonfinite_staged_update_refused_without_any_write(tmp_path):
    a,b=make_pair(tmp_path)
    try:
        wire=a.seal_local(1)
        info,prepared=b._check(wire)
        before=[]
        for g in b.adapter.groups:
            path,_,attr=g.aliases[0].rpartition('.')
            before.append(b.adapter.model.get_submodule(path).parametrizations[attr].original.detach().clone())
        prepared.deltas[5].view(-1)[0]=float('inf')
        with pytest.raises(PatchError,match='nonfinite continual global weights'):
            b._apply_checked(info,prepared)
        for g,prev in zip(b.adapter.groups,before):
            path,_,attr=g.aliases[0].rpartition('.')
            torch.testing.assert_close(b.adapter.model.get_submodule(path).parametrizations[attr].original,prev)
    finally:
        a.journal.close();b.journal.close()


def test_inbound_commit_phase_metrics_report_nonnegative(tmp_path):
    a,b=make_pair(tmp_path)
    try:
        wire=a.seal_local(1)
        async def run():
            async def callback(*args):pass
            pipeline=AsyncInboundPipeline(b,callback)
            pipeline.start()
            assert pipeline.submit('a',wire)
            await pipeline.drain()
            assert pipeline.applied==1
            assert pipeline.last_journal_ms >= 0
            assert pipeline.last_lock_wait_ms >= 0
            assert pipeline.last_weight_apply_ms >= 0
            assert pipeline.last_commit_ms >= (pipeline.last_weight_apply_ms + pipeline.last_journal_ms)
            await pipeline.stop()
        asyncio.run(run())
    finally:
        a.journal.close();b.journal.close()


def test_local_publication_also_uses_batched_synchronization(tmp_path,monkeypatch):
    a,b=make_pair(tmp_path)
    try:
        with torch.no_grad():
            for g in a.adapter.groups:
                g.overlay.local.parts[0].add_(0.005)
        wire=a._make_wire(1,1)
        info,prepared=a._check(wire)
        original=torch.Tensor.item
        calls=[0]
        def counted(t):
            calls[0]+=1
            return original(t)
        with monkeypatch.context() as m:
            m.setattr(torch.Tensor,'item',counted)
            a._apply_checked(info,prepared)
        assert calls[0] <= 2, f'local publication synchronized {calls[0]} times'
        with torch.no_grad():
            for g in a.adapter.groups:
                torch.testing.assert_close(g.overlay.local.delta(),g.overlay.anchor.delta(),atol=1e-6,rtol=1e-6)
    finally:
        a.journal.close();b.journal.close()
