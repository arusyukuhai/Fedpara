import asyncio
from pathlib import Path
import hashlib
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace

from patchlib.core import PatchError, _canonical_json
from patchlib.epoch_hot_swap_v45 import (TrainerGeneration, TrainerStepRouter,
    durable_pointer, checked_cut_and_suffix, checked_paged_cut)
from patchlib.frontier_commitment_v44 import build_frontier_pages, header_digest


def gen(i):
    h=('%064x'%i)
    return TrainerGeneration(i,h,'%064x'%(i+100),object(),object(),object(),SimpleNamespace(genesis=h))

class TestHotSwap(unittest.TestCase):
    def test_paged_4096_all_pages_required(self):
        cuts={f'peer-{i:05d}':i for i in range(4096)}
        header,pages=build_frontier_pages(cuts)
        self.assertEqual(checked_paged_cut(header,pages,header['root'],expected_digest=header_digest(header)),cuts)
        with self.assertRaises(PatchError):checked_paged_cut(header,pages[:-1],header['root'])
        pages[0]['entries'][0][1]+=1
        with self.assertRaises(PatchError):checked_paged_cut(header,pages,header['root'])

    def test_cut_includes_suffix_but_not_duplicate(self):
        db=sqlite3.connect(':memory:')
        db.execute('CREATE TABLE events(origin TEXT,seq INTEGER,digest TEXT)')
        for peer, seq in [('a',1),('a',2),('b',1)]:
            db.execute('INSERT INTO events VALUES(?,?,?)',(peer,seq,'%064x'%(seq+100)))
        hash_=hashlib.sha256()
        for peer,seq in [('a',1),('b',1)]:
            hash_.update(_canonical_json([peer,seq,'%064x'%(seq+100)])+b'\n')
        # Journal sort order is a1,a2,b1 -> cut hash order a1,b1.
        self.assertEqual(checked_cut_and_suffix(db,{'a':1,'b':1},hash_.hexdigest()),[('a',2,'%064x'%102)])
        with self.assertRaises(PatchError):checked_cut_and_suffix(db,{'a':3,'b':1},hash_.hexdigest())
        db.close()

    def test_model_and_optimizer_swap_at_boundary(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                old,new=gen(1),gen(2)
                router=TrainerStepRouter(old)
                started=asyncio.Event();release=asyncio.Event();actions=[]
                async def train():
                    async with router.step() as state:
                        self.assertIs(state,old);started.set();await release.wait()
                    async with router.step() as state:
                        self.assertIs(state,new)
                async def switch():
                    return await router.switch(new,verified=lambda g:True,
                        flush_old=lambda g: actions.append('seal'),
                        carry_old=lambda a,b: actions.append('carry'),
                        attest_new=lambda g:(actions.append('attest') or True),
                        pointer=Path(td)/'active.json')
                job=asyncio.create_task(train());await started.wait()
                transition=asyncio.create_task(switch())
                await asyncio.sleep(.03)
                self.assertFalse(transition.done())
                release.set();self.assertIs(await transition,old);await job
                self.assertEqual(actions,['seal','carry','attest'])
                self.assertEqual(router.current,new)
                self.assertIn('"epoch":2',(Path(td)/'active.json').read_text())
        asyncio.run(run())

    def test_fail_closed_no_pointer_or_swap(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                pointer=Path(td)/'active.json';router=TrainerStepRouter(gen(1))
                with self.assertRaises(PatchError):
                    await router.switch(gen(2),verified=lambda g:False,
                        flush_old=lambda o:None,carry_old=lambda a,b:None,
                        attest_new=lambda n:True,pointer=pointer)
                with self.assertRaises(PatchError):
                    await router.switch(gen(2),verified=lambda g:True,
                        flush_old=lambda o:None,carry_old=lambda a,b:None,
                        attest_new=lambda n:False,pointer=pointer)
                self.assertFalse(pointer.exists());self.assertEqual(router.current.epoch,1)
                async with router.step() as current:self.assertEqual(current.epoch,1)
        asyncio.run(run())

    def test_bad_pointer_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'active'
            with self.assertRaises(PatchError):durable_pointer(p,epoch=1,genesis='bad',checkpoint_hash='a'*64)
            self.assertFalse(p.exists())

if __name__=='__main__':unittest.main()

class TestHotSwapTorch(unittest.TestCase):
    def test_genuine_torch_optimizer_is_new(self):
        import torch
        async def run():
            with tempfile.TemporaryDirectory() as td:
                mdl=torch.nn.Linear(1,1,bias=False)
                opt=torch.optim.AdamW(mdl.parameters(),lr=.01)
                before=TrainerGeneration(1,'%064x'%1,'%064x'%101,mdl,object(),opt,SimpleNamespace(genesis='%064x'%1))
                router=TrainerStepRouter(before)
                async with router.step() as g:
                    loss=g.model(torch.ones(1,1)).square().mean()
                    loss.backward();g.optimizer.step();g.optimizer.zero_grad()
                self.assertTrue(len(opt.state)>0)
                replacement=torch.nn.Linear(1,1,bias=False)
                with torch.no_grad():replacement.weight.fill_(3.)
                new_opt=torch.optim.AdamW(replacement.parameters(),lr=.01)
                after=TrainerGeneration(2,'%064x'%2,'%064x'%102,replacement,object(),new_opt,SimpleNamespace(genesis='%064x'%2))
                await router.switch(after,verified=lambda t:True,
                   flush_old=lambda t:None,carry_old=lambda a,b:None,
                   attest_new=lambda t:True,pointer=Path(td)/'pointer')
                async with router.step() as g:
                    self.assertIs(g.model,replacement)
                    self.assertIs(g.optimizer,new_opt)
                    self.assertEqual(len(g.optimizer.state),0)
                    self.assertAlmostEqual(float(g.model.weight.item()),3.,places=5)
                    loss=g.model(torch.ones(1,1)).square().mean();loss.backward()
                    g.optimizer.step();g.optimizer.zero_grad()
                self.assertLess(float(replacement.weight.item()),3.)
        asyncio.run(run())

    def test_simultaneous_switch_refused(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                router=TrainerStepRouter(gen(1))
                latch=asyncio.Event();entered=asyncio.Event()
                async def attest(t):entered.set();await latch.wait();return True
                kw=dict(verified=lambda t:True,flush_old=lambda t:None,
                        carry_old=lambda a,b:None,attest_new=attest,pointer=Path(td)/'p')
                first=asyncio.create_task(router.switch(gen(2),**kw))
                await entered.wait()
                with self.assertRaises(PatchError):await router.switch(gen(2),**kw)
                latch.set();await first
        asyncio.run(run())

    def test_async_gc_only_calls_verified_runtime(self):
        from patchlib.epoch_hot_swap_v45 import AutoGC
        class Fake:
            def __init__(self):self.count=0
            def poll(self):
                self.count+=1
                return [1] if self.count == 2 else []
        async def run():
            f=Fake();c=AutoGC(f,interval_s=1)
            self.assertEqual(await c.run_once(),[])
            self.assertEqual(await c.run_once(),[1])
        asyncio.run(run())

class TestRealLoHaCarry(unittest.TestCase):
    def test_old_signed_factor_deltas_rebase_as_effective_weights(self):
        from copy import deepcopy
        import torch
        from patchlib.core import UniversalAdapter
        from patchlib.continual import ContinualElasticEngine
        from patchlib.elastic import EventJournal
        class Small(torch.nn.Module):
            def __init__(self):
                super().__init__();self.layer=torch.nn.Linear(4,4,bias=False)
            def forward(self):return self.layer(torch.ones(4)).square().sum()
        with tempfile.TemporaryDirectory() as td:
            torch.manual_seed(123);base=Small()
            old_model=deepcopy(base);new_model=deepcopy(base)
            old_adapter=UniversalAdapter(old_model,'small','revision',kind='loha',rank=2)
            h='a'*64
            original=ContinualElasticEngine(old_adapter,'alice',h,
                EventJournal(Path(td)/'old.sqlite3',h),unsafe=True)
            opt=torch.optim.AdamW(list(old_adapter.trainable_parameters()),lr=.005)
            before=old_model.layer.parametrizations.weight.original.detach().clone()
            original.adapter.model().backward();opt.step();opt.zero_grad(set_to_none=True)
            signed_old=original.seal_local(1)
            old_delta=old_model.layer.parametrizations.weight.original.detach()-before
            # The new certified checkpoint has a different base, but retains
            # exactly the original tensor layout and LoHa hyperparameters.
            with torch.no_grad():
                new_model.layer.weight.add_(.1)
            new_adapter=UniversalAdapter(new_model,'small','revision',kind='loha',rank=2)
            clean_base=new_model.layer.parametrizations.weight.original.detach().clone()
            recipient=ContinualElasticEngine(new_adapter,'bob',h,
                EventJournal(Path(td)/'carry.sqlite3',h),unsafe=True)
            from patchlib.epoch_hot_swap_v45 import rebase_authenticated_additive_suffix
            digest=hashlib.sha256(signed_old).hexdigest()
            applied=rebase_authenticated_additive_suffix(recipient,[signed_old],cutoffs={'alice':0},
                durable_archive_contains=lambda origin,seq,sha:(origin,seq,sha)==('alice',1,digest))
            self.assertEqual(applied,[('alice',1)])
            self.assertTrue(torch.allclose(new_model.layer.parametrizations.weight.original,
                                           clean_base+old_delta,atol=1e-6))
            with self.assertRaises(PatchError):
                rebase_authenticated_additive_suffix(recipient,[signed_old],cutoffs={'alice':0},
                    durable_archive_contains=lambda *args:True)
            with self.assertRaises(PatchError):
                rebase_authenticated_additive_suffix(recipient,[signed_old],cutoffs={'alice':1},
                    durable_archive_contains=lambda *args:True)
            original.journal.close();recipient.journal.close()

    def test_bounded_suffix_scan(self):
        from patchlib.epoch_hot_swap_v45 import stream_cut_summary
        db=sqlite3.connect(':memory:')
        db.execute('CREATE TABLE events(origin TEXT, seq INTEGER, digest TEXT)')
        rows=[(f'p{i:04d}',1,'%064x'%(i+1)) for i in range(4096)]
        db.executemany('INSERT INTO events VALUES(?,?,?)',rows)
        db.commit();sha=hashlib.sha256()
        for p,seq,d in rows[:2048]:sha.update(_canonical_json([p,seq,d])+b'\n')
        cutoffs={p:seq for p,seq,d in rows[:2048]}
        summary=stream_cut_summary(db,cutoffs,sha.hexdigest())
        self.assertEqual(summary['pending_count'],2048)
        db.close()
