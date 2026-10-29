import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from patchlib.core import PatchError
from patchlib.online_generation import _digest
from patchlib.epoch_hot_swap_v45 import TrainerGeneration, TrainerStepRouter
from patchlib.live_epoch_switch_v48 import LiveEpochSwitchService, PreparedEpoch, certified_genesis


def make_gen(epoch):
    g=f'{epoch+10:064x}'
    return TrainerGeneration(epoch,g,f'{epoch+20:064x}',object(),object(),object(),SimpleNamespace(genesis=g))


class Ledger:
    def __init__(self):self.proposal=None
    def latest(self):return {'proposal':self.proposal} if self.proposal else None
    def certified_epoch(self,epoch):
        return {'proposal':self.proposal} if self.proposal and epoch==self.proposal['epoch'] else None


@pytest.mark.asyncio
async def test_live_switch_replaces_optimizer_without_stopping_relay(tmp_path):
    previous=make_gen(0);next_gen=make_gen(1)
    next_gen=TrainerGeneration(1,next_gen.genesis,'a'*64,next_gen.model,
        next_gen.adapter,next_gen.optimizer,next_gen.journal)
    ledger=Ledger();ledger.proposal={'epoch':1,'manifest':'a'*64,'semantic':'b'*64,'cut':'c'*64,'room':'r','parent':'d'*64}
    next_gen=TrainerGeneration(1,certified_genesis(ledger.proposal),'a'*64,next_gen.model,
        next_gen.adapter,next_gen.optimizer,SimpleNamespace(genesis=certified_genesis(ledger.proposal)))
    traces=[]
    ready=asyncio.Event();release=asyncio.Event();in_relay=asyncio.Event()
    async def shadow(p):
        traces.append('shadow')
        return PreparedEpoch(1,_digest(p),'a'*64,next_gen,tmp_path/'manifest',tmp_path/'objects')
    async def flush(_):traces.append('flush')
    async def carry(*_):traces.append('carry')
    async def attest(*_):traces.append('attest');return True
    async def transport(*_):traces.append('transport');return True
    service=LiveEpochSwitchService(ledger=ledger,router=TrainerStepRouter(previous),root=tmp_path,
        build_shadow=shadow,verify_local=lambda p,c:True,flush_unpublished=flush,
        carry_suffix=carry,attest_journal=attest,transport_handoff=transport,validate_shards=False)
    async def training():
        async with service.router.step() as gen:
            assert gen is previous;ready.set();await release.wait()
        async with service.router.step() as gen:
            assert gen is next_gen
    async def relay():
        while not in_relay.is_set():
            traces.append('relay')
            await asyncio.sleep(.005)
    t=asyncio.create_task(training());r=asyncio.create_task(relay())
    await ready.wait()
    transition=asyncio.create_task(service.activate_next())
    await asyncio.sleep(.04)
    assert not transition.done() and 'relay' in traces
    release.set()
    assert await transition is True
    await t
    in_relay.set();await r
    assert traces.index('flush') < traces.index('carry') < traces.index('attest') < traces.index('transport')
    assert service.router.current is next_gen
    assert (tmp_path/'active-epoch.json').exists()
    assert await service.activate_next() is False


@pytest.mark.asyncio
async def test_non_contiguous_certification_rejected(tmp_path):
    old=make_gen(0);new=make_gen(2)
    ledger=Ledger();ledger.proposal={'epoch':2,'manifest':new.checkpoint_hash,'semantic':'b'*64,'room':'r','parent':'d'*64}
    service=LiveEpochSwitchService(ledger=ledger,router=TrainerStepRouter(old),root=tmp_path,
        build_shadow=lambda p: None,verify_local=lambda *a:True,flush_unpublished=lambda *a:None,
        carry_suffix=lambda *a:None,attest_journal=lambda *a:True,transport_handoff=lambda *a:True,validate_shards=False)
    with pytest.raises(PatchError,match='intermediate'):
        await service.activate_next()
    assert service.router.current is old


@pytest.mark.asyncio
async def test_failed_preparation_never_switches(tmp_path):
    old=make_gen(0);new=make_gen(1)
    ledger=Ledger();ledger.proposal={'epoch':1,'manifest':new.checkpoint_hash,'semantic':'b'*64,'room':'r','parent':'d'*64}
    new=TrainerGeneration(1,certified_genesis(ledger.proposal),new.checkpoint_hash,new.model,
                          new.adapter,new.optimizer,SimpleNamespace(genesis=certified_genesis(ledger.proposal)))
    async def shadow(p):
        return PreparedEpoch(1,_digest(p),new.checkpoint_hash,new,tmp_path/'m',tmp_path/'store')
    service=LiveEpochSwitchService(ledger=ledger,router=TrainerStepRouter(old),root=tmp_path,
        build_shadow=shadow,verify_local=lambda *a:False,flush_unpublished=lambda *a:None,
        carry_suffix=lambda *a:None,attest_journal=lambda *a:True,transport_handoff=lambda *a:True,validate_shards=False)
    with pytest.raises(PatchError,match='not locally verified'):
        await service.activate_next()
    assert not service.pointer.exists()
    assert service.router.current is old


@pytest.mark.asyncio
async def test_cancellation_while_waiting_does_not_lock_steps(tmp_path):
    old,new=make_gen(0),make_gen(1)
    router=TrainerStepRouter(old)
    wait=asyncio.Event();entered=asyncio.Event()
    async def step():
        async with router.step():
            entered.set();await wait.wait()
    task=asyncio.create_task(step());await entered.wait()
    attempt=asyncio.create_task(router.switch(new,verified=lambda _:True,
        flush_old=lambda _:None,carry_old=lambda *a:None,attest_new=lambda _:True,
        pointer=tmp_path/'ptr'))
    await asyncio.sleep(.02)
    attempt.cancel()
    with pytest.raises(asyncio.CancelledError):await attempt
    wait.set();await task
    async with asyncio.timeout(.2):
        async with router.step() as current:
            assert current is old
    assert not (tmp_path/'ptr').exists()


def test_real_sharded_checkpoint_validation(tmp_path):
    import torch
    import hashlib
    from patchlib.shards import export_state_sharded, read_manifest, semantic_shard_sha256
    from patchlib.live_epoch_switch_v48 import verify_candidate_files
    model=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Linear(4,1))
    manifest=tmp_path/'base.fmanifest'
    store=tmp_path/'store'
    export_state_sharded(model,store,manifest,model_id='test/model',
                         revision='v1',peer_id='tester',seq=2,shard_bytes=65536)
    p={'manifest':hashlib.sha256(manifest.read_bytes()).hexdigest(),
       'semantic':semantic_shard_sha256(read_manifest(manifest))}
    prepared=SimpleNamespace(manifest=manifest,store=store)
    assert verify_candidate_files(prepared,p)
    blob=next((store/'objects').rglob('*.shard'))
    original=blob.read_bytes()
    blob.write_bytes(b'!' + original[1:])
    with pytest.raises(PatchError):verify_candidate_files(prepared,p)
    blob.write_bytes(original)
    assert verify_candidate_files(prepared,p)
    manifest.write_bytes(manifest.read_bytes()+b' ')
    with pytest.raises(PatchError,match='manifest SHA-256'):
        verify_candidate_files(prepared,p)


def test_deterministic_new_genesis_binds_room_certificate(tmp_path):
    p={'room':'one','epoch':1,'manifest':'a'*64,'semantic':'b'*64,'parent':'c'*64}
    a=certified_genesis(p)
    assert len(a)==64 and certified_genesis(dict(p))==a
    p['room']='two'
    assert certified_genesis(p)!=a
    p['room']='one';p['semantic']='d'*64
    assert certified_genesis(p)!=a

@pytest.mark.asyncio
async def test_live_service_real_pytorch_optimizer_starts_new_step(tmp_path):
    import torch
    torch.manual_seed(7)
    old_model=torch.nn.Linear(2,2,bias=False)
    new_model=torch.nn.Linear(2,2,bias=False)
    with torch.no_grad():new_model.weight.fill_(2.0)
    old_optimizer=torch.optim.AdamW(old_model.parameters(),lr=.01)
    new_optimizer=torch.optim.AdamW(new_model.parameters(),lr=.01)
    p={'epoch':1,'manifest':'a'*64,'semantic':'b'*64,'room':'test','parent':'c'*64}
    current=TrainerGeneration(0,'d'*64,'e'*64,old_model,object(),old_optimizer,
                              SimpleNamespace(genesis='d'*64))
    after=TrainerGeneration(1,certified_genesis(p),p['manifest'],new_model,
                            object(),new_optimizer,SimpleNamespace(genesis=certified_genesis(p)))
    ledger=Ledger();ledger.proposal=p
    async def shadow(proposal):
        return PreparedEpoch(1,_digest(proposal),proposal['manifest'],after,tmp_path/'m',tmp_path/'store')
    gate=TrainerStepRouter(current)
    switch=LiveEpochSwitchService(ledger=ledger,router=gate,root=tmp_path,
        build_shadow=shadow,verify_local=lambda *_:True,
        flush_unpublished=lambda *_:None,carry_suffix=lambda *_:None,
        transport_handoff=lambda *_:True,attest_journal=lambda *_:True,validate_shards=False)
    async with gate.step() as active:
        loss=(active.model(torch.ones(2,2))**2).mean()
        loss.backward();active.optimizer.step();active.optimizer.zero_grad()
    assert old_optimizer.state
    assert await switch.activate_next()
    async with gate.step() as active:
        assert active.model is new_model and active.optimizer is new_optimizer
        assert not new_optimizer.state
        loss=active.model(torch.ones(2,2)).square().mean()
        loss.backward();active.optimizer.step();active.optimizer.zero_grad()
    assert new_optimizer.state
    assert (new_model.weight<2.0).all().item()

@pytest.mark.asyncio
async def test_failed_network_publish_after_pointer_is_fail_stop(tmp_path):
    old,new=make_gen(0),make_gen(1)
    router=TrainerStepRouter(old)
    pointer=tmp_path/'active.json'
    with pytest.raises(PatchError,match='transport activation'):
        await router.switch(new,verified=lambda _:True,
            flush_old=lambda _:None,carry_old=lambda *_:None,
            attest_new=lambda _:True,pointer=pointer,
            activate_new=lambda *_:False)
    # The durable pointer is an unambiguous recovery target; do NOT silently
    # take another old-genesis optimizer step after a networking failure.
    assert pointer.is_file()
    assert router.current is old
    with pytest.raises(PatchError,match='restart from durable pointer'):
        async with router.step():
            pytest.fail('must never execute on retired old epoch')

@pytest.mark.asyncio
async def test_pointer_fsync_failure_blocks_old_steps_and_retries_only_publication(tmp_path,monkeypatch):
    import patchlib.epoch_hot_swap_v45 as hot
    old,new=make_gen(0),make_gen(1)
    router=TrainerStepRouter(old)
    calls=[]
    original=hot.durable_pointer
    def interrupted(path,**kwargs):
        original(path,**kwargs)
        raise OSError('simulated directory fsync failure after rename')
    monkeypatch.setattr(hot,'durable_pointer',interrupted)
    with pytest.raises(OSError):
        await router.switch(new,verified=lambda _:True,
            flush_old=lambda _:calls.append('flush'),carry_old=lambda *_:calls.append('carry'),
            attest_new=lambda _:True,pointer=tmp_path/'ptr',activate_new=lambda *_:True)
    assert router.pending_publication
    with pytest.raises(PatchError,match='publication failed'):
        async with router.step():pass
    monkeypatch.setattr(hot,'durable_pointer',original)
    assert await router.resume_publication()
    async with router.step() as current:assert current is new
    assert calls==['flush','carry']
    assert not router.pending_publication


@pytest.mark.asyncio
async def test_network_failure_retry_does_not_reapply_carry(tmp_path):
    old,new=make_gen(0),make_gen(1)
    router=TrainerStepRouter(old)
    calls=[]
    async def publish(*_):
        calls.append('publish')
        return calls.count('publish')>1
    with pytest.raises(PatchError,match='transport activation failed'):
        await router.switch(new,verified=lambda _:True,
            flush_old=lambda _:calls.append('flush'),carry_old=lambda *_:calls.append('carry'),
            attest_new=lambda _:True,pointer=tmp_path/'ptr',activate_new=publish)
    assert await router.resume_publication()
    assert calls==['flush','carry','publish','publish']
    assert router.current is new


def test_rejoin_walks_certified_intermediate_epochs(tmp_path):
    old=make_gen(0)
    class HistoricalLedger:
        def certified_epoch(self,n):
            if 1<=n<=5:
                return {'proposal':{'epoch':n,'manifest':'a'*64,'semantic':'b'*64,'room':'r','parent':'c'*64}}
        def latest(self):return self.certified_epoch(5)
    s=LiveEpochSwitchService(ledger=HistoricalLedger(),router=TrainerStepRouter(old),root=tmp_path,
        build_shadow=lambda _:None,verify_local=lambda *_:True,flush_unpublished=lambda *_:None,
        carry_suffix=lambda *_:None,transport_handoff=lambda *_:True,attest_journal=lambda *_:True)
    assert s._certificate()['epoch']==1


def test_restart_with_wrong_generation_refused(tmp_path):
    from patchlib.epoch_hot_swap_v45 import durable_pointer
    durable_pointer(tmp_path/'active-epoch.json',epoch=1,genesis='a'*64,checkpoint_hash='b'*64)
    with pytest.raises(PatchError,match='reconstruct durable'):
        LiveEpochSwitchService(ledger=Ledger(),router=TrainerStepRouter(make_gen(0)),root=tmp_path,
            build_shadow=lambda _:None,verify_local=lambda *_:True,flush_unpublished=lambda *_:None,
            carry_suffix=lambda *_:None,transport_handoff=lambda *_:True,attest_journal=lambda *_:True)

@pytest.mark.asyncio
async def test_v52_warm_carry_is_bounded_and_requires_writer_fence(tmp_path):
    # A certified epoch can warm a persistent, independently prepared new
    # trainer while the old trainer keeps running, then finalize at a barrier.
    p={'epoch':1,'manifest':'a'*64,'semantic':'b'*64,'cut':'c'*64,
       'room':'test','parent':'d'*64}
    ledger=Ledger();ledger.proposal=p
    old=make_gen(0)
    after=make_gen(1)
    after=TrainerGeneration(1,certified_genesis(p),p['manifest'],
        after.model,after.adapter,after.optimizer,SimpleNamespace(genesis=certified_genesis(p)))
    class Warm:
        def __init__(self):self.i=0;self.validations=0;self.finished=0
        def validate_cut(self):self.validations+=1
        async def tick(self):
            self.i+=1
            return SimpleNamespace(caught_up=self.i>=4)
        async def finish_under_ingress_barrier(self):self.finished+=1
    warm=Warm();builders=[];frozen=False
    async def shadow(proposal):
        builders.append('new')
        return PreparedEpoch(1,_digest(proposal),p['manifest'],after,tmp_path/'m',tmp_path/'store',warm)
    async def flush(_):
        nonlocal frozen
        frozen=True
    service=LiveEpochSwitchService(ledger=ledger,router=TrainerStepRouter(old),
        root=tmp_path,build_shadow=shadow,verify_local=lambda *_:True,
        flush_unpublished=flush,carry_suffix=lambda *_:None,
        attest_journal=lambda *_:True,transport_handoff=lambda *_:True,
        old_ingress_frozen=lambda _:frozen,warm_carry_batches_per_poll=2,
        validate_shards=False)
    assert await service.activate_next() is False
    assert service.router.current is old
    assert not service.pointer.exists()
    assert await service.activate_next() is True
    assert builders==['new'] and warm.validations==1 and warm.finished==1
    assert service.router.current is after


@pytest.mark.asyncio
async def test_v52_missing_writer_barrier_refuses_streamed_epoch(tmp_path):
    p={'epoch':1,'manifest':'a'*64,'semantic':'b'*64,
       'room':'test','parent':'d'*64}
    ledger=Ledger();ledger.proposal=p
    old=make_gen(0);after=make_gen(1)
    after=TrainerGeneration(1,certified_genesis(p),p['manifest'],
        after.model,after.adapter,after.optimizer,SimpleNamespace(genesis=certified_genesis(p)))
    class Warm:
        def validate_cut(self):pass
        async def tick(self):return SimpleNamespace(caught_up=True)
        async def finish_under_ingress_barrier(self):raise AssertionError('must not apply')
    async def shadow(p):
        return PreparedEpoch(1,_digest(p),p['manifest'],after,tmp_path/'m',tmp_path/'store',Warm())
    service=LiveEpochSwitchService(ledger=ledger,router=TrainerStepRouter(old),
        root=tmp_path,build_shadow=shadow,verify_local=lambda *_:True,
        flush_unpublished=lambda *_:None,carry_suffix=lambda *_:None,
        attest_journal=lambda *_:True,transport_handoff=lambda *_:True,
        old_ingress_frozen=lambda _:False,validate_shards=False)
    with pytest.raises(PatchError,match='old epoch writers'):
        await service.activate_next()
    assert service.router.current is old and not service.pointer.exists()
