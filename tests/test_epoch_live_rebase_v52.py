"""Real signed LoHA + SQLite archive tests of the bounded live carry stage."""
import asyncio
from copy import deepcopy
import hashlib
import pytest
import torch

from patchlib.core import PatchError, UniversalAdapter, _canonical_json
from patchlib.continual import ContinualElasticEngine
from patchlib.elastic import EventJournal
from patchlib.identity import generate_keypair,signer_from_pem,verifier_from_pem
from patchlib.epoch_carry_v46 import CrossEpochCarryArchive
from patchlib.epoch_live_rebase_v52 import StreamingCertifiedRebase,verify_certified_cut


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4,4,bias=False)
    def forward(self):
        return self.layer(torch.ones(4)).square().sum()


def setup(tmp_path):
    torch.manual_seed(42)
    base=Net()
    private=tmp_path/'p.key';public=tmp_path/'p.pub'
    generate_keypair(private,public)
    signer=signer_from_pem(private)
    trusted={'alice':verifier_from_pem(public)}
    def engine(name,g,peer,sign=None):
        model=deepcopy(base)
        adapter=UniversalAdapter(model,'tiny','v1',kind='loha',rank=2)
        journal=EventJournal(tmp_path/f'{name}.db',g,max_bytes=0)
        e=ContinualElasticEngine(adapter,peer,g,journal,trusted=trusted,unsafe=False,signer=sign)
        return e
    original=engine('source','a'*64,'alice',signer)
    target=engine('target','b'*64,'bob')
    def emit():
        opt=torch.optim.AdamW(original.adapter.trainable_parameters(),lr=.01)
        original.adapter.model().backward();opt.step();opt.zero_grad()
        return original.seal_local(1)
    return original,target,emit,engine


def proposal(cutoffs,cut,manifest='c'*64):
    return {'epoch':1,'manifest':manifest,'cutoffs':cutoffs,'cut':cut}


def digest(entries):
    h=hashlib.sha256()
    for origin,seq,digest in sorted(entries):
        h.update(_canonical_json([origin,seq,digest])+b'\n')
    return h.hexdigest()


@pytest.mark.asyncio
async def test_warm_signed_carry_and_disconnect_recovery(tmp_path):
    source,target,emit,maker=setup(tmp_path)
    first=emit()
    p=proposal({'alice':1},digest([('alice',1,hashlib.sha256(first).hexdigest())]))
    archive=CrossEpochCarryArchive(tmp_path/'carry.db',source_genesis=source.genesis,
        target_genesis=target.genesis,certified_manifest=p['manifest'],epoch=1,cutoffs=p['cutoffs'])
    assert archive.replay(target)==0
    runner=StreamingCertifiedRebase(source.journal,archive,target,p,batch_limit=1)
    runner.validate_cut()
    assert (await runner.tick()).carried==0  # certified checkpoint covers seq 1
    second=emit()
    assert (await runner.tick()).carried==1  # training continued during staging
    third=emit()
    attestation=await runner.finish_under_ingress_barrier()
    assert attestation['total_archived']==2
    assert runner.verify_handoff_receipt()==attestation
    final=target.adapter.model.layer.parametrizations.weight.original.detach().clone()
    assert archive.journal.count()==2
    assert (await runner.tick()).carried==2
    # Recreating the shadow from the same certified baseline + archive retains
    # the exact already-carried old deltas, even after a process interruption.
    archive.close();target.journal.close()
    recovered=maker('recovered','b'*64,'bob')
    restored=CrossEpochCarryArchive(tmp_path/'carry.db',source_genesis=source.genesis,
        target_genesis=recovered.genesis,certified_manifest=p['manifest'],epoch=1,cutoffs=p['cutoffs'])
    assert restored.replay(recovered)==2
    assert torch.allclose(recovered.adapter.model.layer.parametrizations.weight.original,final,atol=1e-6)
    restored.close();recovered.journal.close();source.journal.close()


@pytest.mark.asyncio
async def test_missing_or_forged_cut_never_applied(tmp_path):
    source,target,emit,_=setup(tmp_path)
    wire=emit()
    p=proposal({'alice':2},digest([('alice',1,hashlib.sha256(wire).hexdigest())]))
    archive=CrossEpochCarryArchive(tmp_path/'carry.db',source_genesis=source.genesis,
        target_genesis=target.genesis,certified_manifest=p['manifest'],epoch=1,cutoffs=p['cutoffs'])
    archive.replay(target)
    runner=StreamingCertifiedRebase(source.journal,archive,target,p)
    with pytest.raises(PatchError,match='certified-cut'):
        runner.validate_cut()
    with pytest.raises(PatchError,match='validated'):
        await runner.tick()
    assert archive.journal.count()==0
    archive.close();target.journal.close();source.journal.close()


@pytest.mark.asyncio
async def test_bounded_incremental_scan_and_poison_fail_closed(tmp_path):
    source,target,emit,_=setup(tmp_path)
    emit();emit();emit()
    p=proposal({},hashlib.sha256(b'').hexdigest())
    archive=CrossEpochCarryArchive(tmp_path/'carry.db',source_genesis=source.genesis,
        target_genesis=target.genesis,certified_manifest=p['manifest'],epoch=1,cutoffs={})
    archive.replay(target)
    # Small batch allows the active trainer to run between one old signed
    # update and the next, without O(n) auxiliary data or global barrier.
    runner=StreamingCertifiedRebase(source.journal,archive,target,p,batch_limit=1)
    runner.validate_cut()
    assert (await runner.tick()).carried==1
    assert (await runner.tick()).carried==2
    assert (await runner.tick()).carried==3
    assert (await runner.tick()).caught_up
    archive.close();target.journal.close();source.journal.close()

@pytest.mark.asyncio
async def test_service_real_signed_epoch_handoff_and_final_flush(tmp_path):
    """One fully local end-to-end epoch migration with REAL torch/LoHA/events.

    Consensus is provided by a pinned fake ledger with a fixed certificate;
    this test does not pretend to validate real committee networking.
    """
    from types import SimpleNamespace
    from patchlib.shards import export_state_sharded,read_manifest,semantic_shard_sha256
    from patchlib.epoch_hot_swap_v45 import TrainerGeneration,TrainerStepRouter
    from patchlib.live_epoch_switch_v48 import LiveEpochSwitchService,PreparedEpoch,certified_genesis
    from patchlib.online_generation import _digest
    source,unused,emit,maker=setup(tmp_path)
    unused.journal.close()
    manifest=tmp_path/'certified.fmanifest';store=tmp_path/'objects'
    export_state_sharded(source.adapter.model,store,manifest,model_id='tiny',revision='v1',
                         peer_id='alice',seq=0,adapter=source.adapter,shard_bytes=65536)
    manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
    p={'epoch':1,'manifest':manifest_sha,'semantic':semantic_shard_sha256(read_manifest(manifest)),
       'room':'lab','parent':'d'*64,'cut':hashlib.sha256(b'').hexdigest(),
       'cutoffs':{},'replicas':['alice','bob']}
    fresh_genesis=certified_genesis(p)
    target=maker('next',fresh_genesis,'bob')
    archive=CrossEpochCarryArchive(tmp_path/'live-carry.db',source_genesis=source.genesis,
        target_genesis=target.genesis,certified_manifest=p['manifest'],epoch=1,cutoffs={})
    assert archive.replay(target)==0
    rebaser=StreamingCertifiedRebase(source.journal,archive,target,p,batch_limit=1)
    old_optimizer=torch.optim.AdamW(source.adapter.trainable_parameters(),lr=.001)
    new_optimizer=torch.optim.AdamW(target.adapter.trainable_parameters(),lr=.001)
    old=TrainerGeneration(0,source.genesis,'e'*64,source.adapter.model,source.adapter,
                          old_optimizer,source.journal)
    new=TrainerGeneration(1,fresh_genesis,manifest_sha,target.adapter.model,target.adapter,
                          new_optimizer,target.journal)
    class Ledger:
        def latest(self):return {'proposal':p}
        def certified_epoch(self,epoch):return {'proposal':p} if epoch==1 else None
    frozen=False
    def flush(old):
        nonlocal frozen
        assert old is source_gen
        emit()   # final unsealed old update, AFTER the warm carry finished
        frozen=True
    source_gen=old
    async def shadow(proposal):
        return PreparedEpoch(1,_digest(proposal),manifest_sha,new,manifest,store,rebaser)
    def attest(newg,proposal):
        assert newg.journal.genesis==fresh_genesis
        return archive.journal.count()==3
    service=LiveEpochSwitchService(ledger=Ledger(),router=TrainerStepRouter(old),
        root=tmp_path/'epoch-pointer',build_shadow=shadow,
        verify_local=lambda prepared,proposal: True,flush_unpublished=flush,
        carry_suffix=lambda *_:None,attest_journal=attest,
        transport_handoff=lambda *_:True,
        old_ingress_frozen=lambda _:frozen,
        warm_carry_batches_per_poll=1,validate_shards=True)
    emit();emit()
    # First two polls make bounded progress WITHOUT changing active model.
    assert not await service.activate_next()
    assert service.router.current is old
    assert not await service.activate_next()
    assert service.router.current is old
    await service.activate_next()  # the final flush emits one additional signed event
    assert service.router.current is new
    assert archive.journal.count()==3
    assert rebaser.verify_handoff_receipt()['total_archived']==3
    assert (tmp_path/'epoch-pointer'/'active-epoch.json').is_file()
    after=target.adapter.model.layer.parametrizations.weight.original.detach().clone()
    assert not new_optimizer.state
    # Crash-restart replay reconstructs all three original signed old events.
    archive.close();target.journal.close()
    restart=maker('restart',fresh_genesis,'bob')
    archived=CrossEpochCarryArchive(tmp_path/'live-carry.db',source_genesis=source.genesis,
        target_genesis=restart.genesis,certified_manifest=manifest_sha,epoch=1,cutoffs={})
    assert archived.replay(restart)==3
    assert torch.allclose(restart.adapter.model.layer.parametrizations.weight.original,after,atol=1e-5)
    archived.close();restart.journal.close();source.journal.close()


@pytest.mark.asyncio
async def test_invalid_source_bytes_poison_shadow_not_active_trainer(tmp_path):
    source,target,emit,_=setup(tmp_path)
    # SQLite checksum alone is not an authenticated signed LoHA update.
    source.journal.append('alice',1,b'corrupt-wire-without-a-signature')
    p=proposal({},hashlib.sha256(b'').hexdigest())
    archive=CrossEpochCarryArchive(tmp_path/'carry.db',source_genesis=source.genesis,
        target_genesis=target.genesis,certified_manifest=p['manifest'],epoch=1,cutoffs={})
    archive.replay(target)
    runner=StreamingCertifiedRebase(source.journal,archive,target,p)
    runner.validate_cut()
    with pytest.raises(PatchError):await runner.tick()
    with pytest.raises(PatchError,match='poisoned'):
        await runner.tick()
    assert archive.journal.count()==0
    archive.close();target.journal.close();source.journal.close()


def test_certified_cut_cannot_claim_sparse_origin_prefix(tmp_path):
    source,target,emit,_=setup(tmp_path)
    source.journal.append('alice',2,b'stored-but-missing-sequence-one')
    fake_cut=digest([('alice',2,hashlib.sha256(b'stored-but-missing-sequence-one').hexdigest())])
    with pytest.raises(PatchError,match='missing signed sequence'):
        verify_certified_cut(source.journal,{'alice':2},fake_cut)
    target.journal.close();source.journal.close()

@pytest.mark.asyncio
async def test_preparing_shadow_does_not_publish_until_trainer_boundary(tmp_path):
    """The async prewarm phase cannot perform an optimizer or pointer swap."""
    from patchlib.epoch_hot_swap_v45 import TrainerGeneration,TrainerStepRouter
    from patchlib.live_epoch_switch_v48 import LiveEpochSwitchService,PreparedEpoch,certified_genesis
    from patchlib.online_generation import _digest
    source,unused,emit,maker=setup(tmp_path)
    unused.journal.close()
    p={'epoch':1,'manifest':'c'*64,'semantic':'d'*64,'room':'lab',
       'parent':'e'*64,'cut':hashlib.sha256(b'').hexdigest(),
       'cutoffs':{},'replicas':['alice','bob']}
    gen=certified_genesis(p)
    target=maker('shadow',gen,'bob')
    carry=CrossEpochCarryArchive(tmp_path/'carry.db',source_genesis=source.genesis,
        target_genesis=gen,certified_manifest=p['manifest'],epoch=1,cutoffs={})
    carry.replay(target)
    rebaser=StreamingCertifiedRebase(source.journal,carry,target,p)
    old=TrainerGeneration(0,source.genesis,'f'*64,source.adapter.model,source.adapter,
        torch.optim.AdamW(source.adapter.trainable_parameters(),lr=.01),source.journal)
    new=TrainerGeneration(1,gen,p['manifest'],target.adapter.model,target.adapter,
        torch.optim.AdamW(target.adapter.trainable_parameters(),lr=.01),target.journal)
    class Ledger:
        def latest(self):return {'proposal':p}
        def certified_epoch(self,e):return {'proposal':p} if e==1 else None
    frozen=False
    def flush(_):
        nonlocal frozen
        frozen=True
    async def build(_):
        return PreparedEpoch(1,_digest(p),p['manifest'],new,tmp_path/'manifest',tmp_path,rebaser)
    service=LiveEpochSwitchService(ledger=Ledger(),router=TrainerStepRouter(old),
        root=tmp_path/'epoch',build_shadow=build,
        verify_local=lambda *_:True,flush_unpublished=flush,
        carry_suffix=lambda *_:None,transport_handoff=lambda *_:True,
        attest_journal=lambda *_:True,old_ingress_frozen=lambda _:frozen,
        validate_shards=False)
    try:
        assert await service.activate_next(prepare_only=True) is False
        assert service.shadow_ready
        assert not frozen
        assert service.router.current is old
        assert not service.pointer.exists()
        emit() # signed event produced AFTER the independent shadow was staged
        assert await service.activate_next() is True
        assert frozen
        assert not service.shadow_ready
        assert service.router.current is new
        assert carry.journal.count()==1
        assert service.pointer.is_file()
    finally:
        carry.close();target.journal.close();source.journal.close()

@pytest.mark.asyncio
async def test_v55_warm_rebase_reads_large_signed_wire_off_event_loop(tmp_path,monkeypatch):
    source,target,emit,_ = setup(tmp_path)
    try:
        emit()
        p=proposal({},hashlib.sha256(b'').hexdigest())
        archive=CrossEpochCarryArchive(tmp_path/'async-carry.db',source_genesis=source.genesis,
            target_genesis=target.genesis,certified_manifest=p['manifest'],epoch=1,cutoffs={})
        try:
            archive.replay(target)
            runner=StreamingCertifiedRebase(source.journal,archive,target,p)
            runner.validate_cut()
            def forbidden_blocking_get(*_):
                raise AssertionError('blocking journal.get() ran in event loop')
            monkeypatch.setattr(source.journal,'get',forbidden_blocking_get)
            assert (await runner.tick()).carried == 1
        finally:
            archive.close()
    finally:
        target.journal.close();source.journal.close()
