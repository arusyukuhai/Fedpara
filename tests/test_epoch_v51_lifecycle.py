"""v51 local recovery, bounded cleanup and mixed epoch transport regressions."""
import asyncio
from pathlib import Path
import pytest

from patchlib.core import PatchError
from patchlib.epoch_checkpoint_v50 import StagedCheckpoint
from patchlib.epoch_hot_swap_v45 import durable_pointer
from patchlib.epoch_lifecycle_v51 import DurableEpochLifecycle
from patchlib.live_epoch_switch_v48 import certified_genesis


def proposal(epoch):
    return {'room':'room','epoch':epoch,'parent':'0'*64,'manifest':f'{epoch+50:064x}',
            'semantic':f'{epoch+100000:064x}'}


class FakeStager:
    """Only a local lifecycle stub, not a network/security verification mock."""
    def __init__(self,root):
        self.root=root
        self.calls=[]
    async def prepare(self,p):
        self.calls.append(('stage',p['epoch']))
        digest=p['manifest'];view=self.root/'views'/digest
        view.mkdir(parents=True,exist_ok=True)
        (view/'checkpoint.fmanifest').write_bytes(b'verified by separate stager')
        return StagedCheckpoint(view/'checkpoint.fmanifest',view,'checkpoint-'+digest)
    async def retire(self,checkpoint):
        import shutil
        self.calls.append(('retire',checkpoint.owner))
        if checkpoint.store.exists():shutil.rmtree(checkpoint.store)


def manager(tmp_path,stager,**kw):
    return DurableEpochLifecycle(tmp_path/'leases',stager=stager,
        pointer=tmp_path/'pointer',**kw)


def publish(tmp_path,p):
    durable_pointer(tmp_path/'pointer',epoch=p['epoch'],genesis=certified_genesis(p),checkpoint_hash=p['manifest'])


@pytest.mark.asyncio
async def test_abandoned_stages_reclaimed_thousands_of_times(tmp_path):
    stager=FakeStager(tmp_path/'cache');book=manager(tmp_path,stager)
    for i in range(1,501):
        await book.sweep_superseded(i)
        await book.stage(proposal(i))
        assert book.usage().candidates==1
        assert len(book._rows())==1
    await book.sweep_superseded(501)
    assert not book._rows()
    assert not list((stager.root/'views').iterdir())
    assert sum(k=='retire' for k,_ in stager.calls)==500
    book.close()


@pytest.mark.asyncio
async def test_long_lived_active_is_never_collected_as_failed_candidate(tmp_path):
    stager=FakeStager(tmp_path/'cache');book=manager(tmp_path,stager,max_held=1)
    p1=proposal(1)
    await book.stage(p1);publish(tmp_path,p1);await book.published(p1)
    await book.sweep_superseded(3000)
    assert book.usage().active==1
    assert (stager.root/'views'/p1['manifest']).exists()
    p2=proposal(2)
    await book.stage(p2);publish(tmp_path,p2);await book.published(p2)
    assert book.usage().held==1
    with pytest.raises(PatchError,match='retention capacity'):
        await book.stage(proposal(3))
    assert (stager.root/'views'/p1['manifest']).exists()
    book.close()
    reopened=manager(tmp_path,stager,max_held=1)
    await reopened.reconcile_active(p2)
    assert reopened.usage().active==1
    assert reopened.usage().held==1
    reopened.close()


@pytest.mark.asyncio
async def test_reconstruct_missing_lease_from_verified_cache(tmp_path):
    stager=FakeStager(tmp_path/'cache');p=proposal(1)
    publish(tmp_path,p)
    book=manager(tmp_path,stager)
    await book.reconcile_active(p)
    assert book.usage().active==1
    assert len([c for c in stager.calls if c[0]=='stage'])==1
    await book.reconcile_active(p)
    assert len([c for c in stager.calls if c[0]=='stage'])==1
    book.close()


@pytest.mark.asyncio
async def test_crash_between_pointer_publish_and_lease_update(tmp_path):
    stager=FakeStager(tmp_path/'cache');p=proposal(1)
    book=manager(tmp_path,stager)
    await book.stage(p)
    publish(tmp_path,p)
    book.close() # process death before published()
    book=manager(tmp_path,stager)
    await book.sweep_superseded(2)
    assert (stager.root/'views'/p['manifest']).exists() # pointer protects it
    await book.reconcile_active(p)
    assert book.usage().active==1
    book.close()


@pytest.mark.asyncio
async def test_wrong_pointer_and_forged_active_refuse_gc(tmp_path):
    stager=FakeStager(tmp_path/'cache');book=manager(tmp_path,stager)
    p=proposal(1)
    await book.stage(p)
    with pytest.raises(PatchError,match='unpublished'):
        await book.published(p)
    publish(tmp_path,p)
    with pytest.raises(PatchError):
        await book.published({**p,'semantic':'f'*64})
    assert book.usage().candidates==1
    book.close()


@pytest.mark.asyncio
async def test_signed_mixed_generation_link_is_control_only(tmp_path):
    from patchlib.elasticnet import ElasticNode,encode_control
    from patchlib.roomid import auto_identity
    from tests.test_snapshot_request_delivery import FakeChannel
    a,asign,apub,_=auto_identity(tmp_path/'a.pem')
    b,bsign,bpub,_=auto_identity(tmp_path/'b.pem')
    node=ElasticNode('ws://localhost/ws','x'*32,origin=a,remote=b,
        genesis='2'*64,generation_root='0'*64,
        public_key=apub,signer=asign,enable_generation=True)
    ch=FakeChannel();node._attach_channel(ch);ch.handlers['open']()
    def send(kind,counter,genesis='3'*64,**data):
        ch.handlers['message'](encode_control(dict(type=kind,peer=b,genesis=genesis,counter=counter,**data),bsign))
    send('hello',1,pub=bpub,generation_v1=True,generation_root='0'*64)
    assert node.bridge_only and node.remote_supports_generation
    assert node.remote_genesis=='3'*64
    send('generation-head',2,epoch=1,digest='a'*64)
    assert node.generation_controls.get_nowait()==('generation-head',{'epoch':1,'digest':'a'*64})
    assert node.control('generation-need',after=0)
    assert node.control('need',events=[[a,1]]) is False
    assert not any('"type":"need"' in frame for frame in ch.sent)
    assert ch.readyState=='open'
    send('frontier',3,heads={a:10})
    assert ch.readyState=='closed' and node.protocol_failure.is_set()
    await asyncio.sleep(.01)


@pytest.mark.asyncio
async def test_mismatch_without_common_immutable_root_rejected(tmp_path):
    from patchlib.elasticnet import ElasticNode,encode_control
    from patchlib.roomid import auto_identity
    from tests.test_snapshot_request_delivery import FakeChannel
    a,asign,apub,_=auto_identity(tmp_path/'a.pem')
    b,bsign,bpub,_=auto_identity(tmp_path/'b.pem')
    node=ElasticNode('ws://localhost/ws','x'*32,origin=a,remote=b,
        genesis='2'*64,generation_root='0'*64,
        public_key=apub,signer=asign,enable_generation=True)
    ch=FakeChannel();node._attach_channel(ch);ch.handlers['open']()
    ch.handlers['message'](encode_control({'type':'hello','peer':b,'genesis':'3'*64,
        'counter':1,'pub':bpub,'generation_v1':True,'generation_root':'f'*64},bsign))
    assert ch.readyState=='closed'
    await asyncio.sleep(.01)

@pytest.mark.asyncio
async def test_genuine_verified_runtime_gc_releases_old_view_offloop(tmp_path):
    """Actual signed certificate, real sharded weights and SQLite GC proof."""
    import hashlib
    import torch
    from test_epoch_runtime_v43 import prepare as certified_fixture
    from patchlib.epoch_runtime_v43 import EpochRuntimeV43
    from patchlib.online_generation import GenerationLedger, make_proposal, _digest
    from patchlib.shards import export_state_sharded, read_manifest, semantic_shard_sha256
    from patchlib.elastic import EventJournal
    from patchlib.identity import signer_from_pem,verifier_from_pem

    old_owner,old_journal,new_journal,manifest,store,ledgers=certified_fixture(tmp_path)
    p1=ledgers[0].certified_epoch(1)['proposal']
    old_owner.close()
    # The first certified generation's worker journal has no uncovered events.
    # A separate second certified checkpoint may therefore cover the empty cut.
    manifest2=tmp_path/'state2.fmanifest';store2=tmp_path/'shards2'
    model=torch.nn.Linear(2,3)
    with torch.no_grad(): model.weight.add_(4.)
    export_state_sharded(model,store2,manifest2,model_id='example',revision='rev',peer_id='committee-a',seq=2)
    p2=make_proposal(room='room',epoch=2,parent=ledgers[0].head(),
        manifest=hashlib.sha256(manifest2.read_bytes()).hexdigest(),
        semantic=semantic_shard_sha256(read_manifest(manifest2)),
        cut=hashlib.sha256(b'').hexdigest(),cutoffs={},
        replicas=['committee-a','committee-b','committee-c'])
    votes=[ledgers[i].vote(p2,verified_snapshot=True,verified_cut=True) for i in range(2)]
    signed=ledgers[0].certificate(p2,votes)
    for ledger in ledgers:ledger.accept(signed)
    third=tmp_path/'third.sqlite3';nj=EventJournal(third,'f'*64,max_bytes=0)
    with nj.db:
        for key,value in [('generation_certified_epoch','2'),('generation_certified_manifest',p2['manifest']),
                         ('generation_certified_proposal',_digest(p2))]:
            nj.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(key,value))
    nj.close()
    runtime=EpochRuntimeV43(tmp_path/'epoch-runtime',ledgers[0])
    runtime.stage(new_journal,epoch=2)
    runtime.acknowledge_install(epoch=2,manifest=manifest2,store=store2,
                                new_journal=third,loader_confirmation=True)
    runtime.close()
    stager=FakeStager(tmp_path/'cache');book=manager(tmp_path,stager)
    await book.stage(p1);publish(tmp_path,p1);await book.published(p1)
    await book.stage(p2);publish(tmp_path,p2);await book.published(p2)
    assert book.usage().held==1
    def open_runtime():
        members=['committee-a','committee-b','committee-c']
        trusted={name:verifier_from_pem(tmp_path/(name+'.pub')) for name in members}
        ledger=GenerationLedger(tmp_path/'committee-a.gen.sqlite',room='room',peer='committee-a',
             committee=members,signer=signer_from_pem(tmp_path/'committee-a.key'),
             trusted=trusted,genesis='0'*64)
        return EpochRuntimeV43(tmp_path/'epoch-runtime',ledger)
    assert await book.gc_from_runtime(open_runtime,2)==[p1['manifest']]
    assert book.usage().held==0 and book.usage().active==1
    assert not (stager.root/'views'/p1['manifest']).exists()
    assert not new_journal.exists()
    assert third.exists()
    book.close()
    for ledger in ledgers:ledger.close()
