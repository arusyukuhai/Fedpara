import hashlib
import pytest

from patchlib.core import PatchError
from patchlib.epoch_hot_swap_v45 import durable_pointer
from patchlib.epoch_trainer_boot_v52 import resolve_certified_boot,verify_boot_carry
from patchlib.epoch_carry_v46 import CrossEpochCarryArchive
from patchlib.epoch_live_rebase_v52 import StreamingCertifiedRebase
from patchlib.live_epoch_switch_v48 import certified_genesis
from patchlib.shards import export_state_sharded,read_manifest,semantic_shard_sha256
from test_epoch_live_rebase_v52 import setup


@pytest.mark.asyncio
async def test_real_certified_restart_requires_complete_signed_handoff(tmp_path):
    source,unused,emit,maker=setup(tmp_path)
    unused.journal.close()
    root=tmp_path/'live-epoch'
    temp=tmp_path/'prepared.fmanifest'
    export_state_sharded(source.adapter.model,tmp_path/'temp-objects',temp,
        model_id='tiny',revision='v1',peer_id='alice',seq=0,
        adapter=source.adapter,shard_bytes=65536)
    sha=hashlib.sha256(temp.read_bytes()).hexdigest()
    view=root/'checkpoints'/'views'/sha
    view.mkdir(parents=True)
    import shutil
    shutil.copy2(temp,view/'checkpoint.fmanifest')
    shutil.copytree(tmp_path/'temp-objects'/'objects',view/'objects')
    p={'room':'room','epoch':1,'parent':source.genesis,
       'manifest':sha,'semantic':semantic_shard_sha256(read_manifest(temp)),
       'cut':hashlib.sha256(b'').hexdigest(),'cutoffs':{},'replicas':['alice','bob']}
    gen=certified_genesis(p)
    target=maker('next',gen,'bob')
    carry=root/'carries'/'epoch-00000001.sqlite3'
    archive=CrossEpochCarryArchive(carry,source_genesis=source.genesis,
        target_genesis=gen,certified_manifest=sha,epoch=1,cutoffs={})
    assert archive.replay(target)==0
    emit();emit()
    reb=StreamingCertifiedRebase(source.journal,archive,target,p)
    reb.validate_cut()
    assert (await reb.finish_under_ingress_barrier())['total_archived']==2
    durable_pointer(root/'active-epoch.json',epoch=1,genesis=gen,checkpoint_hash=sha)
    class Ledger:
        genesis=source.genesis
        def certified_epoch(self,i):return {'proposal':p} if i==1 else None
    boot=resolve_certified_boot(root=root,journal_base=source.journal.path,
        ledger=Ledger(),root_genesis=source.genesis)
    assert boot is not None
    assert boot.journal_path==root/'journals'/'epoch-00000001.sqlite3'
    assert verify_boot_carry(boot,archive,old_journal=source.journal)
    emit() # An old writer continuing after pointer would be a safety violation
    with pytest.raises(PatchError,match='old journal changed'):
        verify_boot_carry(boot,archive,old_journal=source.journal)
    archive.close();target.journal.close();source.journal.close()


@pytest.mark.asyncio
async def test_restart_rejects_unbacked_pointer(tmp_path):
    root=tmp_path/'live-epoch'
    durable_pointer(root/'active-epoch.json',epoch=1,genesis='a'*64,checkpoint_hash='b'*64)
    class Ledger:
        genesis='c'*64
        def certified_epoch(self,i):return None
    with pytest.raises(PatchError,match='quorum certificate'):
        resolve_certified_boot(root=root,journal_base=tmp_path/'old.db',
            ledger=Ledger(),root_genesis='c'*64)


def test_broken_symlink_pointer_is_not_treated_as_genesis(tmp_path):
    root=tmp_path/'live-epoch'
    root.mkdir()
    (root/'active-epoch.json').symlink_to('nonexistent-durable-epoch-pointer')
    class Ledger:
        genesis='c'*64
        def certified_epoch(self,i):return None
    with pytest.raises(PatchError,match='symlink'):
        resolve_certified_boot(root=root,journal_base=tmp_path/'old.db',
            ledger=Ledger(),root_genesis='c'*64)
