import asyncio
import hashlib
from types import SimpleNamespace
import pytest
import torch
from patchlib.shards import export_state_sharded,read_manifest,semantic_shard_sha256,_blob_path,apply_sharded_state
from patchlib.epoch_checkpoint_v50 import BoundedCheckpointStager
from patchlib.epoch_objects_v50 import CapacityError
from patchlib.live_epoch_switch_v48 import LiveEpochSwitchService,PreparedEpoch,certified_genesis
from patchlib.epoch_hot_swap_v45 import TrainerGeneration,TrainerStepRouter
from patchlib.online_generation import _digest


def snapshot(root):
    model=torch.nn.Linear(2,2,bias=False)
    with torch.no_grad():model.weight.fill_(2.)
    manifest=root/'source.fmanifest';store=root/'source'
    export_state_sharded(model,store,manifest,model_id='test/model',revision='v1',peer_id='tester',shard_bytes=65536)
    raw=manifest.read_bytes()
    p={'epoch':1,'manifest':hashlib.sha256(raw).hexdigest(),'semantic':semantic_shard_sha256(read_manifest(manifest)),
       'room':'room','parent':'f'*64}
    return model,store,raw,p


@pytest.mark.asyncio
async def test_real_download_and_step_boundary_switch(tmp_path):
    source,store,raw,p=snapshot(tmp_path)
    requested=asyncio.Event();release=asyncio.Event();calls=[]
    async def fm(peer,d,limit):
        assert d==p['manifest'] and len(raw)<=limit
        return raw
    async def fb(peer,d,o,n):
        calls.append((d,o,n));requested.set();await release.wait()
        return _blob_path(store,d).read_bytes()[o:o+n]
    stager=BoundedCheckpointStager(tmp_path/'bounded',peers=lambda:['p1'],fetch_manifest=fm,fetch_block=fb,
        quota_bytes=1024*1024,max_manifest_bytes=65536,bytes_per_second=1e9)
    model=torch.nn.Linear(2,2,bias=False)
    old=TrainerGeneration(0,'a'*64,'b'*64,model,object(),torch.optim.SGD(model.parameters(),lr=.01),SimpleNamespace(genesis='a'*64))
    router=TrainerStepRouter(old)
    class Ledger:
        def latest(self):return {'proposal':p}
        def certified_epoch(self,n):return {'proposal':p} if n==1 else None
    built=[]
    async def build(proposal,checkpoint):
        model=torch.nn.Linear(2,2,bias=False)
        await asyncio.to_thread(apply_sharded_state,model,checkpoint.manifest,checkpoint.store,
                               model_id='test/model',revision='v1')
        genesis=certified_genesis(proposal)
        g=TrainerGeneration(1,genesis,p['manifest'],model,object(),torch.optim.SGD(model.parameters(),lr=.01),SimpleNamespace(genesis=genesis))
        built.append(checkpoint)
        return PreparedEpoch(1,_digest(p),p['manifest'],g,checkpoint.manifest,checkpoint.store)
    switch=LiveEpochSwitchService(ledger=Ledger(),router=router,root=tmp_path/'runtime',build_shadow=build,
        verify_local=lambda *_:True,flush_unpublished=lambda *_:None,carry_suffix=lambda *_:None,
        transport_handoff=lambda *_:True,attest_journal=lambda *_:True,checkpoint_stager=stager)
    task=None
    try:
        task=asyncio.create_task(switch.activate_next())
        await asyncio.wait_for(requested.wait(),3)
        for _ in range(4):
            async with router.step() as g:
                assert g is old
                g.optimizer.zero_grad();g.model(torch.ones(1,2)).sum().backward();g.optimizer.step()
        release.set()
        assert await asyncio.wait_for(task,5)
        async with router.step() as g:
            assert g.epoch==1 and torch.equal(g.model.weight,source.weight)
            g.model(torch.ones(1,2)).sum().backward();g.optimizer.step()
        count=len(calls)
        repeated=await stager.prepare(p)
        assert repeated==built[0] and len(calls)==count
        await stager.retire(repeated)
        assert not repeated.store.exists()
    finally:
        release.set()
        if task is not None and not task.done():
            task.cancel();await asyncio.gather(task,return_exceptions=True)
        await stager.close()


@pytest.mark.asyncio
async def test_cache_reopen_avoids_network(tmp_path):
    _,store,raw,p=snapshot(tmp_path);calls=[]
    async def fm(*_):calls.append('manifest');return raw
    async def fb(peer,d,o,n):calls.append('block');return _blob_path(store,d).read_bytes()[o:o+n]
    def make():return BoundedCheckpointStager(tmp_path/'cache-root',peers=lambda:['source'],
        fetch_manifest=fm,fetch_block=fb,max_manifest_bytes=65536,bytes_per_second=1e9)
    first=make()
    try:before=await first.prepare(p)
    finally:await first.close()
    count=len(calls);second=make()
    try:
        assert await second.prepare(p)==before
        assert len(calls)==count
    finally:await second.close()


@pytest.mark.asyncio
async def test_quota_failure_before_tensor_requests(tmp_path):
    _,store,raw,p=snapshot(tmp_path)
    async def fm(*_):return raw
    async def fb(*_):raise AssertionError('tensor requested without capacity')
    stager=BoundedCheckpointStager(tmp_path/'small',peers=lambda:['source'],fetch_manifest=fm,fetch_block=fb,
        quota_bytes=1,max_manifest_bytes=65536,bytes_per_second=1e9)
    try:
        with pytest.raises(CapacityError):await stager.prepare(p)
    finally:await stager.close()
