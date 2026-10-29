"""Live SSD moments: validation, WAL-bound crash restart, COW and replay guards."""
from __future__ import annotations
import asyncio
from copy import deepcopy
from pathlib import Path
import torch
import pytest

from patchlib.core import PatchError, UniversalAdapter
from patchlib.elastic import EventJournal
from patchlib.localprogress import save_working, restore_working, AsyncWorkingSaver, snapshot_path, _tip
from patchlib.live_disk_optimizer_v59 import create_live_disk_optimizer
from test_livefed import Tiny


def make_adapter(seed):
    torch.manual_seed(seed)
    return UniversalAdapter(Tiny(),'model','revision',kind='loha',rank=3)


def run_gradient(adapter, optimizer):
    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(1977)
    loss=adapter.model.forward()
    loss.backward()
    with adapter.lock:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def test_live_disk_moments_match_reference_and_safe_restart(tmp_path):
    adapter=make_adapter(119)
    other=make_adapter(119)
    path=tmp_path/'events.sqlite3'
    journal=EventJournal(path,'a'*64)
    disk=create_live_disk_optimizer(adapter,tmp_path/'moments',lr=.002,weight_decay=.01)
    reference=torch.optim.AdamW(other.trainable_parameters(),lr=.002,weight_decay=.01)
    for _ in range(3):
        run_gradient(adapter,disk)
        run_gradient(other,reference)
    for left,right in zip(adapter.trainable_parameters(),other.trainable_parameters()):
        assert torch.allclose(left,right,atol=1e-6,rtol=1e-5)
    save_working(journal,adapter,disk,3)
    original=[p.detach().clone() for p in adapter.trainable_parameters()]
    old_moments=[(disk.directory/f'{key}.m.f32').read_bytes() for _,_,key in disk.entries]
    # The *active* optimizer continues after a consistent COW capture.
    for _ in range(2):run_gradient(adapter,disk)
    # The previously published snapshot must still be the 3-step version.
    restored=make_adapter(119)
    recovered=create_live_disk_optimizer(restored,tmp_path/'moments',lr=.002,weight_decay=.01)
    assert restore_working(journal,restored,recovered)==3
    for left,right in zip(original,restored.trainable_parameters()):
        assert torch.equal(left,right)
    for i,(_,_,key) in enumerate(recovered.entries):
        assert (recovered.directory/f'{key}.m.f32').read_bytes()==old_moments[i]
    # The next AdamW step exactly matches reference state at saved boundary.
    ref2=make_adapter(119)
    refopt=torch.optim.AdamW(ref2.trainable_parameters(),lr=.002,weight_decay=.01)
    for _ in range(3):run_gradient(ref2,refopt)
    run_gradient(restored,recovered)
    run_gradient(ref2,refopt)
    for a,b in zip(restored.trainable_parameters(),ref2.trainable_parameters()):
        assert torch.allclose(a,b,atol=1e-6,rtol=1e-5)
    disk.close();recovered.close();journal.close()


def test_disk_moments_detect_corruption_without_mutating_parameters(tmp_path):
    adapter=make_adapter(131)
    journal=EventJournal(tmp_path/'j.sqlite3','b'*64)
    disk=create_live_disk_optimizer(adapter,tmp_path/'disk',lr=.001,weight_decay=0)
    run_gradient(adapter,disk)
    save_working(journal,adapter,disk,1)
    folder=next(tmp_path.glob('j.sqlite3.working.safetensors.moments-*'))
    one=next(folder.glob('*.m.f32'))
    with one.open('r+b') as f:
        f.seek(0);v=f.read(1);f.seek(0);f.write(bytes([v[0]^1]))
    dest=make_adapter(131)
    new=create_live_disk_optimizer(dest,tmp_path/'disk',lr=.001,weight_decay=0)
    before=[p.detach().clone() for p in dest.trainable_parameters()]
    with pytest.raises(PatchError,match='damaged optimizer snapshot'):
        restore_working(journal,dest,new)
    assert all(torch.equal(b,p) for b,p in zip(before,dest.trainable_parameters()))
    disk.close();new.close();journal.close()


def test_old_journal_tip_skips_moment_restore(tmp_path):
    adapter=make_adapter(132)
    journal=EventJournal(tmp_path/'j.sqlite3','c'*64)
    disk=create_live_disk_optimizer(adapter,tmp_path/'disk',lr=.001,weight_decay=0)
    run_gradient(adapter,disk)
    save_working(journal,adapter,disk,1)
    assert journal.append('alice',1,b'newer authenticated commit')
    dest=make_adapter(132)
    next_disk=create_live_disk_optimizer(dest,tmp_path/'disk',lr=.001,weight_decay=0)
    assert restore_working(journal,dest,next_disk)==0
    disk.close();next_disk.close();journal.close()


def test_async_disk_checkpoint_keeps_immutable_snapshot(tmp_path):
    async def worker():
        adapter=make_adapter(140)
        journal=EventJournal(tmp_path/'j.sqlite3','d'*64)
        disk=create_live_disk_optimizer(adapter,tmp_path/'disk',lr=.001,weight_decay=0)
        saver=AsyncWorkingSaver()
        run_gradient(adapter,disk)
        assert await saver.checkpoint(journal,adapter,disk,1,journal_tip=_tip(journal))
        run_gradient(adapter,disk)
        await saver.drain()
        dest=make_adapter(140)
        new=create_live_disk_optimizer(dest,tmp_path/'disk',lr=.001,weight_decay=0)
        assert restore_working(journal,dest,new)==1
        disk.close();new.close();journal.close()
    asyncio.run(worker())


def test_disk_layout_or_optimizer_transition_rejected(tmp_path):
    adapter=make_adapter(151)
    journal=EventJournal(tmp_path/'j.sqlite3','e'*64)
    disk=create_live_disk_optimizer(adapter,tmp_path/'disk',lr=.001,weight_decay=0)
    run_gradient(adapter,disk)
    save_working(journal,adapter,disk,1)
    dest=make_adapter(151)
    torch_optim=torch.optim.AdamW(dest.trainable_parameters())
    with pytest.raises(PatchError,match='checkpoint format'):
        restore_working(journal,dest,torch_optim)
    disk.close();journal.close()


def test_live_fedsum_mutates_only_its_private_disk_base(tmp_path):
    from patchlib.continual import ContinualElasticEngine
    from patchlib.layer_offload import install_disk_linear_offload
    from patchlib.live_linear_workspace_v59 import allocate, attach
    from copy import deepcopy
    torch.manual_seed(421)
    template=torch.nn.Sequential(torch.nn.Linear(7,5),torch.nn.Tanh(),torch.nn.Linear(5,3))
    def install(peer):
        m=deepcopy(template)
        ad=UniversalAdapter(m,'toy','one',kind='loha',rank=3)
        work,lease=allocate(tmp_path/'disk_bases')
        info=install_disk_linear_offload(m,ad,work)
        assert info['layers']==2
        attach(m,work,lease)
        j=EventJournal(tmp_path/(peer+'.sqlite3'),'7'*64)
        e=ContinualElasticEngine(ad,peer,'7'*64,j,unsafe=True)
        return e
    first=install('alice')
    second=install('bob')
    assert first.adapter.model[0]._fedpara_disk_path != second.adapter.model[0]._fedpara_disk_path
    original=second.adapter.model[0].parametrizations.weight.original.clone()
    opt=torch.optim.AdamW(first.adapter.trainable_parameters(),lr=.002)
    data=torch.randn(3,7)
    with first.adapter.lock:
        first.adapter.model(data).square().mean().backward()
        opt.step();opt.zero_grad(set_to_none=True)
    wire=first.seal_local(1)
    assert second.ingest(wire)
    assert not torch.equal(second.adapter.model[0].parametrizations.weight.original,original)
    for group1,group2 in zip(first.adapter.groups,second.adapter.groups):
        original1=first.adapter.model.get_submodule(group1.aliases[0].rpartition('.')[0]).parametrizations.weight.original
        original2=second.adapter.model.get_submodule(group2.aliases[0].rpartition('.')[0]).parametrizations.weight.original
        assert torch.allclose(original1,original2,atol=1e-6,rtol=1e-5)
    first.journal.close();second.journal.close()


def test_disk_base_live_sessions_do_not_overwrite_one_another(tmp_path):
    from patchlib.live_linear_workspace_v59 import allocate, _release
    root=tmp_path/'base'
    one,lease_one=allocate(root)
    path=one/'weights.bin'
    path.write_bytes(b'original')
    two,lease_two=allocate(root)
    assert one.exists() and path.read_bytes()==b'original'
    assert two!=one
    _release(lease_one,one)
    _release(lease_two,two)
    assert not one.exists() and not two.exists()


def test_real_elastic_train_live_with_disk_moments(tmp_path,monkeypatch):
    """Exercise the production optimizer factory, seal and working checkpoint."""
    import hashlib
    from types import SimpleNamespace
    from patchlib import elasticnet
    async def disconnected(self):
        raise ConnectionError('test remote stays offline')
    monkeypatch.setattr(elasticnet.ElasticNode,'run',disconnected)
    torch.manual_seed(1331)
    model=Tiny()
    def load(_):
        return model, UniversalAdapter(model,'one-base','fstate-sha256:seed',rank=3)
    def batches(_):
        while True: yield None
    def step(m,adapter,opt,unused,args):
        with adapter.lock:
            torch.manual_seed(132)
            loss=m.forward()
            loss.backward()
            return float(loss.detach())
    args=__import__('types').SimpleNamespace(
        trust_remote_code=False,factory_file=None,allow_legacy_bin=False,resume=None,
        state_base='initial.fstate',grad_accum=1,local_steps=2,rounds=2,
        reinit_every=20,signing_key=None,unsafe_unsigned=True,
        ws='ws://127.0.0.1:8080/ws',link=['bob=pair_room_123456789'],
        peer_id='alice',trusted_peer_key=[],roster_file=None,model_id='one-base',
        journal=tmp_path/'online.db',self_public_key=None,max_journal_gb=1,
        max_delta_rms=1000.,lr=.0001,weight_decay=0.,device='cpu',
        wire_dtype='float16',token='',max_patch_mb=5,peer_timeout=.2,
        heartbeat_interval=.05,boot_grace=0.,sync_timeout=.1,
        inventory_interval=.05,inventory_cycle=.1,shard_state_manifest=None,
        shard_store=None,full_state_output=None,log_steps=100,peers=None,
        checkpoint_every=0,checkpoint_dir=tmp_path/'ck',shard_mb=4,max_state_gb=1,
        disk_optimizer_dir=tmp_path/'live-ssd',disk_optimizer_max_gb=1,
        durable_every=1,working_save_every=1)
    asyncio.run(elasticnet.train_elastic(args,load_model=load,
                                         training_batches=batches,run_step=step))
    assert snapshot_path(SimpleNamespace(path=args.journal)).is_file()
    # Active scratch removed on clean shutdown; durable working snapshot remains.
    assert not list((tmp_path/'live-ssd').rglob('optimizer.json'))
    assert len(list(tmp_path.glob('online.db.working.safetensors.moments-*')))==1


def test_live_cli_model_loader_isolates_shadow_weights(tmp_path,monkeypatch):
    from patchlib import cli
    from patchlib.layer_offload import install_disk_linear_offload
    from types import SimpleNamespace
    torch.manual_seed(327)
    template=torch.nn.Sequential(torch.nn.Linear(4,5),torch.nn.Linear(5,2))
    used=[]
    def fake_loader(opts,use_adapter=True):
        mdl=deepcopy(template)
        adapter=UniversalAdapter(mdl,'m','v',kind='loha',rank=2)
        info=install_disk_linear_offload(mdl,adapter,opts.disk_linear_dir)
        used.append(opts.disk_linear_dir)
        return mdl,adapter
    monkeypatch.setattr(cli,'load_model',fake_loader)
    args=SimpleNamespace(cmd='train-live',disk_linear_dir=tmp_path/'model-base')
    model_a,a=cli.load(args)
    model_b,b=cli.load(args)
    assert len(set(used))==2
    assert all(str(p).startswith(str(args.disk_linear_dir)) for p in used)
    backing_a=model_a[0]._fedpara_disk_source
    backing_b=model_b[0]._fedpara_disk_source
    data_a=Path(backing_a).read_bytes()
    assert Path(backing_b).is_file()
    with torch.no_grad():
        model_b[0].parametrizations.weight.original.add_(1)
    assert Path(backing_a).read_bytes()==data_a
