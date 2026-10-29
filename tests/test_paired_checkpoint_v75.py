"""Crash-boundary checks for v75's *local* paired Actor/Critic snapshot."""
import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from patchlib.core import PatchError
from patchlib.cli import run_step
from patchlib.mixed_learning_v65 import MixedObjectiveTrainer
from patchlib.paired_checkpoint_v75 import (
    AsyncPairedSaver, capture, restore, write, snapshot_path)
from test_mixed_learning_v65 import make_args, setup_model, ToyTokenizer, write as jsonl


class Journal:
    def __init__(self,tmp_path):
        self.path=tmp_path/'journal.sqlite3'
        self.genesis='g'*64
        self.db=sqlite3.connect(self.path)
        self.db.execute('CREATE TABLE events (origin TEXT,seq INTEGER,digest TEXT)')
        self.db.commit()
    def count(self):return self.db.execute('SELECT count(*) FROM events').fetchone()[0]
    def advance(self):
        n=self.count()+1
        self.db.execute('INSERT INTO events VALUES(?,?,?)',('peer',n,'f'*64))
        self.db.commit()
    def close(self):self.db.close()


def setup(tmp_path):
    ppo=jsonl(tmp_path,'ppo.jsonl',[{'prompt':'test','reference':'x'}])
    args=make_args(tmp_path,ppo_jsonl=ppo,critic_state_dir=str(tmp_path/'critic'),
                   critic_kind='mlp',critic_hidden=16)
    model,adapter,optim,batch=setup_model()
    mix=MixedObjectiveTrainer(args,ToyTokenizer());mix._v75_paired=True
    args._v65_mixer=mix;args._v65_mixer_adapter=adapter
    optim.zero_grad(set_to_none=True)
    run_step(model,adapter,optim,batch,args)
    optim.step();optim.zero_grad(set_to_none=True)
    journal=Journal(tmp_path)
    return model,adapter,optim,mix,journal


def test_atomic_single_file_round_trip_preserves_both_optimizers(tmp_path):
    model,a,o,m,j=setup(tmp_path)
    before_a=[p.detach().clone() for p in a.trainable_parameters()]
    before_c=[p.detach().clone() for p in m.critic.parameters()]
    steps=m.critic_steps
    try:
        write(capture(j,a,o,m,1))
        assert snapshot_path(j).is_file()
        with torch.no_grad():
            for p in a.trainable_parameters():p.zero_()
            for p in m.critic.parameters():p.zero_()
        o.state.clear();m.critic_optim.state.clear();m.critic_steps=0
        assert restore(j,a,o,m)==1
        assert m.critic_steps==steps==1
        assert all(torch.equal(x,y) for x,y in zip(before_a,a.trainable_parameters()))
        assert all(torch.equal(x,y) for x,y in zip(before_c,m.critic.parameters()))
        assert len(o.state)>0 and len(m.critic_optim.state)>0
    finally:m.close();j.close()


def test_signed_journal_ahead_fails_closed_without_mutation(tmp_path):
    _,a,o,m,j=setup(tmp_path)
    try:
        write(capture(j,a,o,m,0))
        j.advance()
        saved=[p.detach().clone() for p in a.trainable_parameters()]
        with pytest.raises(PatchError,match='does not match signed journal'):
            restore(j,a,o,m)
        assert all(torch.equal(x,y) for x,y in zip(saved,a.trainable_parameters()))
    finally:m.close();j.close()


def test_metadata_tensor_corruption_fails_before_either_model_mutates(tmp_path):
    _,a,o,m,j=setup(tmp_path)
    try:
        write(capture(j,a,o,m,0))
        path=snapshot_path(j)
        with safe_open(str(path),framework='pt',device='cpu') as f:
            meta=f.metadata();data={k:f.get_tensor(k) for k in f.keys()}
        bad=next(k for k in data if k.startswith('c.p.'))
        data[bad]=torch.zeros(1) # wrong shape / dtype
        save_file(data,str(path),metadata=meta)
        a_before=[p.detach().clone() for p in a.trainable_parameters()]
        c_before=[p.detach().clone() for p in m.critic.parameters()]
        with pytest.raises(PatchError,match='digest mismatch|layout/values'):
            restore(j,a,o,m)
        assert all(torch.equal(x,y) for x,y in zip(a_before,a.trainable_parameters()))
        assert all(torch.equal(x,y) for x,y in zip(c_before,m.critic.parameters()))
    finally:m.close();j.close()


def test_failed_publication_keeps_previous_pair(tmp_path,monkeypatch):
    from patchlib import paired_checkpoint_v75 as paired
    _,a,o,m,j=setup(tmp_path)
    try:
        write(capture(j,a,o,m,0));before=snapshot_path(j).read_bytes()
        def fail(*a,**kw):raise OSError('disk full')
        monkeypatch.setattr(paired.os,'replace',fail)
        with pytest.raises(OSError,match='disk full'):write(capture(j,a,o,m,0))
        assert snapshot_path(j).read_bytes()==before
    finally:m.close();j.close()


def test_async_writer_immutable_capture_and_bounded_tasks(tmp_path,monkeypatch):
    import time
    from patchlib import paired_checkpoint_v75 as paired
    _,a,o,m,j=setup(tmp_path)
    real=paired.write
    def slow(cap):time.sleep(.06);return real(cap)
    monkeypatch.setattr(paired,'write',slow)
    async def runner():
        saver=AsyncPairedSaver()
        assert await saver.checkpoint(j,a,o,m,1)
        assert not await saver.checkpoint(j,a,o,m,2)
        # Background fsync doesn't hold a live parameter reference.
        with torch.no_grad():
            for p in a.trainable_parameters():p.add_(7.)
        await saver.drain()
        assert saver.skipped==1
        assert snapshot_path(j).exists()
        with safe_open(str(snapshot_path(j)),framework='pt',device='cpu') as f:
            assert not torch.equal(f.get_tensor('a.p.0'),list(a.trainable_parameters())[0].detach().cpu())
        assert await saver.checkpoint(j,a,o,m,3,force=True)
        await saver.drain()
        assert restore(j,a,o,m)==3
    try:asyncio.run(runner())
    finally:m.close();j.close()


def test_genesis_and_critic_identity_and_corrupt_tip_fail_closed(tmp_path):
    _,a,o,m,j=setup(tmp_path)
    try:
        write(capture(j,a,o,m,0))
        old=j.genesis;j.genesis='other'
        with pytest.raises(PatchError,match='does not match signed journal'):restore(j,a,o,m)
        j.genesis=old
        m._critic_identity='another-room'
        with pytest.raises(PatchError,match='configuration mismatch'):restore(j,a,o,m)
    finally:m.close();j.close()


def test_finite_tensor_payload_tamper_detected(tmp_path):
    _,a,o,m,j=setup(tmp_path)
    try:
        write(capture(j,a,o,m,0))
        path=snapshot_path(j)
        with safe_open(str(path),framework='pt',device='cpu') as f:
            meta=f.metadata();data={k:f.get_tensor(k) for k in f.keys()}
        key=next(k for k in data if k.startswith('a.p.'))
        data[key]=data[key]+1.0  # finite and correct shape; no header-only check catches it
        save_file(data,str(path),metadata=meta)
        with pytest.raises(PatchError,match='tensor digest mismatch'):
            restore(j,a,o,m)
    finally:m.close();j.close()


def test_new_signed_generation_never_loads_stale_independent_critic(tmp_path,monkeypatch):
    from patchlib import mixed_learning_v65 as mixed
    from patchlib.critic_state_v66 import snapshot_file
    from patchlib import cli
    _, old_adapter, _, old_mixer, old_journal=setup(tmp_path)
    try:
        # An older standalone value checkpoint should never be silently loaded
        # when a v75 paired room switches to a distinct signed generation.
        old_mixer._v75_paired=False
        old_mixer.save_critic()
        old_mixer._v75_paired=True
        assert snapshot_file(old_mixer._critic_dir).exists()
        args=old_mixer.args
        model,new_adapter,opt,batch=setup_model()
        args._v75_paired_active=True
        args._v65_mixer=old_mixer
        args._v65_mixer_adapter=old_adapter
        monkeypatch.setattr(mixed,'build',lambda a:MixedObjectiveTrainer(a,ToyTokenizer()))
        cli.run_step(model,new_adapter,opt,batch,args)
        newer=args._v65_mixer
        assert newer is not old_mixer
        assert newer._v75_paired
        assert newer.critic_steps==1  # not restored old 1 and incremented to 2
        assert not getattr(args,'_v75_skip_legacy_critic',False)
        newer.close()
    finally:
        old_mixer.close();old_journal.close()
