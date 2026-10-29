"""Crash windows: signed event wins; local unpublished paired state is disposable."""
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from patchlib.core import PatchError
from patchlib.paired_checkpoint_v75 import capture, write, snapshot_path
from patchlib.paired_recovery_v76 import recover, inspect
from test_paired_checkpoint_v75 import setup


def _values(module):
    return [p.detach().clone() for p in module]


def test_exact_checkpoint_restores_uncommitted_optimizer_and_critic(tmp_path):
    _,a,o,m,j=setup(tmp_path)
    try:
        write(capture(j,a,o,m,3))
        actor=_values(a.trainable_parameters())
        critic=_values(m.critic.parameters())
        with torch.no_grad():
            for p in a.trainable_parameters():p.zero_()
            for p in m.critic.parameters():p.zero_()
        steps, decision=recover(j,a,o,m,verified_replay=True)
        assert (steps,decision.mode)==(3,'exact')
        assert all(torch.equal(x,y) for x,y in zip(actor,a.trainable_parameters()))
        assert all(torch.equal(x,y) for x,y in zip(critic,m.critic.parameters()))
    finally:m.close();j.close()


def test_advanced_signed_history_resets_local_state_without_rollback(tmp_path):
    _,a,o,m,j=setup(tmp_path)
    try:
        write(capture(j,a,o,m,4))
        j.advance()
        # Mimic engine.replay(): newly constructed base and clean optimizer.
        # No stale saved state may be applied to the replayed base.
        with torch.no_grad():
            for p in a.trainable_parameters():p.add_(10)
            for p in m.critic.parameters():p.sub_(3)
        actor=_values(a.trainable_parameters())
        critic=_values(m.critic.parameters())
        o.state.clear();m.critic_optim.state.clear();m.critic_steps=0
        steps, dec=recover(j,a,o,m,verified_replay=True)
        assert steps==0 and dec.mode=='reset' and dec.previous_checkpoint_local_steps==4
        assert dec.previous_event_count==0 and dec.current_event_count==1
        assert all(torch.equal(x,y) for x,y in zip(actor,a.trainable_parameters()))
        assert all(torch.equal(x,y) for x,y in zip(critic,m.critic.parameters()))
        assert not o.state and not m.critic_optim.state
        assert inspect(j,a,m).mode=='exact'
        assert recover(j,a,o,m,verified_replay=True)[0]==0
    finally:m.close();j.close()


def test_advanced_multiple_signed_events_after_saved_tip(tmp_path):
    _,a,o,m,j=setup(tmp_path)
    try:
        j.advance();write(capture(j,a,o,m,2));j.advance();j.advance()
        steps,decision=recover(j,a,o,m,verified_replay=True)
        assert steps==0 and decision.previous_event_count==1 and decision.current_event_count==3
    finally:m.close();j.close()


def test_fork_at_old_tip_fails_closed(tmp_path):
    _,a,o,m,j=setup(tmp_path)
    try:
        j.advance();write(capture(j,a,o,m,2));j.advance()
        j.db.execute("UPDATE events SET digest=? WHERE rowid=1",('0'*64,));j.db.commit()
        before=snapshot_path(j).read_bytes()
        with pytest.raises(PatchError,match='prefix diverged'):
            recover(j,a,o,m,verified_replay=True)
        assert snapshot_path(j).read_bytes()==before
    finally:m.close();j.close()


def test_rollback_and_same_count_divergence_rejected(tmp_path):
    _,a,o,m,j=setup(tmp_path)
    try:
        j.advance();write(capture(j,a,o,m,5))
        j.db.execute('DELETE FROM events');j.db.commit()
        with pytest.raises(PatchError,match='rolled back or diverged'):
            recover(j,a,o,m,verified_replay=True)
    finally:m.close();j.close()


def test_corrupted_stale_pair_is_not_silently_discarded(tmp_path):
    _,a,o,m,j=setup(tmp_path)
    try:
        write(capture(j,a,o,m,1))
        with safe_open(str(snapshot_path(j)),framework='pt',device='cpu') as reader:
            data={k:reader.get_tensor(k) for k in reader.keys()}
            meta=reader.metadata()
        key=next(k for k in data if k.startswith('c.p.'))
        data[key]=data[key]+0.01
        save_file(data,str(snapshot_path(j)),metadata=meta)
        j.advance()
        with pytest.raises(PatchError,match='digest mismatch'):
            recover(j,a,o,m,verified_replay=True)
    finally:m.close();j.close()


def test_invalid_room_and_journal_replay_proof_rejected(tmp_path):
    _,a,o,m,j=setup(tmp_path)
    try:
        write(capture(j,a,o,m,0));j.advance()
        with pytest.raises(PatchError,match='verified signed journal replay'):
            recover(j,a,o,m)
        m._critic_identity='other-room'
        with pytest.raises(PatchError,match='identity or layout'):
            recover(j,a,o,m,verified_replay=True)
    finally:m.close();j.close()


def test_publication_failure_retries_from_same_stale_snapshot(tmp_path,monkeypatch):
    from patchlib import paired_recovery_v76 as rec
    _,a,o,m,j=setup(tmp_path)
    try:
        write(capture(j,a,o,m,0));j.advance()
        before=snapshot_path(j).read_bytes()
        real=rec.paired.write
        def fail(_):raise OSError('simulated power loss before rename')
        monkeypatch.setattr(rec.paired,'write',fail)
        with pytest.raises(OSError,match='simulated power loss'):
            recover(j,a,o,m,verified_replay=True)
        assert snapshot_path(j).read_bytes()==before
        monkeypatch.setattr(rec.paired,'write',real)
        assert recover(j,a,o,m,verified_replay=True)[1].mode=='reset'
        assert recover(j,a,o,m,verified_replay=True)[1].mode=='exact'
    finally:m.close();j.close()


def test_empty_checkpoint_no_state(tmp_path):
    _,a,o,m,j=setup(tmp_path)
    try:
        assert recover(j,a,o,m,verified_replay=True)==(None,None)
    finally:m.close();j.close()
