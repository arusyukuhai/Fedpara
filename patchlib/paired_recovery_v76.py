"""Fail-closed automatic recovery when the signed journal outruns a v75 pair.

The immutable, verified event replay (performed by the caller BEFORE invoking
this module) is the authority for committed Actor state.  A stale paired file
is *never* applied to that reconstructed Actor.  Only an append-only extension
of the checkpoint's exact journal prefix permits a new local Actor/Critic
checkpoint to be published with freshly initialized optimizer/value state.
This cannot resurrect unpublished local gradients or Critic steps.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from safetensors import safe_open

from .core import PatchError
from .localprogress import _tip
from . import paired_checkpoint_v75 as paired


@dataclass(frozen=True)
class RecoveryDecision:
    mode: str  # exact | reset
    previous_checkpoint_local_steps: int = 0
    previous_event_count: int = 0
    current_event_count: int = 0


def _event_at_index(journal, n):
    """Event in append order, not origin/seq order, with no SQLite thread hop."""
    if n == 0:
        return None
    row = journal.db.execute(
        'SELECT origin,seq,digest FROM events ORDER BY rowid LIMIT 1 OFFSET ?',
        (n - 1,),
    ).fetchone()
    return list(row) if row is not None else None


def inspect(journal, adapter, mixer):
    """Validate a checkpoint fully *before* authorizing discard of its state.

    A journal count alone is insufficient; the exact saved last event must
    still appear at the saved position.  engine.replay() must already have
    verified the event stream and reconstructed the committed Actor base.
    """
    path = paired.snapshot_path(journal)
    if not path.exists():
        return None
    if (path.is_symlink() or path.parent.is_symlink()
            or path.stat().st_size > paired.MAX_BYTES + 1024**2):
        raise PatchError('unsafe paired recovery checkpoint path or size')
    with safe_open(str(path), framework='pt', device='cpu') as reader:
        try:
            meta = json.loads((reader.metadata() or {})['paired'])
        except (KeyError, TypeError, ValueError) as exc:
            raise PatchError('invalid paired recovery checkpoint metadata') from exc
        actor = {'layout':adapter.layout,'kind':adapter.kind,'rank':adapter.rank,
                 'alpha':adapter.alpha,
                 'count':len(list(adapter.trainable_parameters()))}
        if (meta.get('format') != paired.FORMAT or
                meta.get('genesis') != journal.genesis or
                meta.get('actor') != actor or
                meta.get('critic') != paired._identity(mixer) or
                meta.get('critic_count') != len(paired._critic_params(mixer))):
            raise PatchError('paired recovery identity or layout mismatch')
        saved = meta.get('tip')
        if (not isinstance(saved, list) or len(saved) != 2 or
                type(saved[0]) is not int or saved[0] < 0 or
                (saved[0] == 0 and saved[1] is not None) or
                (saved[0] > 0 and (not isinstance(saved[1], list) or
                   len(saved[1]) != 3))):
            raise PatchError('invalid paired recovery journal cursor')
        if type(meta.get('steps')) is not int or meta['steps'] < 0 or (
            type(meta.get('critic_steps')) is not int or meta['critic_steps'] < 0
        ):
            raise PatchError('invalid paired recovery optimizer counters')
        keys = set(reader.keys())
        if paired._tensor_digest({k:reader.get_tensor(k) for k in keys}) != meta.get('tensor_sha256'):
            raise PatchError('paired recovery checkpoint tensor digest mismatch')
        a, ak = paired._validated_params(reader, keys, 'a', list(adapter.trainable_parameters()))
        c, ck = paired._validated_params(reader, keys, 'c', paired._critic_params(mixer))
        if ak | ck != keys:
            raise PatchError('paired recovery checkpoint contains unexpected tensors')
        # The application is all-or-nothing; these validated tensors are NOT
        # applied when the journal has moved beyond the snapshot boundary.
    current = _tip(journal)
    if saved == current:
        return RecoveryDecision('exact', meta['steps'], saved[0], current[0])
    if saved[0] >= current[0]:
        raise PatchError('paired recovery journal rolled back or diverged')
    if _event_at_index(journal, saved[0]) != saved[1]:
        raise PatchError('paired recovery journal prefix diverged; refusing reset')
    return RecoveryDecision('reset', meta['steps'], saved[0], current[0])


def recover(journal, adapter, actor_optimizer, mixer, *, verified_replay=False):
    """Restore exact paired state, or durably reset local-only state.

    This is intentionally not callable as a replacement for engine.replay():
    the caller MUST first verify signed events and replay them to reconstruct
    the authoritative committed Actor model/anchor.  Fresh optimizer and
    Critic objects must be supplied before this function is invoked.
    """
    if not verified_replay:
        raise PatchError('paired recovery requires verified signed journal replay')
    decision = inspect(journal, adapter, mixer)
    if decision is None:
        return None, None
    if decision.mode == 'exact':
        steps = paired.restore(journal, adapter, actor_optimizer, mixer)
        return steps, decision
    # We must NEVER apply the stale Actor factors, their AdamW moments, or the
    # stale Critic head to the already-replayed signed model.  The caller has
    # created clean optimizer/Critic state. Publish a new paired state BEFORE
    # allowing the next training step; atomic rename provides retryability:
    # if interrupted, the old pair is reconsidered on the next restart.
    snap = paired.capture(journal, adapter, actor_optimizer, mixer, 0, journal_tip=_tip(journal))
    paired.write(snap)
    return 0, decision
