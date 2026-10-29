"""Kagami v45: opt-in trainer-boundary hot handoff primitives.

This module intentionally does NOT change the v1 WebRTC genesis negotiation:
calling swap locally without coordinated generation routing is unsafe. It
provides an actual in-process model/optimizer pointer swap, not a mere receipt.
A caller must supply the independently verified certified checkpoint, archive
all uncovered old updates, and route new traffic to the new genesis.

Neither a quorum signature alone nor a success boolean authorizes GC.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Callable

from .core import PatchError, _canonical_json
from .frontier_commitment_v44 import verify_full_frontier, header_digest


@dataclass(frozen=True)
class TrainerGeneration:
    epoch: int
    genesis: str
    checkpoint_hash: str
    model: Any
    adapter: Any
    optimizer: Any
    journal: Any


def _sha256(s):
    return isinstance(s, str) and len(s) == 64 and all(c in '0123456789abcdef' for c in s)


def durable_pointer(path: Path, *, epoch: int, genesis: str, checkpoint_hash: str):
    """Crash-atomic generation pointer. Only call after *all* new bytes exist.

    The consumer MUST verify genesis/checkpoint and journal identity on reload;
    the pointer is a locator, not a substitute for signed certification.
    """
    if type(epoch) is not int or epoch < 1 or not _sha256(genesis) or not _sha256(checkpoint_hash):
        raise PatchError('invalid durable generation pointer')
    p = Path(path)
    if p.is_symlink():
        raise PatchError('refusing symlink pointer')
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_json({'format': 'kagami-hot-pointer-v1', 'epoch': epoch,
                           'genesis': genesis, 'checkpoint': checkpoint_hash})
    temp = p.with_name(p.name+'.new-'+os.urandom(8).hex())
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os,'O_NOFOLLOW',0), 0o600)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, p)
        dfd = os.open(p.parent, os.O_DIRECTORY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
    finally:
        temp.unlink(missing_ok=True)


def checked_cut_and_suffix(db, cutoffs: dict[str, int], cut_digest: str):
    """Stream the exact committed cut and uncovered suffix from a journal DB.

    This checks *all* origin/seq/digest indexes; a missing old event is not
    silently treated as incorporated. The digest is the canonical v1 cut hash.
    The caller MUST separately verify that the snapshot equals this cut.
    """
    if not isinstance(cutoffs, dict) or not _sha256(cut_digest):
        raise PatchError('invalid certified cut')
    digest = hashlib.sha256()
    missing = set(cutoffs)
    suffix = []
    last = {}
    for origin, seq, event_digest in db.execute('SELECT origin,seq,digest FROM events ORDER BY origin,seq'):
        if (not isinstance(origin,str) or type(seq) is not int or
            not _sha256(event_digest) or seq <= last.get(origin,0)):
            raise PatchError('corrupt/nonmonotonic journal index')
        last[origin] = seq
        if seq <= cutoffs.get(origin,0):
            digest.update(_canonical_json([origin,seq,event_digest])+b'\n')
            missing.discard(origin)
        else:
            suffix.append((origin,seq,event_digest))
    if digest.hexdigest() != cut_digest:
        raise PatchError('journal prefix differs from certified checkpoint cut')
    if any(last.get(origin,0) < seq for origin,seq in cutoffs.items()):
        raise PatchError('checkpoint cut is ahead of durable journal')
    return suffix


def checked_paged_cut(header, pages, expected_root: str, *, expected_digest: str | None = None):
    """Refuse partial Merkle frontiers even when their individual proofs pass."""
    if not _sha256(expected_root) or header.get('root') != expected_root:
        raise PatchError('wrong generation boundary root')
    if expected_digest is not None and header_digest(header) != expected_digest:
        raise PatchError('different generation boundary header')
    return verify_full_frontier(header, pages)


class TrainerStepRouter:
    """In-process double-buffered generation pointer with an optimizer barrier.

    Training code must run every forward/backward AND optimizer.step within
    ``async with router.step() as g``. The network relay may continue in a
    separate task. Stage the candidate model off-loop, without changing the
    currently training model. The serialized activation barrier permits no
    partially stepped optimizer to be replaced. It does NOT itself convert
    signed old events, certify checkpoints or rekey WebRTC connections.
    """
    def __init__(self, initial: TrainerGeneration):
        self._check_generation(initial)
        self._current = initial
        self._active = 0
        self._switching = False
        self._fatal = False
        self._pending_publication = None
        self._condition = asyncio.Condition()

    @staticmethod
    def _check_generation(g):
        if (not isinstance(g,TrainerGeneration) or type(g.epoch) is not int or
            not 0 <= g.epoch < 2**32 or not _sha256(g.genesis) or
            not _sha256(g.checkpoint_hash) or g.model is None or
            g.adapter is None or g.optimizer is None or g.journal is None):
            raise PatchError('invalid/partial model+optimizer generation')
        if getattr(g.journal, 'genesis', None) != g.genesis:
            raise PatchError('optimizer generation has a mismatched journal')

    @property
    def current(self):
        return self._current

    @asynccontextmanager
    async def step(self):
        async with self._condition:
            await self._condition.wait_for(lambda: not self._switching or self._fatal)
            if self._fatal:
                raise PatchError('epoch publication failed: restart from durable pointer before training')
            self._active += 1
            generation = self._current
        try:
            yield generation
        finally:
            async with self._condition:
                self._active -= 1
                self._condition.notify_all()

    async def switch(self, target:TrainerGeneration, *,
                     verified:Callable[[TrainerGeneration],bool],
                     flush_old:Callable[[TrainerGeneration],Any],
                     carry_old:Callable[[TrainerGeneration,TrainerGeneration],Any],
                     attest_new:Callable[[TrainerGeneration],Any],
                     pointer:Path,
                     activate_new:Callable[[TrainerGeneration,TrainerGeneration],Any]|None=None) -> TrainerGeneration:
        """Install at next *step* boundary; never closes WebRTC or stops process.

        Each callback MUST be a local trusted action. ``flush_old`` seals
        unpublished gradients, ``carry_old`` durably republishes or archives
        and replays each unincorporated old effective-weight delta exactly once,
        ``attest_new`` verifies installed state + durable new journal. A caller
        which cannot provide these guarantees MUST NOT call switch.
        Old journal remains on disk; local GC is handled separately by the
        verified runtime after old writer ownership has been released.
        """
        self._check_generation(target)
        if any(not callable(c) for c in (verified,flush_old,carry_old,attest_new)):
            raise PatchError('missing trusted migration hooks')
        if verified(target) is not True:
            raise PatchError('new checkpoint not independently verified')
        async with self._condition:
            if self._switching or self._fatal:
                raise PatchError('generation change already in progress or awaiting publication recovery')
            old = self._current
            if target.epoch != old.epoch + 1 or target.genesis == old.genesis:
                raise PatchError('epoch must advance by one to a distinct genesis')
            self._switching = True
        try:
            # Waiting for an in-flight step is cancellable.  Earlier releases
            # performed this wait before the finally clause, leaving all new
            # steps blocked forever if shutdown cancelled the switch task.
            async with self._condition:
                await self._condition.wait_for(lambda: self._active == 0)
            # Never alias the old optimizer or model into a newly certified
            # base. Keeping old AdamW moments is not equivalent to rebasing.
            if target.model is old.model or target.optimizer is old.optimizer or target.adapter is old.adapter:
                raise PatchError('new generation must have independently prepared model and optimizer')
            if target.journal is old.journal:
                raise PatchError('new generation cannot reuse the old event journal')
            await _await_if_needed(flush_old(old))
            await _await_if_needed(carry_old(old,target))
            if await _await_if_needed(attest_new(target)) is not True:
                raise PatchError('new journal/checkpoint/carry not durably acknowledged')
            # Quorum head may advance while we are sealing local gradients;
            # refuse to publish a target that is no longer certified.
            if verified(target) is not True:
                raise PatchError('generation certificate changed during handoff')
            # The durable new locator precedes publishing the live pointer.
            # A crash between them restarts from new checkpoint, not old.
            # From here even an fsync failure can mean rename committed. Never
            # resume the old writer on an uncertain persistent pointer.
            self._pending_publication = (old, target, Path(pointer), activate_new)
            self._fatal = True
            await self._publish_pending()
            self._current = target
            return old
        finally:
            async with self._condition:
                self._switching = False
                self._condition.notify_all()

    async def _publish_pending(self):
        old, target, pointer, activate_new = self._pending_publication
        durable_pointer(pointer, epoch=target.epoch, genesis=target.genesis,
                        checkpoint_hash=target.checkpoint_hash)
        if activate_new is not None and await _await_if_needed(activate_new(old, target)) is not True:
            raise PatchError('new generation transport activation failed')
        self._current = target
        self._pending_publication = None
        self._fatal = False

    @property
    def pending_publication(self):
        return self._pending_publication is not None

    async def resume_publication(self):
        """Retry only pointer/transport publication, never flush or replay twice.

        The locally supplied transport hook MUST be idempotent. Works during
        this process lifetime; after a crash reconstruct from the durable
        checkpoint + carry + journal before creating this router.
        """
        async with self._condition:
            if self._switching:
                raise PatchError('generation publication already in progress')
            if self._pending_publication is None:
                return False
            self._switching = True
        try:
            await self._publish_pending()
            return True
        finally:
            async with self._condition:
                self._switching = False
                self._condition.notify_all()


async def _await_if_needed(value):
    if asyncio.iscoroutine(value) or isinstance(value, asyncio.Future):
        return await value
    return value


def verify_certified_checkpoint(target:TrainerGeneration, *, ledger, manifest:Path, store:Path):
    """Independent LOCAL verification, not an arbitrary user/remote boolean.

    The ledger must have authenticated the quorum certificate. Pinned trust
    configuration and the checkpoint bytes remain required even after voting.
    """
    from .shards import read_manifest, semantic_shard_sha256, verify_shards
    cert=ledger.latest()
    if not cert or not isinstance(cert.get('proposal'),dict):
        raise PatchError('missing accepted quorum certificate')
    p=cert['proposal']
    if p.get('epoch')!=target.epoch or p.get('manifest')!=target.checkpoint_hash:
        raise PatchError('certified generation/checkpoint differs from staged model')
    manifest=Path(manifest)
    if manifest.is_symlink() or not manifest.is_file() or Path(store).is_symlink():
        raise PatchError('unsafe certified snapshot path')
    with manifest.open('rb') as f:
        actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=p['manifest']:
        raise PatchError('checkpoint hash mismatch')
    meta=read_manifest(manifest,max_state_bytes=2*1024**4)
    if semantic_shard_sha256(meta)!=p['semantic']:
        raise PatchError('checkpoint semantic mismatch')
    verify_shards(meta,store)
    return True


class AutoGC:
    """Async GC poller for *already verified* EpochRuntimeV43 state.

    Does not authorize destructive cleanup; runtime.poll() recomputes the
    stricter filesystem/journal checks and rejects live old writers or any
    uncovered post-cut event. The caller owns the lifecycle of this task.
    """
    def __init__(self, runtime, interval_s:float=30.0):
        if not 1 <= interval_s <= 3600:
            raise ValueError('invalid GC polling interval')
        self.runtime=runtime
        self.interval_s=interval_s
        self._task=None
        self.errors=[]

    async def run_once(self):
        # Runtime has sqlite connections bound to the event loop's thread.
        # Disk-heavy deletion can still be moved to a separately opened local
        # runtime in the future; don't move thread-affine SQLite objects now.
        return self.runtime.poll()

    async def run(self, stop:asyncio.Event):
        while not stop.is_set():
            try:
                await self.run_once()
            except (PatchError, OSError) as e:
                self.errors.append(str(e)[:160])
                self.errors[:] = self.errors[-32:]
            try:
                await asyncio.wait_for(stop.wait(),timeout=self.interval_s)
            except asyncio.TimeoutError:
                pass


def stream_cut_summary(db, cutoffs:dict[str,int], cut_digest:str):
    """Bounded-memory full journal scan for a candidate epoch boundary.

    Only O(number of event origins) RAM, irrespective of event count/size.
    The post-cut digest is provenance, NOT sufficient for deleting original
    signed events or for reconstructing their numerical contributions.
    """
    if not isinstance(cutoffs,dict) or not _sha256(cut_digest):
        raise PatchError('invalid certified cut')
    prefix=hashlib.sha256()
    suffix=hashlib.sha256()
    last={}
    count=0
    for origin,seq,digest in db.execute('SELECT origin,seq,digest FROM events ORDER BY origin,seq'):
        if (not isinstance(origin,str) or type(seq) is not int or
            not _sha256(digest) or seq<=last.get(origin,0)):
            raise PatchError('corrupt/nonmonotonic journal index')
        last[origin]=seq
        item=_canonical_json([origin,seq,digest])+b'\n'
        if seq<=cutoffs.get(origin,0):prefix.update(item)
        else:suffix.update(item);count+=1
    if prefix.hexdigest()!=cut_digest or any(last.get(k,0)<v for k,v in cutoffs.items()):
        raise PatchError('journal is incomplete or differs from certified cut')
    return {'pending_count':count,'pending_digest':suffix.hexdigest()}


def rebase_authenticated_additive_suffix(engine, event_wires, *, cutoffs:dict[str,int],
                                         durable_archive_contains:Callable[[str,int,str],bool]):
    """Replay signed old-genesis FedSum *effective-weight* deltas on new base.

    The caller first installs the certified checkpoint into a CLEAN adapter
    of precisely the original layout and uses a verifier-pinned old-genesis
    ContinualElasticEngine. The caller must durably archive the ORIGINAL
    signed event bodies and replay this archive on each restart BEFORE using
    any newly signed epoch events. No factor-space subtraction is allowed.

    This operation does not translate the signature to a new-genesis event;
    current v1 WebRTC peers cannot exchange these carries. It must NOT be used
    as a GC authorization until the new network protocol supports them.
    """
    if not isinstance(cutoffs,dict) or not callable(durable_archive_contains):
        raise PatchError('missing certified cut or durable archive checker')
    seen=set()
    last=None
    applied=[]
    for wire in event_wires:
        # _check authenticates the original peer signature and validates the
        # old genesis, layouts, numerical finiteness and max delta RMS.
        info,prepared=engine._check(wire)
        identity=(info.origin,info.seq)
        if info.seq<=cutoffs.get(info.origin,0):
            raise PatchError('attempted to rebase an update already included in checkpoint')
        if identity in seen or (last is not None and identity<=last):
            raise PatchError('duplicate/unsorted cross-generation update')
        if (identity in engine.applied or
            durable_archive_contains(info.origin,info.seq,info.digest) is not True):
            raise PatchError('unarchived or already applied old-genesis update')
        # The engine's _apply_checked computes the effective tensor delta as
        # f(now_factors)-f(previous_factors), then ADDS it to the new base.
        engine._apply_checked(info,prepared)
        applied.append(identity)
        seen.add(identity);last=identity
    return applied
