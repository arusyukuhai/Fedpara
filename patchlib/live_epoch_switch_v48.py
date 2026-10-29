"""Opt-in, fail-closed online *trainer* epoch activation service.

Bridges signed GenerationLedger certificates to TrainerStepRouter without
stopping an independent relay. An integrator must provide an independently
verified checkpoint->new model/optimizer/journal constructor and transport
handoff: legacy v1 WebRTC links cannot accept a different genesis in place.

This is a *real* model/optimizer step-boundary switch, not a claim that
Kagami's existing ElasticNode is generation-multiplexed. No auto-GC is
permitted without a new-generation peer transport and durable carry proof.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Any

from .core import PatchError, _canonical_json
from .epoch_hot_swap_v45 import TrainerGeneration, TrainerStepRouter
from .online_generation import _digest


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def certified_genesis(proposal):
    """Stable and domain-separated network epoch ID for ALL participating peers."""
    if (not isinstance(proposal,dict) or type(proposal.get('epoch')) is not int
            or not 1 <= proposal['epoch'] < 2**32 or not _sha(proposal.get('manifest'))
            or not _sha(proposal.get('semantic')) or not _sha(proposal.get('parent'))
            or not isinstance(proposal.get('room'),str)):
        raise PatchError('invalid certified genesis derivation fields')
    return hashlib.sha256(_canonical_json(['kagami-certified-epoch-genesis-v1',
        proposal['room'],proposal['epoch'],_digest(proposal),proposal['manifest'],
        proposal['semantic']])).hexdigest()


def verify_candidate_files(prepared,proposal,*,max_state_bytes=2*1024**4):
    """Hash full manifest AND all referenced data-only shards before activation.

    Do NOT trust the remote manifest filename, presence of a file, or a quorum
    signature as proof that the weight bytes on disk are complete.
    """
    from .shards import read_manifest,verify_shards,semantic_shard_sha256
    path=Path(prepared.manifest)
    store=Path(prepared.store)
    if (path.is_symlink() or not path.is_file() or store.is_symlink()
            or not store.is_dir()):
        raise PatchError('certified checkpoint path missing or symlinked')
    hash_=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            hash_.update(chunk)
    if hash_.hexdigest()!=proposal['manifest']:
        raise PatchError('certified manifest SHA-256 mismatch')
    meta=read_manifest(path,max_state_bytes=max_state_bytes)
    if semantic_shard_sha256(meta)!=proposal['semantic']:
        raise PatchError('certified semantic weight hash mismatch')
    verify_shards(meta,store)
    return True


@dataclass(frozen=True)
class PreparedEpoch:
    """Immutable, fully verified candidate; no remote Python code is executed."""
    epoch: int
    proposal_digest: str
    manifest_hash: str
    generation: TrainerGeneration
    manifest: Path
    store: Path
    # Optional v52 streaming suffix staging, backed by a durable carry archive.
    # This is LOCAL state and never transmitted in a signed certificate.
    rebaser: Any = None


class LiveEpochSwitchService:
    """Generation control for a running trainer, using dependency injection.

    The generation-wide quorum proof comes from a pinned GenerationLedger.
    Every candidate's full checkpoint bytes and journal are checked a second
    time at the final step boundary.  The active WebRTC transport MUST be
    generation-aware before `activate` may be called.  `link_handoff` is an
    explicit precondition, not a boolean flag supplied by remote peers.

    All callbacks are locally controlled functions.  At most one activation
    can be in flight; a failed prepare never touches the active model.
    """
    def __init__(self, *, ledger, router: TrainerStepRouter, root: Path,
                 build_shadow: Callable[[dict], Awaitable[PreparedEpoch]],
                 verify_local: Callable[[PreparedEpoch,dict], bool],
                 flush_unpublished: Callable[[TrainerGeneration], Any],
                 carry_suffix: Callable[[TrainerGeneration,TrainerGeneration,dict], Any],
                 transport_handoff: Callable[[TrainerGeneration,TrainerGeneration,dict], Any],
                 attest_journal: Callable[[TrainerGeneration,dict], Any],
                 validate_shards: bool = True, checkpoint_stager=None,
                 checkpoint_lifecycle=None, gc_runtime_factory=None,
                 gc_interval=60.0, old_ingress_frozen=None,
                 warm_carry_batches_per_poll=8):
        self.ledger = ledger
        self.router = router
        self.root = Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self._build = build_shadow
        self._verify = verify_local
        self._flush = flush_unpublished
        self._carry = carry_suffix
        self._transport = transport_handoff
        self._attest = attest_journal
        self.validate_shards=validate_shards
        self.checkpoint_stager=checkpoint_stager
        if not 10.0<=gc_interval<=3600:
            raise ValueError('invalid GC interval')
        self.gc_runtime_factory=gc_runtime_factory
        self.gc_interval=gc_interval
        self._next_gc=0.0
        self._lock = asyncio.Lock()
        self._retry_at = 0.0
        self._failures = 0
        if not 1 <= warm_carry_batches_per_poll <= 128:
            raise ValueError('invalid warm carry budget')
        self.warm_carry_batches_per_poll=warm_carry_batches_per_poll
        self.old_ingress_frozen=old_ingress_frozen
        self._pending_shadow=None
        self.shadow_ready=False  # v53: prewarm without publishing while trainer steps run
        self.pointer = self.root/'active-epoch.json'
        if checkpoint_stager is not None:
            if checkpoint_lifecycle is None:
                from .epoch_lifecycle_v51 import DurableEpochLifecycle
                checkpoint_lifecycle=DurableEpochLifecycle(self.root/'checkpoint-leases',
                    stager=checkpoint_stager,pointer=self.pointer)
            self.checkpoint_lifecycle=checkpoint_lifecycle
        else:
            if checkpoint_lifecycle is not None:
                raise ValueError('checkpoint lifecycle needs a bounded checkpoint stager')
            self.checkpoint_lifecycle=None
        # Restart cannot silently construct an old trainer over a new pointer.
        # The caller must reconstruct the committed generation before admission.
        if self.pointer.exists():
            if self.pointer.is_symlink() or self.pointer.stat().st_size > 4096:
                raise PatchError('unsafe active epoch pointer')
            saved = json.loads(self.pointer.read_text())
            current = router.current
            if (saved.get('format') != 'kagami-hot-pointer-v1' or
                    saved.get('epoch') != current.epoch or
                    saved.get('genesis') != current.genesis or
                    saved.get('checkpoint') != current.checkpoint_hash):
                raise PatchError('reconstruct durable active generation before starting trainer')

    def _certificate(self):
        latest=self.ledger.latest()
        if not latest:return None
        proposal=latest.get('proposal')
        if not isinstance(proposal,dict):raise PatchError('corrupt certified proposal')
        epoch=proposal.get('epoch')
        if type(epoch) is not int or epoch < 1 or epoch >= 2**32:
            raise PatchError('invalid certified epoch')
        # Do not accept an unsigned 'head' notification: read the actual
        # persisted certificate and validate that the proposal is unchanged.
        certified=self.ledger.certified_epoch(epoch)
        if certified is None or certified.get('proposal')!=proposal:
            raise PatchError('quorum head is not durably certified')
        # A disconnected peer may lag many commits. Walk durable certificates
        # one at a time rather than getting stuck forever at the latest head.
        expected = self.router.current.epoch + 1
        if epoch > expected:
            next_certificate = self.ledger.certified_epoch(expected)
            if next_certificate is None:
                raise PatchError('missing intermediate generation; fetch certified chain')
            proposal = next_certificate.get('proposal')
            if not isinstance(proposal, dict) or proposal.get('epoch') != expected:
                raise PatchError('invalid intermediate certified generation')
        return proposal

    async def activate_next(self, *, prepare_only:bool=False):
        """Install the next *certified* epoch, if available; otherwise False.

        Heavy model preparation is performed by build_shadow, outside the
        short step boundary. The transport callback must activate/rekey both
        ingress and egress by epoch and must reject stale writes. It is not
        invoked after any irreversible log deletion (none occurs here).
        """
        async with self._lock:
            if self.router.pending_publication:
                result=await self.router.resume_publication()
                if result and self.checkpoint_lifecycle is not None:
                    cert=self.ledger.certified_epoch(self.router.current.epoch)
                    if cert is None:
                        raise PatchError('missing published generation certificate')
                    await self.checkpoint_lifecycle.published(cert['proposal'])
                return result
            # A process may have crashed after publishing the pointer but
            # before marking its cache lease active. Reconcile on restart.
            if self.checkpoint_lifecycle is not None and self.router.current.epoch>0:
                cert=self.ledger.certified_epoch(self.router.current.epoch)
                if cert is None:
                    raise PatchError('cannot reconstruct cache from uncertified trainer')
                await self.checkpoint_lifecycle.reconcile_active(cert['proposal'])
            proposal=self._certificate()
            if proposal is None or proposal['epoch']<=self.router.current.epoch:
                return False
            # Applying epoch N+2 while N+1 is missing could skip unpublished
            # local updates. Recover the missing certificate first.
            if proposal['epoch']!=self.router.current.epoch+1:
                raise PatchError('missing intermediate generation; fetch certified chain')
            from .online_generation import _digest as _proposal_digest
            pending=self._pending_shadow
            if pending is not None and pending[0]!=_proposal_digest(proposal):
                # Never reuse a partially rebased model for a DIFFERENT cut.
                self._pending_shadow=None
                self.shadow_ready=False
                pending=None
            if pending is not None:
                prepared=pending[1]
            elif self.checkpoint_stager is None:
                prepared=await self._build(proposal)
            elif pending is None:
                if self.checkpoint_lifecycle is not None:
                    await self.checkpoint_lifecycle.sweep_superseded(proposal['epoch'])
                    checkpoint=await self.checkpoint_lifecycle.stage(proposal)
                else:
                    checkpoint=await self.checkpoint_stager.prepare(proposal)
                # Explicit two-argument constructor when using the bounded
                # cache: the builder loads these independently verified files.
                prepared=await self._build(proposal,checkpoint)
                if (not isinstance(prepared,PreparedEpoch) or
                        Path(prepared.manifest)!=checkpoint.manifest or
                        Path(prepared.store)!=checkpoint.store):
                    raise PatchError('builder substituted staged checkpoint paths')
            if (not isinstance(prepared,PreparedEpoch) or
                    prepared.epoch!=proposal['epoch'] or
                    prepared.proposal_digest!=_digest(proposal) or
                    prepared.manifest_hash!=proposal.get('manifest') or
                    not _sha(prepared.manifest_hash) or
                    prepared.generation.epoch!=prepared.epoch or
                    prepared.generation.checkpoint_hash!=prepared.manifest_hash or
                    prepared.generation.genesis!=certified_genesis(proposal)):
                raise PatchError('shadow generation disagrees with certified proposal')
            # A local checksum callback MUST validate the complete content and
            # the exact semantic model before any optimizer can be switched.
            def checked(_candidate):
                fresh=self.ledger.certified_epoch(prepared.epoch)
                if fresh is None or fresh.get('proposal')!=proposal:
                    return False
                return self._verify(prepared,proposal) is True
            # A million-event warm carry may need many polls. Do not hash a
            # multi-terabyte checkpoint again on EVERY one of those polls.
            # Verify when first staged, and AGAIN just before the final swap.
            if pending is None:
                if self.validate_shards:
                    await asyncio.to_thread(verify_candidate_files,prepared,proposal)
                if not checked(prepared.generation):
                    raise PatchError('candidate checkpoint not locally verified')
            if prepared.rebaser is not None:
                # Stage old signed suffix incrementally in the background while
                # the old optimizer and WebRTC relay keep running. Unlike an
                # O(events) one-shot stage, this advances at most a bounded
                # number of events per certificate poll.
                rebaser=prepared.rebaser
                if pending is None or getattr(rebaser,'_validated',True) is False:
                    try:
                        rebaser.validate_cut()
                    except PatchError as exc:
                        # A connected peer may not yet possess every event
                        # within the certified checkpoint cut. Keep the
                        # already-verified shadow rather than allocating a
                        # second massive model on every anti-entropy retry.
                        if ('not all certified-cut events are available' in str(exc)
                                or 'missing signed sequence' in str(exc)):
                            self._pending_shadow=(_proposal_digest(proposal),prepared)
                            self.shadow_ready=False
                            return False
                        raise
                for _ in range(self.warm_carry_batches_per_poll):
                    progress=await rebaser.tick()
                    if progress.caught_up:
                        break
                if not progress.caught_up:
                    self._pending_shadow=(_proposal_digest(proposal),prepared)
                    self.shadow_ready=False
                    return False
                # The final suffix sweep MUST be done after the caller's
                # flush callback has frozen + drained old-genesis ingress.
                if self.old_ingress_frozen is None:
                    raise PatchError('streaming epoch adoption requires an old writer fence')
                # The cached model has changed under signed old suffix replay
                # since preparation. Re-verify all immutable checkpoint shards
                # before attempting the short final writer barrier.
                if self.validate_shards:
                    await asyncio.to_thread(verify_candidate_files,prepared,proposal)
                if not checked(prepared.generation):
                    raise PatchError('certificate/checkpoint changed during warm carry')
            # The production optimizer must request prewarm in a background
            # task and call the irreversible publication at its OWN clean step
            # boundary. An asynchronous stage must never swap mid-backward.
            if prepare_only:
                self._pending_shadow=(_proposal_digest(proposal),prepared)
                self.shadow_ready=True
                return False
            self.shadow_ready=False
            async def carry(old,new):
                if prepared.rebaser is not None:
                    if await _await(self.old_ingress_frozen(old)) is not True:
                        raise PatchError('old epoch writers are not frozen after sealing')
                    await prepared.rebaser.finish_under_ingress_barrier()
                # Critical: the archive is durable before changing any route;
                # old-genesis events after certified cut must be accounted for.
                await _await(self._carry(old,new,proposal))
            async def attest(new):
                # Re-key networking ONLY after catch-up has been durably
                # recorded. If it cannot be made atomic with journal identity,
                # callers must raise rather than silently continue.
                return await _await(self._attest(new,proposal)) is True
            # The trainer's *durable* new pointer is written before new-gen
            # network sends are enabled. If the transport fails at that last
            # phase, TrainerStepRouter fails closed until crash recovery.
            async def publish_network(old,new):
                return await _await(self._transport(old,new,proposal)) is True
            await self.router.switch(prepared.generation, verified=checked,
                flush_old=self._flush,carry_old=carry,attest_new=attest,
                pointer=self.pointer,activate_new=publish_network)
            self._pending_shadow=None
            self.shadow_ready=False
            if self.checkpoint_lifecycle is not None:
                await self.checkpoint_lifecycle.published(proposal)
            return True

    async def close(self):
        """Close lease metadata; the caller still owns the shared disk stager."""
        if self.checkpoint_lifecycle is not None:
            self.checkpoint_lifecycle.close()

    async def run(self, stop:asyncio.Event, *, interval:float=3.0,
                  report:Callable[[BaseException],Any]|None=None):
        """Lightweight watcher; certificate-only reads between transitions."""
        if not 0.2<=interval<=300.0:raise ValueError('bad epoch poll interval')
        while not stop.is_set():
            try:
                if time.monotonic() >= self._retry_at:
                    await self.activate_next()
                    if (self.gc_runtime_factory is not None and
                            self.checkpoint_lifecycle is not None and
                            self.router.current.epoch > 0 and
                            self.pointer.is_file() and
                            time.monotonic()>=self._next_gc):
                        # EpochRuntimeV43 proves that no signed carry or
                        # live old writer is lost, then deletes the old log.
                        # The checkpoint cache is reclaimed only AFTER that.
                        self._next_gc=time.monotonic()+self.gc_interval
                        await self.checkpoint_lifecycle.gc_from_runtime(
                            self.gc_runtime_factory,self.router.current.epoch)
                    self._failures = 0
                    self._retry_at = 0.0
            except (PatchError,OSError,ValueError,RuntimeError) as exc:
                # Failed checkpoints must not cause a full model rebuild every
                # three seconds across thousands of peers. Jitter avoids storms.
                self._failures = min(8, self._failures+1)
                self._retry_at = time.monotonic()+random.uniform(.75,1.25)*min(300,2**self._failures)
                if report is not None:
                    await _await(report(exc))
            try:await asyncio.wait_for(stop.wait(),timeout=interval)
            except asyncio.TimeoutError:pass



async def _await(value):
    if asyncio.isfuture(value) or asyncio.iscoroutine(value):return await value
    return value
