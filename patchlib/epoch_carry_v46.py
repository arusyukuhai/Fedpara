"""Authenticated, crash-replayable cross-generation FedSum carry archive.

This is an epoch-local *additional input* to a verified checkpoint.  It is
not a quorum certificate or authority to delete an old journal.  Original
signed old-genesis LoHa updates are archived, without signature rewriting, in
an EventJournal whose payloads live outside SQLite.  An arriving peer can
reconstruct checkpoint + carried suffix + current-generation journal exactly.

A caller must first verify the signed checkpoint's complete Merkle frontier.
All post-cut events are ordered by (source generation, origin, seq), and
source/target genesis are immutable.  No remote claim authorizes GC: a carry
must either be covered by a later certified snapshot, or kept available for
returning peers.
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from .core import PatchError, _canonical_json
from .elastic import EventJournal


def _hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


class CrossEpochCarryArchive:
    def __init__(self, path: Path, *, source_genesis: str, target_genesis: str,
                 certified_manifest: str, epoch: int, cutoffs: dict[str,int]):
        if (not all(_hex64(s) for s in (source_genesis,target_genesis,certified_manifest))
                or source_genesis == target_genesis or type(epoch) is not int
                or not 1 <= epoch < 2**32 or not isinstance(cutoffs,dict)
                or any(not isinstance(k,str) or type(v) is not int or v < 0
                       for k,v in cutoffs.items())):
            raise PatchError('invalid authenticated carry boundary')
        self.source_genesis=source_genesis
        self.target_genesis=target_genesis
        self.certified_manifest=certified_manifest
        self.epoch=epoch
        self.cutoffs=dict(cutoffs)
        context={'version':1,'old':source_genesis,'new':target_genesis,
                 'manifest':certified_manifest,'epoch':epoch,'cutoffs':sorted(cutoffs.items())}
        self.genesis=hashlib.sha256(_canonical_json(context)).hexdigest()
        self.journal=EventJournal(path,self.genesis,max_bytes=0)
        # Durable PK-index is the membership authority. A Python set of every
        # historically applied event grew without bound on long-lived peers.
        self._replayed_count=0
        self._poisoned=False
        self._async_lock=asyncio.Lock()
        self._replayed=False

    def _validate_engine(self,engine):
        if (getattr(engine,'genesis',None)!=self.target_genesis or
                getattr(engine,'unsafe',True) or not getattr(engine,'trusted',None)):
            raise PatchError('cross-epoch carry requires trusted new-generation engine')

    def _check(self,engine,wire,*,allow_covered=False):
        self._validate_engine(engine)
        info,prepared=engine._check(wire,authenticated_source_genesis=self.source_genesis)
        if not allow_covered and info.seq <= self.cutoffs.get(info.origin,0):
            raise PatchError('carry is already covered by certified checkpoint')
        return info,prepared

    def replay(self,engine):
        """Call ONCE after loading the exact clean certified base, before
        current-generation journal replay or any optimizer step. On recovery,
        rebuild the model from the base; never replay onto a working snapshot.
        """
        if self._replayed:
            raise PatchError('carry archive already replayed on current model')
        self._validate_engine(engine)
        if self._poisoned:
            raise PatchError('poisoned carry archive requires clean base reload')
        self._replayed=True
        count=0
        try:
            for wire in self.journal.events():
                info,prepared=self._check(engine,wire)
                # The journal's (origin, seq) PRIMARY KEY is unique. Replay
                # always starts on a verified *clean* base, never a live model.
                engine._apply_checked(info,prepared)
                count+=1
        except BaseException:
            # An exception can leave a partially modified tensor: no further
            # mutations to this engine until a clean checkpoint is reloaded.
            self._replayed=False
            self._poisoned=True
            raise
        self._replayed_count=count
        return count

    def accept(self,engine,wire:bytes):
        """Archive before application: a crash after append is recovered by
        rebuilding from base and replaying the entire authenticated archive.
        Duplicate signed receipts never apply a second delta.
        """
        if not self._replayed or self._poisoned:
            raise PatchError('must replay durable carries on clean base before receiving')
        if not isinstance(wire,bytes):
            raise PatchError('invalid signed old-generation event')
        info,prepared=self._check(engine,wire)
        key=(info.origin,info.seq)
        existing=self.journal.has(*key)
        if existing:
            if existing[0]!=hashlib.sha256(wire).hexdigest():
                raise PatchError('conflicting carried event identity')
            return False
        try:
            self.journal.append(info.origin,info.seq,wire)
            engine._apply_checked(info,prepared)
        except BaseException:
            self._poisoned=True
            raise
        self._replayed_count+=1
        return True

    async def accept_prechecked(self,engine,wire,info,prepared):
        """Persist authenticated bytes before applying, with O(1) RAM index.

        Concurrent inbound consumers serialize against the archive's own lock,
        rather than relying on the caller to supply a shared commit lock.
        If an append or tensor update fails, the instance is permanently
        poisoned until reconstructed from a clean checkpoint and the log.
        """
        async with self._async_lock:
            if not self._replayed or self._poisoned:
                raise PatchError('carry archive needs clean-base replay')
            self._validate_engine(engine)
            if (getattr(info,'digest',None)!=hashlib.sha256(wire).hexdigest() or
                    info.seq <= self.cutoffs.get(info.origin,0)):
                raise PatchError('unverified/covered cross-generation carry')
            key=(info.origin,info.seq)
            existing=self.journal.has(*key)
            if existing:
                if existing[0]!=info.digest:
                    raise PatchError('conflicting old-generation carry')
                return False
            try:
                # fsync+commit before mutating any tensors. Concurrent calls
                # cannot see an unapplied durable row under this lock.
                if not await self.journal.append_async(info.origin,info.seq,wire):
                    raise PatchError('unexpected duplicate within carry commit lock')
                await asyncio.to_thread(engine._apply_checked,info,prepared)
            except BaseException:
                self._poisoned=True
                raise
            self._replayed_count+=1
            return True

    def close(self):
        self.journal.close()
