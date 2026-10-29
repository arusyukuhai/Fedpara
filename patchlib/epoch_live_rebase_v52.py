"""v52: bounded, exact and crash-replayable warm carry for a certified epoch.

This module does NOT decide that a checkpoint represents a journal cut. That
must be verified independently by the quorum validators. It verifies the
complete locally available signed-index cut, and incrementally transfers every
uncovered old-generation event to a durable carry archive while old training
and WebRTC traffic continue. Only ``finish_under_ingress_barrier`` may attest a
complete local suffix; the caller must prevent ALL further old-genesis inbound
commits/publications until the new data plane is installed.

No O(events) Python lists, no SQL OFFSET, no forced worldwide stop-the-world,
no time-based history deletion. Missing cut events fail closed.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from dataclasses import dataclass

from .core import PatchError, _canonical_json


@dataclass(frozen=True)
class CarryProgress:
    scanned: int
    carried: int
    bytes_carried: int
    cursor: int
    caught_up: bool


def verify_certified_cut(journal, cutoffs: dict[str, int], digest: str) -> tuple[int, int]:
    """O(number of origins) RAM, O(events) sequential indexed read.

    The canonical digest covers EVERY included (origin,sequence,event-digest),
    not just the greatest sequence seen. Missing/extra events in the cut and
    accidental equivocation are detectable. Rows are scanned by the composite
    index, with no unbounded OFFSET pagination.
    """
    if (not isinstance(cutoffs, dict) or len(cutoffs) > 200_000 or
            not isinstance(digest, str) or len(digest) != 64 or
            any(c not in '0123456789abcdef' for c in digest) or
            any(not isinstance(p, str) or not 1 <= len(p) <= 128 or
                type(seq) is not int or seq < 0 or seq >= 2**63
                for p, seq in cutoffs.items())):
        raise PatchError('invalid certified cut metadata')
    sha = hashlib.sha256()
    count = 0
    previous = None
    last = {}
    max_rowid = 0
    for rowid, origin, seq, stored_digest in journal.db.execute(
            'SELECT rowid,origin,seq,digest FROM events ORDER BY origin,seq'):
        if (not isinstance(origin, str) or type(seq) is not int or
                not isinstance(stored_digest, str) or len(stored_digest) != 64 or
                any(c not in '0123456789abcdef' for c in stored_digest) or
                (previous is not None and (origin,seq) <= previous)):
            raise PatchError('corrupt or unordered source journal')
        previous = origin,seq
        max_rowid = max(max_rowid,rowid)
        if seq <= cutoffs.get(origin,0):
            # A per-origin high-water cut must be a COMPLETE prefix; a v1
            # frontier cannot represent missing middle sequences correctly.
            if seq != last.get(origin,0)+1:
                raise PatchError('certified cut has a missing signed sequence')
            sha.update(_canonical_json([origin,seq,stored_digest])+b'\n')
            count += 1
            last[origin] = seq
    if sha.hexdigest() != digest or any(last.get(origin,0) < seq for origin,seq in cutoffs.items()):
        raise PatchError('not all certified-cut events are available locally')
    return count,max_rowid


class StreamingCertifiedRebase:
    """Warm-stage signed suffix with incremental disk usage and zero DB offsets.

    ``archive`` must be a CrossEpochCarryArchive already replayed onto a
    NEW, clean certified base. It commits old signed bytes durably BEFORE
    applying their effects. A crash requires a clean-base reload followed by
    the archive's deterministic replay; it is unsafe to reuse partially
    modified shadow model in-process on error.
    """
    def __init__(self, source, archive, engine, proposal: dict, *, batch_limit=16,
                 max_batch_bytes=16 * 1024**2):
        if (not 1 <= batch_limit <= 128 or not 64*1024 <= max_batch_bytes <= 256*1024**2):
            raise ValueError('invalid carry memory budget')
        if (not isinstance(proposal,dict) or not isinstance(proposal.get('cutoffs'),dict)
                or not isinstance(proposal.get('cut'),str)):
            raise PatchError('missing certified journal cut')
        if (archive.source_genesis != source.genesis or
                archive.target_genesis != engine.genesis or
                archive.certified_manifest != proposal.get('manifest') or
                archive.epoch != proposal.get('epoch') or
                archive.cutoffs != proposal['cutoffs']):
            raise PatchError('carry archive does not match certified boundary')
        if not getattr(archive,'_replayed',False):
            raise PatchError('replay carry archive on clean base before staging')
        self.source = source
        self.archive = archive
        self.engine = engine
        self.proposal = proposal
        self.batch_limit=batch_limit
        self.max_batch_bytes=max_batch_bytes
        self.cursor=0
        self.scanned=0
        self.carried=0
        self.bytes_carried=0
        self._validated=False
        self._poisoned=False
        self._verified_cut=None
        self._cut_guard=None
        self._cached_inventory=None

    def validate_cut(self):
        """Fully validate once, then reuse the *same certified digest* only
        while the underlying indexed rows remain unmodified.

        New rows appended after a successfully checked, gap-free per-origin
        frontier are necessarily outside that frontier (PK + no REPLACE).
        Ordinary post-cut gossip therefore does not cause another O(N) scan.
        In a legacy/custom journal without a durable mutation guard, always
        perform the original complete scan instead.
        """
        if self._poisoned:raise PatchError('poisoned shadow, reload clean checkpoint')
        guard_fn=getattr(self.source,'epoch_cut_guard',None)
        before=guard_fn() if callable(guard_fn) else None
        if before is not None and self._validated and before==self._cut_guard:
            return self._verified_cut
        previously_validated=self._validated
        try:
            result=verify_certified_cut(self.source,self.proposal['cutoffs'],self.proposal['cut'])
            after=guard_fn() if callable(guard_fn) else None
            if before is not None and before!=after:
                # The SQL cursor may have read a stale snapshot.
                raise PatchError('certified-cut source mutated during validation')
        except BaseException:
            self._validated=False
            self._verified_cut=None
            self._cut_guard=None
            if previously_validated:
                # A proven prefix became unverifiable. Any already applied
                # shadow deltas might have been based on a corrupted cut.
                self._poisoned=True
            raise
        self._verified_cut=result
        self._cut_guard=after
        self._validated=True
        return result

    async def tick(self) -> CarryProgress:
        """Do bounded work; the trainer can run steps between ticks.

        Each new event row has a monotonically increasing rowid, even when an
        offline peer supplies a much older (origin,seq). A keyset cursor thus
        cannot accidentally skip a newly discovered post-cut event.
        """
        if self._poisoned:
            raise PatchError('poisoned shadow, reload clean certified checkpoint')
        if not self._validated:
            raise PatchError('rebase requires a validated cut and clean shadow')
        guard_fn=getattr(self.source,'epoch_cut_guard',None)
        if (self._cut_guard is not None and callable(guard_fn)
                and guard_fn()!=self._cut_guard):
            self._poisoned=True
            raise PatchError('certified-cut index changed during warm rebase')
        # Query only indexed event metadata, keeping huge payloads outside SQL.
        # A row may itself exceed the byte budget; refuse it before allocation.
        rows = self.source.db.execute(
            'SELECT rowid,origin,seq,wire_len,digest FROM events WHERE rowid>? '
            'ORDER BY rowid LIMIT ?', (self.cursor,self.batch_limit)).fetchall()
        spent=0
        try:
            for rowid,origin,seq,wire_len,wire_digest in rows:
                if (type(wire_len) is not int or wire_len <= 0 or
                        wire_len > self.max_batch_bytes):
                    raise PatchError('carry event exceeds configured byte cap')
                if spent and spent + wire_len > self.max_batch_bytes:
                    break
                if seq > self.proposal['cutoffs'].get(origin,0):
                    # An interrupted attempt can resume after reconstructing
                    # a clean shadow and replaying its durable carry archive.
                    # Avoid decoding/allocating tens of millions of already
                    # authenticated factor deltas on a long-lived peer.
                    existing=self.archive.journal.has(origin,seq)
                    if existing is not None:
                        if existing[0]!=wire_digest:
                            raise PatchError('conflicting source event in durable carry')
                        spent+=wire_len
                        self.cursor=rowid
                        self.scanned+=1
                        continue
                    # Signed LoHA records may be several MiB. Do not block
                    # the trainer watcher/event loop on sidecar disk reads.
                    get_async=getattr(self.source,'get_async',None)
                    if callable(get_async):
                        wire=await get_async(origin,seq)
                    else:
                        wire=await asyncio.to_thread(self.source.get,origin,seq)
                    if not isinstance(wire,bytes) or len(wire)!=wire_len:
                        raise PatchError('source carry body missing')
                    # Signature validation and expensive factor decoding can
                    # safely run off the WebRTC event loop.
                    info,parts=await asyncio.to_thread(self.archive._check,self.engine,wire)
                    if (info.origin,info.seq)!=(origin,seq):
                        raise PatchError('source index differs from signed update')
                    if await self.archive.accept_prechecked(self.engine,wire,info,parts):
                        self.carried+=1
                        self.bytes_carried+=wire_len
                spent+=wire_len
                self.cursor=rowid
                self.scanned+=1
        except BaseException:
            # Applying a delta may fail after an earlier destination tensor
            # changed. Never admit training on the shadow after such an error.
            self._poisoned=True
            raise
        consumed = self.cursor == rows[-1][0] if rows else True
        return CarryProgress(self.scanned,self.carried,self.bytes_carried,self.cursor,
                             consumed and len(rows)<self.batch_limit)

    def _inventory_stamp(self):
        """Detect all normal SQLite writes, including archive worker commits.

        total_changes observes the current connection and data_version sees
        other connections. This avoids a write-amplifying per-INSERT counter.
        A changed stamp causes a new full digest, never a guessed digest.
        """
        db=self.archive.journal.db
        return (db.total_changes,
                db.execute('PRAGMA data_version').fetchone()[0],
                db.execute('PRAGMA schema_version').fetchone()[0])

    def archive_inventory_digest(self):
        """O(1) RAM, exact canonical archive-index commitment, cacheable only
        if no SQLite writer has changed the index since that verified scan.
        """
        before=self._inventory_stamp()
        cache=self._cached_inventory
        if cache is not None and cache[2]==before:
            return cache[:2]
        sha=hashlib.sha256()
        count=0
        for origin,seq,digest in self.archive.journal.db.execute(
                'SELECT origin,seq,digest FROM events ORDER BY origin,seq'):
            sha.update(_canonical_json([origin,seq,digest])+b'\n')
            count+=1
        after=self._inventory_stamp()
        if before!=after:
            self._cached_inventory=None
            raise PatchError('carry archive changed during inventory scan')
        result=(count,sha.hexdigest(),after)
        self._cached_inventory=result
        return result[:2]

    def _receipt_path(self):
        return Path(str(self.archive.journal.path)+'.handoff.json')

    def verify_handoff_receipt(self):
        """Proof of locally archived bytes, NOT authorization to GC old history."""
        path=self._receipt_path()
        if not path.is_file() or path.is_symlink() or path.stat().st_size>4096:
            raise PatchError('missing or unsafe local carry handoff receipt')
        data=json.loads(path.read_bytes())
        if (not isinstance(data,dict) or data.get('format')!='kagami-local-carry-attestation-v1'
                or data.get('source_genesis')!=self.source.genesis
                or data.get('target_genesis')!=self.engine.genesis
                or data.get('certified_manifest')!=self.proposal['manifest']
                or data.get('epoch')!=self.proposal['epoch']
                or data.get('archive_genesis')!=self.archive.genesis):
            raise PatchError('receipt belongs to another certified epoch')
        count,digest=self.archive_inventory_digest()
        if (type(data.get('total_archived')) is not int or
                data['total_archived']!=count or data.get('archive_digest')!=digest):
            raise PatchError('carry archive differs from durable local handoff receipt')
        return data

    def _durable_receipt(self,record):
        """Atomic/fsynced local receipt after source has been frozen + drained."""
        path=self._receipt_path()
        if path.is_symlink():raise PatchError('unsafe handoff receipt symlink')
        raw=_canonical_json(record)
        if len(raw)>4096:raise PatchError('handoff receipt too large')
        tmp=path.with_name(path.name+'.tmp-'+os.urandom(8).hex())
        fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
        try:
            with os.fdopen(fd,'wb') as writer:
                writer.write(raw);writer.flush();os.fsync(writer.fileno())
            os.replace(tmp,path)
            directory=os.open(path.parent,os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)
        finally:
            tmp.unlink(missing_ok=True)

    async def finish_under_ingress_barrier(self):
        """Final small carry after caller has drained and FROZEN old writers.

        A producer inserting later into ``source`` INVALIDATES this attestation.
        The caller must hold its old generation's ingest + local seal barrier
        until the durable epoch pointer and replacement transport are live.
        This method cannot enforce that external ownership requirement.
        """
        if not self._validated:raise PatchError('unvalidated cut')
        while True:
            before=self.cursor
            progress=await self.tick()
            if progress.caught_up:
                break
            if before==self.cursor:
                raise PatchError('no progress processing certified suffix')
        # Revalidate that no claimed cut row was lost or modified by a writer
        # racing snapshot staging. This is deliberately an epoch-only scan.
        self.validate_cut()
        final_rowid=self.source.db.execute('SELECT COALESCE(MAX(rowid),0) FROM events').fetchone()[0]
        if final_rowid > self.cursor:
            raise PatchError('old journal grew during final carry; ingress is not frozen')
        count,digest=self.archive_inventory_digest()
        receipt={'format':'kagami-local-carry-attestation-v1',
                'source_genesis':self.source.genesis,
                'target_genesis':self.engine.genesis,
                'certified_manifest':self.proposal['manifest'],
                'epoch':self.proposal['epoch'],'source_rowid':final_rowid,
                'archived_events_this_run':self.carried,
                'total_archived':count,'archive_digest':digest,
                'archive_genesis':self.archive.genesis}
        await asyncio.to_thread(self._durable_receipt,receipt)
        return self.verify_handoff_receipt()
