"""Durable, bounded lifecycle for candidate/active checkpoint *cache views*.

A checkpoint being superseded is NOT proof that its journal is disposable. The
only unconditional reclamation is a candidate never published as active. A
formerly active base needs a separate, locally verified EpochRuntimeV43 GC
proof, and the successor must already be the durable active trainer.

A finite disk cannot provide infinite disconnected, uncommitted updates. When
proofs are missing this layer refuses further epochs at its configured limit
rather than destroy necessary recovery state.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
from typing import Callable
from dataclasses import dataclass

from .core import PatchError
from .epoch_checkpoint_v50 import StagedCheckpoint
from .epoch_objects_v50 import digest_ok
from .live_epoch_switch_v48 import certified_genesis


@dataclass(frozen=True)
class EpochCacheUsage:
    active: int
    held: int
    candidates: int
    retired: int


class DurableEpochLifecycle:
    """Crash-tolerant local cache-retention policy; no network-side GC commands.

    SQLite is used exclusively on the service event-loop thread. The checkpoint
    cache itself runs on its own worker thread. Only one service may own the
    directory, because the stager's cache already enforces process flock.
    """
    def __init__(self, root: Path, *, stager, pointer: Path, max_held: int = 2,
                 max_candidates: int = 2, metadata_bytes: int = 4*1024**2):
        if (not 0 <= max_held <= 100 or not 1 <= max_candidates <= 16 or
                type(metadata_bytes) is not int or not 128*1024<=metadata_bytes<=64*1024**2):
            raise ValueError('invalid epoch retention budget')
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.stager = stager
        self.pointer = Path(pointer)
        self.max_held = max_held
        self.max_candidates = max_candidates
        self.lock = asyncio.Lock()
        self.db = sqlite3.connect(self.root/'epoch-leases.sqlite3',timeout=20)
        self.db.execute('PRAGMA auto_vacuum=INCREMENTAL')
        self.db.execute('PRAGMA journal_mode=DELETE')
        pagesize=self.db.execute('PRAGMA page_size').fetchone()[0]
        self.db.execute(f'PRAGMA max_page_count={max(32,metadata_bytes//pagesize)}')
        self.db.execute('PRAGMA synchronous=FULL')
        with self.db:
            self.db.execute('''CREATE TABLE IF NOT EXISTS lease(
                manifest TEXT PRIMARY KEY, epoch INTEGER NOT NULL,
                state TEXT NOT NULL CHECK (state IN
                  ('preparing','staged','active','held','abandoning','retiring','retired')),
                genesis TEXT NOT NULL)''')
            self.db.execute('CREATE INDEX IF NOT EXISTS lease_epoch ON lease(epoch)')

    def close(self):
        self.db.close()

    def _rows(self):
        return self.db.execute('SELECT manifest,epoch,state,genesis FROM lease ORDER BY epoch').fetchall()

    def usage(self):
        rows=self._rows()
        return EpochCacheUsage(sum(r[2]=='active' for r in rows),
            sum(r[2]=='held' for r in rows),
            sum(r[2] in ('preparing','staged','abandoning') for r in rows),
            sum(r[2]=='retired' for r in rows))

    def _pointer(self):
        if not self.pointer.exists():return None
        if self.pointer.is_symlink() or self.pointer.stat().st_size > 4096:
            raise PatchError('unsafe active pointer during checkpoint cleanup')
        obj=json.loads(self.pointer.read_text())
        if (not isinstance(obj,dict) or obj.get('format')!='kagami-hot-pointer-v1'
                or type(obj.get('epoch')) is not int or not digest_ok(obj.get('genesis'))
                or not digest_ok(obj.get('checkpoint'))):
            raise PatchError('unverified active checkpoint pointer')
        return obj

    def _checkpoint(self, digest):
        root=self.stager.root/'views'/digest
        return StagedCheckpoint(root/'checkpoint.fmanifest',root,'checkpoint-'+digest)

    def _assert_active_pointer(self, *, epoch, digest, genesis):
        ptr=self._pointer()
        if ptr is None or (ptr['epoch'],ptr['checkpoint'],ptr['genesis'])!=(epoch,digest,genesis):
            raise PatchError('checkpoint cache cannot acknowledge an unpublished epoch')
        return ptr

    async def _retire_candidate(self,digest,epoch):
        # Abandoning never includes an active/published pointer, even in the
        # crash window between pointer rename and directory fsync.
        ptr=self._pointer()
        if ptr and ptr['checkpoint']==digest:
            raise PatchError('refusing to abandon pointer-owned checkpoint')
        row=self.db.execute('SELECT state,epoch FROM lease WHERE manifest=?',(digest,)).fetchone()
        if row is None or row[1]!=epoch or row[0] not in ('preparing','staged','abandoning'):
            raise PatchError('cannot abandon active checkpoint')
        with self.db:
            self.db.execute("UPDATE lease SET state='abandoning' WHERE manifest=?",(digest,))
        await self.stager.retire(self._checkpoint(digest))
        with self.db:
            self.db.execute('DELETE FROM lease WHERE manifest=? AND state=?',(digest,'abandoning'))

    async def sweep_superseded(self, next_epoch:int):
        """Recover a crashed candidate transfer and reclaim failed older ones.

        Must run while no other local preparation/switch can be in flight.
        Never prunes the pointer's candidate on an uncertain fsync outcome.
        """
        async with self.lock:
            for digest,epoch,state,_ in self._rows():
                if (state in ('preparing','staged','abandoning') and epoch<next_epoch):
                    ptr=self._pointer()
                    if ptr and ptr['checkpoint']==digest:
                        continue # pointer publication may predate lease commit
                    await self._retire_candidate(digest,epoch)

    async def stage(self, proposal):
        digest=proposal['manifest'];epoch=proposal['epoch']
        genesis=certified_genesis(proposal)
        if not digest_ok(digest) or type(epoch) is not int:
            raise PatchError('invalid staged epoch')
        async with self.lock:
            ptr=self._pointer()
            if ptr and ptr['epoch']>=epoch:
                raise PatchError('refusing to download behind durable active generation')
            rows=self._rows()
            other_epochs=[r for r in rows if r[1]==epoch and r[0]!=digest and r[2]!='retired']
            if other_epochs:
                raise PatchError('conflicting staged checkpoint at same generation')
            held=sum(r[2] in ('held','active') for r in rows)
            active=sum(r[2]=='active' for r in rows)
            candidates=sum(r[2] in ('preparing','staged','abandoning') and r[0]!=digest for r in rows)
            if held-active>=self.max_held or candidates>=self.max_candidates:
                raise PatchError('checkpoint retention capacity reached: verified GC required')
            with self.db:
                self.db.execute("INSERT OR IGNORE INTO lease VALUES(?,?,'preparing',?)",
                                (digest,epoch,genesis))
            checkpoint=await self.stager.prepare(proposal)
            if (checkpoint.owner!='checkpoint-'+digest or
                    checkpoint!=self._checkpoint(digest)):
                raise PatchError('stager substituted checkpoint view')
            with self.db:
                self.db.execute("UPDATE lease SET state='staged' WHERE manifest=? AND state IN ('preparing','staged')",(digest,))
            return checkpoint

    async def published(self,proposal):
        """Record the REAL pointer after successful live trainer/network handoff.

        Safe to retry after process death when startup has reconstructed the
        exact certified model and optimizer from durable on-disk state.
        """
        digest=proposal['manifest'];epoch=proposal['epoch'];genesis=certified_genesis(proposal)
        async with self.lock:
            self._assert_active_pointer(epoch=epoch,digest=digest,genesis=genesis)
            row=self.db.execute('SELECT epoch,state,genesis FROM lease WHERE manifest=?',(digest,)).fetchone()
            if row is None or row[0]!=epoch or row[2]!=genesis or row[1] not in ('staged','preparing','active'):
                raise PatchError('no locally verified candidate checkpoint for active pointer')
            with self.db:
                self.db.execute("UPDATE lease SET state='held' WHERE state='active' AND manifest!=?",(digest,))
                self.db.execute("UPDATE lease SET state='active' WHERE manifest=?",(digest,))

    async def reconcile_active(self,proposal):
        """Recover lease commit after pointer/SQLite crash boundary.

        A new lifecycle DB can also adopt an older, independently verified
        v50 cache view. Never equate pointer presence with tensor validity.
        """
        digest=proposal['manifest'];epoch=proposal['epoch']
        genesis=certified_genesis(proposal)
        async with self.lock:
            self._assert_active_pointer(epoch=epoch,digest=digest,genesis=genesis)
            row=self.db.execute('SELECT epoch,state,genesis FROM lease WHERE manifest=?',(digest,)).fetchone()
            if row is not None and row[1]=='active' and row[0]==epoch and row[2]==genesis:
                return
            if row is None:
                # Run the complete manifest+semantic+shard checks before
                # reconstructing a lost lease, even if it costs a re-download.
                checkpoint=await self.stager.prepare(proposal)
                if checkpoint!=self._checkpoint(digest):
                    raise PatchError('recovered cache view does not match pointer')
                with self.db:
                    self.db.execute("INSERT INTO lease VALUES(?,?,'staged',?)",(digest,epoch,genesis))
        await self.published(proposal)

    async def gc_from_runtime(self, open_runtime: Callable, current_epoch: int):
        """Run independent retirement proofs in a disk thread, not on WebRTC.

        ``open_runtime`` creates a NEW EpochRuntimeV43 inside that thread,
        with its own fresh ledger/SQLite handles. The runtime verifies the
        checkpoint/carry/journal and exclusive old-writer lock before deleting
        anything. A signed remote message or a caller-supplied bool is never
        a substitute for that proof. The new active pointer is rechecked.
        """
        removed=[]
        async with self.lock:
            ptr=self._pointer()
            if ptr is None or ptr['epoch'] != current_epoch:
                raise PatchError('trainer/pointer mismatch in automated GC')
            for digest,epoch,state,_ in self._rows():
                if state not in ('held','retiring') or epoch>=current_epoch:
                    continue
                def verify():
                    from .epoch_runtime_v43 import EpochRuntimeV43
                    runtime=open_runtime()
                    if not isinstance(runtime,EpochRuntimeV43):
                        raise PatchError('retirement requires EpochRuntimeV43 proof')
                    try:
                        return (runtime.collect(epoch=epoch+1) is True and
                                runtime.status(epoch+1).get('state')=='retired')
                    finally:
                        try:
                            runtime.close()
                        finally:
                            runtime.ledger.close()
                if not await asyncio.to_thread(verify):
                    continue
                # A crash might have durably renamed a new pointer; never
                # unlink a view if it now belongs to the running generation.
                if self._pointer()['checkpoint']==digest:
                    raise PatchError('active checkpoint changed during GC')
                with self.db:
                    self.db.execute("UPDATE lease SET state='retiring' WHERE manifest=?",(digest,))
                await self.stager.retire(self._checkpoint(digest))
                with self.db:
                    self.db.execute("DELETE FROM lease WHERE manifest=? AND state='retiring'",(digest,))
                removed.append(digest)
            if removed:
                self.db.execute('PRAGMA incremental_vacuum(128)')
        return removed
