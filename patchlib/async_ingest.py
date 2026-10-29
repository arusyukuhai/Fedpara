"""Bounded, ordered, non-blocking signed-event ingestion for live WebRTC Gossip.

The RTC callback only enqueues bytes. A single background coroutine performs
CPU-heavy signature/delta checks off the event loop, journals on the SQLite
owner thread, then applies under the existing per-adapter lock. The trainer
NEVER seals its local LoHa block merely because a remote event arrived.

Unbounded speculative tasks and unchecked acknowledgement are intentionally
forbidden. A full queue leaves the event in the network receive queue; if that
also fills, signed inventory/need retransmits it from a durable peer.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from typing import Awaitable, Callable

from .core import PatchError


class AsyncInboundPipeline:
    def __init__(self, engine, on_committed: Callable[..., Awaitable[None]], *,
                 max_queue: int = 32, commit_lock: asyncio.Lock | None = None,
                 max_queue_bytes: int = 128 * 1024 * 1024,
                 carry_archive=None):
        if not isinstance(max_queue, int) or not 1 <= max_queue <= 256:
            raise ValueError('inbound queue capacity must be between 1 and 256')
        if type(max_queue_bytes) is not int or max_queue_bytes < 1024:
            raise ValueError('inbound queue byte budget must be at least 1 KiB')
        self.max_queue_bytes = max_queue_bytes
        self.pending_bytes = 0
        self.max_observed_bytes = 0
        self.engine = engine
        # Optional v46 signed old-generation carry lane.  It must be replayed
        # on the clean certified base BEFORE this pipeline starts.
        self.carry_archive = carry_archive
        if carry_archive is not None and not getattr(carry_archive,'_replayed',False):
            raise PatchError('carry archive must be replayed before training resumes')
        self.on_committed = on_committed
        self.queue: asyncio.Queue[tuple[str,bytes]] = asyncio.Queue(maxsize=max_queue)
        self.commit_lock = commit_lock or asyncio.Lock()
        self.task: asyncio.Task | None = None
        self.received = 0
        self.aggregate_batches = 0
        self.aggregate_events = 0
        self.applied = 0
        self.duplicates = 0
        self.max_observed = 0
        self.check_ms_total = 0.0
        self.commit_ms_total = 0.0
        self.check_count = 0
        self.last_check_ms = 0.0
        self.last_commit_ms = 0.0
        self.last_lock_wait_ms = 0.0
        self.last_journal_ms = 0.0
        self.last_weight_apply_ms = 0.0
        # Digest of the EXACT authenticated, committed bytes -> event metadata.
        # This bounded hot cache avoids repeatedly decoding + rebuilding huge
        # low-rank tensors for gossip retransmissions across multiple links.
        self._recent: OrderedDict[bytes, object] = OrderedDict()
        self._recent_limit = 2048

    def start(self):
        if self.task is not None:
            raise RuntimeError('pipeline already started')
        self.task = asyncio.create_task(self._run(), name='fedpara-async-inbound')
        return self.task

    def check(self):
        if self.task is not None and self.task.done():
            # Never silently continue on a verification, disk or apply failure.
            self.task.result()
            raise RuntimeError('inbound pipeline unexpectedly exited')

    def submit(self, sender: str, wire: bytes) -> bool:
        self.check()
        if not isinstance(wire, bytes):
            raise TypeError('signed inbound payload must be bytes')
        if self.queue.full():
            return False
        # A single legitimate event may exceed the soft budget. Admit it only
        # when there is no queued OR in-flight message; otherwise keep it in
        # the bounded transport receive queue until space is available.
        if self.pending_bytes + len(wire) > self.max_queue_bytes and self.pending_bytes:
            return False
        self.queue.put_nowait((sender,wire))
        self.pending_bytes += len(wire)
        self.max_observed_bytes = max(self.max_observed_bytes, self.pending_bytes)
        self.received += 1
        self.max_observed = max(self.max_observed, self.queue.qsize())
        return True

    async def _run(self):
        while True:
            sender,wire = await self.queue.get()
            try:
                from .aggregate_wire import is_aggregate, unpack_aggregate
                if is_aggregate(wire):
                    # Unpack only after strict compressed/decompressed size checks;
                    # each original signed event is then verified separately.
                    limit = getattr(self.engine, 'max_wire_bytes', 512 * 1024 * 1024)
                    events = await asyncio.to_thread(unpack_aggregate, wire, max_bytes=limit)
                    await self._ingest_aggregate(sender, events)
                    continue
                # A previously authenticated/committed EXACT byte sequence
                # needs no repeated Ed25519+safetensors+dense reconstruction.
                # SHA-256 collision resistance binds this hit to signed bytes.
                digest = hashlib.sha256(wire).digest()
                previous = self._recent.get(digest)
                if previous is not None:
                    self._recent.move_to_end(digest)
                    self.duplicates += 1
                    await self.on_committed(previous,wire,sender,False)
                    continue
                # CPU verification and low-rank delta preparation run WITHOUT
                # the training lock; only the application is serialized.
                started = time.monotonic()
                is_carry = False
                try:
                    info, parts = await asyncio.to_thread(self.engine._check, wire)
                except PatchError as exc:
                    # Do not reinterpret arbitrary invalid new-epoch wires as
                    # carries. Only the precise genesis-validation failure
                    # permits a second, *authenticated* old-genesis check.
                    if (self.carry_archive is None or
                            'genesis' not in str(exc)):
                        raise
                    info, parts = await asyncio.to_thread(
                        self.carry_archive._check, self.engine, wire,
                        allow_covered=True)
                    is_carry = True
                self.last_check_ms = (time.monotonic()-started)*1000
                self.check_ms_total += self.last_check_ms
                self.check_count += 1
                started = time.monotonic()
                if is_carry:
                    if info.seq <= self.carry_archive.cutoffs.get(info.origin,0):
                        self.duplicates += 1
                        await self.on_committed(info,wire,sender,False)
                        continue
                    async with self.commit_lock:
                        # FSYNC signed original wire BEFORE model mutation.
                        # Different source epochs have independent ID spaces.
                        fresh = await self.carry_archive.accept_prechecked(
                            self.engine, wire, info, parts)
                    if fresh:
                        self.applied += 1
                    else:
                        self.duplicates += 1
                    await self.on_committed(info,wire,sender,fresh)
                    continue
                async with self.commit_lock:
                    locked_at = time.monotonic()
                    self.last_lock_wait_ms = (locked_at-started)*1000
                    self.last_journal_ms = 0.0
                    self.last_weight_apply_ms = 0.0
                    ident = (info.origin,info.seq)
                    if ident in self.engine.applied:
                        current = self.engine.journal.has(*ident)
                        if not current or current[0] != info.digest:
                            raise PatchError('conflicting event ID')
                        fresh = False
                        self.duplicates += 1
                    else:
                        # Crash contract: durable append precedes model apply.
                        # A crash after append reconstructs the same event on replay.
                        journal_started = time.monotonic()
                        await self.engine.journal.append_async(info.origin,info.seq,wire)
                        self.last_journal_ms = (time.monotonic()-journal_started)*1000
                        apply_started = time.monotonic()
                        await asyncio.to_thread(self.engine._apply_checked,info,parts)
                        self.last_weight_apply_ms = (time.monotonic()-apply_started)*1000
                        self.applied += 1
                        fresh = True
                self.last_commit_ms = (time.monotonic()-started)*1000
                self.commit_ms_total += self.last_commit_ms
                self._recent[digest] = info
                if len(self._recent) > self._recent_limit:
                    self._recent.popitem(last=False)
                await self.on_committed(info,wire,sender,fresh)
            finally:
                self.pending_bytes -= len(wire)
                self.queue.task_done()

    async def _ingest_aggregate(self, sender: str, wires: list[bytes]):
        """Exactly-once cross-origin FedSum, including partially seen bundles.

        All events authenticate independently (the relay cannot mint source
        updates). Durable per-event append precedes the SINGLE summed model
        application; a crash replays the original signed journal entries.
        """
        if not 2 <= len(wires) <= 8:
            raise PatchError('invalid aggregate cardinality')
        prepared=[]
        cached=[]
        first=time.monotonic()
        for wire in wires:
            digest=hashlib.sha256(wire).digest()
            old=self._recent.get(digest)
            if old is not None:
                cached.append((old,wire))
                continue
            info,parts=await asyncio.to_thread(self.engine._check,wire)
            prepared.append((info,parts,wire,digest))
        self.last_check_ms=(time.monotonic()-first)*1000
        self.check_ms_total+=self.last_check_ms
        self.check_count+=len(prepared)
        if len({(info.origin,info.seq) for info,_,_,_ in prepared}) != len(prepared):
            raise PatchError('conflicting event identities inside aggregate')
        started=time.monotonic()
        fresh=[]
        duplicate=[]
        async with self.commit_lock:
            self.last_lock_wait_ms=(time.monotonic()-started)*1000
            self.last_journal_ms=0.0
            self.last_weight_apply_ms=0.0
            # A single fsync + SQLite transaction for the entire bundle.
            # All prechecks run before durable insertion, and per-ID duplicate
            # filtering preserves exact FedSum semantics on partial overlap.
            journal_start=time.monotonic()
            flags=await self.engine.journal.append_many_async(
                [(info.origin,info.seq,wire) for info,_,wire,_ in prepared]) if prepared else []
            self.last_journal_ms=(time.monotonic()-journal_start)*1000
            for (info,parts,wire,digest), appended in zip(prepared,flags):
                if appended:
                    fresh.append((info,parts,wire))
                else:
                    duplicate.append((info,wire))
                    self.duplicates+=1
            if fresh:
                apply_start=time.monotonic()
                await asyncio.to_thread(self.engine._apply_checked_many,
                                        [(info,parts) for info,parts,_ in fresh])
                self.last_weight_apply_ms=(time.monotonic()-apply_start)*1000
                self.applied+=len(fresh)
        self.last_commit_ms=(time.monotonic()-started)*1000
        self.commit_ms_total+=self.last_commit_ms
        self.aggregate_batches+=1
        self.aggregate_events+=len(wires)
        for info,_,wire,digest in prepared:
            self._recent[digest]=info
            if len(self._recent)>self._recent_limit:
                self._recent.popitem(last=False)
        self.duplicates+=len(cached)
        for info,wire in [*cached,*duplicate]:
            await self.on_committed(info,wire,sender,False)
        for info,_,wire in fresh:
            await self.on_committed(info,wire,sender,True)

    async def drain(self):
        self.check()
        while self.queue.qsize():
            done,pending=await asyncio.wait(
                {asyncio.create_task(self.queue.join()),self.task},
                return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                if task is not self.task: task.cancel()
            self.check()
        # Also wait for the item already dequeued by the worker.
        wait=asyncio.create_task(self.queue.join())
        done,pending=await asyncio.wait({wait,self.task},return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            if task is not self.task:task.cancel()
        self.check()

    async def stop(self, *, drain_timeout: float = 20.):
        if self.task is None:return
        if not self.task.done():
            try:
                await asyncio.wait_for(self.drain(),timeout=drain_timeout)
            except (asyncio.TimeoutError,Exception):
                # Incomplete events remain requested via signed gossip inventory.
                pass
            self.task.cancel()
        await asyncio.gather(self.task,return_exceptions=True)
