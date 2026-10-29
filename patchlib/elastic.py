"""Elastic, additive asynchronous federated updates with durable anti-entropy.

NOT strict synchronized FedAvg. Each update carries its denominator, chosen by
its producer's recent verified peer liveness. Immutable genesis and append-only
signed event IDs allow disconnected replicas to eventually reconcile when they
receive the same set of events. Offline peers are NEVER part of a quorum.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
import asyncio
import json
import math
import os
import errno
import sqlite3
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

from .core import PEER_ID, FactorDelta, PatchError, UniversalAdapter, _canonical_json, decode_patch
from .livefed import HEX64, _delta_block
from .numerics import squared_l2
from .event_archive import append_frame, read_frame, recover_tail, payload_path

MAGIC = b'FPEL1\0'
MAX_ENVELOPE_HEADER = 4096
PROTOCOL = 'fedpara-elastic-v1'


def pack_event(patch: bytes, origin: str, signer=None, *, public_key: str | None = None) -> bytes:
    digest = hashlib.sha256(patch).hexdigest()
    head = {'origin': origin, 'sha256': digest, 'size': len(patch),
            'signature': signer(len(patch), digest) if signer is not None else None}
    if public_key is not None:
        # Self-certifying identity: archive verifier travels with every event.
        from .roomid import identity_from_public_b64
        identity_from_public_b64(public_key, origin)
        if signer is None: raise PatchError('self-certifying event must be signed')
        head['pub'] = public_key
    raw = _canonical_json(head)
    if len(raw) > MAX_ENVELOPE_HEADER:
        raise PatchError('elastic event header too large')
    return MAGIC + struct.pack('!I', len(raw)) + raw + patch


def unpack_event(wire: bytes, trusted: dict, *, unsafe: bool = False):
    if not isinstance(wire, bytes) or not wire.startswith(MAGIC) or len(wire) < len(MAGIC) + 4:
        raise PatchError('invalid elastic envelope')
    n = struct.unpack('!I', wire[len(MAGIC):len(MAGIC)+4])[0]
    if not (0 < n <= MAX_ENVELOPE_HEADER) or len(wire) <= len(MAGIC)+4+n:
        raise PatchError('invalid elastic envelope header length')
    try:
        hdr = json.loads(wire[len(MAGIC)+4:len(MAGIC)+4+n]); patch = wire[len(MAGIC)+4+n:]
    except (ValueError, UnicodeDecodeError) as e:
        raise PatchError('invalid elastic envelope JSON') from e
    if not isinstance(hdr, dict) or set(hdr) not in ({'origin','sha256','size','signature'}, {'origin','sha256','size','signature','pub'}):
        raise PatchError('invalid elastic envelope header')
    origin, digest = hdr['origin'], hdr['sha256']
    if not isinstance(origin,str) or not PEER_ID.fullmatch(origin):
        raise PatchError('invalid origin')
    if type(hdr['size']) is not int or len(patch) != hdr['size'] or hashlib.sha256(patch).hexdigest() != digest:
        raise PatchError('event checksum/size mismatch')
    if not unsafe:
        if 'pub' in hdr:
            from .roomid import identity_from_public_b64
            verifier = identity_from_public_b64(hdr['pub'], origin)
        else:
            verifier = trusted.get(origin)
        if verifier is None:
            raise PatchError('untrusted event origin')
        verifier({'size':len(patch),'sha256':digest,'sig':hdr['signature']})
    meta, weights = decode_patch(patch)
    if meta.get('peer_id') != origin or type(meta.get('seq')) is not int:
        raise PatchError('patch origin / sequence differs from envelope')
    return origin, meta, weights, patch


@dataclass(frozen=True)
class EventInfo:
    origin: str
    seq: int
    digest: str
    denominator: int


class EventJournal:
    """SQLite metadata + fsync-before-commit binary event archive.

    New events are never stored as large BLOBs inside SQLite. Older inline
    rows remain readable, so a v13 journal may be opened without migration.
    Never expires events silently.

    The genesis model is an immutable external .fstate; after a crash replay
    all archived events onto it to restore the exact local observed event set.
    An optional nonzero hard cap refuses writes rather than silently deleting
    data needed by returning peers. ``max_bytes=0`` disables the artificial
    ceiling; physical free disk space remains the true finite constraint.
    """
    def __init__(self, path: Path, genesis: str, *, max_bytes: int = 20*1024**3, sync: str = "FULL"):
        if not HEX64.fullmatch(genesis) or type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('invalid genesis / log cap')
        if sync not in ('FULL', 'NORMAL'):
            raise ValueError('journal sync must be FULL or NORMAL')
        self.sync = sync
        self._write_lock = threading.RLock()
        self._async_slots = None
        self._journal_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fedpara-journal")
        self._worker_connection = None
        self.path = Path(path)
        if Path(str(self.path)+'.retired.json').exists():
            raise PatchError('journal belongs to a retired Kagami generation; load the certified replacement checkpoint')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # SQLite permits multiple writers, but two trainers MUST NOT share an
        # identity and append conflicting gradients to the same peer journal.
        self._lock_fd = None
        # Preserve the explicit immutable-genesis diagnostic even if a live
        # trainer currently holds the exclusive write lock.
        if self.path.exists():
            try:
                lookup=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True,timeout=2)
                try:
                    row=lookup.execute("SELECT value FROM meta WHERE name='genesis'").fetchone()
                finally:
                    lookup.close()
                if row and row[0] != genesis:
                    raise PatchError('local update journal belongs to a DIFFERENT genesis checkpoint')
            except sqlite3.OperationalError:
                pass  # A new DB may not yet have initialized its meta table.
        try:
            import fcntl
            lockfile = Path(str(self.path)+'.train.lock')
            self._lock_fd = os.open(lockfile, os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
            try: fcntl.flock(self._lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError as e:
                if e.errno in (errno.EWOULDBLOCK,errno.EAGAIN,errno.EACCES):
                    raise PatchError(f'peer journal is already used by a live trainer: {self.path}') from e
                raise
        except ImportError:
            raise RuntimeError('safe journal locking requires POSIX flock')
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous='+self.sync)
        # Live workers checkpoint the WAL on a separate connection. SQLite's
        # default 1000-page auto-checkpoint runs synchronously inside INSERT,
        # often adding seconds for multi-megabyte signed factor payloads.
        self.db.execute('PRAGMA wal_autocheckpoint=0')
        self.db.execute('CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.db.execute('''CREATE TABLE IF NOT EXISTS events (
          origin TEXT NOT NULL, seq INTEGER NOT NULL, digest TEXT NOT NULL,
          wire BLOB NOT NULL, wire_len INTEGER NOT NULL,
          PRIMARY KEY(origin, seq))''')
        # Non-destructive migration: retain inline BLOB for v13 rows but index
        # all newly committed payloads by their append-only sidecar position.
        columns = {row[1] for row in self.db.execute('PRAGMA table_info(events)')}
        if 'log_offset' not in columns:
            self.db.execute('ALTER TABLE events ADD COLUMN log_offset INTEGER')
        if 'log_size' not in columns:
            self.db.execute('ALTER TABLE events ADD COLUMN log_size INTEGER')
        self.db.execute('CREATE INDEX IF NOT EXISTS event_origin_seq ON events(origin,seq)')
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta(name,value) VALUES('archive_used_bytes', (SELECT CAST(COALESCE(SUM(wire_len),0) AS TEXT) FROM events))")
        old = self.db.execute("SELECT value FROM meta WHERE name='genesis'").fetchone()
        if old and old[0] != genesis:
            self.db.close()
            self._journal_executor.shutdown(wait=True)
            if self._lock_fd is not None: os.close(self._lock_fd)
            raise PatchError('local update journal belongs to a DIFFERENT genesis checkpoint')
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta(name,value) VALUES('genesis',?)", (genesis,))
        self.genesis = genesis
        self.max_bytes = max_bytes
        # The trainer holds the exclusive .train.lock for the whole lifetime.
        # Recovery is therefore safe even after a power loss mid-append.
        self.log_path = payload_path(self.path)
        self._log_fd = None
        try:
            self._log_fd = os.open(self.log_path, os.O_RDWR | os.O_CREAT |
                                   getattr(os, 'O_NOFOLLOW', 0), 0o600)
            self._committed_end = recover_tail(self.db, self._log_fd)
            self._install_cut_audit()
        except BaseException:
            self.db.close()
            self._journal_executor.shutdown(wait=True)
            if self._log_fd is not None:
                os.close(self._log_fd)
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None
            raise

    def _install_cut_audit(self):
        """Track revisions of already-indexed events for fast certified-cut reuse.

        INSERTs of *new* event IDs cannot affect a previously verified complete
        per-origin prefix. UPDATE/DELETE/REPLACE of indexed rows can, and must
        invalidate the cut witness. Triggers run in the mutating transaction,
        including commits from the async journal worker or another SQLite
        connection. Unlike a Python counter, this survives process crashes.

        The INSERT guard prohibits SQL INSERT OR REPLACE silently overwriting
        a previously verified event when recursive_triggers is disabled.
        This mechanism assumes a trusted local SQLite database/DDL: it is NOT
        an authentication scheme against an attacker with direct DB access.
        """
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta(name,value) VALUES('epoch_index_mutation', '0')")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS epoch_no_event_replace_v55
                BEFORE INSERT ON events
                WHEN EXISTS (SELECT 1 FROM events
                             WHERE origin=NEW.origin AND seq=NEW.seq)
                BEGIN SELECT RAISE(ABORT, 'event ID replacement forbidden'); END""")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS epoch_event_update_v55
                AFTER UPDATE ON events BEGIN
                UPDATE meta SET value=CAST(value AS INTEGER)+1
                    WHERE name='epoch_index_mutation'; END""")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS epoch_event_delete_v55
                AFTER DELETE ON events BEGIN
                UPDATE meta SET value=CAST(value AS INTEGER)+1
                    WHERE name='epoch_index_mutation'; END""")

    def epoch_cut_guard(self):
        """O(1) transactional mutation witness for locally certified prefixes.

        Coupled with a previously fully verified contiguous prefix, inserts
        cannot create a new in-prefix key. A changed schema or mutation count
        forces revalidation. The caller must still hold its writer fence at
        final handoff: this method does not freeze concurrent training.
        """
        schema = self.db.execute('PRAGMA schema_version').fetchone()[0]
        row = self.db.execute("SELECT value FROM meta WHERE name='epoch_index_mutation'").fetchone()
        if row is None or not str(row[0]).isdigit():
            return None
        return schema, int(row[0])

    def close(self):
        # Worker connection is thread-affine, so close it on its own thread.
        self._journal_executor.submit(self._close_worker_connection).result()
        self._journal_executor.shutdown(wait=True)
        os.close(self._log_fd)
        self.db.close()
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    def total_bytes(self):
        return int(self.db.execute("SELECT value FROM meta WHERE name='archive_used_bytes'").fetchone()[0])

    def next_seq(self, origin):
        row = self.db.execute('SELECT MAX(seq) FROM events WHERE origin=?', (origin,)).fetchone()
        return (row[0] or 0) + 1

    def has(self, origin, seq):
        return self.db.execute('SELECT digest FROM events WHERE origin=? AND seq=?',(origin,seq)).fetchone()

    def _append_using(self, db, origin: str, seq: int, wire: bytes) -> bool:
        """Serialize writers and commit small indexed rows only after fsync.

        The on-disk order is: append signed frame, fsync frame, commit SQLite
        metadata. If the process crashes between these stages, recover_tail
        discards the unindexed frame before the next append/replay.
        """
        if type(seq) is not int or not 1 <= seq < 2**63:
            raise PatchError('invalid sequence')
        digest_raw = hashlib.sha256(wire).digest()
        digest = digest_raw.hex()
        with self._write_lock:
            row = db.execute('SELECT digest FROM events WHERE origin=? AND seq=?',
                             (origin, seq)).fetchone()
            if row:
                if row[0] != digest:
                    raise PatchError('equivocation: different signed events with same origin/seq')
                return False
            used = int(db.execute("SELECT value FROM meta WHERE name='archive_used_bytes'").fetchone()[0])
            if self.max_bytes and used + len(wire) > self.max_bytes:
                raise PatchError('durable update log is full; create & distribute a trusted new checkpoint/epoch')
            # A failed insertion in this process may have left an orphan tail.
            # Trim it only after querying durable metadata under writer lock.
            if os.fstat(self._log_fd).st_size != self._committed_end:
                self._committed_end = recover_tail(db, self._log_fd)
            offset, frame_len = append_frame(self._log_fd, wire,
                                             durable=(self.sync == 'FULL'),digest=digest_raw)
            # Existing schema has NOT NULL wire for backwards compatibility;
            # storing empty bytes keeps old tools able to enumerate metadata.
            with db:
                db.execute('''INSERT INTO events
                   (origin,seq,digest,wire,wire_len,log_offset,log_size)
                   VALUES(?,?,?,?,?,?,?)''',
                   (origin, seq, digest, b'', len(wire), offset, frame_len))
                db.execute("UPDATE meta SET value=CAST(value AS INTEGER)+? WHERE name='archive_used_bytes'", (len(wire),))
            self._committed_end = offset + frame_len
            return True

    def _append_many_using(self, db, entries: list[tuple[str, int, bytes]]) -> list[bool]:
        """Commit independently signed updates with ONE archive sync and DB transaction.

        Validate *all* identities and disk limits before writing any frames.
        The archive is synced before committing its small SQLite index; any
        uncommitted tail is trimmed on the next append/restart. Returns one
        freshness flag per event, preserving constituent-event replay.
        """
        if not 1 <= len(entries) <= 8:
            raise ValueError('batch size must be 1..8')
        seen = set()
        prepared = []
        for origin, seq, wire in entries:
            if (not isinstance(origin, str) or not PEER_ID.fullmatch(origin)
                    or type(seq) is not int or not 1 <= seq < 2**63
                    or not isinstance(wire, bytes) or not wire):
                raise PatchError('invalid event batch identity/payload')
            ident = (origin, seq)
            if ident in seen:
                raise PatchError('duplicate event ID in archive batch')
            seen.add(ident)
            digest_raw = hashlib.sha256(wire).digest()
            prepared.append((origin, seq, wire, digest_raw))
        with self._write_lock:
            new = []
            flags = []
            for origin, seq, wire, digest_raw in prepared:
                existing = db.execute(
                    'SELECT digest FROM events WHERE origin=? AND seq=?',
                    (origin, seq)).fetchone()
                if existing is not None:
                    if existing[0] != digest_raw.hex():
                        raise PatchError('equivocation: different signed events with same origin/seq')
                    flags.append(False)
                else:
                    flags.append(True)
                    new.append((origin, seq, wire, digest_raw))
            if not new:
                return flags
            used = int(db.execute(
                "SELECT value FROM meta WHERE name='archive_used_bytes'").fetchone()[0])
            extra = sum(len(wire) for _, _, wire, _ in new)
            if self.max_bytes and used + extra > self.max_bytes:
                raise PatchError('durable update log is full; create a trusted new epoch')
            if os.fstat(self._log_fd).st_size != self._committed_end:
                self._committed_end = recover_tail(db, self._log_fd)
            rows = []
            for origin, seq, wire, digest_raw in new:
                offset, frame_len = append_frame(
                    self._log_fd, wire, durable=False, digest=digest_raw)
                rows.append((origin, seq, digest_raw.hex(), b'', len(wire), offset, frame_len))
            # A single FULL sync covers every signed payload BEFORE its DB index.
            if self.sync == 'FULL':
                os.fsync(self._log_fd)
            with db:
                db.executemany("INSERT INTO events "
                    "(origin,seq,digest,wire,wire_len,log_offset,log_size) "
                    "VALUES(?,?,?,?,?,?,?)", rows)
                db.execute("UPDATE meta SET value=CAST(value AS INTEGER)+? "
                           "WHERE name='archive_used_bytes'", (extra,))
            self._committed_end = rows[-1][-2] + rows[-1][-1]
            return flags

    def append_many(self, entries: list[tuple[str, int, bytes]]) -> list[bool]:
        return self._append_many_using(self.db, entries)

    async def append_many_async(self, entries: list[tuple[str, int, bytes]]) -> list[bool]:
        """Serialize a bounded batch through the existing journal worker."""
        if self._async_slots is None:
            self._async_slots = asyncio.Semaphore(8)
        async with self._async_slots:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._journal_executor, self._append_many_worker, entries)

    def _append_many_worker(self, entries):
        return self._append_many_using(self._worker_db(), entries)

    def append(self, origin: str, seq: int, wire: bytes) -> bool:
        return self._append_using(self.db, origin, seq, wire)

    def _worker_db(self):
        if self._worker_connection is None:
            db = sqlite3.connect(self.path, timeout=30)
            db.execute('PRAGMA synchronous=' + self.sync)
            db.execute('PRAGMA wal_autocheckpoint=0')
            self._worker_connection = db
        return self._worker_connection

    def _close_worker_connection(self):
        if self._worker_connection is not None:
            self._worker_connection.close()
            self._worker_connection = None

    def _append_worker(self, origin: str, seq: int, wire: bytes) -> bool:
        """Persistent thread-owned connection prevents per-event reconnect."""
        return self._append_using(self._worker_db(), origin, seq, wire)

    async def append_async(self, origin: str, seq: int, wire: bytes) -> bool:
        """Bounded one-writer async pipeline; return only after durable commit.

        Queue capacity limits uncommitted bytes; signatures and gossip remain
        on their own coroutines. No event is ACKed before its index commit.
        """
        if self._async_slots is None:
            self._async_slots = asyncio.Semaphore(8)
        async with self._async_slots:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._journal_executor,
                                              self._append_worker, origin, seq, wire)

    def migrate_legacy(self, *, limit: int | None = None) -> int:
        """Optional offline-only migration of v13 inline BLOBs to frames.

        Each row is moved in a crash-safe fsync-then-index transaction.
        Run VACUUM separately if physical DB space must be reclaimed.
        """
        moved = 0
        with self._write_lock:
            while limit is None or moved < limit:
                row = self.db.execute(
                    'SELECT origin,seq,digest,wire,wire_len FROM events '
                    'WHERE log_offset IS NULL ORDER BY rowid LIMIT 1').fetchone()
                if row is None:
                    break
                origin, seq, digest, wire, n = row
                if len(wire) != n or hashlib.sha256(wire).hexdigest() != digest:
                    raise PatchError('legacy inline event is corrupt; refusing migration')
                if os.fstat(self._log_fd).st_size != self._committed_end:
                    self._committed_end = recover_tail(self.db, self._log_fd)
                offset, size = append_frame(self._log_fd, bytes(wire),
                                            durable=(self.sync == 'FULL'),
                                            digest=bytes.fromhex(digest))
                with self.db:
                    self.db.execute(
                        'UPDATE events SET wire=?,log_offset=?,log_size=? '
                        'WHERE origin=? AND seq=? AND log_offset IS NULL',
                        (b'', offset, size, origin, seq))
                self._committed_end = offset + size
                moved += 1
        return moved

    def passive_checkpoint(self):
        """A separate connection checkpoints committed WAL frames in the background.

        PASSIVE never waits for active readers/writers. Any remaining frames
        are still crash-durable in the FULL-synchronous WAL.
        """
        with sqlite3.connect(self.path, timeout=.2) as db:
            db.execute('PRAGMA synchronous='+self.sync)
            return tuple(db.execute('PRAGMA wal_checkpoint(PASSIVE)').fetchone())

    def inventory(self, offset=0, limit=64):
        if not 0 <= offset or not 1 <= limit <= 96:
            raise ValueError('invalid inventory page')
        rows = self.db.execute('SELECT origin,seq,digest FROM events ORDER BY rowid LIMIT ? OFFSET ?',
                               (limit+1,offset)).fetchall()
        return [list(x) for x in rows[:limit]], len(rows) <= limit

    def frontiers(self):
        """Compact per-origin high water marks for gap-aware anti-entropy.

        The full inventory remains the authority for digest/equivocation checks.
        This is only an *advertisement*, never proof that intermediate events
        exist at the advertiser. In particular it cannot authorize an update.
        """
        return [[origin, int(seq)] for origin, seq in self.db.execute(
            'SELECT origin, MAX(seq) FROM events GROUP BY origin ORDER BY origin')]

    def missing_range(self, origin: str, first: int, last: int):
        """Find absent sequences inside a bounded, advertised interval.

        Query the indexed origin/seq key once instead of issuing a SQLite
        lookup for every sequence number. Callers must enforce a small window.
        """
        if (not PEER_ID.fullmatch(origin) or not 1 <= first <= last < 2**63 or
                last - first >= 256):
            raise ValueError('invalid frontier reconciliation range')
        present = {int(row[0]) for row in self.db.execute(
            'SELECT seq FROM events WHERE origin=? AND seq BETWEEN ? AND ?',
            (origin, first, last))}
        return [(origin, seq) for seq in range(first, last + 1) if seq not in present]

    def _wire_from_row(self, row):
        wire, offset, frame_len, wire_len, digest = row
        if offset is None:
            data = bytes(wire)
            if len(data) != wire_len or hashlib.sha256(data).hexdigest() != digest:
                raise PatchError('legacy journal event failed checksum')
            return data
        return read_frame(self._log_fd, offset, frame_len, wire_len, digest)

    def events(self):
        cursor = self.db.execute('SELECT wire,log_offset,log_size,wire_len,digest FROM events ORDER BY rowid')
        for row in cursor:
            yield self._wire_from_row(row)

    def get(self, origin, seq):
        row = self.db.execute('SELECT wire,log_offset,log_size,wire_len,digest FROM events WHERE origin=? AND seq=?', (origin,seq)).fetchone()
        return self._wire_from_row(row) if row else None

    def _get_worker(self, origin, seq):
        row = self._worker_db().execute(
            'SELECT wire,log_offset,log_size,wire_len,digest FROM events '
            'WHERE origin=? AND seq=?', (origin,seq)).fetchone()
        return self._wire_from_row(row) if row else None

    async def get_async(self, origin, seq):
        """Do not hash/load multi-MiB NEED responses on the RTC event loop."""
        loop=asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._journal_executor, self._get_worker, origin, seq)

    def count(self):
        return self.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]


class AppliedJournalIndex:
    """Small hot-event cache with SQLite as the authority for older events.

    Keeping the entire set of (peer, sequence) tuples in memory grows as
    O(total events), even though the indexed journal already stores the IDs.
    During replay, all durable IDs exist BEFORE they are applied, so replay
    explicitly disables disk membership checks until it is finished.
    """
    def __init__(self, journal, capacity=8192):
        if capacity < 1:
            raise ValueError('invalid applied index capacity')
        self.journal=journal
        self.capacity=capacity
        self.hot=OrderedDict()
        self.replaying=False

    def __contains__(self, ident):
        if ident in self.hot:
            self.hot.move_to_end(ident)
            return True
        if self.replaying:
            return False
        return self.journal.has(*ident) is not None

    def add(self, ident):
        self.hot[ident]=None
        self.hot.move_to_end(ident)
        if len(self.hot)>self.capacity:
            self.hot.popitem(last=False)

    def update(self, identities):
        for ident in identities:
            self.add(ident)


class ElasticEngine:
    def __init__(self, adapter: UniversalAdapter, self_id: str, genesis: str,
                 journal: EventJournal, *, signer=None, trusted=None, unsafe=False,
                 max_delta_rms=10., max_denominator=256, block_elements=524288,
                 event_public: str | None = None):
        if not PEER_ID.fullmatch(self_id) or journal.genesis != genesis:
            raise ValueError('invalid local ID or genesis')
        existing_owner = journal.db.execute("SELECT value FROM meta WHERE name='owner'").fetchone()
        if existing_owner is not None and existing_owner[0] != self_id:
            raise PatchError('journal belongs to a different local peer ID')
        with journal.db:
            journal.db.execute("INSERT OR IGNORE INTO meta(name,value) VALUES('owner',?)", (self_id,))
        self.adapter, self.self_id, self.genesis = adapter, self_id, genesis
        self.journal, self.signer, self.trusted, self.unsafe = journal, signer, trusted or {}, unsafe
        self.event_public=event_public
        self.max_delta_rms, self.max_denominator, self.block_elements = max_delta_rms,max_denominator,block_elements
        if not math.isfinite(max_delta_rms) or max_delta_rms <= 0 or max_denominator < 1:
            raise ValueError('invalid delta or denominator limits')
        self.applied = AppliedJournalIndex(journal)

    def _check(self, wire):
        origin, meta, weights, patch = unpack_event(wire, self.trusted,unsafe=self.unsafe)
        if (meta.get('elastic_protocol') != PROTOCOL or meta.get('genesis') != self.genesis or
            type(meta.get('denominator')) is not int or not 1 <= meta['denominator'] <= self.max_denominator):
            raise PatchError('invalid elastic protocol/genesis/denominator')
        if meta['seq'] < 1:
            raise PatchError('elastic event sequence must be >=1')
        parts = self.adapter._validate(meta,weights)
        # Factor multiplications can overflow although the factors are finite.
        with self.adapter.lock, torch.no_grad():
            for gi,g in enumerate(self.adapter.groups):
                spec = g.overlay.local
                d = [t.to(spec.parts[0].device,dtype=torch.float32) for t in parts[gi]]
                n0 = g.shape[0] if g.shape else 1
                row_elements = math.prod(g.shape[1:]) if len(g.shape)>1 else 1
                rows = max(1,self.block_elements//row_elements)
                power=0.0
                for lo in range(0,n0,rows):
                    delta = _delta_block(spec,d,lo,min(lo+rows,n0))
                    if not torch.isfinite(delta).all().item():
                        raise PatchError('nonfinite reconstructed elastic update')
                    power += squared_l2(delta)
                rms = math.sqrt(power/math.prod(g.shape))
                if not math.isfinite(rms) or rms > self.max_delta_rms:
                    raise PatchError('excessive elastic update RMS')
        return EventInfo(origin,meta['seq'],hashlib.sha256(wire).hexdigest(),meta['denominator']),parts

    def _apply_checked(self, info, parts):
        self._apply_checked_many([(info, parts)])

    def _apply_checked_many(self, updates):
        """Accumulate validated signed contributions from *different origins*.

        All constituent wires must be authenticated and durably journaled by
        the caller before touching any weights. One model-lock acquisition,
        one finite-value check and one adapter refresh per tensor block make
        a multi-origin FedSum cheaper than applying each event separately.
        Partial overlap is filtered by the journal BEFORE this method.
        """
        if not updates:
            return
        identities = [(info.origin, info.seq) for info, _ in updates]
        if len(set(identities)) != len(identities):
            raise PatchError('duplicate event in aggregated update')
        with self.adapter.lock, torch.no_grad():
            for gi, g in enumerate(self.adapter.groups):
                path, _, attr = g.aliases[0].rpartition('.')
                dest_param = self.adapter.model.get_submodule(path).parametrizations[attr].original
                spec = g.overlay.local
                device = dest_param.device
                prepared = [(info, [p.to(device, dtype=torch.float32) for p in parts[gi]])
                            for info, parts in updates]
                n0 = g.shape[0] if g.shape else 1
                rows = max(1, self.block_elements //
                           (math.prod(g.shape[1:]) if len(g.shape)>1 else 1))
                for lo in range(0, n0, rows):
                    hi = min(lo+rows, n0)
                    dest = dest_param[lo:hi] if g.shape else dest_param
                    summed = None
                    for info, factors in prepared:
                        update = _delta_block(spec, factors, lo, hi) / info.denominator
                        summed = update if summed is None else summed + update
                    val = dest.float() + summed
                    if not torch.isfinite(val).all().item():
                        raise PatchError('nonfinite summed model weights')
                    dest.copy_(val.to(dest.dtype))
                from .relora import _refresh_spec
                _refresh_spec(spec)
            self.applied.update(identities)

    def ingest(self, wire: bytes) -> bool:
        info, parts = self._check(wire)
        if (info.origin,info.seq) in self.applied:
            # _check also verifies signature on retransmissions.
            existing = self.journal.has(info.origin,info.seq)
            if not existing or existing[0] != info.digest:
                raise PatchError('conflicting event ID')
            return False
        self.journal.append(info.origin,info.seq,wire)
        self._apply_checked(info,parts)
        return True

    def seal_local(self, denominator: int, *, dtype='float16', max_wire_bytes: int | None = None) -> bytes:
        if type(denominator) is not int or not 1 <= denominator <= self.max_denominator:
            raise PatchError('invalid live peer count')
        seq = self.journal.next_seq(self.self_id)
        patch = self.adapter.export(self.self_id,seq=seq,dtype=dtype,
                                    extra_meta={'elastic_protocol':PROTOCOL, 'genesis':self.genesis,
                                                'denominator':denominator})
        wire = pack_event(patch,self.self_id,self.signer,public_key=self.event_public)
        if max_wire_bytes is not None and len(wire)>max_wire_bytes:
            raise PatchError('outgoing elastic event exceeds --max-patch-mb')
        if not self.ingest(wire):
            raise PatchError('new local update had duplicate sequence')
        return wire

    async def ingest_responsive(self, wire: bytes) -> bool:
        """Ingest a signed event without starving aiortc's asyncio loop.

        Ed25519 checks and dense tensor reconstruction can take seconds on a
        contended laptop. SQLite access stays on the trainer/event-loop thread
        (default sqlite3 check_same_thread=True); only tensor operations move
        to a worker. Awaiting each stage serializes all model mutations.
        """
        import time
        started = time.monotonic()
        info, parts = await asyncio.to_thread(self._check, wire)
        check_ms = (time.monotonic() - started)*1000
        ident = (info.origin, info.seq)
        if ident in self.applied:
            existing = self.journal.has(*ident)
            if not existing or existing[0] != info.digest:
                raise PatchError('conflicting event ID')
            return False
        started = time.monotonic()
        await self.journal.append_async(info.origin, info.seq, wire)
        journal_ms = (time.monotonic() - started)*1000
        started = time.monotonic()
        await asyncio.to_thread(self._apply_checked, info, parts)
        apply_ms = (time.monotonic() - started)*1000
        self.last_ingest_stages_ms = (check_ms, journal_ms, apply_ms)
        return True

    async def seal_local_responsive(self, denominator: int, *, dtype='float16',
                                    max_wire_bytes: int | None = None) -> bytes:
        """Export/validate/apply model deltas off-loop; SQLite stays on-loop."""
        if type(denominator) is not int or not 1 <= denominator <= self.max_denominator:
            raise PatchError('invalid live peer count')
        seq = self.journal.next_seq(self.self_id)
        patch = await asyncio.to_thread(
            self.adapter.export, self.self_id, seq=seq, dtype=dtype,
            extra_meta={'elastic_protocol': PROTOCOL, 'genesis': self.genesis,
                        'denominator': denominator})
        wire = await asyncio.to_thread(pack_event, patch, self.self_id,
                                       self.signer, public_key=self.event_public)
        if max_wire_bytes is not None and len(wire) > max_wire_bytes:
            raise PatchError('outgoing elastic event exceeds --max-patch-mb')
        if not await self.ingest_responsive(wire):
            raise PatchError('new local update had duplicate sequence')
        return wire

    def replay(self):
        """Restore applied base from immutable --state-base plus durable update log."""
        import time
        self.applied.replaying=True
        seen=0
        last_report=time.monotonic()
        try:
            for wire in self.journal.events():
                info,parts = self._check(wire)
                if (info.origin,info.seq) in self.applied:
                    raise PatchError('duplicate journal event')
                self._apply_checked(info,parts)
                seen += 1
                # A large cold restart is expensive: show truthful progress in
                # the GUI log rather than appearing permanently unresponsive.
                now=time.monotonic()
                if now-last_report >= 5.0:
                    print(f'KAGAMI RESUME replaying signed history: {seen} events applied',flush=True)
                    last_report=now
            if seen:
                print(f'KAGAMI RESUME restored {seen} durable signed updates',flush=True)
        finally:
            self.applied.replaying=False

    def reset_factors(self):
        with self.adapter.lock,torch.no_grad():
            for g in self.adapter.groups:
                from .relora import _refresh_spec
                _refresh_spec(g.overlay.local)
