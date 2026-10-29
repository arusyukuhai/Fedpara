"""Durable bounded content-addressed epoch transfer cache.

Single process ownership, SQLite owner pins, sequential resumable blocks,
SHA-256 before publication, byte reservations before download. This store is
for checkpoints only: it MUST NOT collect event journals or carry archives.
It is independent of torch so crash/quota/transfer tests need no accelerator.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import os
import sqlite3
import time
from pathlib import Path


class CacheError(ValueError):
    pass


class CapacityError(CacheError):
    pass


def digest_ok(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class EpochObjectStore:
    """Owns a private directory; one writer process enforced by flock.

    All methods are synchronous and serialized by the caller. Open and use in
    the same thread. Payload reservations are limited by quota; SQLite has a
    separate hard page limit and incremental vacuum. Owner pins never expire
    on a wall clock: an offline trainer's base must not be evicted by a TTL.
    """
    def __init__(self, root, *, quota_bytes=8*1024**3, max_objects=100_000,
                 max_object_bytes=64*1024**2, metadata_bytes=64*1024**2):
        import fcntl
        for n in (quota_bytes, max_objects, max_object_bytes, metadata_bytes):
            if type(n) is not int or n <= 0:
                raise ValueError('positive integer cache limits required')
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise CacheError('cache root cannot be a symlink')
        self.lock = (self.root/'owner.lock').open('a+b')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise CacheError('epoch cache already has an owner')
        self.quota = quota_bytes
        self.max_objects = max_objects
        self.max_object_bytes = max_object_bytes
        self.db = None
        try:
            for name in ('objects', 'partial'):
                p = self.root/name
                p.mkdir(exist_ok=True)
                if p.is_symlink():
                    raise CacheError('unsafe cache directory')
            self.db = sqlite3.connect(self.root/'cache.sqlite3', timeout=5)
            self.db.execute('PRAGMA auto_vacuum=INCREMENTAL')
            # DELETE keeps metadata disk bounded without accumulating a WAL.
            self.db.execute('PRAGMA journal_mode=DELETE')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('PRAGMA foreign_keys=ON')
            page_size = self.db.execute('PRAGMA page_size').fetchone()[0]
            self.db.execute(f'PRAGMA max_page_count={max(32, metadata_bytes//page_size)}')
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS objects(
                    digest TEXT PRIMARY KEY, size INTEGER NOT NULL,
                    received INTEGER NOT NULL DEFAULT 0,
                    complete INTEGER NOT NULL DEFAULT 0,
                    accessed REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS owners(
                    owner TEXT NOT NULL, digest TEXT NOT NULL REFERENCES objects(digest),
                    PRIMARY KEY(owner,digest));
                CREATE INDEX IF NOT EXISTS owner_digest ON owners(digest);
                CREATE INDEX IF NOT EXISTS lru ON objects(accessed);
            ''')
            self._recover()
            if self.reserved_bytes > self.quota or self.count > self.max_objects:
                self.collect()
                if self.reserved_bytes > self.quota or self.count > self.max_objects:
                    raise CapacityError('new limit smaller than pinned state')
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def reserved_bytes(self):
        return self.db.execute('SELECT COALESCE(SUM(size),0) FROM objects').fetchone()[0]

    @property
    def count(self):
        return self.db.execute('SELECT COUNT(*) FROM objects').fetchone()[0]

    def _path(self, digest, *, partial=False):
        if not digest_ok(digest):
            raise CacheError('invalid content digest')
        return self.root/('partial' if partial else 'objects')/(digest+'.blob')

    @staticmethod
    def _hash_file(path):
        with path.open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()

    def _recover(self):
        # Files unknown to the committed index can only be orphaned cache data.
        # Never traverse or collect files outside these two owned directories.
        for dirname in ('objects', 'partial'):
            for p in (self.root/dirname).iterdir():
                digest = p.name.removesuffix('.blob')
                if (not digest_ok(digest) or p.is_symlink() or not p.is_file()):
                    raise CacheError('unexpected file in private epoch cache')
                if not self.db.execute('SELECT 1 FROM objects WHERE digest=?', (digest,)).fetchone():
                    p.unlink()
        for digest, size, received, complete in self.db.execute(
                'SELECT digest,size,received,complete FROM objects').fetchall():
            final = self._path(digest)
            part = self._path(digest, partial=True)
            # Crash after rename but before SQL completion: finish publication.
            if final.exists():
                if final.stat().st_size == size and self._hash_file(final) == digest:
                    with self.db:
                        self.db.execute('UPDATE objects SET received=size,complete=1 WHERE digest=?', (digest,))
                    part.unlink(missing_ok=True)
                    continue
                final.unlink()
                received = 0
            if not part.exists() or part.stat().st_size < received or complete:
                received = 0
            if part.exists():
                with part.open('r+b') as f:
                    f.truncate(received)
                    f.flush()
                    os.fsync(f.fileno())
            # Crash after the last block's SQL commit but before verification
            # or rename must not strand a permanently "full but incomplete"
            # download with no remaining range to request.
            if received == size and part.exists():
                if self._hash_file(part) == digest:
                    os.replace(part, final)
                    fsync_dir(final.parent)
                    fsync_dir(part.parent)
                    with self.db:
                        self.db.execute('UPDATE objects SET received=size,complete=1 WHERE digest=?', (digest,))
                    continue
                part.unlink()
                fsync_dir(part.parent)
                received = 0
            with self.db:
                self.db.execute('UPDATE objects SET received=?,complete=0 WHERE digest=?', (received, digest))

    def pin(self, owner, specs):
        """Atomically reserve/pin a whole checkpoint; no partial pin on failure.

        Caller supplies a verified manifest's UNIQUE {digest:size} mapping.
        Calling again adds pins idempotently, it never unpins the active base.
        """
        if not isinstance(owner, str) or not 0 < len(owner) <= 128:
            raise CacheError('invalid owner')
        if not isinstance(specs, dict) or not specs or len(specs) > self.max_objects:
            raise CacheError('invalid checkpoint object count')
        if any(not digest_ok(d) or type(s) is not int or not 0 < s <= self.max_object_bytes
               for d, s in specs.items()):
            raise CacheError('invalid object specification')
        missing = {}
        for d, size in specs.items():
            row = self.db.execute('SELECT size FROM objects WHERE digest=?', (d,)).fetchone()
            if row is not None and row[0] != size:
                raise CacheError('same digest with conflicting size')
            if row is None:
                missing[d] = size
        need = sum(missing.values())
        if need > self.quota:
            raise CapacityError('checkpoint exceeds cache quota')
        # Protect requested objects from eviction before they receive new pins.
        self.collect(required_bytes=need, required_objects=len(missing), protect=set(specs))
        if self.reserved_bytes+need > self.quota or self.count+len(missing) > self.max_objects:
            raise CapacityError('active/pending checkpoint pins exhaust quota')
        with self.db:
            for digest, size in missing.items():
                self.db.execute('INSERT INTO objects VALUES(?,?,0,0,?)', (digest, size, time.time()))
            self.db.executemany('INSERT OR IGNORE INTO owners VALUES(?,?)', ((owner,d) for d in specs))

    def release(self, owner):
        """Local lifecycle only: call AFTER durable adoption/retirement proof.

        Never expose this as a remote message handler or expire by peer age.
        """
        with self.db:
            self.db.execute('DELETE FROM owners WHERE owner=?', (owner,))

    def collect(self, *, required_bytes=0, required_objects=0, protect=frozenset(), all_unpinned=False):
        freed = 0
        used, count = self.reserved_bytes, self.count
        for digest, size in self.db.execute('''SELECT digest,size FROM objects o
            WHERE NOT EXISTS(SELECT 1 FROM owners p WHERE p.digest=o.digest)
            ORDER BY accessed''').fetchall():
            if not all_unpinned and used+required_bytes <= self.quota and count+required_objects <= self.max_objects:
                break
            if digest in protect:
                continue
            # Files first: a crash leaves a missing unpinned entry, repaired at
            # startup. SQL first could leave unaccounted payloads at full quota.
            self._path(digest).unlink(missing_ok=True)
            self._path(digest, partial=True).unlink(missing_ok=True)
            with self.db:
                self.db.execute('DELETE FROM objects WHERE digest=?', (digest,))
            used -= size
            count -= 1
            freed += size
        if freed:
            fsync_dir(self.root/'objects')
            fsync_dir(self.root/'partial')
            self.db.execute('PRAGMA incremental_vacuum(128)')
        return freed

    def status(self, digest):
        row = self.db.execute('SELECT size,received,complete FROM objects WHERE digest=?', (digest,)).fetchone()
        if row is None:
            raise CacheError('unreserved object')
        return row[0], row[1], bool(row[2])

    def append(self, digest, offset, data):
        """Persist one requested block. Replay offsets are rejected, never added twice."""
        size, received, complete = self.status(digest)
        if complete:
            raise CacheError('object already complete')
        if (type(offset) is not int or offset != received or not isinstance(data, bytes)
                or not 0 < len(data) <= 256*1024 or offset+len(data) > size):
            raise CacheError('unsolicited/oversized/noncontiguous object block')
        path = self._path(digest, partial=True)
        if path.is_symlink():
            raise CacheError('unsafe partial path')
        with path.open('r+b' if path.exists() else 'w+b') as f:
            f.seek(offset)
            f.write(data)
            f.truncate()
            f.flush()
            os.fsync(f.fileno())
        fsync_dir(path.parent)
        end = offset+len(data)
        with self.db:
            self.db.execute('UPDATE objects SET received=?,accessed=? WHERE digest=?', (end,time.time(),digest))
        if end == size:
            if self._hash_file(path) != digest:
                # Discard corrupt partial bytes; leave the pin/reservation.
                path.unlink()
                fsync_dir(path.parent)
                with self.db:
                    self.db.execute('UPDATE objects SET received=0 WHERE digest=?', (digest,))
                raise CacheError('downloaded checkpoint object hash mismatch')
            target = self._path(digest)
            os.replace(path, target)
            fsync_dir(target.parent)
            fsync_dir(path.parent)
            with self.db:
                self.db.execute('UPDATE objects SET complete=1 WHERE digest=?', (digest,))
        return end == size

    def path(self, digest):
        if not self.status(digest)[2]:
            raise CacheError('object not yet verified')
        return self._path(digest)

    def read_block(self, digest, offset, length):
        size, _, complete = self.status(digest)
        if (not complete or type(offset) is not int or type(length) is not int
                or not 0 <= offset < size or not 1 <= length <= 256*1024):
            raise CacheError('invalid object range')
        with self.path(digest).open('rb') as f:
            f.seek(offset)
            return f.read(min(length, size-offset))


class ByteRateLimiter:
    """One shared limiter per NODE, not per neighbor. Bounded burst + byte rate."""
    def __init__(self, bytes_per_second, *, burst=256*1024):
        if (not isinstance(bytes_per_second, (int,float)) or not math.isfinite(bytes_per_second)
                or bytes_per_second <= 0 or type(burst) is not int or burst <= 0):
            raise ValueError('invalid bandwidth budget')
        self.rate = bytes_per_second
        self.burst = burst
        self.tokens = float(burst)
        self.last = time.monotonic()
        self.lock = asyncio.Lock()

    async def take(self, size):
        if type(size) is not int or not 0 < size <= self.burst:
            raise ValueError('request larger than bandwidth burst')
        async with self.lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.burst, self.tokens+(now-self.last)*self.rate)
                self.last = now
                if self.tokens >= size:
                    self.tokens -= size
                    return
                await asyncio.sleep(min(1.0, (size-self.tokens)/self.rate))


class ResumableEpochFetch:
    """Pull only from current neighbors, at most max_neighbors; no room broadcast.

    fetch(peer,digest,offset,length) must use an authenticated bounded RPC.
    It returns exactly the requested bytes; transport must reject oversized
    frames BEFORE buffering. This class bounds requests and disk, not an
    underlying WebRTC implementation's internal buffers.
    """
    def __init__(self, store, fetch, *, limiter, max_neighbors=8, block_bytes=64*1024,
                 timeout=10., max_attempts=4):
        if (type(max_neighbors) is not int or not 1 <= max_neighbors <= 32
                or type(block_bytes) is not int or not 1 <= block_bytes <= min(256*1024,limiter.burst)
                or not math.isfinite(timeout) or timeout <= 0
                or type(max_attempts) is not int or not 1 <= max_attempts <= 32):
            raise ValueError('invalid fetch limits')
        self.store, self.fetch, self.limiter = store, fetch, limiter
        self.max_neighbors, self.block_bytes = max_neighbors, block_bytes
        self.timeout, self.max_attempts = timeout, max_attempts
        self.lock = asyncio.Lock()
        self.cursor = 0

    async def _store(self, method, *args):
        result = getattr(self.store, method)(*args)
        return await result if inspect.isawaitable(result) else result

    async def download(self, digest, peers):
        # One in-flight block per cache worker. Multiple callers do not multiply
        # bandwidth, duplicate a partial offset, or grow a persistent work queue.
        async with self.lock:
            candidates = []
            for peer in peers:
                if peer not in candidates:
                    candidates.append(peer)
                if len(candidates) == self.max_neighbors:
                    break
            size, offset, complete = await self._store('status', digest)
            if complete:
                return await self._store('path', digest)
            if not candidates:
                raise ConnectionError('no connected checkpoint source')
            failures = 0
            while offset < size:
                peer = candidates[self.cursor % len(candidates)]
                length = min(self.block_bytes, size-offset)
                await self.limiter.take(length)  # failed traffic counts too
                try:
                    data = await asyncio.wait_for(self.fetch(peer,digest,offset,length),self.timeout)
                    if not isinstance(data, bytes) or len(data) != length:
                        raise CacheError('invalid checkpoint range response')
                    await self._store('append', digest, offset, data)
                except (OSError, TimeoutError, CacheError):
                    failures += 1
                    self.cursor += 1
                    size, offset, complete = await self._store('status', digest)
                    if failures >= self.max_attempts:
                        raise
                    await asyncio.sleep(min(2., .05*2**(failures-1)))
                    continue
                offset += length
            return await self._store('path', digest)
