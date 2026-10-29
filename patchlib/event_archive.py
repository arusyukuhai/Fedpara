"""Crash-consistent append-only payload archive for signed Fedpara events.

A frame is [8-byte magic][uint64 payload length][sha256][exact signed bytes].
A frame is fdatasync/fsync'd BEFORE its small SQLite index row is committed.
Unindexed tail frames are discarded at the next exclusive journal open. An
indexed but damaged/missing frame is always fatal; never silently skip events.
"""
from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path

from .core import PatchError

MAGIC = b'FPEV14\0\0'
HEADER = struct.Struct('<8sQ32s')


def payload_path(db_path: Path) -> Path:
    return Path(str(db_path) + '.fplog')


def append_frame(fd: int, wire: bytes, *, durable: bool = True,
                 digest: bytes | None = None) -> tuple[int, int]:
    """Append a complete, checksummed frame and sync before returning."""
    start = os.lseek(fd, 0, os.SEEK_END)
    if digest is None:
        digest = hashlib.sha256(wire).digest()
    elif len(digest) != 32:
        raise ValueError('archive digest must have 32 bytes')
    # Avoid allocating/copying a second giant wire object just to prefix it.
    head = HEADER.pack(MAGIC, len(wire), digest)
    for chunk in (head, wire):
        view = memoryview(chunk)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError('append-only archive could not write data')
            view = view[written:]
    if durable:
        os.fsync(fd)
    return start, HEADER.size + len(wire)


def read_frame(fd: int, offset: int, frame_len: int, wire_len: int,
               digest: str) -> bytes:
    if (offset is None or offset < 0 or frame_len != HEADER.size + wire_len or
            wire_len < 0 or frame_len < HEADER.size):
        raise PatchError('invalid indexed event archive range')
    header = os.pread(fd, HEADER.size, offset)
    if len(header) != HEADER.size:
        raise PatchError('committed event archive frame is missing')
    magic, n, sha = HEADER.unpack(header)
    if magic != MAGIC or n != wire_len or sha.hex() != digest:
        raise PatchError('committed event archive frame header is corrupt')
    remaining = wire_len
    cursor = offset + HEADER.size
    chunks = []
    while remaining:
        chunk = os.pread(fd, min(4 * 1024 * 1024, remaining), cursor)
        if not chunk:
            raise PatchError('committed event archive payload is truncated')
        chunks.append(chunk)
        cursor += len(chunk)
        remaining -= len(chunk)
    wire = b''.join(chunks)
    if hashlib.sha256(wire).digest() != sha:
        raise PatchError('committed event archive payload checksum mismatch')
    return wire


def read_stored_wire(db_path: Path, wire: bytes, offset: int | None,
                     frame_len: int | None, wire_len: int, digest: str) -> bytes:
    """Read both old inline SQLite rows and new external sidecar rows.

    Used by offline analysis that cannot acquire the running trainer's lock.
    """
    if offset is None:
        data = bytes(wire)
        if len(data) != wire_len or hashlib.sha256(data).hexdigest() != digest:
            raise PatchError('legacy SQLite event checksum mismatch')
        return data
    path = payload_path(db_path)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    except OSError as exc:
        raise PatchError('committed event archive is missing') from exc
    try:
        return read_frame(fd, offset, frame_len, wire_len, digest)
    finally:
        os.close(fd)


def recover_tail(db, fd: int):
    """Roll back incomplete/unindexed tail; refuse missing durable data."""
    last = db.execute('''SELECT log_offset, log_size, wire_len, digest FROM events
                         WHERE log_offset IS NOT NULL
                         ORDER BY log_offset DESC LIMIT 1''').fetchone()
    end = 0 if last is None else last[0] + last[1]
    size = os.fstat(fd).st_size
    if end > size:
        raise PatchError('committed event archive is truncated; refusing unsafe replay')
    if last is not None:
        read_frame(fd, last[0], last[1], last[2], last[3])
    if size > end:
        os.ftruncate(fd, end)
        os.fsync(fd)
    return end


def _main():
    import argparse
    import sqlite3
    parser = argparse.ArgumentParser(description='Offline v13 -> v14 event payload migration')
    parser.add_argument('command', choices=['migrate'])
    parser.add_argument('journal', type=Path, help='Existing events.sqlite3 journal path')
    parser.add_argument('--vacuum', action='store_true',
                        help='Reclaim old SQLite BLOB storage; requires free disk space')
    args = parser.parse_args()
    with sqlite3.connect(f'file:{args.journal.resolve()}?mode=ro', uri=True) as db:
        row=db.execute("SELECT value FROM meta WHERE name='genesis'").fetchone()
    if row is None:
        parser.error('journal does not have a genesis')
    from .elastic import EventJournal
    journal=EventJournal(args.journal, row[0], sync='FULL')
    try:
        n=journal.migrate_legacy()
        if args.vacuum:
            journal.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            journal.db.execute('VACUUM')
        print(f'Migrated {n} inline events; payload archive: {journal.log_path}')
    finally:
        journal.close()


if __name__ == '__main__':
    _main()
