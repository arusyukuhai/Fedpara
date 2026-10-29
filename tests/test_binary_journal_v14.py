"""Crash/recovery, migration and bounded writer regressions for binary event log."""
from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from patchlib.core import PatchError
from patchlib.elastic import EventJournal
from patchlib.event_archive import HEADER, read_stored_wire


def test_new_payload_is_not_in_sqlite_and_survives_restart(tmp_path):
    path = tmp_path / 'events.sqlite3'
    wire = os.urandom(2_500_000)
    j = EventJournal(path, 'a'*64)
    try:
        assert j.append('alice', 1, wire)
        data = j.db.execute('SELECT LENGTH(wire),wire_len,log_offset,log_size,digest FROM events').fetchone()
        assert data[:2] == (0, len(wire))
        assert data[2] == 0 and data[3] == len(wire) + HEADER.size
        assert data[4] == hashlib.sha256(wire).hexdigest()
        assert j.get('alice', 1) == wire
        assert j.total_bytes() == len(wire)
        assert list(j.events()) == [wire]
    finally:
        j.close()
    assert path.stat().st_size < len(wire) // 10
    reopened = EventJournal(path, 'a'*64)
    try:
        assert reopened.count() == 1
        assert reopened.get('alice', 1) == wire
        assert not reopened.append('alice', 1, wire)
        with pytest.raises(PatchError, match='equivocation'):
            reopened.append('alice', 1, b'different')
    finally:
        reopened.close()


def test_orphan_frame_and_partial_write_are_trimmed(tmp_path, monkeypatch):
    import patchlib.elastic as mod
    path = tmp_path / 'journal.db'
    j = EventJournal(path, 'b'*64)
    j.append('alice', 1, b'committed')
    valid_size = j.log_path.stat().st_size
    original = mod.append_frame
    def fail_after_binary_write(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('power loss before SQLite commit')
    monkeypatch.setattr(mod, 'append_frame', fail_after_binary_write)
    with pytest.raises(RuntimeError, match='power loss'):
        j.append('alice', 2, b'not-committed')
    assert j.count() == 1
    j.close()
    monkeypatch.setattr(mod, 'append_frame', original)
    reopened=EventJournal(path,'b'*64)
    try:
        assert reopened.log_path.stat().st_size == valid_size
        assert reopened.get('alice',1) == b'committed'
        assert reopened.get('alice',2) is None
        assert reopened.append('alice',2,b'after-recovery')
    finally:
        reopened.close()


def test_committed_frame_corruption_fails_closed(tmp_path):
    path = tmp_path / 'events.sqlite3'
    j=EventJournal(path,'c'*64)
    j.append('alice',1,b'valuable')
    sidecar=j.log_path
    j.close()
    with open(sidecar, 'r+b') as f:
        f.seek(-1,os.SEEK_END)
        f.write(b'!')
    with pytest.raises(PatchError,match='checksum'):
        EventJournal(path,'c'*64)


def test_missing_committed_sidecar_fails_closed(tmp_path):
    path = tmp_path / 'journal.db'
    j=EventJournal(path,'d'*64)
    j.append('alice',1,b'not disposable')
    sidecar=j.log_path
    j.close()
    sidecar.unlink()
    with pytest.raises(PatchError,match='truncated'):
        EventJournal(path,'d'*64)


def test_legacy_sqlite_rows_survive_and_migrate(tmp_path):
    path=tmp_path/'legacy.sqlite3'
    db=sqlite3.connect(path)
    db.execute('CREATE TABLE meta(name TEXT PRIMARY KEY,value TEXT NOT NULL)')
    db.execute('INSERT INTO meta VALUES (?,?)',('genesis','e'*64))
    db.execute('CREATE TABLE events(origin TEXT NOT NULL,seq INTEGER NOT NULL,digest TEXT NOT NULL,wire BLOB NOT NULL,wire_len INTEGER NOT NULL,PRIMARY KEY(origin,seq))')
    old=b'legacy-signed-wire'*10000
    db.execute('INSERT INTO events VALUES (?,?,?,?,?)',('alice',1,hashlib.sha256(old).hexdigest(),old,len(old)))
    db.commit();db.close()
    j=EventJournal(path,'e'*64)
    try:
        assert j.get('alice',1) == old
        assert j.append('alice',2,b'new')
        assert list(j.events()) == [old,b'new']
        assert j.total_bytes() == len(old) + 3
        assert j.migrate_legacy() == 1
        assert j.migrate_legacy() == 0
        assert j.db.execute('SELECT LENGTH(wire) FROM events WHERE seq=1').fetchone()[0] == 0
        assert list(j.events()) == [old,b'new']
    finally:
        j.close()
    reopened=EventJournal(path,'e'*64)
    try:
        assert list(reopened.events()) == [old,b'new']
        assert reopened.get('alice',1) == old
    finally:
        reopened.close()


def test_async_writer_orders_distinct_writes_with_one_thread(tmp_path):
    async def run():
        j=EventJournal(tmp_path/'async.db','f'*64)
        try:
            results=await asyncio.gather(*(j.append_async('alice',s,bytes([s])*100000)
                                           for s in range(1,21)))
            assert all(results)
            assert j.count()==20
            for s in range(1,21):
                assert j.get('alice',s)==bytes([s])*100000
            assert j.total_bytes()==2_000_000
        finally:
            j.close()
    asyncio.run(run())


def test_external_read_only_helper(tmp_path):
    path=tmp_path/'j.db'
    j=EventJournal(path,'a'*64)
    try:
        j.append('alice',1,b'wire')
        row=j.db.execute('SELECT wire,log_offset,log_size,wire_len,digest FROM events').fetchone()
        assert read_stored_wire(path,*row)==b'wire'
    finally:
        j.close()
