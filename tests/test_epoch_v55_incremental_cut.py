"""Failure-injection regression for the v55 incremental certified cut witness.

A first complete certified digest is indispensable. The optimization only
avoids rehashing that SAME complete prefix while signed suffix events append.
"""
import hashlib
import sqlite3
from types import SimpleNamespace

import pytest

from patchlib.core import PatchError, _canonical_json
from patchlib.elastic import EventJournal
from patchlib.epoch_live_rebase_v52 import StreamingCertifiedRebase, verify_certified_cut
import patchlib.epoch_live_rebase_v52 as rebase_module


def _journal(tmp_path, stem):
    return EventJournal(tmp_path / (stem + '.db'), 'a' * 64)


def _digest(rows):
    h = hashlib.sha256()
    for origin, seq, digest in sorted(rows):
        h.update(_canonical_json([origin, seq, digest]) + b'\n')
    return h.hexdigest()


def _cut_runner(journal, rows):
    # We test the cut-proof method in isolation, without constructing a giant
    # Tensor/LoHA shadow just to inspect local certified index metadata.
    runner = object.__new__(StreamingCertifiedRebase)
    runner.source = journal
    runner.proposal = {'cutoffs': {'alice': len(rows)}, 'cut': _digest(rows)}
    runner._poisoned = False
    runner._validated = False
    runner._verified_cut = None
    runner._cut_guard = None
    runner._cached_inventory = None
    return runner


def test_signed_cut_fully_checked_once_and_new_suffix_does_not_rehash(tmp_path, monkeypatch):
    journal = _journal(tmp_path, 'cut')
    try:
        wires = [b'a', b'b', b'c']
        for i, wire in enumerate(wires[:2], 1):
            journal.append('alice', i, wire)
        rows = [('alice', i, hashlib.sha256(w).hexdigest()) for i, w in enumerate(wires[:2], 1)]
        runner = _cut_runner(journal, rows)
        original = rebase_module.verify_certified_cut
        calls = []
        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        monkeypatch.setattr(rebase_module, 'verify_certified_cut', counted)
        assert runner.validate_cut() == (2, 2)
        assert len(calls) == 1
        # Appending a completely new signed suffix MUST NOT mutate an
        # earlier complete high-water prefix, even if it came from another peer.
        journal.append('alice', 3, wires[2])
        journal.append('bob', 1, b'd')
        assert runner.validate_cut() == (2, 2)
        assert len(calls) == 1
    finally:
        journal.close()


def test_old_event_tamper_fails_closed_instead_of_using_cached_cut(tmp_path):
    journal = _journal(tmp_path, 'cut')
    try:
        journal.append('alice', 1, b'a')
        runner = _cut_runner(journal, [('alice', 1, hashlib.sha256(b'a').hexdigest())])
        runner.validate_cut()
        with journal.db:
            journal.db.execute('UPDATE events SET digest=? WHERE origin=? AND seq=?', ('f'*64, 'alice', 1))
        with pytest.raises(PatchError, match='certified-cut events'):
            runner.validate_cut()
    finally:
        journal.close()


def test_external_sqlite_connection_tamper_invalidate_guard(tmp_path):
    journal = _journal(tmp_path, 'cut')
    try:
        journal.append('alice', 1, b'a')
        runner = _cut_runner(journal, [('alice', 1, hashlib.sha256(b'a').hexdigest())])
        runner.validate_cut()
        with sqlite3.connect(journal.path) as external:
            external.execute("UPDATE events SET digest=? WHERE origin='alice' AND seq=1", ('f'*64,))
        with pytest.raises(PatchError, match='certified-cut events'):
            runner.validate_cut()
    finally:
        journal.close()


def test_sql_replace_cannot_bypass_mutation_witness(tmp_path):
    journal = _journal(tmp_path, 'cut')
    try:
        journal.append('alice', 1, b'a')
        runner = _cut_runner(journal, [('alice', 1, hashlib.sha256(b'a').hexdigest())])
        runner.validate_cut()
        with pytest.raises(sqlite3.IntegrityError, match='replacement forbidden'):
            with journal.db:
                journal.db.execute("INSERT OR REPLACE INTO events "
                                   "(origin,seq,digest,wire,wire_len) "
                                   "VALUES ('alice',1,? ,X'61',1)", ('f'*64,))
        assert runner.validate_cut()[0] == 1
    finally:
        journal.close()


def test_schema_change_invalidates_witness_and_requires_full_recheck(tmp_path, monkeypatch):
    journal = _journal(tmp_path, 'cut')
    try:
        journal.append('alice', 1, b'a')
        runner = _cut_runner(journal, [('alice', 1, hashlib.sha256(b'a').hexdigest())])
        runner.validate_cut()
        # Removing the guard also changes SQLite's schema cookie, forcing the
        # expensive safe path. Never trust a witness from an earlier schema.
        with journal.db:
            journal.db.execute('DROP TRIGGER epoch_event_update_v55')
            journal.db.execute("UPDATE events SET digest=? WHERE origin='alice' AND seq=1", ('f'*64,))
        with pytest.raises(PatchError, match='certified-cut events'):
            runner.validate_cut()
    finally:
        journal.close()


def test_legacy_source_without_guard_still_runs_full_validation(tmp_path, monkeypatch):
    journal = _journal(tmp_path, 'cut')
    try:
        journal.append('alice', 1, b'a')
        # A legacy/custom source has no epoch_cut_guard. It must not benefit
        # from this fast path without durable audit triggers.
        legacy = SimpleNamespace(db=journal.db)
        runner = _cut_runner(legacy, [('alice', 1, hashlib.sha256(b'a').hexdigest())])
        original = rebase_module.verify_certified_cut
        calls = []
        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        monkeypatch.setattr(rebase_module, 'verify_certified_cut', counted)
        runner.validate_cut()
        runner.validate_cut()
        assert len(calls) == 2
    finally:
        journal.close()


@pytest.mark.asyncio
async def test_archive_index_cached_until_worker_appends_more_events(tmp_path, monkeypatch):
    archive = _journal(tmp_path, 'archive')
    try:
        archive.append('alice', 1, b'a')
        runner = object.__new__(StreamingCertifiedRebase)
        runner.archive = SimpleNamespace(journal=archive)
        runner._cached_inventory = None
        first = runner.archive_inventory_digest()
        original_json = rebase_module._canonical_json
        calls = []
        def counted(*args, **kwargs):
            calls.append(1)
            return original_json(*args, **kwargs)
        monkeypatch.setattr(rebase_module, '_canonical_json', counted)
        assert runner.archive_inventory_digest() == first
        assert not calls  # no full scan/re-hash
        await archive.append_async('alice', 2, b'b')
        second = runner.archive_inventory_digest()
        assert second[0] == 2 and second[1] != first[1]
        assert calls  # worker-side SQLite write correctly invalidated cache
    finally:
        archive.close()


def test_archive_direct_index_edit_invalidates_digest_cache(tmp_path):
    archive = _journal(tmp_path, 'archive')
    try:
        archive.append('alice', 1, b'a')
        runner = object.__new__(StreamingCertifiedRebase)
        runner.archive = SimpleNamespace(journal=archive)
        runner._cached_inventory = None
        old = runner.archive_inventory_digest()
        with archive.db:
            archive.db.execute("UPDATE events SET digest=? WHERE origin='alice'", ('f'*64,))
        new = runner.archive_inventory_digest()
        assert new[1] != old[1]
    finally:
        archive.close()

@pytest.mark.asyncio
async def test_cut_mutation_during_warm_rebase_poison_shadow(tmp_path):
    journal = _journal(tmp_path, 'cut')
    try:
        journal.append('alice', 1, b'a')
        runner = _cut_runner(journal, [('alice', 1, hashlib.sha256(b'a').hexdigest())])
        runner.validate_cut()
        # Even if a warm carry already passed its initial proof, tampering
        # during subsequent training must fail *before* touching shadow.
        with journal.db:
            journal.db.execute('UPDATE events SET digest=? WHERE seq=1', ('f'*64,))
        with pytest.raises(PatchError, match='index changed'):
            await runner.tick()
        assert runner._poisoned
    finally:
        journal.close()


def test_failed_reverification_poison_previously_validated_shadow(tmp_path):
    journal = _journal(tmp_path, 'cut')
    try:
        journal.append('alice', 1, b'a')
        runner = _cut_runner(journal, [('alice', 1, hashlib.sha256(b'a').hexdigest())])
        runner.validate_cut()
        with journal.db:
            journal.db.execute('UPDATE events SET digest=? WHERE seq=1', ('f'*64,))
        with pytest.raises(PatchError):
            runner.validate_cut()
        assert runner._poisoned and not runner._validated
    finally:
        journal.close()
