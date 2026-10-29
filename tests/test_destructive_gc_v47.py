"""Regression tests for Kagami v47 GC after verified generation cutover.

These tests deliberately do not claim to exercise autonomous live-worker
cutover: the v46 trainer does not yet provide that capability.
"""
from pathlib import Path
import hashlib

from patchlib.elastic import EventJournal
from patchlib.epoch_runtime_v43 import EpochRuntimeV43
from patchlib.online_generation import make_proposal

from test_epoch_runtime_v43 import prepare


def _activated(tmp_path, *, close_old=True):
    owner, old, new, manifest, store, ledgers = prepare(tmp_path)
    manager=EpochRuntimeV43(tmp_path/'runtime',ledgers[0])
    manager.stage(old)
    manager.acknowledge_install(epoch=1,manifest=manifest,store=store,
                                new_journal=new,loader_confirmation=True)
    if close_old:
        owner.close()
    return owner,manager,old,new,manifest,store,ledgers


def _close(manager,ledgers):
    manager.close()
    for ledger in ledgers:
        ledger.close()


def test_gc_fails_closed_if_snapshot_shard_disappears_after_install(tmp_path):
    _,runtime,old,new,manifest,store,ledgers=_activated(tmp_path)
    shard=next(store.rglob('*.shard'))
    shard.unlink()
    assert runtime.poll()==[]
    assert old.exists() and Path(str(old)+'.fplog').exists()
    _close(runtime,ledgers)


def test_gc_fails_closed_if_snapshot_shard_is_tampered(tmp_path):
    _,runtime,old,new,manifest,store,ledgers=_activated(tmp_path)
    shard=next(store.rglob('*.shard'))
    raw=shard.read_bytes();shard.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    assert runtime.poll()==[]
    assert old.exists()
    _close(runtime,ledgers)


def test_collecting_before_tombstone_must_reverify_late_events(tmp_path):
    _,runtime,old,new,manifest,store,ledgers=_activated(tmp_path)
    with runtime.adoption.db:
        runtime.adoption.db.execute("UPDATE epochs SET state='collecting' WHERE epoch=1")
    # The old writer has returned; one more publisher races before GC intent.
    j=EventJournal(old,'d'*64,max_bytes=0)
    j.append('alice',2,b'late event')
    j.close()
    assert not Path(str(old)+'.retired.json').exists()
    assert runtime.poll()==[]
    assert old.exists()
    _close(runtime,ledgers)


def test_gc_rejects_replacement_journal_reusing_original_genesis(tmp_path):
    _,runtime,old,new,manifest,store,ledgers=_activated(tmp_path)
    import sqlite3
    with sqlite3.connect(new) as db:
        db.execute("UPDATE meta SET value=? WHERE name='genesis'",('d'*64,))
    assert runtime.poll()==[]
    assert old.exists()
    _close(runtime,ledgers)


def test_old_certified_epoch_still_collectable_after_newer_epoch(tmp_path):
    _,runtime,old,new,manifest,store,ledgers=_activated(tmp_path)
    old_cert=ledgers[0].certified_epoch(1)
    parent=ledgers[0].head()
    p2=make_proposal(room='room',epoch=2,parent=parent,
                     manifest=old_cert['proposal']['manifest'],
                     semantic=old_cert['proposal']['semantic'],
                     cut=old_cert['proposal']['cut'],
                     cutoffs={'alice':1},replicas=['committee-a','committee-b','committee-c'])
    votes=[l.vote(p2,verified_snapshot=True,verified_cut=True) for l in ledgers[:2]]
    cert=ledgers[0].certificate(p2,votes)
    for ledger in ledgers: ledger.accept(cert)
    assert ledgers[0].latest()['proposal']['epoch']==2
    assert runtime.poll()==[1]
    assert not old.exists()
    assert new.exists()
    _close(runtime,ledgers)


def test_gc_rejects_corrupt_replacement_database(tmp_path):
    _,runtime,old,new,manifest,store,ledgers=_activated(tmp_path)
    new.write_bytes(b'not a sqlite database')
    assert runtime.poll()==[]
    assert old.exists()
    _close(runtime,ledgers)
