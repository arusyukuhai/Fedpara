"""v58: actual torch LowRank/COW export, suffix, tamper, and budget regressions."""
from __future__ import annotations
import asyncio
import hashlib
import threading
import sqlite3
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from patchlib.core import UniversalAdapter,PatchError
from patchlib.elastic import EventJournal
from patchlib.epoch_frozen_export_v58 import capture
from patchlib.epoch_export_v57 import export_at_step_boundary,verify_receipt
from patchlib.identity import generate_keypair,signer_from_pem,verifier_from_pem
from patchlib.shards import read_manifest,verify_shards,_blob_path
from patchlib.shard_runtime import effective_chunks,effective_state_sources


def setup(tmp_path):
    model=torch.nn.Sequential(torch.nn.Linear(8,4),torch.nn.Linear(4,2))
    adapter=UniversalAdapter(model,'frozen-tiny','r1',kind='loha',rank=2)
    journal=EventJournal(tmp_path/'events.sqlite3','a'*64)
    key=tmp_path/'a.key';pub=tmp_path/'a.pub'
    generate_keypair(key,pub)
    return model,adapter,journal,signer_from_pem(key),{'a':verifier_from_pem(pub)}


def contents(model,adapter):
    return {n:b''.join(effective_chunks(t,o,chunk_bytes=65536))
            for n,(t,o) in effective_state_sources(model,adapter).items()}


def export(tmp_path,model,adapter,journal,signer,frozen):
    return export_at_step_boundary(journal=journal,model=model,adapter=adapter,
        store=tmp_path/'store',incoming=tmp_path/'incoming',room='test',epoch=1,
        parent='0'*64,replicas=['a','b'],signer=signer,peer='a',
        max_state_bytes=1024**2,shard_bytes=65536,frozen=frozen)


def read_payload(meta,store):
    return {spec['name']:b''.join(_blob_path(store,c['sha256']).read_bytes()
           for c in spec['shards']) for spec in meta['tensors']}


def test_frozen_factors_are_independent_while_training_proceeds(tmp_path):
    model,adapter,journal,signer,trusted=setup(tmp_path)
    try:
        assert journal.append('alice',1,b'old-update')
        frozen=capture(journal=journal,model=model,adapter=adapter)
        expected=contents(model,adapter)
        with torch.no_grad():
            for group in adapter.groups:
                for part in group.overlay.local.parts:
                    part.add_(.1)
        assert journal.append('alice',2,b'post-capture-update')
        assert contents(model,adapter)!=expected
        p,pages,receipt=export(tmp_path,model,adapter,journal,signer,frozen)
        assert verify_receipt(receipt,p,trusted=trusted,committee=['a','b'])
        meta=read_manifest(tmp_path/'incoming'/(p['manifest']+'.fmanifest'))
        verify_shards(meta,tmp_path/'store')
        assert read_payload(meta,tmp_path/'store')==expected
        assert meta['seq']==1  # not the later journal size
        assert frozen.frontier=={'alice':1}
        assert frozen.factor_bytes>0
    finally:journal.close()


def test_base_in_place_mutation_never_receives_a_signature(tmp_path):
    model,adapter,journal,signer,_=setup(tmp_path)
    try:
        frozen=capture(journal=journal,model=model,adapter=adapter)
        with torch.no_grad():
            model[0].parametrizations.weight.original.add_(1.)
        with pytest.raises(PatchError,match='base tensor changed'):
            export(tmp_path,model,adapter,journal,signer,frozen)
        assert not list((tmp_path/'incoming').glob('*.fmanifest')) if (tmp_path/'incoming').exists() else True
    finally:journal.close()


def test_byte_budget_rejects_full_dense_fallback(tmp_path):
    model,adapter,journal,signer,_=setup(tmp_path)
    try:
        with pytest.raises(PatchError,match='memory budget'):
            capture(journal=journal,model=model,adapter=adapter,max_factor_bytes=1)
    finally:journal.close()


def test_cut_mutation_during_async_export_is_rejected(tmp_path):
    model,adapter,journal,signer,_=setup(tmp_path)
    try:
        assert journal.append('alice',1,b'a')
        frozen=capture(journal=journal,model=model,adapter=adapter)
        # Authorized append is fine; modifying a prefix is not.
        # New journal v55 guards existing-row updates and replacements.
        p,_,_=export(tmp_path,model,adapter,journal,signer,frozen)
        assert p['inventory']==frozen.inventory
        # A trusted DB writer can modify the index. It must invalidate the
        # captured signed prefix even if the audit trigger is present.
        journal.db.execute("UPDATE events SET digest=? WHERE origin='alice'",('f'*64,))
        journal.db.commit()
        with pytest.raises(PatchError,match='certified-cut'):
            export(tmp_path,model,adapter,journal,signer,frozen)
    finally:journal.close()


def test_remote_overlay_install_is_frozen(tmp_path):
    model,adapter,journal,signer,_=setup(tmp_path)
    try:
        first=adapter.groups[0]
        parts=[torch.ones_like(p)*.25 for p in first.overlay.local.parts]
        first.overlay.install('remote',parts)
        expected=contents(model,adapter)
        frozen=capture(journal=journal,model=model,adapter=adapter)
        first.overlay.remove('remote')
        first.overlay.install('other',[p*2 for p in parts])
        p,_,_=export(tmp_path,model,adapter,journal,signer,frozen)
        meta=read_manifest(tmp_path/'incoming'/(p['manifest']+'.fmanifest'))
        assert read_payload(meta,tmp_path/'store')==expected
    finally:journal.close()


@pytest.mark.asyncio
async def test_background_export_does_not_block_training_or_p2p_loop(tmp_path,monkeypatch):
    from patchlib import epoch_export_v57
    import time
    model,adapter,journal,signer,_=setup(tmp_path)
    try:
        frozen=capture(journal=journal,model=model,adapter=adapter)
        original=epoch_export_v57.export_state_sharded
        started=threading.Event()
        def slow(*args,**kwargs):
            started.set()
            time.sleep(.25)  # Simulated slow multi-shard fsync/offload
            return original(*args,**kwargs)
        monkeypatch.setattr(epoch_export_v57,'export_state_sharded',slow)
        def background():
            with sqlite3.connect(journal.path) as db:
                view=SimpleNamespace(db=db,count=lambda: db.execute('SELECT COUNT(*) FROM events').fetchone()[0])
                return export(tmp_path,model,adapter,view,signer,frozen)
        task=asyncio.create_task(asyncio.to_thread(background))
        while not started.is_set():
            await asyncio.sleep(.005)
        ticks=0
        for _ in range(15):
            with torch.no_grad():
                for group in adapter.groups:
                    for part in group.overlay.local.parts:
                        part.add_(.001)
            if ticks==0: await journal.append_async('alice',1,b'signed-post-cut-event')
            ticks+=1
            await asyncio.sleep(.005)
        assert ticks==15 and not task.done()
        p,_,_=await task
        meta=read_manifest(tmp_path/'incoming'/(p['manifest']+'.fmanifest'))
        assert meta['seq']==0
    finally: journal.close()


def test_wal_pinned_deferred_cut_never_absorbs_post_capture_events(tmp_path):
    from patchlib.epoch_frozen_export_v58 import finish_deferred_frontier
    model,adapter,journal,signer,trusted=setup(tmp_path)
    db=sqlite3.connect(journal.path,check_same_thread=False)
    try:
        assert journal.append('alice',1,b'first')
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        db.execute('SELECT COALESCE(MAX(rowid),0) FROM events').fetchone()
        frozen=capture(journal=SimpleNamespace(db=db),model=model,
                       adapter=adapter,defer_frontier=True)
        with pytest.raises(PatchError,match='unfinalized'):
            export(tmp_path,model,adapter,journal,signer,frozen)
        assert journal.append('alice',2,b'after-first-cut')
        final=finish_deferred_frontier(frozen,db)
        assert final.frontier=={'alice':1}
        assert final.event_count==1
        db.close()
        p,_,receipt=export(tmp_path,model,adapter,journal,signer,final)
        assert verify_receipt(receipt,p,trusted=trusted,committee=['a','b'])
        meta=read_manifest(tmp_path/'incoming'/(p['manifest']+'.fmanifest'))
        assert meta['seq']==1
    finally:
        db.close()
        journal.close()


def test_deferred_cut_keeps_origin_gap_validation(tmp_path):
    from patchlib.epoch_frozen_export_v58 import finish_deferred_frontier
    model,adapter,journal,_,_=setup(tmp_path)
    db=sqlite3.connect(journal.path,check_same_thread=False)
    try:
        # Simulate an impossible gap in a damaged local journal.
        journal.db.execute('INSERT INTO events(origin,seq,digest,wire,wire_len) VALUES(?,?,?,?,?)',
                           ('alice',2,'a'*64,b'x',1))
        journal.db.commit()
        db.execute('BEGIN')
        db.execute('SELECT COALESCE(MAX(rowid),0) FROM events').fetchone()
        frozen=capture(journal=SimpleNamespace(db=db),model=model,adapter=adapter,
                       defer_frontier=True)
        with pytest.raises(PatchError,match='noncontiguous'):
            finish_deferred_frontier(frozen,db)
    finally:
        db.close()
        journal.close()
