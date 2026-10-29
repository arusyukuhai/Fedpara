"""Crash-safe, bounded generation handoff state machine (Kagami v43).

This is a coordinator, NOT an untrusted remote command to delete journal data.
Every step requires positive, independently verified runtime acknowledgements.
The current continual v1 worker is not automatically hot-swappable: callers
must provide an actual trained-model installer / new-journal writer, whose
result this controller validates rather than believing boolean flags.

A fresh process resumes a previously interrupted transition by reading this
SQLite WAL. Old journal payloads remain until an exclusive-owner check proves
the old worker is gone and the new worker is durable.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Callable

from .core import PatchError, _canonical_json
from .online_generation import GenerationLedger, _digest
from .live_generation_adoption import OnlineAdoptionStore

STATES = ('certified','quarantined','installed','retired')


def _hash_file(path:Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


class EpochRuntimeV43:
    """Monotonic crash-consistent generation lifecycle with local GC polling.

    Network consensus and tensor installation are separate trust boundaries.
    `install` is an application-supplied callback; it MUST finish its own
    optimizer and worker handoff, create new journal and return its paths.
    Every operation is idempotent and survives process death.
    """
    def __init__(self, root:Path, ledger:GenerationLedger):
        self.root=Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.ledger=ledger
        self.adoption=OnlineAdoptionStore(self.root/'adoption.sqlite3',ledger)
        self.db=sqlite3.connect(self.root/'runtime.sqlite3',timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        with self.db:
            self.db.execute('''CREATE TABLE IF NOT EXISTS handoffs(
                epoch INTEGER PRIMARY KEY, proposal TEXT NOT NULL,
                state TEXT NOT NULL, old_journal TEXT NOT NULL,
                manifest TEXT, store TEXT, new_journal TEXT,
                updated REAL NOT NULL)''')

    def close(self):
        self.adoption.close();self.db.close()

    def _certificate(self,epoch):
        cert=self.ledger.certified_epoch(epoch)
        if cert is None:
            raise PatchError('handoff requires independently verified committed certificate')
        return cert['proposal']

    def _row(self,epoch):
        return self.db.execute('SELECT proposal,state,old_journal,manifest,store,new_journal FROM handoffs WHERE epoch=?',(epoch,)).fetchone()

    def stage(self,old_journal:Path, *, epoch=None):
        cert=self.ledger.latest()
        if cert is None:raise PatchError('no verified generation certificate')
        p=cert['proposal']
        if epoch is not None and epoch!=p['epoch']:raise PatchError('generation mismatch')
        path=Path(old_journal).resolve()
        if not path.is_file() or path.is_symlink():raise PatchError('old journal not found')
        prev=self._row(p['epoch'])
        if prev and (prev[0]!=_digest(p) or Path(prev[2])!=path):
            raise PatchError('generation reconfiguration or old journal substitution')
        # The child adoption index is persistent and may contain local old
        # events past the certified cut. Never infer those events are covered.
        status=self.adoption.prepare(path,epoch=p['epoch'])
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO handoffs(epoch,proposal,state,old_journal,updated) VALUES(?,?,?,?,?)',
                            (p['epoch'],_digest(p),'quarantined',str(path),time.time()))
        return {**status,'epoch':p['epoch'],'state':self._row(p['epoch'])[1]}

    def acknowledge_install(self, *, epoch:int, manifest:Path, store:Path,
                            new_journal:Path, loader_confirmation:bool):
        """An actual trainer calls this AFTER loading tensor bytes and journaling.

        Verification includes the manifest/semantic bytes, ALL shard data,
        and the new journal's trainer-written generation fields. Arbitrary
        UI checkboxes or remote signed claims cannot advance this step.
        """
        if loader_confirmation is not True:
            raise PatchError('the local training worker did not confirm checkpoint installation')
        p=self._certificate(epoch)
        row=self._row(epoch)
        if not row or row[1] not in ('quarantined','installed'):
            raise PatchError('unprepared generation')
        manifest=Path(manifest).resolve();store=Path(store).resolve();new_journal=Path(new_journal).resolve()
        if manifest.is_symlink() or not manifest.is_file() or not store.is_dir() or new_journal==Path(row[2]):
            raise PatchError('invalid new generation files')
        if _hash_file(manifest)!=p['manifest']:
            raise PatchError('manifest does not match quorum certificate')
        from .shards import read_manifest,verify_shards,semantic_shard_sha256
        meta=read_manifest(manifest,max_state_bytes=2*1024**4)
        if semantic_shard_sha256(meta)!=p['semantic']:
            raise PatchError('model weights differ from quorum certificate')
        verify_shards(meta,store)
        self.adoption.mark_active(epoch=epoch,manifest=manifest,new_journal=new_journal,
                                  load_confirmed=True,new_journal_confirmed=True)
        with self.db:
            self.db.execute('UPDATE handoffs SET state=?,manifest=?,store=?,new_journal=?,updated=? WHERE epoch=?',
                            ('installed',str(manifest),str(store),str(new_journal),time.time(),epoch))
        return self.status(epoch)

    def status(self,epoch):
        row=self._row(epoch)
        if not row:return {'epoch':epoch,'state':'missing'}
        result={'epoch':epoch,'state':row[1], 'old_journal':row[2],
                'new_journal':row[5],'manifest':row[3]}
        result.update({k:v for k,v in self.adoption.status(epoch).items() if k in ('pending','carried')})
        return result

    def collect(self,*,epoch:int):
        """Delete local archived events ONLY after independently proven cutover.

        IMPORTANT: existing v1 journals with post-cut events are NEVER deleted.
        Those updates require a new-gen wire format and verified carry replay;
        merely storing receipts cannot establish mathematical equivalence.
        """
        row=self._row(epoch)
        if not row or row[1] not in ('installed','retired'):
            return False
        if row[1]=='retired':return True
        adoption_state=self.adoption.status(epoch)['state']
        if adoption_state != 'collected':
            # A signed manifest is NOT proof that the actual tensor shards
            # still exist. Re-verify every referenced byte and semantic digest
            # BEFORE destructive GC, including when recovering 'collecting'.
            # Do the potentially slow work only for an eligible older log;
            # this method runs in a background worker, not the RTC loop.
            if adoption_state != 'collecting' and not self.adoption.gc_eligible(epoch,Path(row[2])):
                return False
            from .shards import read_manifest,verify_shards,semantic_shard_sha256
            manifest=Path(row[3]);store=Path(row[4]);p=self._certificate(epoch)
            if (not manifest.is_file() or manifest.is_symlink() or
                    not store.is_dir() or store.is_symlink() or
                    _hash_file(manifest)!=p['manifest']):
                return False
            meta=read_manifest(manifest,max_state_bytes=2*1024**4)
            if semantic_shard_sha256(meta)!=p['semantic']:
                return False
            verify_shards(meta,store)
            if not self.adoption.request_gc(epoch,Path(row[2])):return False
        with self.db:
            self.db.execute('UPDATE handoffs SET state=?,updated=? WHERE epoch=?',
                            ('retired',time.time(),epoch))
        return True

    def poll(self):
        """Idempotent GC when the old process relinquishes its journal lock.

        One bounded SQL page of ready epochs per call; leaves *all* uncertain
        events untouched. Intended to run beside WebRTC's normal heartbeat.
        """
        retired=[]
        for (epoch,) in self.db.execute("SELECT epoch FROM handoffs WHERE state='installed' ORDER BY epoch LIMIT 8"):
            try:
                if self.collect(epoch=epoch):retired.append(epoch)
            except (OSError,PatchError,sqlite3.Error):
                pass
        return retired
