"""Crash-recoverable handoff records and conservative GC for online epochs.

Network agreement does not mean every peer's private/unpublished gradients
were incorporated. Events after a certified vector cut must be deliberately
quarantined or re-authored against the new base; silently applying them to a
new genesis would corrupt training. A caller supplies a trusted runtime hook
that returns durable proof of new-generation ingestion for EVERY such event.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3

from .core import PatchError, _canonical_json
from .online_generation import GenerationLedger, _digest


class OnlineAdoptionStore:
    """Stores a durable, idempotent preparation and rebase index in SQLite.

    There is intentionally NO implicit fallback that calls `unlink` when
    required proof is unavailable. GC is local, not a global delete command.
    """
    def __init__(self, path: Path, ledger:GenerationLedger):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.ledger=ledger
        self.db=sqlite3.connect(self.path,timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS epochs(epoch INTEGER PRIMARY KEY, proposal TEXT NOT NULL, state TEXT NOT NULL, new_journal TEXT, checkpoint TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS carries(epoch INTEGER NOT NULL,origin TEXT NOT NULL,seq INTEGER NOT NULL,digest TEXT NOT NULL,status TEXT NOT NULL,receipt TEXT,PRIMARY KEY(epoch,origin,seq))')

    def close(self):self.db.close()

    def prepare(self, old_journal:Path, *, epoch=None):
        cert=self.ledger.latest()
        if not cert:raise PatchError('cannot stage an uncommitted generation')
        p=cert['proposal']
        if epoch is not None and epoch != p['epoch']:
            raise PatchError('generation mismatch')
        old=Path(old_journal)
        if not old.is_file() or old.is_symlink():
            raise PatchError('missing/symlink old event journal')
        # Read indexed signatures only: the big update payload stays in the
        # external append-only file until a transfer/receipt is durable.
        with self.db:
            existing=self.db.execute('SELECT proposal FROM epochs WHERE epoch=?',(p['epoch'],)).fetchone()
            if existing and existing[0]!=_digest(p):raise PatchError('conflicting local generation adoption')
            self.db.execute('INSERT OR IGNORE INTO epochs(epoch,proposal,state) VALUES(?,?,?)',
                            (p['epoch'],_digest(p),'staged'))
            # Stream over SQLite cursor; do not materialize years of events.
            with sqlite3.connect(f'file:{old.resolve()}?mode=ro',uri=True) as journal:
                for origin,seq,digest in journal.execute('SELECT origin,seq,digest FROM events ORDER BY origin,seq'):
                    if seq > p['cutoffs'].get(origin,0):
                        self.db.execute('INSERT OR IGNORE INTO carries(epoch,origin,seq,digest,status) VALUES(?,?,?,?,?)',
                                        (p['epoch'],origin,seq,digest,'pending'))
        return self.status(p['epoch'])

    def status(self, epoch):
        counts=dict(self.db.execute('SELECT status,COUNT(*) FROM carries WHERE epoch=? GROUP BY status',(epoch,)))
        state=self.db.execute('SELECT state FROM epochs WHERE epoch=?',(epoch,)).fetchone()
        return {'epoch':epoch,'state':state[0] if state else 'unknown',
                'pending':counts.get('pending',0), 'carried':counts.get('carried',0)}

    def record_carry(self, *, epoch:int, origin:str, seq:int, digest:str, new_event_receipt:str):
        """Receipt MUST identify a *durably stored* newly signed event.

        The model-runtime caller performs old-wire signature validation,
        gradient/delta rebasing and new event journal acknowledgement. This
        function records the resulting link; no remote claim is sufficient.
        """
        if (not isinstance(new_event_receipt,str) or len(new_event_receipt)!=64
                or any(c not in '0123456789abcdef' for c in new_event_receipt)):
            raise PatchError('invalid new-generation durable event receipt')
        with self.db:
            row=self.db.execute('SELECT digest,status,receipt FROM carries WHERE epoch=? AND origin=? AND seq=?',
                                (epoch,origin,seq)).fetchone()
            if not row or row[0]!=digest:raise PatchError('unknown/conflicting post-cut event')
            if row[1]=='carried' and row[2]!=new_event_receipt:raise PatchError('conflicting old-to-new mapping')
            self.db.execute("UPDATE carries SET status='carried',receipt=? WHERE epoch=? AND origin=? AND seq=?",
                            (new_event_receipt,epoch,origin,seq))

    def mark_active(self, *, epoch:int, manifest:Path, new_journal:Path,
                    load_confirmed:bool, new_journal_confirmed:bool):
        if not load_confirmed or not new_journal_confirmed:
            raise PatchError('cannot adopt before verified weights and durable new journal')
        cert=self.ledger.certified_epoch(epoch)
        if not cert:raise PatchError('no committed generation certificate')
        p=cert['proposal']
        manifest=Path(manifest);new_journal=Path(new_journal)
        if hashlib.sha256(manifest.read_bytes()).hexdigest()!=p['manifest']:
            raise PatchError('checkpoint hash changed since certification')
        if not new_journal.is_file() or new_journal.resolve()==Path(self.path).resolve():
            raise PatchError('missing/invalid activated generation journal')
        with sqlite3.connect(f'file:{new_journal.resolve()}?mode=ro',uri=True) as db:
            # Must be initialized with a *different* immutable genesis and
            # acknowledged by the trainer, not merely by this controller.
            fields=dict(db.execute("SELECT name,value FROM meta WHERE name IN ('genesis','generation_certified_epoch','generation_certified_manifest','generation_certified_proposal')"))
        if (fields.get('generation_certified_epoch')!=str(epoch) or
            fields.get('generation_certified_manifest')!=p['manifest'] or
            fields.get('generation_certified_proposal')!=_digest(p)):
            raise PatchError('actual trainer has not acknowledged generation activation')
        with self.db:
            current=self.db.execute('SELECT state,proposal FROM epochs WHERE epoch=?',(epoch,)).fetchone()
            if not current or current[1]!=_digest(p):raise PatchError('missing staged generation')
            self.db.execute("UPDATE epochs SET state='active',new_journal=?,checkpoint=? WHERE epoch=?",
                            (str(new_journal.resolve()),str(manifest.resolve()),epoch))

    def gc_eligible(self, epoch, old_journal, *, allow_collecting=False):
        """Re-scan live old journal because arrivals may follow `prepare()`."""
        cert=self.ledger.certified_epoch(epoch)
        if not cert:return False
        p=cert['proposal'];row=self.db.execute('SELECT state,new_journal,checkpoint FROM epochs WHERE epoch=?',(epoch,)).fetchone()
        if not row or row[0] not in (('active','collecting') if allow_collecting else ('active',)):return False
        if not row[1] or not row[2] or Path(row[1]).resolve()==Path(old_journal).resolve():return False
        if not Path(row[1]).is_file() or not Path(row[2]).is_file():return False
        # An arbitrary string supplied as a 'new event receipt' does not
        # prove mathematical equivalence of the old LoHa delta after rebase.
        # Until the trainer emits and verifies a cryptographically linked
        # carry event, ALL suffix-bearing generations remain ineligible.
        if self.db.execute("SELECT COUNT(*) FROM carries WHERE epoch=?",(epoch,)).fetchone()[0]:return False
        with sqlite3.connect(f'file:{Path(old_journal).resolve()}?mode=ro',uri=True) as old:
            committed_hash = hashlib.sha256()
            seen = {}
            for origin,seq,digest in old.execute('SELECT origin,seq,digest FROM events ORDER BY origin,seq'):
                if (type(seq) is not int or seq != seen.get(origin,0)+1 or
                        not isinstance(digest,str) or len(digest)!=64 or
                        any(c not in '0123456789abcdef' for c in digest)):
                    return False
                seen[origin] = seq
                if seq <= p['cutoffs'].get(origin,0):
                    committed_hash.update(_canonical_json([origin,seq,digest])+b'\n')
                else:
                    receipt=self.db.execute('SELECT digest,status FROM carries WHERE epoch=? AND origin=? AND seq=?',
                                             (epoch,origin,seq)).fetchone()
                    if not receipt or receipt!=(digest,'carried'):
                        return False
            if committed_hash.hexdigest() != p['cut'] or any(
                    seen.get(origin,0)<cut for origin,cut in p['cutoffs'].items()):
                return False
        return True

    def collect_ready(self, old_journals:dict[int,Path]):
        """Safe local GC polling: execute only after completed handoff.

        Never deletes when old training is active, suffix is unswitched, or
        snapshot/new journal disappeared. The caller may poll at a low rate.
        """
        retired=[]
        for epoch, path in old_journals.items():
            try:
                if self.request_gc(epoch,path):retired.append(epoch)
            except (PatchError,sqlite3.Error,OSError):
                pass  # Leave evidence untouched; retry after next activation.
        return retired

    def request_gc(self, epoch, old_journal:Path):
        """Crash-recoverable local deletion, never remotely triggerable.

        The `collecting` state and retired-journal tombstone are durable BEFORE
        touching archived payloads. A crash partway through deletion may be
        safely retried without recreating an old empty genesis journal.
        """
        import fcntl
        path=Path(old_journal)
        if path.is_symlink():raise PatchError('refusing symlink journal')
        row=self.db.execute('SELECT state,new_journal,checkpoint FROM epochs WHERE epoch=?',(epoch,)).fetchone()
        if row is None or row[0] not in ('active','collecting','collected'):
            raise PatchError('new generation not yet active')
        if row[0]=='collected':return True
        cert=self.ledger.certified_epoch(epoch)
        if not cert:
            raise PatchError('no committed generation certificate')
        p=cert['proposal']
        if (not row[1] or not row[2] or not Path(row[1]).is_file()
            or not Path(row[2]).is_file() or path.resolve()==Path(row[1]).resolve()):
            raise PatchError('replacement journal / checkpoint unavailable')
        if hashlib.sha256(Path(row[2]).read_bytes()).hexdigest()!=p['manifest']:
            raise PatchError('replacement checkpoint differs from certificate')
        # The replacement may have been removed or overwritten while GC was
        # recovering. Re-validate its durable generation identity each time;
        # an old state row is not a perpetual authorization to delete bytes.
        with sqlite3.connect(f'file:{Path(row[1]).resolve()}?mode=ro',uri=True) as newdb:
            fields=dict(newdb.execute("SELECT name,value FROM meta WHERE name='genesis' OR name LIKE 'generation_certified_%'"))
            # A valid metadata row is not enough: SQLite must still open as a
            # healthy read-only DB before the previous generation is removed.
            if newdb.execute('PRAGMA quick_check(1)').fetchone() != ('ok',):
                raise PatchError('replacement journal failed sqlite integrity check')
        if (not isinstance(fields.get('genesis'),str) or
                len(fields['genesis'])!=64 or
                any(c not in '0123456789abcdef' for c in fields['genesis'])):
            raise PatchError('replacement journal lacks a valid genesis')
        if (fields.get('generation_certified_epoch')!=str(epoch) or
            fields.get('generation_certified_manifest')!=p['manifest'] or
            fields.get('generation_certified_proposal')!=_digest(p)):
            raise PatchError('replacement trainer journal lost certified generation identity')
        fd=os.open(str(path)+'.train.lock',os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
        try:
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise PatchError('old trainer still owns the journal') from exc
            # The marker is a durable phase boundary. A crash after writing
            # `collecting` but BEFORE writing a tombstone MUST NOT bypass a
            # new journal scan: late writes may have arrived meanwhile.
            tombstone=Path(str(path)+'.retired.json')
            marker={'epoch':epoch,'manifest':p['manifest'],'new_journal':row[1]}
            raw=_canonical_json(marker)
            if not tombstone.exists():
                if not path.is_file() or not self.gc_eligible(
                        epoch,path,allow_collecting=True):
                    raise PatchError('uncovered update / old checkpoint boundary not verified')
                with sqlite3.connect(f'file:{path.resolve()}?mode=ro',uri=True) as olddb:
                    old_identity=olddb.execute("SELECT value FROM meta WHERE name='genesis'").fetchone()
                if not old_identity or old_identity[0]==fields['genesis']:
                    raise PatchError('replacement journal is not a distinct model genesis')
                if row[0]=='active':
                    with self.db:
                        self.db.execute("UPDATE epochs SET state='collecting' WHERE epoch=? AND state='active'",(epoch,))
            # Never let a previous-process restart recreate an empty original
            # log after a successful rollover. Marker is kept after GC.
            if tombstone.exists():
                if tombstone.is_symlink() or tombstone.read_bytes()!=raw:
                    raise PatchError('conflicting existing retired journal marker')
            else:
                flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0)
                tfd=os.open(tombstone,flags,0o600)
                try:
                    with os.fdopen(tfd,'wb') as stream:
                        stream.write(raw);stream.flush();os.fsync(stream.fileno())
                finally:
                    # fdopen owns tfd if successful.
                    pass
                dirfd=os.open(tombstone.parent,os.O_DIRECTORY)
                try:os.fsync(dirfd)
                finally:os.close(dirfd)
            for target in [Path(str(path)+'.fplog'),path,
                           Path(str(path)+'-wal'),Path(str(path)+'-shm')]:
                if target.is_symlink():raise PatchError('unsafe archive symlink')
                target.unlink(missing_ok=True)
            dirfd=os.open(path.parent,os.O_DIRECTORY)
            try:os.fsync(dirfd)
            finally:os.close(dirfd)
            with self.db:
                self.db.execute("UPDATE epochs SET state='collected' WHERE epoch=?",(epoch,))
            return True
        finally:
            fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
