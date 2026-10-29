"""Recover from multi-epoch offline intervals without skipping certified cuts.

Certificate synchronization is distinct from *model/update* recovery. This
module can make monotone progress through adjacent already-certified epochs,
using v53's existing signed carry/rebase for EACH hop. It never simply jumps to
latest weights and discards offline updates. Missing evidence leaves old state
intact; dead archive holders cannot be compensated by a bare signature.
"""
from __future__ import annotations
import asyncio
import json
from pathlib import Path

from .core import PatchError
from .online_generation import _digest


class CertifiedChain:
    """Verify every certificate edge (including the very first genesis)."""
    def __init__(self, ledger):
        self.ledger=ledger

    def path(self, start, end, *, max_hops=16384):
        if (type(start) is not int or type(end) is not int or start<0 or
                end<start or end-start>max_hops):
            raise PatchError('invalid certified recovery range')
        previous=(self.ledger.genesis if start==0 else self._proof(start)['proposal'])
        digest=previous if isinstance(previous,str) else _digest(previous)
        result=[]
        for epoch in range(start+1,end+1):
            cert=self._proof(epoch)
            if cert['proposal']['parent']!=digest:
                raise PatchError('certified recovery chain has broken parent')
            result.append(cert)
            digest=_digest(cert['proposal'])
        return result

    def _proof(self,epoch):
        cert=self.ledger.certified_epoch(epoch)
        if cert is None:raise PatchError(f'missing authenticated certificate for epoch {epoch}')
        return cert


class MultiEpochCatchup:
    """Drive adjacent safe trainer handoffs, never a multi-epoch unsafe leap.

    Any remotely fetched certificate must first be inserted/verified by the
    ledger. `activate_next` verifies local checkpoint, exporter, signed carry,
    and writer fences. Failure is propagated, preserving its fail-closed state.
    """
    def __init__(self,service,*,max_hops_per_poll=2,require_prewarmed=False):
        if not 1<=max_hops_per_poll<=64:raise ValueError('invalid catch-up budget')
        self.service=service
        self.max_hops_per_poll=max_hops_per_poll
        self.require_prewarmed=require_prewarmed
        self._lock=asyncio.Lock()
        self._verified_tip_epoch=0
        self._verified_tip_digest=self.service.ledger.genesis

    async def poll(self):
        async with self._lock:
            done=0
            for _ in range(self.max_hops_per_poll):
                current=self.service.router.current.epoch
                latest=self.service.ledger.latest()
                target=latest['proposal']['epoch'] if latest else 0
                if current>=target:break
                # The full signed chain is checked before starting even one
                # irreversible transition, not just the newest remote head.
                # An append-only, locally pinned ledger permits a verified
                # prefix to be cached. Re-check its tip on every poll and
                # validate only genuinely new certificates; no O(E²) scans.
                if self._verified_tip_epoch:
                    old=self.service.ledger.certified_epoch(self._verified_tip_epoch)
                    if old is None or _digest(old['proposal'])!=self._verified_tip_digest:
                        raise PatchError('previously verified chain prefix changed')
                start=max(current,self._verified_tip_epoch)
                if target>start:
                    verified=CertifiedChain(self.service.ledger).path(start,target)
                    self._verified_tip_epoch=target
                    self._verified_tip_digest=_digest(verified[-1]['proposal'])
                if self.require_prewarmed and not self.service.shadow_ready:
                    break
                advanced=await self.service.activate_next()
                if not advanced:
                    break
                if self.service.router.current.epoch!=current+1:
                    raise PatchError('recovery attempted to skip a certified epoch')
                done+=1
            return done


class DurableOfflineOutbox:
    """Bounded disk-backed quarantine for signed old-epoch updates.

    These are ORIGINAL signed wires, not replayed onto a newer base without
    their old-genesis validator. No TTL, quorum head, or age deletes them.
    If quota is exhausted the producer must backpressure, not drop evidence.
    """
    def __init__(self, root:Path, *, max_bytes=4*1024**3):
        import sqlite3
        if type(max_bytes) is not int or max_bytes<1024:raise ValueError('invalid retention budget')
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(root/'pending.sqlite3',timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS pending (epoch INTEGER, origin TEXT, seq INTEGER, wire BLOB, PRIMARY KEY(epoch,origin,seq))')
        self.db.execute('CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY,v TEXT)')
        # One-time upgrade scan only. Every later quota check is O(1), even
        # after millions of signed events or a restart.
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO meta VALUES(?,?)',('used_bytes',
                str(self.db.execute('SELECT COALESCE(SUM(length(wire)),0) FROM pending').fetchone()[0])))
            self.db.execute('CREATE TRIGGER IF NOT EXISTS offline_insert AFTER INSERT ON pending '
                "BEGIN UPDATE meta SET v=CAST(v AS INTEGER)+length(NEW.wire) WHERE k='used_bytes'; END")
            self.db.execute('CREATE TRIGGER IF NOT EXISTS offline_delete AFTER DELETE ON pending '
                "BEGIN UPDATE meta SET v=CAST(v AS INTEGER)-length(OLD.wire) WHERE k='used_bytes'; END")
        self.max_bytes=max_bytes

    def hold(self,*,epoch,origin,seq,wire:bytes):
        if (type(epoch) is not int or epoch<0 or type(seq) is not int or seq<1 or
                not isinstance(origin,str) or not 1<=len(origin)<=128 or
                not isinstance(wire,bytes) or not wire or len(wire)>16*1024**2):
            raise PatchError('invalid offline signed update')
        import hashlib
        digest=hashlib.sha256(wire).hexdigest()
        with self.db:
            existing=self.db.execute('SELECT wire FROM pending WHERE epoch=? AND origin=? AND seq=?',
                (epoch,origin,seq)).fetchone()
            if existing is not None:
                if hashlib.sha256(existing[0]).hexdigest()!=digest:
                    raise PatchError('equivocating offline update')
                return False
            # Sum content sizes in SQLite; no Python set of old origins.
            used=int(self.db.execute("SELECT v FROM meta WHERE k='used_bytes'").fetchone()[0])
            if used+len(wire)>self.max_bytes:
                raise PatchError('offline outbox at capacity: stop publisher safely')
            self.db.execute('INSERT INTO pending VALUES(?,?,?,?)',(epoch,origin,seq,wire))
        return True

    def iter_epoch(self,epoch):
        return self.db.execute('SELECT origin,seq,wire FROM pending WHERE epoch=? ORDER BY origin,seq',(epoch,))

    def acknowledge(self,*,epoch,origin,seq,verified_archive):
        """Only caller's independently durable signed-carry acknowledgment permits removal."""
        if not verified_archive:
            raise PatchError('cannot release offline evidence without durable signed carry')
        # Caller must have checked archive identity and durability, and supply
        # a real local `CrossEpochCarryArchive` object with the stored key.
        from .epoch_carry_v46 import CrossEpochCarryArchive
        if not isinstance(verified_archive,CrossEpochCarryArchive) or verified_archive.epoch!=epoch+1:
            raise PatchError('foreign signed carry receipt')
        row=self.db.execute('SELECT wire FROM pending WHERE epoch=? AND origin=? AND seq=?',
                            (epoch,origin,seq)).fetchone()
        if row is None:return False
        # Check ORIGINAL signed bytes, not merely origin/seq identity.
        archived=verified_archive.journal.get(origin,seq)
        if archived is None or archived!=row[0]:
            raise PatchError('carry bytes are missing or differ from quarantined signed update')
        with self.db:self.db.execute('DELETE FROM pending WHERE epoch=? AND origin=? AND seq=?',(epoch,origin,seq))
        return True

    def close(self):self.db.close()
