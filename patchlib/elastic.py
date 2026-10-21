"""Elastic, additive asynchronous federated updates with durable anti-entropy.

NOT strict synchronized FedAvg. Each update carries its denominator, chosen by
its producer's recent verified peer liveness. Immutable genesis and append-only
signed event IDs allow disconnected replicas to eventually reconcile when they
receive the same set of events. Offline peers are NEVER part of a quorum.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

from .core import PEER_ID, FactorDelta, PatchError, UniversalAdapter, _canonical_json, decode_patch
from .livefed import HEX64, _delta_block

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
    """SQLite WAL durable update archive. Never expires events silently.

    The genesis model is an immutable external .fstate; after a crash replay
    all archived events onto it to restore the exact local observed event set.
    To bound storage it *refuses new events* at configured hard cap, rather than
    discarding changes needed by a peer that returns after a long absence.
    """
    def __init__(self, path: Path, genesis: str, *, max_bytes: int = 20*1024**3):
        if not HEX64.fullmatch(genesis) or max_bytes <= 0:
            raise ValueError('invalid genesis / log cap')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.db.execute('''CREATE TABLE IF NOT EXISTS events (
          origin TEXT NOT NULL, seq INTEGER NOT NULL, digest TEXT NOT NULL,
          wire BLOB NOT NULL, wire_len INTEGER NOT NULL,
          PRIMARY KEY(origin, seq))''')
        self.db.execute('CREATE INDEX IF NOT EXISTS event_origin_seq ON events(origin,seq)')
        old = self.db.execute("SELECT value FROM meta WHERE name='genesis'").fetchone()
        if old and old[0] != genesis:
            raise PatchError('local update journal belongs to a DIFFERENT genesis checkpoint')
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO meta(name,value) VALUES('genesis',?)", (genesis,))
        self.genesis = genesis
        self.max_bytes = max_bytes

    def close(self):
        self.db.close()

    def total_bytes(self):
        return self.db.execute('SELECT COALESCE(SUM(wire_len),0) FROM events').fetchone()[0]

    def next_seq(self, origin):
        row = self.db.execute('SELECT MAX(seq) FROM events WHERE origin=?', (origin,)).fetchone()
        return (row[0] or 0) + 1

    def has(self, origin, seq):
        return self.db.execute('SELECT digest FROM events WHERE origin=? AND seq=?',(origin,seq)).fetchone()

    def append(self, origin: str, seq: int, wire: bytes) -> bool:
        if type(seq) is not int or not 1 <= seq < 2**63:
            raise PatchError('invalid sequence')
        digest = hashlib.sha256(wire).hexdigest()
        current = self.has(origin,seq)
        if current:
            if current[0] != digest:
                raise PatchError('equivocation: different signed events with same origin/seq')
            return False
        if self.total_bytes() + len(wire) > self.max_bytes:
            raise PatchError('durable update log is full; create & distribute a trusted new checkpoint/epoch')
        with self.db:
            self.db.execute('INSERT INTO events(origin,seq,digest,wire,wire_len) VALUES(?,?,?,?,?)',
                            (origin,seq,digest,wire,len(wire)))
        return True

    def inventory(self, offset=0, limit=64):
        if not 0 <= offset or not 1 <= limit <= 96:
            raise ValueError('invalid inventory page')
        rows = self.db.execute('SELECT origin,seq,digest FROM events ORDER BY rowid LIMIT ? OFFSET ?',
                               (limit+1,offset)).fetchall()
        return [list(x) for x in rows[:limit]], len(rows) <= limit

    def events(self):
        cursor = self.db.execute('SELECT wire FROM events ORDER BY rowid')
        for (wire,) in cursor:
            yield bytes(wire)

    def get(self, origin, seq):
        row = self.db.execute('SELECT wire FROM events WHERE origin=? AND seq=?',(origin,seq)).fetchone()
        return bytes(row[0]) if row else None

    def count(self):
        return self.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]


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
        self.applied = set()

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
                    power += delta.double().square().sum().item()
                rms = math.sqrt(power/math.prod(g.shape))
                if not math.isfinite(rms) or rms > self.max_delta_rms:
                    raise PatchError('excessive elastic update RMS')
        return EventInfo(origin,meta['seq'],hashlib.sha256(wire).hexdigest(),meta['denominator']),parts

    def _apply_checked(self, info, parts):
        # Called only after full preflight and durable append. Updates are additive;
        # replay from genesis after process termination to recover atomic semantics.
        with self.adapter.lock, torch.no_grad():
            for gi,g in enumerate(self.adapter.groups):
                path,_,attr = g.aliases[0].rpartition('.')
                dest_param = self.adapter.model.get_submodule(path).parametrizations[attr].original
                spec = g.overlay.local
                device = dest_param.device
                tensors = [p.to(device,dtype=torch.float32) for p in parts[gi]]
                n0 = g.shape[0] if g.shape else 1
                rows = max(1,self.block_elements // (math.prod(g.shape[1:]) if len(g.shape)>1 else 1))
                for lo in range(0,n0,rows):
                    dest = dest_param[lo:min(lo+rows,n0)] if g.shape else dest_param
                    val = dest.float() + _delta_block(spec,tensors,lo,min(lo+rows,n0)) / info.denominator
                    if not torch.isfinite(val).all().item():
                        raise PatchError('nonfinite averaged model weights')
                    dest.copy_(val.to(dest.dtype))
                # Inbound updates invalidate local factors based on previous base.
                fresh = FactorDelta(g.shape,self.adapter.kind,self.adapter.rank,self.adapter.alpha)
                for p,init in zip(spec.parts,fresh.parts):
                    p.copy_(init.to(device=p.device,dtype=p.dtype))
                    p.grad = None
            self.applied.add((info.origin,info.seq))

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

    def replay(self):
        """Restore applied base from immutable --state-base plus durable update log."""
        for wire in self.journal.events():
            info,parts = self._check(wire)
            if (info.origin,info.seq) in self.applied:
                raise PatchError('duplicate journal event')
            self._apply_checked(info,parts)

    def reset_factors(self):
        with self.adapter.lock,torch.no_grad():
            for g in self.adapter.groups:
                new = FactorDelta(g.shape,self.adapter.kind,self.adapter.rank,self.adapter.alpha)
                for p,v in zip(g.overlay.local.parts,new.parts):
                    p.copy_(v.to(p.device));p.grad=None
