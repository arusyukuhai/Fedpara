"""Crash-fault quorum certificate and offline-peer recovery for future Kagami epochs.

IMPORTANT: This is a *control plane*, not an in-place v1 gossip rebase.
Existing v1 trainer MUST NOT discard events based on these certificates alone.

Safety model: fixed, pinned voting committee; < majority crash failures;
no Byzantine committee; unique durable voting identity; content-verified
snapshots; one vote per height/parent; no remote Python execution.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Callable

from .core import PatchError, _canonical_json

PROTOCOL = 'kagami-online-generation-control-v1'
MAX_VOTERS = 31
MAX_MEMBERS = 256
MAX_WIRE = 40 * 1024
HEX = set('0123456789abcdef')


def _sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in HEX for c in value)


def _digest(obj) -> str:
    return hashlib.sha256(_canonical_json(obj)).hexdigest()


def _sign(payload: dict, peer: str, signer: Callable) -> dict:
    raw = _canonical_json(payload)
    if len(raw) > MAX_WIRE - 512:
        raise PatchError('generation message too large')
    signed = {'payload': payload, 'peer': peer}
    raw = _canonical_json(signed)
    signed['signature'] = signer(len(raw), hashlib.sha256(raw).hexdigest())
    return signed


def _verify(signed: dict, trusted: dict) -> dict:
    if not isinstance(signed, dict) or set(signed) != {'payload','peer','signature'}:
        raise PatchError('invalid generation signed envelope')
    peer = signed['peer']
    if not isinstance(peer, str) or peer not in trusted:
        raise PatchError('unpinned generation signer')
    if not isinstance(signed['signature'], str) or len(signed['signature']) > 128:
        raise PatchError('invalid generation signature')
    to_sign = {'payload':signed['payload'], 'peer':peer}
    raw = _canonical_json(to_sign)
    if len(raw) > MAX_WIRE - 512:
        raise PatchError('generation message too large')
    verifier = trusted[peer]
    verifier({'size':len(raw), 'sha256':hashlib.sha256(raw).hexdigest(),
              'sig':signed['signature']})
    return signed['payload']


def make_proposal(*, room: str, epoch: int, parent: str, manifest: str,
                  semantic: str, cut: str, cutoffs: dict[str,int],
                  replicas: list[str]) -> dict:
    """A new base is accepted only at a well-defined immutable journal cut.

    Replica receipts are verified separately during voting. This proposal
    never claims all peers have the same weights during async training.
    """
    if not isinstance(room,str) or not 1 <= len(room) <= 160:
        raise PatchError('invalid room')
    if type(epoch) is not int or not 1 <= epoch < 2**32:
        raise PatchError('invalid epoch')
    if not all(map(_sha,(parent,manifest,semantic,cut))):
        raise PatchError('invalid generation digest')
    if not isinstance(cutoffs,dict) or len(cutoffs) > MAX_MEMBERS:
        raise PatchError('invalid frontier cut')
    if any(not isinstance(k,str) or not 1 <= len(k) <= 128 or
           type(v) is not int or v < 0 or v >= 2**63
           for k,v in cutoffs.items()):
        raise PatchError('invalid frontier entry')
    if not isinstance(replicas,list) or len(replicas)!=len(set(replicas)) or not 2 <= len(replicas) <= MAX_VOTERS:
        raise PatchError('checkpoint needs at least two independent replicas')
    if any(not isinstance(x,str) or not 1 <= len(x) <= 128 for x in replicas):
        raise PatchError('invalid replica identity')
    return {'protocol':PROTOCOL,'room':room,'epoch':epoch,'parent':parent,
            'manifest':manifest,'semantic':semantic,'cut':cut,
            'cutoffs':dict(sorted(cutoffs.items())),
            'replicas':sorted(replicas)}


class GenerationLedger:
    """Durable quorum approval of *one* immutable checkpoint per epoch.

    No automatic mutation of training weights or journal occurs here. Every
    replica MUST independently check the checkpoint and boundary before vote.
    Persist votes before answering, including on rejected conflicting proposals.
    """
    def __init__(self, path: Path, *, room: str, peer: str, committee: list[str],
                 signer: Callable, trusted: dict, genesis: str, min_replicas: int = 2):
        if (not isinstance(committee,list) or len(committee)!=len(set(committee))
                or not 3 <= len(committee) <= MAX_VOTERS or not isinstance(peer,str)
                or not 1 <= len(peer) <= 128 or not set(committee).issubset(trusted)
                or (peer in committee and not callable(signer))
                or not _sha(genesis) or not 2 <= min_replicas <= len(committee)):
            raise PatchError('requires pinned odd/even crash-fault voting committee')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.peer=peer;self.room=room;self.committee=tuple(sorted(committee))
        self.signer=signer;self.trusted=trusted;self.genesis=genesis
        self.majority=len(committee)//2+1
        self.min_replicas=min_replicas
        self.db=sqlite3.connect(self.path,timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS config (name TEXT PRIMARY KEY, value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS votes (epoch INTEGER, peer TEXT, proposal TEXT NOT NULL, wire TEXT NOT NULL, PRIMARY KEY(epoch,peer))')
        self.db.execute('CREATE TABLE IF NOT EXISTS commits (epoch INTEGER PRIMARY KEY, digest TEXT UNIQUE NOT NULL, wire TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        fingerprint=_digest([room,genesis,list(self.committee),min_replicas])
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO config VALUES(?,?)',('fingerprint',fingerprint))
        if self.db.execute("SELECT value FROM config WHERE name='fingerprint'").fetchone()[0]!=fingerprint:
            raise PatchError('generation committee/config changed on an existing ledger')

    def close(self):self.db.close()

    def latest(self):
        row=self.db.execute('SELECT wire FROM commits ORDER BY epoch DESC LIMIT 1').fetchone()
        return json.loads(row[0]) if row else None

    def certified_epoch(self, epoch: int):
        """Read a historical COMMITTED quorum certificate, not just the tip.

        Automatic GC can lag multiple generations while an old trainer shuts
        down. A newer certificate must not make an already-verified previous
        generation impossible to collect. Never treat the unverified local
        `votes` table as proof of commitment.
        """
        if type(epoch) is not int or epoch < 1:
            return None
        row = self.db.execute('SELECT digest,wire FROM commits WHERE epoch=?', (epoch,)).fetchone()
        if row is None:
            return None
        try:
            cert = json.loads(row[1])
            if (cert['proposal']['epoch'] != epoch or cert.get('protocol') != PROTOCOL
                    or cert['proposal'].get('room') != self.room
                    or row[0] != _digest(cert['proposal'])):
                raise PatchError('corrupt committed epoch record')
            self._verify_certificate_without_tip(cert)
            return cert
        except (KeyError, TypeError, ValueError) as exc:
            raise PatchError('corrupt committed epoch record') from exc

    def head(self):
        latest=self.latest()
        return _digest(latest['proposal']) if latest else self.genesis

    def _validate_proposal(self, p):
        if not isinstance(p,dict) or set(p)!={'protocol','room','epoch','parent','manifest','semantic','cut','cutoffs','replicas'}:
            raise PatchError('bad generation proposal fields')
        canonical=make_proposal(room=p['room'],epoch=p['epoch'],parent=p['parent'],
                       manifest=p['manifest'],semantic=p['semantic'],cut=p['cut'],
                       cutoffs=p['cutoffs'],replicas=p['replicas'])
        if canonical!=p or p['room']!=self.room:
            raise PatchError('noncanonical/wrong room generation')
        latest=self.latest()
        if p['epoch']!=(latest['proposal']['epoch']+1 if latest else 1) or p['parent']!=self.head():
            raise PatchError('stale or disconnected epoch proposal')
        if len(set(p['replicas']).intersection(self.committee)) < self.min_replicas:
            raise PatchError('not enough pinned snapshot holders')

    def vote(self, proposal:dict, *, verified_snapshot:bool, verified_cut:bool) -> dict:
        """Vote only when this peer actually validated BOTH snapshot & journal cut.

        verified flags must come from local hash/shard verification, not
        untrusted remote claims. Caller cannot vote for a missing checkpoint.
        """
        if self.peer not in self.committee:
            raise PatchError('ordinary peers cannot cast committee votes')
        self._validate_proposal(proposal)
        if not (verified_snapshot and verified_cut and self.peer in proposal['replicas']):
            raise PatchError('refusing to vote without local verified checkpoint/cut')
        digest=_digest(proposal)
        existing=self.db.execute('SELECT proposal,wire FROM votes WHERE epoch=? AND peer=?',
                                 (proposal['epoch'],self.peer)).fetchone()
        if existing:
            if existing[0]!=digest:
                raise PatchError('refusing conflicting generation vote')
            return json.loads(existing[1])
        signed=_sign({'protocol':PROTOCOL,'kind':'vote','proposal':digest,
                     'room':self.room,'epoch':proposal['epoch']},self.peer,self.signer)
        with self.db:
            self.db.execute('INSERT INTO votes VALUES(?,?,?,?)',
                            (proposal['epoch'],self.peer,digest,json.dumps(signed,sort_keys=True)))
        return signed

    def certificate(self, proposal:dict, votes:list[dict]) -> dict:
        self._validate_proposal(proposal)
        digest=_digest(proposal)
        by_peer={}
        for signed in votes:
            payload=_verify(signed,self.trusted)
            if (not isinstance(payload,dict) or payload!=
                 {'protocol':PROTOCOL,'kind':'vote','proposal':digest,
                  'room':self.room,'epoch':proposal['epoch']}):
                raise PatchError('vote does not bind exact checkpoint')
            peer=signed['peer']
            if peer not in self.committee or peer not in proposal['replicas'] or peer in by_peer:
                raise PatchError('untrusted/duplicated checkpoint vote')
            by_peer[peer]=signed
        if len(by_peer)<max(self.majority,self.min_replicas):
            raise PatchError('quorum/replicas insufficient')
        return {'protocol':PROTOCOL,'proposal':proposal,
                'votes':[by_peer[k] for k in sorted(by_peer)]}

    def accept(self, cert:dict) -> bool:
        if not isinstance(cert,dict) or set(cert)!={'protocol','proposal','votes'} or cert['protocol']!=PROTOCOL:
            raise PatchError('invalid generation certificate')
        p=cert['proposal']
        prior=self.db.execute('SELECT digest FROM commits WHERE epoch=?',
                              (p.get('epoch',-1) if isinstance(p,dict) else -1,)).fetchone()
        if prior:
            if prior[0]!=_digest(p):raise PatchError('conflicting certified epoch')
            # Idempotent, but still authenticate the presented certificate.
            self._verify_certificate_without_tip(cert)
            return False
        self._validate_proposal(p)
        self._verify_certificate_without_tip(cert)
        with self.db:
            self.db.execute('INSERT INTO commits VALUES(?,?,?)',
                            (p['epoch'],_digest(p),json.dumps(cert,sort_keys=True)))
        return True

    def _verify_certificate_without_tip(self,cert):
        p=cert['proposal'];digest=_digest(p);peers=set()
        for wire in cert['votes']:
            payload=_verify(wire,self.trusted)
            if payload!={'protocol':PROTOCOL,'kind':'vote','proposal':digest,
                         'room':self.room,'epoch':p['epoch']}:
                raise PatchError('certificate contains vote for other proposal')
            peer=wire['peer']
            if peer in peers or peer not in self.committee or peer not in p['replicas']:
                raise PatchError('duplicate/foreign generation signer')
            peers.add(peer)
        if len(peers)<max(self.majority,self.min_replicas):
            raise PatchError('certificate lacks pinned quorum')

    def adoption_status(self, *, local_manifest_sha:str|None,
                        local_semantic_sha:str|None, pending_local_events:int,
                        new_journal_ready:bool) -> dict:
        """Offline returners must fetch latest certified *base*, never replay old
        events into an unrelated next epoch. Unsynced local events block GC.
        """
        cert=self.latest()
        if cert is None:return {'action':'no-committed-generation','gc_allowed':False}
        p=cert['proposal']
        verified=(local_manifest_sha==p['manifest'] and local_semantic_sha==p['semantic'])
        if not verified:
            return {'action':'fetch-certified-snapshot','epoch':p['epoch'],
                    'manifest':p['manifest'],'replicas':p['replicas'],
                    'gc_allowed':False}
        if pending_local_events:
            return {'action':'quarantine-or-rebase-uncommitted-updates',
                    'epoch':p['epoch'],'pending':pending_local_events,'gc_allowed':False}
        if not new_journal_ready:
            return {'action':'initialize-and-verify-new-journal','epoch':p['epoch'],
                    'gc_allowed':False}
        return {'action':'adopted','epoch':p['epoch'],'gc_allowed':True}

    def recovery_announcement(self) -> dict:
        """Small state advertisement suitable for bounded gossip to late peers."""
        cert=self.latest()
        if not cert:return {'protocol':PROTOCOL,'kind':'no-generation','room':self.room}
        return {'protocol':PROTOCOL,'kind':'generation-head','room':self.room,
                'epoch':cert['proposal']['epoch'],'digest':_digest(cert['proposal']),
                'manifest':cert['proposal']['manifest'],'replicas':cert['proposal']['replicas']}


def atomic_publish_certificate(path:Path, certificate:dict) -> None:
    """Persist independently checked cert before announcing the new generation."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=_canonical_json(certificate)
    fd,tmp=tempfile.mkstemp(prefix='.next-generation-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as out:
            out.write(raw);out.flush();os.fsync(out.fileno())
        os.replace(tmp,path)
        dfd=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(dfd)
        finally:os.close(dfd)
    finally:Path(tmp).unlink(missing_ok=True)


class AutomaticGenerationControl:
    """Bounded epidemic control-plane messages (transport agnostic).

    Call `receive` for authenticated + independently verified envelopes from
    any neighbor; call `drain` to gossip immutable messages to up to k neighbors.
    This allows thousands of ordinary peers with a small pinned committee.
    Snapshot/data-plane activation is intentionally external and gated.
    """
    def __init__(self, ledger:GenerationLedger, snapshot_valid:Callable[[dict],bool],
                 cut_valid:Callable[[dict],bool],max_messages:int=64):
        if not 4 <= max_messages <= 256:raise ValueError('bounded gossip cache required')
        self.ledger=ledger
        self.snapshot_valid=snapshot_valid
        self.cut_valid=cut_valid
        self.max_messages=max_messages
        self.messages={}
        self.proposals={}
        self.votes={}

    def _remember(self, msg):
        if len(_canonical_json(msg))>MAX_WIRE:raise PatchError('oversize epoch gossip')
        digest=_digest(msg)
        if digest not in self.messages:
            if len(self.messages)>=self.max_messages:
                # Drop stale advertisements before recent certificates.
                oldest=next(iter(self.messages))
                self.messages.pop(oldest)
            self.messages[digest]=msg
        return digest

    def propose(self, p):
        self.ledger._validate_proposal(p)
        if not (self.snapshot_valid(p) and self.cut_valid(p)):
            raise PatchError('cannot propose unavailable or inconsistent checkpoint')
        msg={'kind':'proposal','data':p}
        self.receive(msg)
        return msg

    def receive(self,msg):
        if not isinstance(msg,dict) or set(msg) != {'kind','data'}:
            raise PatchError('unknown generation control wire')
        kind=msg['kind'];data=msg['data']
        if kind=='proposal':
            self.ledger._validate_proposal(data)
            digest=_digest(data)
            if digest not in self.proposals and len(self.proposals)>=self.max_messages:
                self.proposals.pop(next(iter(self.proposals)))
            self.proposals[digest]=data
            self._remember(msg)
            if self.ledger.peer in data['replicas'] and self.snapshot_valid(data) and self.cut_valid(data):
                vote=self.ledger.vote(data,verified_snapshot=True,verified_cut=True)
                self.receive({'kind':'vote','data':vote})
        elif kind=='vote':
            payload=_verify(data,self.ledger.trusted)
            if (payload.get('protocol')!=PROTOCOL or payload.get('kind')!='vote'
                or payload.get('room')!=self.ledger.room):
                raise PatchError('foreign generation vote')
            digest=payload.get('proposal')
            if not _sha(digest):raise PatchError('bad vote proposal digest')
            peer=data['peer']
            if digest not in self.votes and len(self.votes)>=self.max_messages:
                self.votes.pop(next(iter(self.votes)))
            votes=self.votes.setdefault(digest,{})
            if peer in votes and votes[peer]!=data:
                raise PatchError('conflicting signed voter statement')
            votes[peer]=data
            self._remember(msg)
            candidate=self.proposals.get(digest)
            if candidate and len(votes)>=self.ledger.majority:
                try:cert=self.ledger.certificate(candidate,list(votes.values()))
                except PatchError:return
                self.receive({'kind':'certificate','data':cert})
        elif kind=='certificate':
            changed=self.ledger.accept(data)
            if changed:
                # A quorum checkpoint is now certified, NOT automatically
                # installed in an active v1 optimizer.
                self.proposals.clear();self.votes.clear()
                self.messages.clear()
            self._remember(msg)
        else:
            raise PatchError('unsupported generation message')

    def drain(self, *, maximum:int=8):
        if type(maximum) is not int or not 1<=maximum<=64:
            raise ValueError('invalid gossip budget')
        return list(self.messages.values())[-maximum:]


def mark_local_generation_loaded(*, ledger:GenerationLedger, new_journal:Path,
                                 manifest:Path, store:Path,
                                 weights_loaded_and_verified:bool):
    """Record activation only AFTER trainer has loaded verified new weights.

    This must be called by the actual training process while it owns the new
    journal (rather than a GUI that merely copied files to disk). This module
    does NOT call it automatically in the old v1 trainer.
    """
    from .shards import read_manifest, verify_shards, semantic_shard_sha256
    if not weights_loaded_and_verified:
        raise PatchError('weight installation has not been confirmed by trainer')
    cert=ledger.latest()
    if not cert:raise PatchError('no certified new-generation checkpoint')
    p=cert['proposal']
    manifest=Path(manifest)
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=p['manifest']:
        raise PatchError('loaded manifest is not the certified base')
    meta=read_manifest(manifest,max_state_bytes=2*1024**4)
    verify_shards(meta,store)
    if semantic_shard_sha256(meta)!=p['semantic']:
        raise PatchError('semantic checkpoint does not match certified base')
    with sqlite3.connect(new_journal,timeout=3) as db:
        db.execute('PRAGMA busy_timeout=3000')
        old_genesis=db.execute("SELECT value FROM meta WHERE name='genesis'").fetchone()
        if not old_genesis or not _sha(old_genesis[0]):
            raise PatchError('new journal not initialized')
        with db:
            db.execute("INSERT OR REPLACE INTO meta(name,value) VALUES('generation_certified_epoch',?)",(str(p['epoch']),))
            db.execute("INSERT OR REPLACE INTO meta(name,value) VALUES('generation_certified_manifest',?)",(p['manifest'],))
            db.execute("INSERT OR REPLACE INTO meta(name,value) VALUES('generation_certified_proposal',?)",(_digest(p),))
    return {'epoch':p['epoch'],'manifest':p['manifest']}


def retire_adopted_offline_journal(*, ledger:GenerationLedger,
                                    old_journal:Path, checkpoint_manifest:Path,
                                    new_journal:Path, local_semantic:str,
                                    pending_local_events:int, keep_old:bool=False):
    """ONLY after locally persisted new base, old event cut, and old lock.

    Never remove a current v1 journal, nor drop events past the signed cut.
    This is NOT a network-wide concurrent compaction method.
    """
    import fcntl
    old=Path(old_journal);new=Path(new_journal)
    if old.is_symlink() or new.is_symlink():raise PatchError('unsafe symlink journal')
    old=old.resolve();new=new.resolve()
    if old==new or keep_old or not old.is_file() or not new.is_file():
        raise PatchError('need distinct old/new durable journals; no implicit GC')
    if type(pending_local_events) is not int or pending_local_events < 0:
        raise PatchError('unknown pending local updates')
    cert=ledger.latest()
    if cert is None:raise PatchError('uncertified epoch cannot retire history')
    p=cert['proposal']
    actual=hashlib.sha256(Path(checkpoint_manifest).read_bytes()).hexdigest()
    with sqlite3.connect(f'file:{new}?mode=ro',uri=True) as db:
        fields=dict(db.execute("SELECT name,value FROM meta WHERE name LIKE 'generation_certified_%'"))
    if (fields.get('generation_certified_manifest')!=p['manifest'] or
        fields.get('generation_certified_proposal')!=_digest(p) or
        fields.get('generation_certified_epoch')!=str(p['epoch'])):
        raise PatchError('new trainer has not acknowledged verified generation')
    state=ledger.adoption_status(local_manifest_sha=actual,
                 local_semantic_sha=local_semantic,pending_local_events=pending_local_events,
                 new_journal_ready=True)
    if not state['gc_allowed']:
        raise PatchError('refuse old journal GC: '+state['action'])
    lock=Path(str(old)+'.train.lock')
    fd=os.open(lock,os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise PatchError('cannot GC active trainer journal') from e
        with sqlite3.connect(f'file:{old}?mode=ro',uri=True) as db:
            digest=hashlib.sha256()
            extras=0
            for origin,seq,hash_ in db.execute('SELECT origin,seq,digest FROM events ORDER BY origin,seq'):
                if seq>p['cutoffs'].get(origin,0):extras+=1
                else:digest.update(_canonical_json([origin,seq,hash_])+b'\n')
            if extras:
                raise PatchError('uncovered signed updates exist in old journal; quarantine before GC')
            if digest.hexdigest()!=p['cut']:
                raise PatchError('old journal does not match certified checkpoint frontier')
        targets=[Path(str(old)+'.fplog'),old]
        for target in targets:
            if target.is_symlink():raise PatchError('refusing symlink history')
        for target in targets:target.unlink(missing_ok=True)
        return {'retired':True,'epoch':p['epoch'],'old':str(old)}
    finally:
        fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)


class OfflineSnapshotRecovery:
    """Content-verified recovery path for a peer returning after many epochs.

    Control messages/certificates may be fetched over signaling or gossip,
    model chunks from any pinned certified replica. No trust in a URL alone.
    The hosting trainer must install the verified weights as a new base before
    it may publish new updates or retire old ones.
    """
    def __init__(self, ledger:GenerationLedger):
        self.ledger=ledger

    def ingest_chain(self, certificates:list[dict]) -> int:
        if not isinstance(certificates,list) or len(certificates)>1024:
            raise PatchError('invalid certificate catch-up batch')
        previous=self.ledger.latest()
        n=0
        for cert in certificates:
            p=cert.get('proposal',{}) if isinstance(cert,dict) else {}
            if previous and p.get('epoch',0)<=previous['proposal']['epoch']:
                # Idempotent only if identical, and still verify signatures.
                self.ledger.accept(cert)
                continue
            if self.ledger.accept(cert):n+=1
            previous=cert
        return n

    def fetch_verified(self, fetch_from_replica:Callable, *, max_state_bytes:int=2*1024**4):
        """Fetch callback(replica, manifest-sha) -> (manifest-path, shard-store).

        Integrity is checked after download. Retry other replicas on failure;
        never accept a newer-looking but uncertified checkpoint.
        """
        from .shards import read_manifest,verify_shards,semantic_shard_sha256
        cert=self.ledger.latest()
        if cert is None:raise PatchError('no certified checkpoint for offline recovery')
        p=cert['proposal'];errors=[]
        for replica in p['replicas']:
            try:
                path,store=fetch_from_replica(replica,p['manifest'])
                path=Path(path);store=Path(store)
                if hashlib.sha256(path.read_bytes()).hexdigest()!=p['manifest']:
                    raise PatchError('replica checkpoint manifest hash mismatch')
                meta=read_manifest(path,max_state_bytes=max_state_bytes)
                verify_shards(meta,store)
                if semantic_shard_sha256(meta)!=p['semantic']:
                    raise PatchError('replica checkpoint semantic mismatch')
                return {'manifest':path,'store':store,'epoch':p['epoch'],
                        'replica':replica,'proposal':p}
            except (PatchError, OSError, ValueError, TypeError, KeyError) as exc:
                errors.append(f'{replica}: {type(exc).__name__}: {exc}')
        raise PatchError('no verified checkpoint replica available: '+'; '.join(errors)[:300])

    def restore(self, *, fetch_from_replica:Callable,
                install:Callable, pending_local_events:int):
        """Fail closed when stale local updates haven't been safely resolved.

        install(manifest, store, epoch) is the model-runtime-specific loader;
        it MUST validate actual weight application and its new-gen journal.
        """
        if type(pending_local_events) is not int or pending_local_events<0:
            raise PatchError('unknown unsent local updates')
        if pending_local_events:
            return {'action':'quarantine-or-rebase-uncommitted-updates',
                    'pending':pending_local_events,'restored':False}
        source=self.fetch_verified(fetch_from_replica)
        confirmed=install(source['manifest'],source['store'],source['epoch'])
        if confirmed is not True:
            raise PatchError('model runtime did not confirm successful checkpoint install')
        return {'action':'restored','restored':True,'epoch':source['epoch'],
                'manifest':str(source['manifest'])}
