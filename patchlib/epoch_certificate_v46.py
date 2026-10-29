"""Kagami v46 compact quorum certificates over complete Merkle frontiers.

Proposal votes bind only a constant-sized root, NOT 256 inline origins.  A
committee member must verify every requested frontier page and the actual
local cut / snapshot receipt before voting.  This fixed-membership crash-fault
quorum is not Byzantine consensus and does not itself change active weights.
"""
from __future__ import annotations
import hashlib
import json
import sqlite3
from pathlib import Path
from .core import PatchError, _canonical_json
from .frontier_commitment_v44 import (build_frontier_pages, verify_full_frontier,
                                      header_digest)
from .online_generation import _digest, _sign, _verify, _sha

PROTOCOL = 'kagami-compact-epoch-certificate-v2'
MAX_COMMITTEE = 31


def proposal_v2(*, room, epoch, parent, manifest, semantic, inventory, frontier_header, replicas):
    if (not isinstance(room,str) or not 1 <= len(room) <= 160 or
            type(epoch) is not int or not 1 <= epoch < 2**32 or
            not all(_sha(x) for x in (parent,manifest,semantic,inventory)) or
            not isinstance(replicas,list) or not 2 <= len(replicas) <= MAX_COMMITTEE or
            len(set(replicas))!=len(replicas) or
            any(not isinstance(x,str) or not x or len(x)>128 for x in replicas)):
        raise PatchError('invalid compact epoch proposal')
    digest=header_digest(frontier_header)
    return {'protocol':PROTOCOL,'room':room,'epoch':epoch,'parent':parent,
            'manifest':manifest,'semantic':semantic,'inventory':inventory,
            'frontier_header':frontier_header,'frontier_digest':digest,
            'replicas':sorted(replicas)}


def verify_paged_candidate(proposal, pages, journal_db):
    """Complete frontier (not a sampled proof) and exact indexed prefix.

    Caller separately checks that a *locked* snapshot exporter attested
    checkpoint contents at this cut. Never accept an arbitrary model file.
    """
    frontier=verify_full_frontier(proposal['frontier_header'],pages)
    h=hashlib.sha256()
    seen={}
    for origin,seq,digest in journal_db.execute('SELECT origin,seq,digest FROM events ORDER BY origin,seq'):
        if seq <= frontier.get(origin,0):
            if seq != seen.get(origin,0) + 1:
                raise PatchError('certified prefix has missing or out-of-order event')
            if not _sha(digest):
                raise PatchError('invalid event digest in certified cut')
            h.update(_canonical_json([origin,seq,digest])+b'\n')
            seen[origin]=seq
    if (h.hexdigest()!=proposal['inventory'] or
            any(seen.get(peer,0)<seq for peer,seq in frontier.items() if seq>0)):
        raise PatchError('local journal lacks exact certified frontier')
    return frontier


class CompactEpochLedger:
    """One durable signed vote per epoch; fail closed on competing proposals.

    Snapshots have to be checked by a trusted local exporter before calling
    vote(...,verified_snapshot=True). A remote Boolean must NEVER call this API.
    """
    def __init__(self,path:Path,*,room:str,peer:str,committee:list[str],signer,trusted:dict,
                 genesis:str):
        if (not _sha(genesis) or not isinstance(room,str) or not room or
                not isinstance(committee,list) or len(committee)!=len(set(committee)) or
                not 3 <= len(committee) <= MAX_COMMITTEE or
                not set(committee).issubset(trusted)):
            raise PatchError('invalid compact epoch committee or genesis')
        self.room=room;self.peer=peer;self.committee=tuple(sorted(committee));self.signer=signer
        self.trusted=trusted;self.genesis=genesis;self.majority=len(committee)//2+1
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path,timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS votes(epoch INTEGER,peer TEXT,proposal TEXT,wire TEXT,PRIMARY KEY(epoch,peer))')
        self.db.execute('CREATE TABLE IF NOT EXISTS commits(epoch INTEGER PRIMARY KEY,digest TEXT,wire TEXT)')
        fp=_digest([room,genesis,self.committee])
        with self.db:self.db.execute('INSERT OR IGNORE INTO config VALUES(?,?)',('fingerprint',fp))
        if self.db.execute("SELECT value FROM config WHERE key='fingerprint'").fetchone()[0]!=fp:
            raise PatchError('compact epoch committee fingerprint changed')

    def latest(self):
        r=self.db.execute('SELECT wire FROM commits ORDER BY epoch DESC LIMIT 1').fetchone()
        return json.loads(r[0]) if r else None

    def certified_epoch(self, epoch:int):
        """Read, verify and return a durable historical commit (never a hint)."""
        if type(epoch) is not int or epoch<1:return None
        row=self.db.execute('SELECT digest,wire FROM commits WHERE epoch=?',(epoch,)).fetchone()
        if row is None:return None
        try:
            cert=json.loads(row[1]);p=cert['proposal']
            if (p['epoch']!=epoch or p['room']!=self.room or
                    cert['protocol']!=PROTOCOL or row[0]!=_digest(p)):
                raise PatchError('corrupt compact certificate')
            self._check_votes(p,cert['votes'])
            predecessor=(self.genesis if epoch==1 else
                self.db.execute('SELECT digest FROM commits WHERE epoch=?',(epoch-1,)).fetchone())
            if predecessor is None:raise PatchError('missing compact predecessor')
            parent=predecessor if isinstance(predecessor,str) else predecessor[0]
            if p['parent']!=parent:raise PatchError('broken compact certificate parent chain')
            return cert
        except (KeyError,TypeError,ValueError,IndexError) as exc:
            raise PatchError('invalid persisted compact certificate') from exc

    def head(self):
        current=self.latest()
        return _digest(current['proposal']) if current else self.genesis

    def _validate(self,p):
        if not isinstance(p,dict) or set(p)!={'protocol','room','epoch','parent','manifest','semantic','inventory','frontier_header','frontier_digest','replicas'}:
            raise PatchError('invalid compact certificate fields')
        expected=proposal_v2(room=p['room'],epoch=p['epoch'],parent=p['parent'],
            manifest=p['manifest'],semantic=p['semantic'],inventory=p['inventory'],
            frontier_header=p['frontier_header'],replicas=p['replicas'])
        last=self.latest()
        if (expected!=p or p['room']!=self.room or
                p['epoch']!=(last['proposal']['epoch']+1 if last else 1) or
                p['parent']!=self.head() or
                len(set(p['replicas']) & set(self.committee))<self.majority):
            raise PatchError('stale/noncanonical compact epoch proposal')

    def vote(self,p,*,pages,journal_db,manifest_path:Path,shard_store:Path):
        self._validate(p)
        if self.peer not in self.committee or self.peer not in p['replicas'] or not callable(self.signer):
            raise PatchError('not an authorized local committee checkpoint holder')
        # The locally pinned snapshot must exist and verify byte-for-byte;
        # never trust an ungrounded caller-supplied verified=True Boolean.
        from .shards import read_manifest, verify_shards, semantic_shard_sha256
        path=Path(manifest_path); store=Path(shard_store)
        if (not path.is_file() or path.is_symlink() or not store.is_dir() or store.is_symlink()):
            raise PatchError('missing/unsafe local checkpoint shards')
        with path.open('rb') as f:
            sha=hashlib.file_digest(f,'sha256').hexdigest()
        if sha!=p['manifest']:
            raise PatchError('candidate checkpoint manifest digest differs from voted bytes')
        snapshot=read_manifest(path,max_state_bytes=2*1024**4)
        verify_shards(snapshot,store)
        if semantic_shard_sha256(snapshot)!=p['semantic']:
            raise PatchError('candidate checkpoint semantics do not match vote')
        verify_paged_candidate(p,pages,journal_db)
        digest=_digest(p)
        existing=self.db.execute('SELECT proposal,wire FROM votes WHERE epoch=? AND peer=?',
                                 (p['epoch'],self.peer)).fetchone()
        if existing:
            if existing[0]!=digest:raise PatchError('conflicting vote persisted')
            return json.loads(existing[1])
        payload={'protocol':PROTOCOL,'kind':'vote','room':self.room,'epoch':p['epoch'],'proposal':digest}
        wire=_sign(payload,self.peer,self.signer)
        with self.db:self.db.execute('INSERT INTO votes VALUES(?,?,?,?)',
                                    (p['epoch'],self.peer,digest,json.dumps(wire,sort_keys=True)))
        return wire

    def _check_votes(self,p,votes):
        seen=set()
        for wire in votes:
            payload=_verify(wire,self.trusted)
            expected={'protocol':PROTOCOL,'kind':'vote','room':self.room,'epoch':p['epoch'],'proposal':_digest(p)}
            if payload!=expected or wire['peer'] not in self.committee or wire['peer'] not in p['replicas'] or wire['peer'] in seen:
                raise PatchError('invalid/duplicate compact quorum vote')
            seen.add(wire['peer'])
        if len(seen)<self.majority:raise PatchError('not enough independent quorum signatures')

    def certificate(self,p,votes):
        self._validate(p);self._check_votes(p,votes)
        return {'protocol':PROTOCOL,'proposal':p,'votes':sorted(votes,key=lambda w:w['peer'])}

    def accept(self,cert):
        if not isinstance(cert,dict) or set(cert)!={'protocol','proposal','votes'} or cert['protocol']!=PROTOCOL:
            raise PatchError('malformed compact epoch certificate')
        p=cert['proposal'];epoch=p.get('epoch') if isinstance(p,dict) else None
        previous=self.db.execute('SELECT digest FROM commits WHERE epoch=?',(epoch,)).fetchone()
        if previous:
            if previous[0]!=_digest(p):raise PatchError('equivocating committed epoch')
            self._check_votes(p,cert['votes'])
            return False
        self._validate(p);self._check_votes(p,cert['votes'])
        with self.db:self.db.execute('INSERT INTO commits VALUES(?,?,?)',
                                    (epoch,_digest(p),json.dumps(cert,sort_keys=True)))
        return True

    def close(self):self.db.close()
