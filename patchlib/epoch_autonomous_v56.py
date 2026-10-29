"""Bounded, durable, crash-fault quorum assembly for a *paged* epoch frontier.

This is a v2 control-plane for a fixed pinned committee (<=31) overseeing an
arbitrarily larger population (default 4096 origins). It cannot synthesize a
valid live optimizer/snapshot receipt: `local_export_verified` is a LOCAL,
trusted callback and must stay False until the real exporter proves the cut.

All proposal/page envelopes must arrive over an already authenticated
GenerationWebRTCBridge. Votes are independently signed and quorum-verified.
No page, bare head, or unsigned proposal can activate new model weights.
"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import sqlite3

from .core import PatchError, _canonical_json
from .online_generation import _digest, _verify, _sign
from .epoch_certificate_v46 import PROTOCOL, CompactEpochLedger, proposal_v2
from .frontier_commitment_v44 import verify_frontier_page

MAX_ORIGINS = 4096
MAX_PROPOSAL_BYTES = 8192
MAX_ACTIVE_PROPOSALS = 2


class AutonomousPagedEpoch:
    """Indexed pages and signed votes, bounded by one candidate per epoch.

    Used through its bridge's signed generation-wire channel. All disk records
    are transactional. A process crash requires no peer-local volatile state.
    """
    def __init__(self, ledger:CompactEpochLedger, path:Path, *,
                 local_checkpoint, local_export_verified, max_origins=MAX_ORIGINS,
                 defer_votes=False):
        if not isinstance(ledger,CompactEpochLedger) or not 1<=max_origins<=MAX_ORIGINS:
            raise ValueError('invalid autonomous paged certificate configuration')
        if not callable(local_checkpoint) or not callable(local_export_verified):
            raise ValueError('a trusted local exporter/checkpoint is required')
        self.ledger=ledger
        self.local_checkpoint=local_checkpoint
        self.local_export_verified=local_export_verified
        self.max_origins=max_origins
        self.defer_votes=bool(defer_votes)
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path,timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS config (k TEXT PRIMARY KEY,v TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS proposals (digest TEXT PRIMARY KEY, epoch INTEGER NOT NULL, wire TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS pages (digest TEXT NOT NULL, idx INTEGER NOT NULL, wire TEXT NOT NULL, PRIMARY KEY(digest,idx))')
        self.db.execute('CREATE TABLE IF NOT EXISTS votes (digest TEXT NOT NULL, peer TEXT NOT NULL, wire TEXT NOT NULL, PRIMARY KEY(digest,peer))')
        self.db.execute('CREATE TABLE IF NOT EXISTS endorsements (digest TEXT PRIMARY KEY, wire TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS exporter_receipts (digest TEXT PRIMARY KEY, wire TEXT NOT NULL)')
        fingerprint=_digest([ledger.room, ledger.genesis, ledger.committee,max_origins])
        with self.db:self.db.execute('INSERT OR IGNORE INTO config VALUES(?,?)',('fingerprint',fingerprint))
        if self.db.execute("SELECT v FROM config WHERE k='fingerprint'").fetchone()[0]!=fingerprint:
            raise PatchError('epoch coordinator config changed')
        self._busy=False
        self._seed_epoch=1
        self._page_scan_epoch=1

    def close(self):self.db.close()

    def _candidate(self,digest):
        row=self.db.execute('SELECT wire FROM proposals WHERE digest=?',(digest,)).fetchone()
        return json.loads(row[0]) if row else None

    def _validate_proposal(self,p):
        if not isinstance(p,dict) or set(p)!={'protocol','room','epoch','parent','manifest','semantic','inventory','frontier_header','frontier_digest','replicas'}:
            raise PatchError('invalid paged candidate')
        expected=proposal_v2(room=p['room'],epoch=p['epoch'],parent=p['parent'],
            manifest=p['manifest'],semantic=p['semantic'],inventory=p['inventory'],
            frontier_header=p['frontier_header'],replicas=p['replicas'])
        if (expected!=p or p['room']!=self.ledger.room or
                p['frontier_header']['total']>self.max_origins or
                len(_canonical_json(p))>MAX_PROPOSAL_BYTES):
            raise PatchError('invalid / oversized frontier candidate')
        committed=self.ledger.certified_epoch(p['epoch'])
        if committed is not None:
            if committed['proposal'] != p:
                raise PatchError('conflicting already certified compact epoch')
            return _digest(p)
        self.ledger._validate(p)
        return _digest(p)

    def receive_export_receipt(self,wire,p):
        from .epoch_export_v57 import verify_receipt
        verify_receipt(wire,p,trusted=self.ledger.trusted,committee=self.ledger.committee)
        digest=self._validate_proposal(p)
        raw=json.dumps(wire,sort_keys=True,separators=(',',':'))
        row=self.db.execute('SELECT wire FROM exporter_receipts WHERE digest=?',(digest,)).fetchone()
        if row is not None and row[0]!=raw:
            raise PatchError('conflicting signed exporter receipts')
        with self.db:self.db.execute('INSERT OR IGNORE INTO exporter_receipts VALUES(?,?)',(digest,raw))
        return True

    def has_export_receipt(self,p):
        from .epoch_export_v57 import verify_receipt
        wire=self.db.execute('SELECT wire FROM exporter_receipts WHERE digest=?',(_digest(p),)).fetchone()
        if wire is None:return False
        verify_receipt(json.loads(wire[0]),p,trusted=self.ledger.trusted,committee=self.ledger.committee)
        return True

    def offer(self,p,pages):
        digest=self._validate_proposal(p)
        if len(pages)!=p['frontier_header']['pages']:
            raise PatchError('cannot offer incomplete candidate')
        # Verify the whole frontier: no duplicate origin across pages.
        from .frontier_commitment_v44 import verify_full_frontier
        verify_full_frontier(p['frontier_header'], pages)
        if (self.ledger.peer not in self.ledger.committee or
                self.ledger.peer not in p['replicas'] or not self.local_export_verified(p) or
                self.local_checkpoint(p) is None):
            raise PatchError('only a locally verified committee exporter may offer a candidate')
        self._remember_proposal(p)
        endorsement=_sign({'protocol':PROTOCOL,'kind':'candidate','digest':digest,
                           'epoch':p['epoch'],'room':self.ledger.room},
                           self.ledger.peer,self.ledger.signer)
        self._remember_endorsement(digest,endorsement)
        for page in pages:self._remember_page(digest,page)
        self._maybe_vote(digest)
        return digest

    def _remember_endorsement(self,digest,wire):
        p=self._candidate(digest)
        if p is None:raise PatchError('endorsement without candidate')
        payload=_verify(wire,self.ledger.trusted)
        if (payload!={'protocol':PROTOCOL,'kind':'candidate','digest':digest,
                     'epoch':p['epoch'],'room':self.ledger.room} or
                wire['peer'] not in self.ledger.committee or wire['peer'] not in p['replicas']):
            raise PatchError('unsigned/noncommittee candidate')
        raw=json.dumps(wire,sort_keys=True,separators=(',',':'))
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO endorsements VALUES(?,?)',(digest,raw))

    def purge_committed(self):
        """Never discard certified frontier pages while trainer recovery needs them.

        Compact certificates contain a root only; deleting their pages here
        would permanently strand offline trainers with no way to reconstruct
        the 4096-origin carry boundary. A future GC requires an explicit
        replicated, audited page-retention/lease protocol.
        """
        return 0

    def certified_pages(self, proposal):
        """Read only complete Merkle-verified pages of a quorum-certified cut."""
        cert=self.ledger.certified_epoch(proposal.get('epoch'))
        if cert is None or cert['proposal']!=proposal:
            raise PatchError('unsigned compact trainer boundary')
        digest=_digest(proposal)
        self._seed_committed(proposal)
        pages=self._pages(digest)
        if pages is not None:
            from .frontier_commitment_v44 import verify_full_frontier
            verify_full_frontier(proposal['frontier_header'],pages)
        return pages

    def _seed_committed(self, proposal):
        """A signed quorum cert suffices to request missing historical pages."""
        self._validate_proposal(proposal)
        self._remember_proposal(proposal)

    def _remember_proposal(self,p):
        digest=self._validate_proposal(p)
        row=self.db.execute('SELECT digest FROM proposals WHERE epoch=?',(p['epoch'],)).fetchone()
        if row is not None and row[0]!=digest:
            raise PatchError('conflicting candidate for same epoch; manual fork resolution required')
        with self.db:
            cur=self.db.execute('INSERT OR IGNORE INTO proposals VALUES(?,?,?)',
                (digest,p['epoch'],json.dumps(p,sort_keys=True,separators=(',',':'))))
        # Rewind ONLY when a genuinely new historical proposal appears.
        if cur.rowcount and p['epoch']<self._page_scan_epoch:
            self._page_scan_epoch=p['epoch']
        return digest

    def _remember_page(self,digest,page):
        p=self._candidate(digest)
        if p is None:raise PatchError('unknown paged candidate')
        verify_frontier_page(p['frontier_header'],page)
        row=self.db.execute('SELECT wire FROM pages WHERE digest=? AND idx=?',
                           (digest,page['index'])).fetchone()
        raw=json.dumps(page,sort_keys=True,separators=(',',':'))
        if row is not None and row[0]!=raw:
            raise PatchError('conflicting valid frontier page')
        with self.db:self.db.execute('INSERT OR IGNORE INTO pages VALUES(?,?,?)',(digest,page['index'],raw))

    def _pages(self,digest):
        p=self._candidate(digest)
        if p is None:return None
        rows=self.db.execute('SELECT idx,wire FROM pages WHERE digest=? ORDER BY idx',(digest,)).fetchall()
        if len(rows)!=p['frontier_header']['pages'] or any(i!=k for k,(i,_) in enumerate(rows)):
            return None
        return [json.loads(raw) for _,raw in rows]

    def ready_for_vote(self):
        """Choose one locally receipted, fully paged candidate for worker I/O.

        No model hashing or journal scanning occurs on the WebRTC event loop.
        """
        for digest,wire in self.db.execute('SELECT digest,wire FROM proposals ORDER BY epoch'):
            p=json.loads(wire)
            if (self.ledger.peer not in p['replicas'] or
                    self.ledger.peer not in self.ledger.committee or
                    self.ledger.certified_epoch(p['epoch']) is not None or
                    self.db.execute('SELECT 1 FROM votes WHERE digest=? AND peer=?',
                         (digest,self.ledger.peer)).fetchone()):continue
            if not self.local_export_verified(p):continue
            if self.local_checkpoint(p) is None:continue
            pages=self._pages(digest)
            if pages is not None:return p,pages
        return None

    def _maybe_vote(self,digest):
        if self.defer_votes:return
        p=self._candidate(digest)
        if p is None or self.ledger.peer not in p['replicas'] or self.ledger.peer not in self.ledger.committee:
            return
        # A signed local vote is already durable; never rehash an entire
        # multi-terabyte snapshot on every anti-entropy tick or neighbor.
        if self.db.execute('SELECT 1 FROM votes WHERE digest=? AND peer=?',
                (digest,self.ledger.peer)).fetchone():return
        pages=self._pages(digest)
        if pages is None:return
        # NEVER use a network-supplied Boolean to assert that weights were
        # exported from exactly this certified cut.
        if not self.local_export_verified(p):return
        sources=self.local_checkpoint(p)
        if sources is None:return
        manifest_path,shard_store,journal_db=sources
        wire=self.ledger.vote(p,pages=pages,journal_db=journal_db,
            manifest_path=manifest_path,shard_store=shard_store)
        self._remember_vote(digest,wire)

    def _remember_vote(self,digest,wire):
        p=self._candidate(digest)
        if p is None:raise PatchError('unknown compact vote proposal')
        payload=_verify(wire,self.ledger.trusted)
        expected={'protocol':PROTOCOL,'kind':'vote','room':self.ledger.room,
                  'epoch':p['epoch'],'proposal':digest}
        if (payload!=expected or wire['peer'] not in self.ledger.committee
                or wire['peer'] not in p['replicas']):
            raise PatchError('invalid compact voter')
        raw=json.dumps(wire,sort_keys=True,separators=(',',':'))
        row=self.db.execute('SELECT wire FROM votes WHERE digest=? AND peer=?',(digest,wire['peer'])).fetchone()
        if row is not None and row[0]!=raw:
            raise PatchError('equivocating compact quorum vote')
        with self.db:self.db.execute('INSERT OR IGNORE INTO votes VALUES(?,?,?)',(digest,wire['peer'],raw))
        votes=[json.loads(x[0]) for x in self.db.execute('SELECT wire FROM votes WHERE digest=?',(digest,))]
        if len(votes)>=self.ledger.majority and self.ledger.certified_epoch(p['epoch']) is None:
            cert=self.ledger.certificate(p,votes)
            self.ledger.accept(cert)

    def receive(self,sender,msg):
        """Only call after authenticating sender through the transport."""
        if not isinstance(msg,dict) or set(msg)!={'kind','data'}:
            raise PatchError('invalid paged control envelope')
        kind,data=msg['kind'],msg['data']
        if kind=='compact-proposal':
            if not isinstance(data,dict) or set(data)!={'proposal','endorsement'}:
                raise PatchError('unendorsed compact candidate')
            p=data['proposal'];digest=self._validate_proposal(p)
            payload=_verify(data['endorsement'],self.ledger.trusted)
            if (payload!={'protocol':PROTOCOL,'kind':'candidate','digest':digest,
                         'epoch':p['epoch'],'room':self.ledger.room} or
                    data['endorsement']['peer'] not in self.ledger.committee or
                    data['endorsement']['peer'] not in p['replicas']):
                raise PatchError('candidate not endorsed by pinned committee')
            self._remember_proposal(p)
            self._remember_endorsement(digest,data['endorsement'])
            if self.ledger.certified_epoch(p['epoch']) is None:
                self._maybe_vote(digest)
        elif kind=='compact-export-receipt':
            if not isinstance(data,dict) or set(data)!={'proposal','receipt'}:
                raise PatchError('malformed signed export receipt')
            self.receive_export_receipt(data['receipt'],data['proposal'])
        elif kind=='compact-page':
            if not isinstance(data,dict) or set(data)!={'digest','page'}:
                raise PatchError('malformed frontier page envelope')
            self._remember_page(data['digest'],data['page'])
            if self.ledger.certified_epoch(self._candidate(data['digest'])['epoch']) is None:
                self._maybe_vote(data['digest'])
        elif kind=='compact-vote':
            if not isinstance(data,dict):raise PatchError('invalid vote envelope')
            payload=_verify(data,self.ledger.trusted)
            if (not isinstance(payload,dict) or payload.get('kind')!='vote'
                    or payload.get('protocol')!=PROTOCOL):
                raise PatchError('not a signed compact vote')
            self._remember_vote(payload['proposal'],data)
        elif kind=='compact-page-need':
            if not isinstance(data,dict) or set(data)!={'digest','index'} or type(data['index']) is not int:
                raise PatchError('invalid compact page request')
            p=self._candidate(data['digest'])
            if p is None or not 0<=data['index']<p['frontier_header']['pages']:
                raise PatchError('unknown compact page request')
            # Stateless reply is scheduled by bridge tick (no per-sender queues).
            return {'kind':'compact-page','data':{'digest':data['digest'],
                'page':self.page(data['digest'],data['index'])}} if self.page(data['digest'],data['index']) else None
        else:raise PatchError('unsupported compact epoch message')
        return None

    def page(self,digest,index):
        row=self.db.execute('SELECT wire FROM pages WHERE digest=? AND idx=?',(digest,index)).fetchone()
        return json.loads(row[0]) if row else None

    def announcements(self):
        """Per-neighbor caller should send a few messages per tick; O(degree)."""
        if self._busy:return []
        self._busy=True
        try:
            here=self.ledger.latest()
            current=here['proposal']['epoch'] if here else 0
            pending=self.db.execute('SELECT digest,wire FROM proposals WHERE epoch>? ORDER BY epoch LIMIT 1',(current,)).fetchone()
            if pending is None:return []
            digest,wire=pending
            p=json.loads(wire)
            self._maybe_vote(digest)
            endorsement=self.db.execute('SELECT wire FROM endorsements WHERE digest=?',(digest,)).fetchone()
            if endorsement is None:
                # Signed certificates may arrive before their optional
                # candidate envelope; no candidate broadcast without endorsement.
                return []
            result=[{'kind':'compact-proposal','data':{'proposal':p,'endorsement':json.loads(endorsement[0])}}]
            receipt=self.db.execute('SELECT wire FROM exporter_receipts WHERE digest=?',(digest,)).fetchone()
            if receipt is not None:
                result.append({'kind':'compact-export-receipt',
                    'data':{'proposal':p,'receipt':json.loads(receipt[0])}})
            for (vote,) in self.db.execute('SELECT wire FROM votes WHERE digest=? ORDER BY peer',(digest,)):
                result.append({'kind':'compact-vote','data':json.loads(vote)})
            return result
        finally:self._busy=False

    def missing_pages(self,limit=4):
        # Also recover pages for committed epochs. A reconnecting trainer
        # needs them to reconstruct the exact signed carry cut.
        if type(limit) is not int or not 1<=limit<=32:
            raise ValueError('invalid page request budget')
        head=self.ledger.latest()
        if head is not None:
            # Lazy historical seeding: at most 4 committed certificates per
            # tick, not an O(number of epochs) full scan on every neighbor.
            for epoch in range(self._seed_epoch,
                               min(head['proposal']['epoch']+1,self._seed_epoch+4)):
                cert=self.ledger.certified_epoch(epoch)
                if cert is None:raise PatchError('missing compact certificate chain')
                self._seed_committed(cert['proposal'])
                self._seed_epoch=epoch+1
        row=None
        # Bound idle control-plane work independently of the number of
        # historical certified epochs; never walk all page sets per tick.
        for _ in range(8):
            found=self.db.execute('SELECT epoch,digest,wire FROM proposals WHERE epoch>=? '
                'ORDER BY epoch LIMIT 1',(self._page_scan_epoch,)).fetchone()
            if found is None:break
            epoch,digest,raw=found;p=json.loads(raw)
            n=self.db.execute('SELECT COUNT(*) FROM pages WHERE digest=?',(digest,)).fetchone()[0]
            if n<p['frontier_header']['pages']:
                row=(digest,raw);break
            self._page_scan_epoch=epoch+1
        if row is None:return []
        digest,raw=row;p=json.loads(raw)
        existing={x[0] for x in self.db.execute('SELECT idx FROM pages WHERE digest=?',(digest,))}
        return [{'kind':'compact-page-need','data':{'digest':digest,'index':i}}
                for i in range(p['frontier_header']['pages']) if i not in existing][:limit]
