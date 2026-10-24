"""Crash-fault-tolerant, authenticated epoch decision protocol for Fedpara v2.

A *single-decree Paxos* ballot is scoped to an immutable checkpoint/roster.
All protocol statements are Ed25519 signed, durable BEFORE acknowledgements,
and can be relayed unchanged via existing WebRTC control DataChannels.

This protocol certifies an immutable update-set manifest, NOT model quality,
Byzantine correctness, or snapshot replication. Final model materialization
requires every manifest update and an independently validated checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Callable

from .core import PatchError, _canonical_json

MAX_WIRE = 16384
MAX_UPDATES = 96
MAX_PEERS = 96


def _hash(value):
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha(text):
    return isinstance(text,str) and len(text)==64 and all(c in '0123456789abcdef' for c in text)


def make_value(epoch: str, base: str, committee: tuple[str,...], updates: list[dict], *,
               eta: float = 1.0, ridge: float = 0.0) -> dict:
    """Explicitly specify the *exact* selected epoch updates, not a prefix tip.

    Each update must be signed/validated in the normal event journal before
    voting. Nothing may be appended to a value after a ballot is accepted.
    """
    import math
    if not _sha(epoch) or not _sha(base) or len(committee)!=len(set(committee)) or not 1<=len(committee)<=MAX_PEERS:
        raise PatchError('invalid immutable epoch/committee')
    if any(not isinstance(p,str) or not p for p in committee):
        raise PatchError('bad committee peer')
    if not (math.isfinite(eta) and eta>0 and math.isfinite(ridge) and ridge>=0):
        raise PatchError('bad epoch aggregation coefficients')
    if len(updates)>MAX_UPDATES:
        raise PatchError('epoch update cap; use smaller cohorts')
    canonical=[]
    keys=set()
    for item in updates:
        if (not isinstance(item,dict) or set(item)!={'origin','seq','digest','samples'} or
            item['origin'] not in committee or type(item['seq']) is not int or item['seq']<=0 or
            not _sha(item['digest']) or type(item['samples']) is not int or
            not 1<=item['samples']<=65536):
            raise PatchError('invalid epoch event descriptor')
        key=(item['origin'],item['seq'])
        if key in keys: raise PatchError('duplicate epoch event')
        keys.add(key)
        canonical.append(dict(item))
    canonical.sort(key=lambda x:(x['origin'],x['seq']))
    return {'format':'fedpara-epoch-value-v2','epoch':epoch,'base':base,
            'committee_sha256':_hash(sorted(committee)),
            'updates':canonical,'eta':float(eta),'ridge':float(ridge)}


class EpochPaxos:
    """One immutable epoch instance. Explicit majority/rotating leader.

    Crash failures only: safety assumes signers follow the durable promise
    rule, one persistent identity per peer, stable committee, and trusted keys.
    Reconfiguration and malicious quorum equivocation are not supported.
    """
    def __init__(self, db_path: Path, *, peer: str, committee: list[str],
                 epoch: str, base: str, signer: Callable, trusted: dict,
                 available: Callable[[dict],bool] | None = None):
        from .elasticnet import encode_control, decode_control
        if not _sha(epoch) or not _sha(base):raise PatchError('bad epoch or base digest')
        self.committee=tuple(sorted(committee))
        if (peer not in self.committee or len(set(self.committee))!=len(self.committee)
            or not 2<=len(self.committee)<=MAX_PEERS or signer is None):
            raise PatchError('epoch requires a pinned fixed committee and a signer')
        self.peer=peer; self.epoch=epoch; self.base=base
        self.signer=signer; self.trusted=trusted; self.available=available or (lambda _:True)
        self.majority=len(self.committee)//2+1
        self.encode=encode_control; self.decode=decode_control
        self.members_sha=_hash(list(self.committee))
        self.db_path=Path(db_path)
        self.db_path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.db_path)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS epoch_state (epoch TEXT PRIMARY KEY, config TEXT NOT NULL, promised INTEGER NOT NULL, accepted_ballot INTEGER NOT NULL, accepted TEXT, committed TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS epoch_messages (epoch TEXT, digest TEXT, wire TEXT, PRIMARY KEY(epoch,digest))')
        config=_hash([self.epoch,self.base,list(self.committee)])
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO epoch_state VALUES(?,?, -1,-1,NULL,NULL)',(epoch,config))
        row=self.db.execute('SELECT config,promised,accepted_ballot,accepted,committed FROM epoch_state WHERE epoch=?',(epoch,)).fetchone()
        if row[0]!=config:raise PatchError('epoch configuration changed after restart')
        self.promised,self.accepted_ballot=row[1:3]
        self.accepted=json.loads(row[3]) if row[3] else None
        self.committed=json.loads(row[4]) if row[4] else None
        self.ballot=-1
        self.candidate=None
        self.promises={}
        self.accepts={}
        self.sent_accept=False
        self.outbox=[]
        self.known=set(x[0] for x in self.db.execute('SELECT digest FROM epoch_messages WHERE epoch=?',(epoch,)))

    def close(self):
        self.db.close()

    def _persist(self):
        with self.db:
            self.db.execute('UPDATE epoch_state SET promised=?,accepted_ballot=?,accepted=?,committed=? WHERE epoch=?',
                (self.promised,self.accepted_ballot,json.dumps(self.accepted,separators=(',',':')) if self.accepted else None,
                 json.dumps(self.committed,separators=(',',':')) if self.committed else None,self.epoch))

    def _wire(self, kind, **fields):
        message={'type':'epoch-v2-'+kind,'peer':self.peer,'epoch':self.epoch,
                 'base':self.base,'committee':self.members_sha, **fields}
        return self.encode(message,self.signer)

    def _broadcast(self, kind, **fields):
        wire=self._wire(kind,**fields)
        self._retain(wire)
        self.outbox.append(wire)
        return wire

    def _retain(self,wire):
        sha=hashlib.sha256(wire.encode()).hexdigest()
        if sha not in self.known:
            if len(self.known)>=4096:
                raise PatchError('epoch signed-message budget exhausted')
            with self.db:
                self.db.execute('INSERT OR IGNORE INTO epoch_messages VALUES(?,?,?)',(self.epoch,sha,wire))
            self.known.add(sha)
        return sha

    def messages(self):
        """Durably stored verified envelope inventory for opportunistic relaying."""
        return [x[0] for x in self.db.execute('SELECT wire FROM epoch_messages WHERE epoch=? ORDER BY digest',(self.epoch,))]

    def _check_value(self,value):
        if not isinstance(value,dict) or set(value)!={'format','epoch','base','committee_sha256','updates','eta','ridge'}:
            raise PatchError('invalid epoch value schema')
        if not isinstance(value['updates'],list) or type(value['eta']) not in (int,float) or type(value['ridge']) not in (int,float):
            raise PatchError('malformed epoch value')
        fresh=make_value(self.epoch,self.base,self.committee,value['updates'],eta=value['eta'],ridge=value['ridge'])
        if fresh!=value:raise PatchError('noncanonical or mismatched epoch value')
        if not self.available(value):raise PatchError('epoch updates/checkpoint not locally verified yet')

    def propose(self, value: dict, *, view: int):
        """Rotate coordinator every view. Different leaders may try higher views.

        A proposal only progresses if the leader collects majority promises.
        Caller must wait for a timeout to advance the view, not spam ballots.
        """
        if type(view) is not int or view<0 or view>=2**31 or self.committee[view%len(self.committee)]!=self.peer:
            raise PatchError('not designated coordinator of this view')
        if self.committed:raise PatchError('epoch already committed; cannot re-propose')
        self._check_value(value)
        if view<=self.ballot:raise PatchError('ballot view must increase')
        self.ballot=view;self.candidate=value
        self.promises={};self.accepts={};self.sent_accept=False
        self._broadcast('prepare',ballot=view)

    def receive(self,wire: str) -> bool:
        """Process and durably retain authenticated origin envelope.

        Returns true for a newly observed envelope. Forward ONLY this original
        signed wire; never rewrite peer identity when relaying.
        """
        if not isinstance(wire,str) or len(wire)>MAX_WIRE:
            raise PatchError('oversized epoch envelope')
        try:
            body=json.loads(wire)
        except (TypeError,ValueError) as e:raise PatchError('malformed epoch JSON') from e
        origin=body.get('peer') if isinstance(body,dict) else None
        if origin not in self.committee:raise PatchError('epoch signer outside committee')
        verify=self.trusted.get(origin)
        if verify is None:raise PatchError('epoch signer public key not pinned')
        msg=self.decode(wire,verify,False)
        if (msg.get('peer')!=origin or msg.get('epoch')!=self.epoch or msg.get('base')!=self.base
            or msg.get('committee')!=self.members_sha):
            raise PatchError('epoch ID / signer / committee mismatch')
        kind=msg.get('type')
        if kind not in ('epoch-v2-prepare','epoch-v2-promise','epoch-v2-accept','epoch-v2-accepted','epoch-v2-commit'):
            raise PatchError('unsupported epoch message')
        # Only persist a message AFTER every field and certificate is validated.
        newly_seen = hashlib.sha256(wire.encode()).hexdigest() not in self.known
        if kind=='epoch-v2-prepare':
            ballot=msg.get('ballot')
            if type(ballot) is not int or ballot<0 or ballot>=2**31 or self.committee[ballot%len(self.committee)]!=origin:
                raise PatchError('invalid prepare leader/view')
            if ballot>self.promised and self.committed is None:
                self.promised=ballot;self._persist()
                self._broadcast('promise',ballot=ballot,accepted_ballot=self.accepted_ballot,
                                accepted=self.accepted)
        elif kind=='epoch-v2-promise':
            ballot=msg.get('ballot'); accepted_ballot=msg.get('accepted_ballot'); accepted=msg.get('accepted')
            if (type(ballot) is not int or ballot<0 or ballot>=2**31 or
                type(accepted_ballot) is not int or accepted_ballot < -1 or accepted_ballot>ballot):
                raise PatchError('invalid epoch promise')
            if accepted is not None:self._check_value(accepted)
            if accepted is None and accepted_ballot!=-1:raise PatchError('missing promised value')
            if (origin!=self.peer and self.peer==self.committee[ballot%len(self.committee)]
                and ballot==self.ballot and not self.sent_accept):
                self.promises[origin]=(accepted_ballot,accepted)
                # The proposer's own promise arrives through its own loopback.
                self._try_accept()
        elif kind=='epoch-v2-accept':
            ballot=msg.get('ballot');value=msg.get('value')
            if type(ballot) is not int or ballot<0 or ballot>=2**31 or self.committee[ballot%len(self.committee)]!=origin:
                raise PatchError('invalid accept leader/view')
            self._check_value(value)
            if ballot>=self.promised and self.committed is None:
                if self.accepted_ballot==ballot and self.accepted!=value:
                    raise PatchError('coordinator equivocation within a ballot')
                same=(self.accepted_ballot==ballot and self.accepted==value)
                self.promised=ballot;self.accepted_ballot=ballot;self.accepted=value
                self._persist()
                if not same:
                    self._broadcast('accepted',ballot=ballot,digest=_hash(value))
        elif kind=='epoch-v2-accepted':
            ballot=msg.get('ballot');digest=msg.get('digest')
            if type(ballot) is not int or ballot<0 or ballot>=2**31 or not _sha(digest):raise PatchError('invalid accepted proof')
            if self.peer==self.committee[ballot%len(self.committee)] and ballot==self.ballot:
                if self.candidate is not None and _hash(self.candidate)==digest:
                    self.accepts[origin]=wire
                    if len(self.accepts)>=self.majority and not self.committed:
                        self._broadcast('commit',ballot=ballot,value=self.candidate,
                                        proofs=[self.accepts[p] for p in sorted(self.accepts)][:self.majority])
        elif kind=='epoch-v2-commit':
            ballot=msg.get('ballot');value=msg.get('value');proofs=msg.get('proofs')
            if type(ballot) is not int or ballot<0 or ballot>=2**31 or not isinstance(proofs,list) or not self.majority<=len(proofs)<=len(self.committee):
                raise PatchError('invalid epoch commit certificate')
            self._check_value(value)
            digest=_hash(value); witnesses=set()
            for signed in proofs:
                if not isinstance(signed,str) or len(signed)>MAX_WIRE:raise PatchError('invalid certificate witness')
                try: witness=json.loads(signed)
                except ValueError as e:raise PatchError('invalid witness JSON') from e
                who=witness.get('peer')
                if who in witnesses or who not in self.committee:raise PatchError('duplicate/unknown certificate signer')
                authenticated=self.decode(signed,self.trusted[who],False)
                if (authenticated.get('type')!='epoch-v2-accepted' or authenticated.get('peer')!=who or
                    authenticated.get('epoch')!=self.epoch or authenticated.get('base')!=self.base or
                    authenticated.get('committee')!=self.members_sha or
                    authenticated.get('ballot')!=ballot or authenticated.get('digest')!=digest):
                    raise PatchError('invalid accept certificate')
                witnesses.add(who)
            if self.committed is not None and self.committed['digest']!=digest:
                raise PatchError('SAFETY VIOLATION: conflicting epoch certificate')
            self.committed={'epoch':self.epoch,'base':self.base,'ballot':ballot,
                            'digest':digest,'value':value,'proofs':proofs}
            self._persist()
        self._retain(wire)
        return newly_seen

    def _try_accept(self):
        if self.sent_accept or self.candidate is None or len(self.promises)<self.majority-1:
            return
        # Include leader's durable promise: prepare self-message must be
        # processed, preventing a coordinator from bypassing local ballot lock.
        if self.promised!=self.ballot:return
        promises=list(self.promises.values())+[(self.accepted_ballot,self.accepted)]
        best=max(n for n,_ in promises)
        if best>=0:
            candidates=[value for n,value in promises if n==best]
            if any(value!=candidates[0] for value in candidates):
                raise PatchError('same accepted ballot contains conflicting values')
            chosen=candidates[0]
            self._check_value(chosen)
            self.candidate=chosen
        self.sent_accept=True
        self._broadcast('accept',ballot=self.ballot,value=self.candidate)

    def status(self):
        return {'epoch':self.epoch,'peer':self.peer,'view':self.ballot,
                'promised':self.promised,'accepted_ballot':self.accepted_ballot,
                'committed':self.committed is not None,'digest':self.committed['digest'] if self.committed else None,
                'promise_count':len(self.promises)+int(self.promised==self.ballot and self.ballot>=0),
                'accept_count':len(self.accepts),'quorum':self.majority}
