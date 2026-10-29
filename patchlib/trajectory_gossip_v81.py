"""Kagami v81: signed, content-addressed behavior checkpoint + verified PPO IS.

KCP81 pages are independently authenticated, journaled, and requested through
existing bounded trajectory frontier/need gossip. KTR81 references their exact
checkpoint digest. A missing checkpoint NEVER produces an applied receipt.
"""
from __future__ import annotations
import hashlib
import json
import math
import struct
from .core import PatchError, PEER_ID, _canonical_json
from .trajectory_gossip_v80 import (TrajectoryGossip, validate_rollout,
    MAX_WIRE, MAX_HEAD, _id)
from .behavior_checkpoint_v81 import (PAGE_BYTES, MAX_CHECKPOINT_PAGES,
    MAX_CHECKPOINT_BYTES, digest as checkpoint_digest, replay_logps)

TR_MAGIC=b'KTR81\0'
CP_MAGIC=b'KCP81\0'
TR_DOMAIN=b'KAGAMI-V81-CHECKPOINTED-TRAJECTORY\0'
CP_DOMAIN=b'KAGAMI-V81-CHECKPOINT-PAGE\0'


def _seal(head,payload,signer,magic,domain):
    signed=domain+_canonical_json(head)
    head=dict(head)
    head['signature']=signer(len(signed),hashlib.sha256(signed).hexdigest())
    encoded=_canonical_json(head)
    frame=magic+struct.pack('!I',len(encoded))+encoded+payload
    if len(encoded)>MAX_HEAD or len(frame)>MAX_WIRE:
        raise PatchError('checkpointed trajectory frame exceeds wire limit')
    return frame


def _open(wire,magic,domain,genesis,epochs,trusted):
    if not isinstance(wire,bytes) or not wire.startswith(magic) or not len(magic)+4<len(wire)<=MAX_WIRE:
        raise PatchError('invalid v81 trajectory/checkpoint frame')
    hsize=struct.unpack('!I',wire[len(magic):len(magic)+4])[0]
    if not 0<hsize<=MAX_HEAD or len(wire)<=len(magic)+4+hsize:
        raise PatchError('invalid v81 event header size')
    try:head=json.loads(wire[len(magic)+4:len(magic)+4+hsize])
    except (UnicodeDecodeError,TypeError,ValueError) as exc:raise PatchError('invalid v81 signed header') from exc
    if not isinstance(head,dict):raise PatchError('invalid v81 header fields')
    origin=head.get('origin');seq=head.get('seq');epoch=head.get('actor_epoch')
    if (not isinstance(origin,str) or not PEER_ID.fullmatch(origin) or
        type(seq) is not int or not 1<=seq<2**63 or
        head.get('genesis')!=genesis or not _id(epoch) or epoch not in epochs):
        raise PatchError('v81 origin or certified Actor epoch mismatch')
    payload=wire[len(magic)+4+hsize:]
    if (type(head.get('size')) is not int or head['size']!=len(payload) or
        not _id(head.get('sha256')) or hashlib.sha256(payload).hexdigest()!=head['sha256']):
        raise PatchError('v81 payload commitment mismatch')
    if 'pub' in head:
        from .roomid import identity_from_public_b64
        verify=identity_from_public_b64(head['pub'],origin)
    else:verify=(trusted or {}).get(origin)
    if verify is None:raise PatchError('unauthenticated v81 event origin')
    sig=head.get('signature')
    signed=domain+_canonical_json({k:v for k,v in head.items() if k!='signature'})
    verify({'size':len(signed),'sha256':hashlib.sha256(signed).hexdigest(),'sig':sig})
    return head,payload


def _head(origin,seq,genesis,epoch,payload,pub):
    if not isinstance(origin,str) or not PEER_ID.fullmatch(origin) or type(seq) is not int or not 1<=seq<2**63 or not _id(genesis) or not _id(epoch):
        raise PatchError('invalid v81 signed event identity')
    h={'origin':origin,'seq':seq,'genesis':genesis,'actor_epoch':epoch,
       'size':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    if pub is not None:
        from .roomid import identity_from_public_b64
        identity_from_public_b64(pub,origin)
        h['pub']=pub
    return h


def page_wire(origin,seq,genesis,epoch,blob,index,signer,*,pub=None):
    if not isinstance(blob,bytes) or not blob or len(blob)>MAX_CHECKPOINT_BYTES:
        raise PatchError('invalid v81 behavior checkpoint')
    total=(len(blob)+PAGE_BYTES-1)//PAGE_BYTES
    if total>MAX_CHECKPOINT_PAGES or type(index) is not int or not 0<=index<total:
        raise PatchError('v81 checkpoint page index overflow')
    payload=blob[index*PAGE_BYTES:(index+1)*PAGE_BYTES]
    head=_head(origin,seq,genesis,epoch,payload,pub)
    head.update(checkpoint=checkpoint_digest(blob),index=index,total=total,checkpoint_size=len(blob))
    return _seal(head,payload,signer,CP_MAGIC,CP_DOMAIN)


def parse_page(wire,genesis,epochs,trusted=None):
    h,p=_open(wire,CP_MAGIC,CP_DOMAIN,genesis,epochs,trusted)
    if (set(h)-{'origin','seq','genesis','actor_epoch','size','sha256','pub','signature',
               'checkpoint','index','total','checkpoint_size'} or
        not {'checkpoint','index','total','checkpoint_size','signature'}<=set(h) or
        not _id(h['checkpoint']) or type(h['index']) is not int or
        type(h['total']) is not int or type(h['checkpoint_size']) is not int or
        not 1<=h['total']<=MAX_CHECKPOINT_PAGES or
        not 0<=h['index']<h['total'] or not 0<h['checkpoint_size']<=MAX_CHECKPOINT_BYTES or
        (h['checkpoint_size']+PAGE_BYTES-1)//PAGE_BYTES!=h['total'] or
        len(p)!=min(PAGE_BYTES,h['checkpoint_size']-h['index']*PAGE_BYTES)):
        raise PatchError('invalid v81 checkpoint page commitment')
    return h,p


def rollout_wire(origin,seq,genesis,epoch,rollout,checkpoint,signer,*,pub=None,actor_tip=None):
    validate_rollout(rollout,max_length=4096)
    if not _id(checkpoint):raise PatchError('missing behavior Actor checkpoint hash')
    payload=_canonical_json(rollout)
    head=_head(origin,seq,genesis,epoch,payload,pub)
    head['checkpoint']=checkpoint
    if actor_tip is not None:
        if (not isinstance(actor_tip,list) or len(actor_tip)!=2 or type(actor_tip[0]) is not int or
            actor_tip[0]<0 or not (actor_tip[1] is None or
            (isinstance(actor_tip[1],list) and len(actor_tip[1])==3 and
             isinstance(actor_tip[1][0],str) and PEER_ID.fullmatch(actor_tip[1][0]) and
             type(actor_tip[1][1]) is int and actor_tip[1][1]>0 and _id(actor_tip[1][2])))):
            raise PatchError('invalid v81 Actor journal cursor')
        head['actor_tip']=actor_tip
    return _seal(head,payload,signer,TR_MAGIC,TR_DOMAIN)


def parse_rollout(wire,genesis,epochs,trusted=None,*,reward_mode,max_length):
    h,p=_open(wire,TR_MAGIC,TR_DOMAIN,genesis,epochs,trusted)
    if (set(h)-{'origin','seq','genesis','actor_epoch','size','sha256','pub','signature','checkpoint','actor_tip'} or
        'signature' not in h or not _id(h.get('checkpoint'))):
        raise PatchError('missing v81 authenticated behavior checkpoint commitment')
    if 'actor_tip' in h:
        tip=h['actor_tip']
        if not (isinstance(tip,list) and len(tip)==2 and type(tip[0]) is int and tip[0]>=0 and
                (tip[1] is None or (isinstance(tip[1],list) and len(tip[1])==3 and
                isinstance(tip[1][0],str) and PEER_ID.fullmatch(tip[1][0]) and
                type(tip[1][1]) is int and tip[1][1]>0 and _id(tip[1][2])))):
            raise PatchError('invalid v81 signed journal cursor')
    try:d=json.loads(p)
    except (ValueError,UnicodeDecodeError,TypeError) as exc:raise PatchError('malformed v81 trajectory') from exc
    validate_rollout(d,max_length=max_length,reward_mode=reward_mode)
    return h,d


class VerifiedTrajectoryGossip(TrajectoryGossip):
    """Fail-closed receiver of KCP81 pages and KTR81 trajectories.

    Deduplicates signed pages; stores only page indices (not duplicate blobs).
    Rebuilds an incomplete index from the durable event journal on restart.
    """
    def __init__(self,*args,**kwargs):
        kwargs['_lane_tag']='v81'
        super().__init__(*args,**kwargs)
        self.journal.db.execute('''CREATE TABLE IF NOT EXISTS v81_pages
            (origin TEXT NOT NULL, checkpoint TEXT NOT NULL, idx INTEGER NOT NULL,
             total INTEGER NOT NULL, checkpoint_size INTEGER NOT NULL,
             actor_epoch TEXT NOT NULL, seq INTEGER NOT NULL,
             PRIMARY KEY(origin,checkpoint,idx))''')
        self.journal.db.execute('''CREATE TABLE IF NOT EXISTS v81_rollouts
            (origin TEXT NOT NULL, seq INTEGER NOT NULL, checkpoint TEXT NOT NULL,
             actor_epoch TEXT NOT NULL, PRIMARY KEY(origin,seq))''')
        self.journal.db.commit()
        self._reindex()

    def _reindex(self):
        # Rebuild orphaned index rows after a crash between EventJournal's
        # fsynced append and v81 index commit (no cross-file transaction).
        for origin,seq in self.journal.db.execute('''SELECT e.origin,e.seq FROM events e
            LEFT JOIN v81_rollouts r ON e.origin=r.origin AND e.seq=r.seq
            LEFT JOIN v81_pages p ON e.origin=p.origin AND e.seq=p.seq
            WHERE r.seq IS NULL AND p.seq IS NULL ORDER BY e.origin,e.seq''').fetchall():
            wire=self.journal.get(origin,seq)
            if wire is None:raise PatchError('missing immutable v81 event archive')
            self._index(wire,origin,seq)

    def _index(self,wire,origin,seq):
        if wire.startswith(CP_MAGIC):
            h,_=parse_page(wire,self.genesis,self.allowed_epochs,self.trusted)
            if (h['origin'],h['seq'])!=(origin,seq):raise PatchError('invalid v81 stored page identity')
            prior=self.journal.db.execute('SELECT seq FROM v81_pages WHERE origin=? AND checkpoint=? AND idx=?',
                (origin,h['checkpoint'],h['index'])).fetchone()
            if prior is not None and prior[0]!=seq:
                raise PatchError('conflicting behavior checkpoint page index')
            with self.journal.db:
                self.journal.db.execute('''INSERT OR IGNORE INTO v81_pages
                    (origin,checkpoint,idx,total,checkpoint_size,actor_epoch,seq)
                    VALUES(?,?,?,?,?,?,?)''',
                    (origin,h['checkpoint'],h['index'],h['total'],h['checkpoint_size'],h['actor_epoch'],seq))
        elif wire.startswith(TR_MAGIC):
            h,_=parse_rollout(wire,self.genesis,self.allowed_epochs,self.trusted,
                               reward_mode=self.reward_mode,max_length=self.max_length)
            if (h['origin'],h['seq'])!=(origin,seq):raise PatchError('invalid v81 stored rollout identity')
            with self.journal.db:
                self.journal.db.execute('INSERT OR IGNORE INTO v81_rollouts(origin,seq,checkpoint,actor_epoch) VALUES(?,?,?,?)',
                                      (origin,seq,h['checkpoint'],h['actor_epoch']))
        else:raise PatchError('mixed v80/v81 or Actor event in v81 trajectory journal')

    def local(self,rollout,*,checkpoint=None,actor_tip=None):
        validate_rollout(rollout,max_length=self.max_length,reward_mode=self.reward_mode)
        d=checkpoint_digest(checkpoint)
        total=(len(checkpoint)+PAGE_BYTES-1)//PAGE_BYTES
        frames=[]
        existing=self.journal.db.execute('''SELECT idx,actor_epoch,total,checkpoint_size FROM v81_pages
            WHERE origin=? AND checkpoint=?''',(self.self_id,d)).fetchall()
        if any(e[1]!=self.actor_epoch or e[2]!=total or e[3]!=len(checkpoint) for e in existing):
            raise PatchError('v81 checkpoint was already committed under a different Actor epoch')
        have={e[0] for e in existing}
        # Content addressing avoids re-shipping unchanged policies and also
        # resumes a partially journaled checkpoint after a local crash.
        for i in range(total):
            if i in have:continue
            seq=self.journal.next_seq(self.self_id)
            wire=page_wire(self.self_id,seq,self.genesis,self.actor_epoch,checkpoint,i,self.signer,pub=self.pub)
            if not self.journal.append(self.self_id,seq,wire):raise PatchError('duplicate local v81 checkpoint page')
            self._index(wire,self.self_id,seq)
            frames.append(wire)
        seq=self.journal.next_seq(self.self_id)
        wire=rollout_wire(self.self_id,seq,self.genesis,self.actor_epoch,rollout,d,self.signer,pub=self.pub,actor_tip=actor_tip)
        if not self.journal.append(self.self_id,seq,wire):raise PatchError('duplicate local v81 trajectory')
        self._index(wire,self.self_id,seq)
        self.persist((self.self_id,seq))
        frames.append(wire)
        return frames

    def ingest(self,wire):
        if wire.startswith(CP_MAGIC):
            h,_=parse_page(wire,self.genesis,self.allowed_epochs,self.trusted)
        elif wire.startswith(TR_MAGIC):
            h,_=parse_rollout(wire,self.genesis,self.allowed_epochs,self.trusted,
                               reward_mode=self.reward_mode,max_length=self.max_length)
        else:raise PatchError('v81 room rejects legacy or unsigned trajectories')
        origin,seq=h['origin'],h['seq']
        # Do not poison the immutable event archive with a second signed
        # page for the same checkpoint offset at a different journal seq.
        if wire.startswith(CP_MAGIC):
            old=self.journal.db.execute('SELECT seq FROM v81_pages WHERE origin=? AND checkpoint=? AND idx=?',
                (origin,h['checkpoint'],h['index'])).fetchone()
            if old is not None and old[0]!=seq:
                raise PatchError('conflicting behavior checkpoint page index')
        prior=self.journal.has(origin,seq)
        if prior:
            if prior[0]!=hashlib.sha256(wire).hexdigest():raise PatchError('v81 event equivocation')
            self._index(wire,origin,seq)
            return False
        if not self.journal.append(origin,seq,wire):raise PatchError('v81 event index race')
        self._index(wire,origin,seq)
        return True

    def pending(self,limit=4):
        rows=self.journal.db.execute('''SELECT r.origin,r.seq FROM v81_rollouts r
            LEFT JOIN trajectory_applied a ON a.origin=r.origin AND a.seq=r.seq
            WHERE a.origin IS NULL ORDER BY r.origin,r.seq LIMIT ?''',(limit,)).fetchall()
        return [(o,s,self.journal.get(o,s)) for o,s in rows]

    def _checkpoint(self,origin,checkpoint,epoch):
        rows=self.journal.db.execute('''SELECT idx,total,checkpoint_size,actor_epoch,seq
            FROM v81_pages WHERE origin=? AND checkpoint=? ORDER BY idx''',(origin,checkpoint)).fetchall()
        if not rows:return None
        total=rows[0][1];size=rows[0][2]
        if (len(rows)<total or len(rows)>total or total>MAX_CHECKPOINT_PAGES or
            size>MAX_CHECKPOINT_BYTES):return None
        chunks=[]
        for i,(idx,n,sz,ep,seq) in enumerate(rows):
            if idx!=i or n!=total or sz!=size or ep!=epoch:
                raise PatchError('v81 checkpoint index/epoch equivocation')
            wire=self.journal.get(origin,seq)
            if wire is None:raise PatchError('v81 missing archived checkpoint page')
            h,p=parse_page(wire,self.genesis,self.allowed_epochs,self.trusted)
            if h['checkpoint']!=checkpoint or h['index']!=i:
                raise PatchError('v81 checkpoint page archive changed')
            chunks.append(p)
        blob=b''.join(chunks)
        if len(blob)!=size or hashlib.sha256(blob).hexdigest()!=checkpoint:
            raise PatchError('v81 behavior checkpoint digest mismatch')
        return blob

    def apply_pending(self,model,device,limit=1):
        applied=0
        for origin,seq,wire in self.pending(max(4,limit)):
            if applied>=limit:break
            if wire is None:raise PatchError('v81 missing durable trajectory')
            head,data=parse_rollout(wire,self.genesis,self.allowed_epochs,self.trusted,
                                    reward_mode=self.reward_mode,max_length=self.max_length)
            blob=self._checkpoint(origin,head['checkpoint'],head['actor_epoch'])
            if blob is None:continue  # future anti-entropy completes it; NEVER ACK
            before=self._snapshot()
            try:
                checked_logps=replay_logps(model,blob,data,self.mixer.args,device)
                # Use the independently RECOMPUTED mu, not the signed claim.
                # The 2e-3 cross-backend comparison tolerance must never
                # enter the IS estimator as a systematic log-ratio error.
                verified=dict(data,behavior_logp=checked_logps)
                self.mixer.train_imported_rollout(model,device,verified)
                self.persist((origin,seq))
            except BaseException:
                with self.journal.db:
                    self.journal.db.execute('''INSERT OR REPLACE INTO trajectory_state
                        (id,tensors,digest,critic_steps) VALUES(1,?,?,?)''',before)
                self._restore()
                raise
            applied+=1
        return applied
