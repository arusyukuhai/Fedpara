"""Opt-in signed, bounded PPO trajectories for recipient-side Critic IS replay.

This is NOT a claim of deterministic global Critic convergence or exact
finite-sample/off-policy correction. Source behavior log probabilities are
signed claims, not independently attested executions of a past Actor.
"""
from __future__ import annotations
import hashlib
import json
import math
import struct

import torch
from safetensors.torch import save as safe_save, load as safe_load
from .core import PatchError, PEER_ID, _canonical_json
from .elastic import EventJournal
from .paired_checkpoint_v75 import _record

MAGIC=b'KTR80\0'
DOMAIN=b'KAGAMI-V80-SIGNED-TRAJECTORY\0'
MAX_WIRE=65536
MAX_HEAD=4096
MAX_STATE=8*1024*1024


def _id(x):
    return isinstance(x,str) and len(x)==64 and all(y in '0123456789abcdef' for y in x)


def validate_rollout(data,*,max_length=1024,reward_mode=None):
    if not isinstance(data,dict) or set(data)!={'prompt','tokens','behavior_logp','reward','reference','reward_mode'}:
        raise PatchError('invalid or missing trajectory fields')
    prompt=data['prompt'];tokens=data['tokens'];logps=data['behavior_logp'];ref=data['reference']
    if not isinstance(prompt,list) or not 1<=len(prompt)<=max_length-1:
        raise PatchError('invalid trajectory prompt length')
    if not isinstance(tokens,list) or not 1<=len(tokens)<=128 or len(prompt)+len(tokens)>max_length:
        raise PatchError('invalid trajectory generated length')
    if (not isinstance(ref,list) or not 1<=len(ref)<=1024 or
        any(type(n) is not int or not 0<=n<2**31 for n in prompt+tokens+ref)):
        raise PatchError('invalid trajectory token ids')
    if not isinstance(logps,list) or len(logps)!=len(tokens) or any(
            type(x) not in (int,float) or not math.isfinite(x) or x>1e-5 or x<-1e6 for x in logps):
        raise PatchError('invalid behavior action log probabilities')
    if data['reward_mode'] not in ('exact','token_match') or (reward_mode is not None and reward_mode!=data['reward_mode']):
        raise PatchError('unsupported or mismatched trajectory reward mode')
    expected=(float(tokens==ref) if data['reward_mode']=='exact' else
              float(sum(a==b for a,b in zip(tokens,ref))/max(len(tokens),len(ref))))
    reward=data['reward']
    if type(reward) not in (float,int) or not math.isfinite(reward) or abs(float(reward)-expected)>1e-6:
        raise PatchError('unverified trajectory reward')
    return data


def make_wire(origin,seq,genesis,actor_epoch,rollout,signer,*,pub=None,actor_tip=None):
    if not isinstance(origin,str) or not PEER_ID.fullmatch(origin) or type(seq) is not int or not 1<=seq<2**63 or not _id(genesis) or not _id(actor_epoch):
        raise PatchError('invalid rollout event identity')
    validate_rollout(rollout,max_length=4096)
    payload=_canonical_json(rollout)
    head={'origin':origin,'seq':seq,'genesis':genesis,'actor_epoch':actor_epoch,
          'size':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    if actor_tip is not None:
        if (not isinstance(actor_tip,list) or len(actor_tip)!=2 or type(actor_tip[0]) is not int or
            actor_tip[0]<0 or not (actor_tip[1] is None or (isinstance(actor_tip[1],list) and
            len(actor_tip[1])==3 and isinstance(actor_tip[1][0],str) and PEER_ID.fullmatch(actor_tip[1][0]) and
            type(actor_tip[1][1]) is int and actor_tip[1][1]>0 and _id(actor_tip[1][2])))):
            raise PatchError('invalid signed behavior Actor journal position')
        head['actor_tip']=actor_tip
    if pub is not None:
        from .roomid import identity_from_public_b64
        identity_from_public_b64(pub,origin)
        head['pub']=pub
    signing=DOMAIN+_canonical_json(head)
    head['signature']=signer(len(signing),hashlib.sha256(signing).hexdigest())
    header=_canonical_json(head)
    wire=MAGIC+struct.pack('!I',len(header))+header+payload
    if len(header)>MAX_HEAD or len(wire)>MAX_WIRE:
        raise PatchError('trajectory event exceeds bounded frame')
    return wire


def parse_wire(wire,genesis,allowed_epochs,trusted=None,*,reward_mode,max_length=1024):
    if not isinstance(wire,bytes) or not wire.startswith(MAGIC) or not len(MAGIC)+4<len(wire)<=MAX_WIRE:
        raise PatchError('invalid trajectory frame magic/size')
    count=struct.unpack('!I',wire[len(MAGIC):len(MAGIC)+4])[0]
    if not 0<count<=MAX_HEAD or len(wire)<=len(MAGIC)+4+count:
        raise PatchError('invalid trajectory header length')
    try:
        head=json.loads(wire[len(MAGIC)+4:len(MAGIC)+4+count])
        payload=wire[len(MAGIC)+4+count:]
        data=json.loads(payload)
    except (ValueError,UnicodeDecodeError,TypeError) as exc:
        raise PatchError('invalid trajectory JSON') from exc
    if not isinstance(head,dict) or set(head)-{'origin','seq','genesis','actor_epoch','actor_tip','size','sha256','pub','signature'} or not {'origin','seq','genesis','actor_epoch','size','sha256','signature'}<=set(head):
        raise PatchError('bad trajectory authenticated metadata')
    origin=head['origin']; seq=head['seq']
    if (not isinstance(origin,str) or not PEER_ID.fullmatch(origin) or type(seq) is not int or not 1<=seq<2**63 or
        head['genesis']!=genesis or not _id(head['actor_epoch']) or head['actor_epoch'] not in allowed_epochs or
        type(head['size']) is not int or head['size']!=len(payload) or
        not _id(head['sha256']) or head['sha256']!=hashlib.sha256(payload).hexdigest()):
        raise PatchError('trajectory commitment / lineage mismatch')
    if 'actor_tip' in head:
        tip=head['actor_tip']
        if (not isinstance(tip,list) or len(tip)!=2 or type(tip[0]) is not int or
            tip[0]<0 or not (tip[1] is None or (isinstance(tip[1],list) and
            len(tip[1])==3 and isinstance(tip[1][0],str) and PEER_ID.fullmatch(tip[1][0]) and
            type(tip[1][1]) is int and tip[1][1]>0 and _id(tip[1][2])))):
            raise PatchError('invalid signed behavior Actor journal position')
    if 'pub' in head:
        from .roomid import identity_from_public_b64
        verify=identity_from_public_b64(head['pub'],origin)
    else:
        verify=(trusted or {}).get(origin)
    if verify is None:raise PatchError('unauthenticated trajectory origin')
    signed={k:v for k,v in head.items() if k!='signature'}
    signing=DOMAIN+_canonical_json(signed)
    verify({'size':len(signing),'sha256':hashlib.sha256(signing).hexdigest(),'sig':head['signature']})
    return origin,seq,validate_rollout(data,max_length=max_length,reward_mode=reward_mode)


class TrajectoryGossip:
    """Durable arrival + replay with atomic local value/AdamW state + receipt.

    Every recipient has its OWN changing Actor and therefore its own value
    estimator. Snapshot+receipt are a single SQLite transaction. A crash
    after compute but before commit reloads the prior snapshot; it does not
    apply a permanently half-committed value update twice.
    """
    def __init__(self,path,root_genesis,mixer,self_id,signer,*,pub=None,trusted=None,allowed_epochs,actor_epoch,max_bytes=256*1024**2,_lane_tag="v80"):
        if not _id(root_genesis):raise PatchError('invalid rollout room genesis')
        self.genesis=hashlib.sha256(('kagami-signed-rollout-'+_lane_tag+':'+root_genesis).encode()).hexdigest()
        self.journal=EventJournal(path,self.genesis,max_bytes=max_bytes)
        self.mixer=mixer;self.self_id=self_id;self.signer=signer;self.pub=pub;self.trusted=trusted or {}
        self.allowed_epochs=allowed_epochs;self.actor_epoch=actor_epoch
        self.reward_mode=mixer.args.ppo_reward;self.max_length=int(mixer.args.max_length)
        self.journal.db.execute('CREATE TABLE IF NOT EXISTS trajectory_applied (origin TEXT NOT NULL, seq INTEGER NOT NULL, PRIMARY KEY(origin,seq))')
        self.journal.db.execute('CREATE TABLE IF NOT EXISTS trajectory_state (id INTEGER PRIMARY KEY CHECK(id=1), tensors BLOB NOT NULL, digest TEXT NOT NULL, critic_steps INTEGER NOT NULL)')
        self.journal.db.commit()
        try:
            self._restore()
        except BaseException:
            self.journal.close()
            raise

    def _params(self):
        from .distributed_critic_v78 import critic_params
        return critic_params(self.mixer)

    def _snapshot(self):
        tensors=_record('c',self._params(),self.mixer.critic_optim)
        blob=safe_save(tensors)
        if len(blob)>MAX_STATE:raise PatchError('trajectory replay state exceeds 8MiB')
        return blob,hashlib.sha256(blob).hexdigest(),int(self.mixer.critic_steps)

    def _restore(self):
        row=self.journal.db.execute('SELECT tensors,digest,critic_steps FROM trajectory_state WHERE id=1').fetchone()
        if row is None:
            if self.journal.db.execute('SELECT COUNT(*) FROM trajectory_applied').fetchone()[0]:
                raise PatchError('trajectory receipts exist without checkpoint')
            self.persist()
            return
        blob,digest,steps=row
        if hashlib.sha256(blob).hexdigest()!=digest or len(blob)>MAX_STATE:
            raise PatchError('corrupt local trajectory replay checkpoint')
        try:tensors=safe_load(blob)
        except Exception as exc:raise PatchError('invalid trajectory checkpoint tensors') from exc
        params=self._params()
        # Validate every parameter and AdamW state before changing anything.
        expected=set();prepared=[]
        for i,p in enumerate(params):
            key=f'c.p.{i}';expected.add(key)
            if key not in tensors:raise PatchError('missing replay value parameter')
            w=tensors[key]
            if w.shape!=p.shape or w.dtype!=p.dtype or not bool(torch.isfinite(w).all()):
                raise PatchError('invalid replay value tensor')
            moment={};seen=[]
            for name in ('step','exp_avg','exp_avg_sq'):
                k=f'c.o.{i}.{name}'
                if k in tensors:
                    expected.add(k);seen.append(name);v=tensors[k]
                    if not bool(torch.isfinite(v).all()):raise PatchError('nonfinite replay moment')
                    if name=='step' and (v.numel()!=1 or float(v.item())<0):raise PatchError('invalid step')
                    if name!='step' and (v.shape!=p.shape or v.dtype!=p.dtype):raise PatchError('bad moment layout')
                    moment[name]=v
            if seen and len(seen)!=3:raise PatchError('incomplete replay optimizer')
            prepared.append((p,w.to(p.device),{k:v.to('cpu' if k=='step' else p.device) for k,v in moment.items()}))
        if set(tensors)!=expected or type(steps) is not int or steps<0:raise PatchError('unexpected replay checkpoint content')
        with torch.no_grad():
            self.mixer.critic_optim.state.clear()
            for p,w,states in prepared:
                p.copy_(w);p.grad=None
                if states:self.mixer.critic_optim.state[p]=states
        self.mixer.critic_steps=steps
        self.mixer.cached=None;self.mixer.reuses=0

    def persist(self,receipt=None):
        state=self._snapshot()
        with self.journal.db:
            self.journal.db.execute('INSERT OR REPLACE INTO trajectory_state(id,tensors,digest,critic_steps) VALUES(1,?,?,?)',state)
            if receipt is not None:
                self.journal.db.execute('INSERT INTO trajectory_applied(origin,seq) VALUES(?,?)',receipt)

    def local(self,rollout,*,actor_tip=None):
        validate_rollout(rollout,max_length=self.max_length,reward_mode=self.reward_mode)
        seq=self.journal.next_seq(self.self_id)
        wire=make_wire(self.self_id,seq,self.genesis,self.actor_epoch,rollout,self.signer,pub=self.pub,actor_tip=actor_tip)
        if not self.journal.append(self.self_id,seq,wire):raise PatchError('duplicate local trajectory')
        self.persist((self.self_id,seq))
        return wire

    def ingest(self,wire):
        origin,seq,_=parse_wire(wire,self.genesis,self.allowed_epochs,self.trusted,
                    reward_mode=self.reward_mode,max_length=self.max_length)
        found=self.journal.has(origin,seq)
        if found:
            if found[0]!=hashlib.sha256(wire).hexdigest():raise PatchError('trajectory equivocation')
            return False
        if not self.journal.append(origin,seq,wire):raise PatchError('trajectory index race')
        return True

    def pending(self,limit=4):
        rows=self.journal.db.execute('''SELECT e.origin,e.seq FROM events e LEFT JOIN trajectory_applied a
          ON (a.origin=e.origin AND a.seq=e.seq) WHERE a.origin IS NULL
          ORDER BY e.origin,e.seq LIMIT ?''',(limit,)).fetchall()
        return [(origin,seq,self.journal.get(origin,seq)) for origin,seq in rows]

    def apply_pending(self,model,device,limit=1):
        n=0
        for origin,seq,wire in self.pending(limit):
            if wire is None:raise PatchError('missing durable trajectory event')
            _,_,data=parse_wire(wire,self.genesis,self.allowed_epochs,self.trusted,
                                reward_mode=self.reward_mode,max_length=self.max_length)
            # Do not mutate state or acknowledge on unrepresentable IS ratios.
            # The caller must be under adapter.lock and the PPO step fence.
            before=self._snapshot()
            try:
                self.mixer.train_imported_rollout(model,device,data)
                self.persist((origin,seq))
            except BaseException:
                # Recover both weights AND optimizer on any failure, including
                # SQLite commit errors, without marking the event as applied.
                self.journal.db.execute('INSERT OR REPLACE INTO trajectory_state(id,tensors,digest,critic_steps) VALUES(1,?,?,?)',before)
                self.journal.db.commit()
                self._restore()
                raise
            n+=1
        return n

    def frontiers(self):return self.journal.frontiers()
    def need(self,heads,limit=16):
        if not isinstance(heads,list) or len(heads)>128:raise PatchError('malformed trajectory frontiers')
        needed=[]
        for item in heads:
            if (not isinstance(item,list) or len(item)!=2 or not isinstance(item[0],str) or
                not PEER_ID.fullmatch(item[0]) or type(item[1]) is not int or not 1<=item[1]<2**63):
                raise PatchError('bad trajectory frontier identity')
            origin,end=item
            for key in self.journal.missing_range(origin,1,min(end, self.journal.db.execute(
                'SELECT COALESCE(MAX(seq),0) FROM events WHERE origin=?',(origin,)).fetchone()[0]+64)):
                needed.append(key)
                if len(needed)>=limit:return needed
        return needed
    def close(self):self.journal.close()
