"""Opt-in signed, durable, additive P2P gossip for PPO value heads.

v77 is deliberately a SEPARATE journal/namespace from Actor FedSum. All
critic parameters use the same pinned initial weights, signed per-step deltas
and deterministic single-application replay. AdamW moments stay local, and are
reset whenever foreign value updates enter the current critic.

It is NOT a fully atomic multi-journal checkpoint / V-trace algorithm.
"""
from __future__ import annotations
import hashlib
import json
import struct
from pathlib import Path
from collections import OrderedDict

import torch
from safetensors.torch import save as save_tensors, load as load_tensors
from .core import PatchError, PEER_ID, _canonical_json
from .elastic import EventJournal

MAGIC=b'KVC77\0'
MAX_EVENT=2*1024*1024
MAX_HEAD=4096
MAX_PARAMS=128
MAX_TENSORS_BYTES=1024*1024


def critic_params(mixer):
    if mixer is None or mixer.critic is None:
        raise PatchError('a PPO critic is required for signed value gossip')
    p=(mixer.critic.trainable_parameters() if getattr(mixer.args,'critic_kind','mlp')=='hf'
       else mixer.critic.parameters())
    p=list(p)
    if not p or len(p)>MAX_PARAMS or len({id(x) for x in p})!=len(p):
        raise PatchError('invalid critic trainable parameter layout')
    if any(not x.dtype.is_floating_point or x.numel()==0 for x in p):
        raise PatchError('Critic gossip requires nonempty floating point tensors')
    return p


def critic_identity(mixer, actor_genesis):
    from .paired_checkpoint_v75 import _identity
    data={'domain':'critic-signed-additive-v77','actor_genesis':actor_genesis,
          'critic':_identity(mixer),'architecture':[(list(x.shape),str(x.dtype)) for x in critic_params(mixer)]}
    return hashlib.sha256(_canonical_json(data)).hexdigest()


def pack(origin,seq,genesis,deltas,signer,public_key=None,actor_tip=None):
    if not PEER_ID.fullmatch(origin) or type(seq) is not int or seq<1 or seq>=2**63:
        raise PatchError('invalid critic event identity')
    if not isinstance(genesis,str) or len(genesis)!=64:
        raise PatchError('invalid critic genesis')
    if not 1<=len(deltas)<=MAX_PARAMS:
        raise PatchError('invalid critic tensor count')
    tensors={}
    size=0
    for i,t in enumerate(deltas):
        if not isinstance(t,torch.Tensor) or not t.dtype.is_floating_point or not bool(torch.isfinite(t).all()):
            raise PatchError('nonfinite/nonfloating critic update')
        # CPU float32 values preserve signed additive semantics independent of
        # source BF16/FP16; on apply always round only once per incoming event.
        val=t.detach().to('cpu',dtype=torch.float32).contiguous()
        size+=val.numel()*val.element_size()
        tensors[f'd.{i}']=val
    if size>MAX_TENSORS_BYTES:
        raise PatchError('critic event exceeds 1MiB tensor budget')
    payload=save_tensors(tensors)
    sha=hashlib.sha256(payload).hexdigest()
    if not callable(signer):raise PatchError('critic gossip requires signed identity')
    head={'origin':origin,'seq':seq,'genesis':genesis,'size':len(payload),
          'sha256':sha}
    if public_key is not None:
        from .roomid import identity_from_public_b64
        identity_from_public_b64(public_key,origin)
        head['pub']=public_key
    if actor_tip is not None:
        if (not isinstance(actor_tip,list) or len(actor_tip)!=2 or
                type(actor_tip[0]) is not int or actor_tip[0]<0):
            raise PatchError('invalid actor lineage marker')
        head['actor_tip']=actor_tip
    # Domain-separated signed commitment covers all metadata + payload hash.
    signing=b'KAGAMI-V77-CRITIC\0'+_canonical_json(head)
    head['signature']=signer(len(signing),hashlib.sha256(signing).hexdigest())
    raw=_canonical_json(head)
    if len(raw)>MAX_HEAD or len(payload)+len(raw)+len(MAGIC)+4>MAX_EVENT:
        raise PatchError('critic event too large')
    return MAGIC+struct.pack('!I',len(raw))+raw+payload


def unpack(wire,genesis,params,trusted=None,unsafe=False):
    if not isinstance(wire,bytes) or not wire.startswith(MAGIC) or len(wire)>MAX_EVENT:
        raise PatchError('bad critic wire magic/size')
    if len(wire)<len(MAGIC)+4:
        raise PatchError('truncated critic header')
    hsize=struct.unpack('!I',wire[len(MAGIC):len(MAGIC)+4])[0]
    if not 0<hsize<=MAX_HEAD or len(wire)<=len(MAGIC)+4+hsize:
        raise PatchError('bad critic header bounds')
    try:head=json.loads(wire[len(MAGIC)+4:len(MAGIC)+4+hsize])
    except (ValueError,UnicodeDecodeError) as e:raise PatchError('invalid critic JSON') from e
    payload=wire[len(MAGIC)+4+hsize:]
    if (not isinstance(head,dict) or
            set(head)-{'origin','seq','genesis','size','sha256','signature','pub','actor_tip'} or
            not {'origin','seq','genesis','size','sha256','signature'}<=set(head)):
        raise PatchError('unknown/missing critic header fields')
    origin=head['origin'];seq=head['seq']
    if (not isinstance(origin,str) or not PEER_ID.fullmatch(origin) or
            type(seq) is not int or not 1<=seq<2**63 or head['genesis']!=genesis or
            type(head['size']) is not int or head['size']!=len(payload) or
            not isinstance(head['sha256'],str) or len(head['sha256'])!=64 or
            hashlib.sha256(payload).hexdigest()!=head['sha256']):
        raise PatchError('critic payload identity/genesis/checksum mismatch')
    if 'actor_tip' in head:
        tip=head['actor_tip']
        if (not isinstance(tip,list) or len(tip)!=2 or
                type(tip[0]) is not int or not 0<=tip[0]<2**63 or
                not (tip[1] is None or (isinstance(tip[1],list) and len(tip[1])==3
                     and isinstance(tip[1][0],str) and isinstance(tip[1][2],str)))):
            raise PatchError('invalid critic actor lineage')
    if not unsafe:
        if 'pub' in head:
            from .roomid import identity_from_public_b64
            verify=identity_from_public_b64(head['pub'],origin)
        else:
            verify=(trusted or {}).get(origin)
        if verify is None:raise PatchError('untrusted Critic origin')
        bare={k:v for k,v in head.items() if k!='signature'}
        signing=b'KAGAMI-V77-CRITIC\0'+_canonical_json(bare)
        verify({'size':len(signing),'sha256':hashlib.sha256(signing).hexdigest(),
                'sig':head['signature']})
    try:tensors=load_tensors(payload)
    except Exception as e:raise PatchError('invalid critic safetensors') from e
    if len(tensors)!=len(params) or set(tensors)!={f'd.{i}' for i in range(len(params))}:
        raise PatchError('critic parameter count mismatch')
    result=[]
    for i,p in enumerate(params):
        val=tensors[f'd.{i}']
        if (val.shape!=p.shape or val.dtype!=torch.float32 or
                not bool(torch.isfinite(val).all())):
            raise PatchError('critic delta shape, dtype or finite check failed')
        if val.numel()>0 and float(val.abs().max())>10:
            raise PatchError('excessive critic delta magnitude')
        result.append(val)
    return origin,seq,result


class CriticGossip:
    """One writer per peer, signed independent critic log + bounded gossip.

    The caller MUST begin with a common base value head, then replay. The
    receiving thread must serialize with local training on the actor lock.
    """
    def __init__(self, path, actor_genesis, mixer, self_id, *, signer, public_key=None,
                 trusted=None, unsafe=False, max_bytes=2*1024**3, lock=None, baseline=None):
        if signer is None and not unsafe:raise PatchError('Critic P2P requires a signer')
        self.mixer=mixer
        self.params=critic_params(mixer)
        self._original=[x.detach().to('cpu').clone() for x in (baseline if baseline is not None else self.params)]
        if len(self._original)!=len(self.params) or any(a.shape!=b.shape or a.dtype!=b.dtype for a,b in zip(self._original,self.params)):
            raise PatchError('critic baseline layout mismatch')
        self._initial_hash=self.base_digest()
        self.genesis=hashlib.sha256((critic_identity(mixer,actor_genesis)+self._initial_hash).encode('ascii')).hexdigest()
        self.journal=EventJournal(Path(path),self.genesis,max_bytes=max_bytes)
        self.self_id=self_id
        self.signer=signer
        self.public_key=public_key
        self.trusted=trusted or {}
        self.unsafe=unsafe
        self.lock=lock
        self._applied=set()
        self._baseline=[p.detach().clone() for p in self._original]
        try:
            self._install_base_hash()
            self.replay()
        except BaseException:
            self.journal.close()
            raise

    def base_digest(self):
        h=hashlib.sha256()
        for p in self._original:
            t=p.to(torch.float32).contiguous()
            h.update(str(tuple(t.shape)).encode());h.update(memoryview(t.reshape(-1).view(torch.uint8).numpy()))
        return h.hexdigest()

    def _install_base_hash(self):
        row=self.journal.db.execute("SELECT value FROM meta WHERE name='critic_initial_sha'").fetchone()
        if row and row[0]!=self._initial_hash:
            raise PatchError('different initial Critic head for the same signed journal')
        with self.journal.db:
            self.journal.db.execute("INSERT OR IGNORE INTO meta(name,value) VALUES('critic_initial_sha',?)",(self._initial_hash,))

    def _apply(self,values):
        # Verify entire resulting state before mutating ANY parameter.
        proposal=[]
        for p,d in zip(self.params,values):
            value=p.detach().float()+d.to(p.device)
            if not bool(torch.isfinite(value).all()):raise PatchError('critic update overflow')
            proposal.append(value.to(p.dtype))
        with torch.no_grad():
            for p,value in zip(self.params,proposal):p.copy_(value)

    def replay(self):
        # Caller supplies a NEW baseline, not a partially saved previous
        # version. Each event is authenticated before it touches the head.
        with torch.no_grad():
            for p,initial in zip(self.params,self._baseline):p.copy_(initial)
        self._applied.clear()
        for wire in self.journal.events():
            origin,seq,deltas=unpack(wire,self.genesis,self.params,self.trusted,self.unsafe)
            key=(origin,seq)
            if key in self._applied:raise PatchError('duplicate durable critic event')
            self._apply(deltas)
            self._applied.add(key)
        self.reset_optimizer()
        return len(self._applied)

    def reset_optimizer(self):
        # AdamW moments of one local gradient history do not represent a
        # synchronized optimizer state after foreign signed deltas.
        if self.mixer.critic_optim is not None:
            self.mixer.critic_optim.state.clear()
            self.mixer.critic_optim.zero_grad(set_to_none=True)
        self.mixer.cached=None
        self.mixer.reuses=0

    def record_local(self,before,actor_tip=None):
        if len(before)!=len(self.params):raise PatchError('critic before/after layout changed')
        delta=[]
        for earlier,current in zip(before,self.params):
            if earlier.shape!=current.shape:raise PatchError('critic local parameter layout changed')
            difference=current.detach().float().cpu()-earlier.detach().float().cpu()
            if not bool(torch.isfinite(difference).all()):raise PatchError('nonfinite local critic update')
            delta.append(difference)
        seq=self.journal.next_seq(self.self_id)
        wire=pack(self.self_id,seq,self.genesis,delta,self.signer,self.public_key,actor_tip)
        # Already applied by local optimizer; commit ONLY, never reapply.
        if not self.journal.append(self.self_id,seq,wire):
            raise PatchError('duplicate local critic update')
        self._applied.add((self.self_id,seq))
        return wire

    def ingest(self,wire):
        origin,seq,delta=unpack(wire,self.genesis,self.params,self.trusted,self.unsafe)
        key=(origin,seq)
        if key in self._applied:
            found=self.journal.has(origin,seq)
            if not found or found[0]!=hashlib.sha256(wire).hexdigest():
                raise PatchError('critic equivocation')
            return False
        # Durable before apply, so interruption permits clean baseline replay.
        self.journal.append(origin,seq,wire)
        self._apply(delta)
        self._applied.add(key)
        if origin!=self.self_id:self.reset_optimizer()
        return True

    def need(self,heads,limit=16):
        if not isinstance(heads,list) or len(heads)>128:raise PatchError('invalid critic frontier')
        result=[]
        for item in heads:
            if (not isinstance(item,list) or len(item)!=2 or
                    not isinstance(item[0],str) or not PEER_ID.fullmatch(item[0]) or
                    type(item[1]) is not int or not 1<=item[1]<2**63):
                raise PatchError('malformed critic frontier')
            # Request missing *gaps*, not merely the new maximum.
            count,local=self.journal.db.execute('SELECT COUNT(*),COALESCE(MAX(seq),0) FROM events WHERE origin=?',(item[0],)).fetchone()
            if count==local:
                first=int(local)+1
            else:
                # First missing ordinal: COUNT(ids<=mid)==mid exactly when
                # the entire 1..mid prefix exists. Indexed binary search.
                low,high=1,int(local)
                while low<high:
                    mid=(low+high)//2
                    present=self.journal.db.execute('SELECT COUNT(*) FROM events WHERE origin=? AND seq<=?',(item[0],mid)).fetchone()[0]
                    if present==mid:low=mid+1
                    else:high=mid
                first=low

            last=min(int(item[1]),first+63)
            if first>last:continue
            for key in self.journal.missing_range(item[0],first,last):
                result.append(key)
                if len(result)>=limit:return result
        return result

    def close(self):self.journal.close()
