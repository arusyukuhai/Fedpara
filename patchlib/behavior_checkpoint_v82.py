"""Kagami v82: exact LoHA/LoRA behavior-Actor reconstruction.

Unlike v81, remote Overlay contributions and ReLoRA anchors are NOT assumed
identical on every peer. Their actual factor bytes are part of the signed,
content-addressed behavior checkpoint. The frozen original weights/non-adapter
buffers still require an independently pinned *identical* base; a different
base is deliberately refused rather than silently substituted.

This attests a replayable policy distribution, NOT honest random sampling.
The caller must serialize capture/replay with adapter.lock and all model writes.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import save as safe_save, load as safe_load

from .core import PatchError, Overlay, FrozenDelta, _canonical_json
from .behavior_checkpoint_v81 import (
    PAGE_BYTES, MAX_CHECKPOINT_PAGES, MAX_CHECKPOINT_BYTES,
    LOGP_ATOL, _tensor_digest, digest,
)

FORMAT='actor-loha-overlay-complete-v82'
MAX_GROUPS=8192
MAX_REMOTE_PER_GROUP=64  # guard against unbounded topology state / memory
MAX_BASE_REPLAY_BYTES=256*1024**2  # limited in-process fallback; large bases need shadow model


def _overlays(model):
    # An offloaded/virtual base can have a different effective value than its
    # PyTorch original Parameter. Never attest the original as the sampled
    # Actor until a certified disk-source/ledger snapshot codec is available.
    if (getattr(getattr(model,'_fedpara_lazy_mutable_weights',None),
                '_fedpara_is_factor_ledger',False) or
            any(getattr(m,'_fedpara_disk_path',None) for m in model.modules())):
        raise PatchError('LoHA behavior replay requires a certified virtual/disk base snapshot; streamed or factor-ledger base not supported by v82')
    # named_modules() deduplicates tied parametrizations and shared Overlays.
    groups=[(path, mod) for path, mod in model.named_modules() if isinstance(mod, Overlay)]
    if len(groups)>MAX_GROUPS:
        raise PatchError('too many LoHA overlay groups for behavior replay')
    if not groups or len({id(m) for _,m in groups})!=len(groups):
        raise PatchError('missing or aliased LoHA overlay registration')
    return groups


def _base_and_live(model,groups):
    excluded_p={id(p) for _,mod in groups for p in mod.parameters()}
    excluded_b={id(b) for _,mod in groups for b in mod.buffers()}
    live=[];base=[]
    for name,p in model.named_parameters():
        if p.requires_grad:
            live.append((name,p))
        elif id(p) not in excluded_p:
            base.append(('parameter:'+name,p))
    for name,b in model.named_buffers():
        if id(b) not in excluded_b:
            base.append(('buffer:'+name,b))
    if not live:raise PatchError('no trainable Actor parameters for behavior checkpoint')
    for _,ov in groups:
        if any(not p.requires_grad for p in ov.local.parts):
            raise PatchError('frozen LoHA local parts are unsupported for behavior replay')
    return live,base


def _schema(path,mod):
    spec=mod.local
    if spec.kind not in ('lora','loha') or not 1<=spec.rank<=256 or not math.isfinite(spec.alpha):
        raise PatchError('invalid LoHA group specification')
    remotes=sorted(mod.remotes.keys())
    if len(remotes)>MAX_REMOTE_PER_GROUP or any(not isinstance(key,str) or
           len(key)!=64 or any(c not in '0123456789abcdef' for c in key) for key in remotes):
        raise PatchError('invalid or excessive LoHA remote overlays')
    if hasattr(mod,'anchor') and not isinstance(mod.anchor,FrozenDelta):
        raise PatchError('unknown LoHA anchor type')
    if any(not isinstance(r,FrozenDelta) for r in mod.remotes.values()):
        raise PatchError('unknown LoHA remote type')
    return {'path':path, 'shape':list(spec.shape), 'kind':spec.kind,
            'rank':spec.rank,'alpha':spec.alpha,'dense':spec.dense,
            'anchor':hasattr(mod,'anchor'),'remotes':remotes}


def _frozen_parts(mod,member):
    if (member.kind!=mod.local.kind or member.rank!=mod.local.rank or
            member.alpha!=mod.local.alpha or member.dense!=mod.local.dense or
            tuple(member.shape)!=tuple(mod.local.shape)):
        raise PatchError('remote/anchor LoHA tensor factor specification mismatch')
    parts=[getattr(member,'part'+str(i)) for i in range(member.nparts)]
    if len(parts)!=len(mod.local.parts):
        raise PatchError('incorrect LoHA frozen factor count')
    return parts


def capture(model):
    """Call before first sampled token, inside the model/adapter write fence."""
    groups=_overlays(model)
    live,base=_base_and_live(model,groups)
    tensors={n:p.detach().to('cpu').contiguous().clone() for n,p in live}
    if len(tensors)!=len(live):raise PatchError('duplicate trainable Actor tensor name')
    descriptors=[]
    for j,(path,mod) in enumerate(groups):
        entry=_schema(path,mod)
        for category,member in (([('anchor',mod.anchor)] if entry['anchor'] else [])+
                                [('remote:'+key,mod.remotes[key]) for key in entry['remotes']]):
            for k,t in enumerate(_frozen_parts(mod,member)):
                name=f'v82.overlay.{j}.{category}.{k}'
                if name in tensors:raise PatchError('LoHA snapshot key collision')
                tensors[name]=t.detach().to('cpu').contiguous().clone()
        descriptors.append(entry)
    if not all(bool(torch.isfinite(t).all()) for t in tensors.values()):
        raise PatchError('nonfinite LoHA behavior tensor')
    meta={'version':FORMAT,'base_sha256':_tensor_digest(base),
          'live_weights':sorted(n for n,_ in live),'overlays':descriptors,
          'weights':sorted(tensors)}
    blob=safe_save(tensors,metadata={'behavior':_canonical_json(meta).decode('utf8')})
    if len(blob)>MAX_CHECKPOINT_BYTES:
        raise PatchError('LoHA behavior checkpoint exceeds 64MiB bound; snapshot sharding required')
    return blob



def capture_frozen_base(model):
    """Optional content-addressed old-epoch base artifact, NOT per-rollout wire.

    Intended for a pinned actor-epoch snapshot/archive. The caller persists it
    once per distinct SHA-256 base and distributes via a separate certified
    snapshot/manifest path. Never attach this dense blob to each KTR82 rollout.
    """
    groups=_overlays(model)
    _,base=_base_and_live(model,groups)
    base_hash=_tensor_digest(base)
    tensors={name:t.detach().cpu().contiguous().clone() for name,t in base}
    wire=safe_save(tensors,metadata={'base_sha256':base_hash,
                                    'version':'kagami-frozen-actor-base-v82'})
    if len(wire)>MAX_BASE_REPLAY_BYTES:
        raise PatchError('historical full base requires external sharded epoch snapshot')
    return base_hash,wire


def _validated_base(wire,base,expected_hash):
    if not isinstance(wire,bytes) or not 8<len(wire)<=MAX_BASE_REPLAY_BYTES:
        raise PatchError('historical base blob missing or exceeds bounded replay budget')
    header_len=int.from_bytes(wire[:8],'little')
    if not 0<header_len<=1024*1024 or header_len>len(wire)-8:
        raise PatchError('invalid historical base artifact header')
    try:
        h=json.loads(wire[8:8+header_len]);meta=h['__metadata__']
        tensors=safe_load(wire)
    except (ValueError,TypeError,KeyError,RuntimeError,UnicodeDecodeError) as exc:
        raise PatchError('invalid historical base artifact') from exc
    if (not isinstance(meta,dict) or meta.get('version')!='kagami-frozen-actor-base-v82'
            or meta.get('base_sha256')!=expected_hash or
            set(tensors)!={name for name,_ in base} or
            _tensor_digest(tensors.items())!=expected_hash):
        raise PatchError('historical base contents fail commitment verification')
    result=[]
    for name,param in base:
        value=tensors[name]
        if value.dtype!=param.dtype or value.shape!=param.shape:
            raise PatchError('historical base shape/dtype mismatch')
        result.append((param,value.to(param.device)))
    return result


class FrozenBaseCAS:
    """Local immutable CAS for optional historical base reconstruction.

    Explicitly enrolled, certified epoch snapshots only. This is storage, not a
    signed peer transfer protocol. The caller authenticates the epoch before
    enrolling or using a snapshot; do not use arbitrary web-provided base files.
    """
    def __init__(self,folder):
        self.folder=Path(folder)
        self.folder.mkdir(parents=True,exist_ok=True)

    def put(self,sha,blob):
        if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
            raise PatchError('bad historical base commitment')
        if not isinstance(blob,bytes) or len(blob)>MAX_BASE_REPLAY_BYTES:
            raise PatchError('oversized historical base object')
        if len(blob)<8:raise PatchError('truncated historical base')
        head=int.from_bytes(blob[:8],'little')
        if not 0<head<=1024*1024 or head>len(blob)-8:
            raise PatchError('invalid base object')
        # Verify manifest against the actual stored tensors, NOT SHA(blob).
        try:
            meta=json.loads(blob[8:8+head])['__metadata__']
            values=safe_load(blob)
        except (ValueError,TypeError,KeyError,RuntimeError,UnicodeDecodeError) as exc:
            raise PatchError('corrupt historical base CAS object') from exc
        if (meta.get('base_sha256')!=sha or meta.get('version')!='kagami-frozen-actor-base-v82'
                or _tensor_digest(values.items())!=sha):
            raise PatchError('historical base CAS digest mismatch')
        path=self.folder/(sha+'.safetensors')
        if path.is_file():
            if path.read_bytes()!=blob:raise PatchError('historical base equivocation')
            return path
        path_tmp=self.folder/(sha+'.'+os.urandom(8).hex()+'.tmp')
        try:
            with path_tmp.open('xb') as fh:
                fh.write(blob);fh.flush();os.fsync(fh.fileno())
            # Atomic replace is safe because content-addressed objects are
            # verified before and after retrieval; concurrent writers must
            # produce same canonical bytes (not assumed without validation).
            if path.exists() and path.read_bytes()!=blob:
                raise PatchError('base CAS concurrent equivocation')
            os.replace(path_tmp,path)
            directory_fd=os.open(self.folder,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
            try:os.fsync(directory_fd)
            finally:os.close(directory_fd)
            return path
        finally:path_tmp.unlink(missing_ok=True)

    def __call__(self,sha):
        if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
            raise PatchError('invalid frozen base SHA-256')
        p=self.folder/(sha+'.safetensors')
        return p.read_bytes() if p.is_file() and p.stat().st_size<=MAX_BASE_REPLAY_BYTES else None

def _decode(blob):
    digest(blob)
    if len(blob)<8:raise PatchError('truncated LoHA checkpoint')
    hsize=int.from_bytes(blob[:8],'little')
    if not 0<hsize<=1024*1024 or hsize>len(blob)-8:
        raise PatchError('invalid LoHA checkpoint header')
    try:
        hdr=json.loads(blob[8:8+hsize])
        meta=json.loads(hdr['__metadata__']['behavior'])
        tensors=safe_load(blob)
    except (UnicodeDecodeError,KeyError,ValueError,TypeError,RuntimeError) as exc:
        raise PatchError('invalid LoHA checkpoint serialization') from exc
    if (not isinstance(meta,dict) or set(meta)!={'version','base_sha256','live_weights','overlays','weights'}
            or meta['version']!=FORMAT or not isinstance(meta['overlays'],list)
            or not isinstance(meta['live_weights'],list) or not isinstance(meta['weights'],list)
            or meta['weights']!=sorted(tensors) or len(meta['overlays'])>MAX_GROUPS):
        raise PatchError('noncanonical LoHA checkpoint metadata')
    return meta,tensors


def _validated(model,blob,base_resolver=None):
    meta,tensors=_decode(blob)
    groups=_overlays(model)
    live,base=_base_and_live(model,groups)
    if len(groups)!=len(meta['overlays']):raise PatchError('LoHA overlay group layout mismatch')
    if meta['live_weights']!=sorted(n for n,_ in live):
        raise PatchError('LoHA live factor layout mismatch')
    base_update=[]
    if meta['base_sha256']!=_tensor_digest(base):
        if base_resolver is None:
            raise PatchError('frozen base differs; supply certified old base epoch snapshot before replay')
        old_base=base_resolver(meta['base_sha256'])
        if old_base is None:
            raise PatchError('historical frozen base unavailable (pending certified epoch snapshot)')
        base_update=_validated_base(old_base,base,meta['base_sha256'])
    staged=[]; expected=set(meta['live_weights']); overlays=[]
    for name,p in live:
        t=tensors.get(name)
        if t is None or t.shape!=p.shape or t.dtype!=p.dtype or not bool(torch.isfinite(t).all()):
            raise PatchError('invalid LoHA live factor')
        staged.append((p,t.to(p.device)))
    for j,((path,ov),desc) in enumerate(zip(groups,meta['overlays'])):
        if (not isinstance(desc,dict) or set(desc)!={'path','shape','kind','rank','alpha','dense','anchor','remotes'} or
            desc['path']!=path or desc['shape']!=list(ov.local.shape) or
            desc['kind']!=ov.local.kind or type(desc['rank']) is not int or desc['rank']!=ov.local.rank or
            type(desc['alpha']) not in (float,int) or desc['alpha']!=ov.local.alpha or
            type(desc['dense']) is not bool or desc['dense']!=ov.local.dense or
            type(desc['anchor']) is not bool or not isinstance(desc['remotes'],list) or
            len(desc['remotes'])>MAX_REMOTE_PER_GROUP or desc['remotes']!=sorted(set(desc['remotes'])) or
            any(not isinstance(key,str) or len(key)!=64 or
                any(c not in '0123456789abcdef' for c in key) for key in desc['remotes'])):
            raise PatchError('LoHA factor layout/checkpoint metadata mismatch')
        def restore_frozen(category):
            # Require exact expected factor shape and dtype before ANY mutation.
            parts=[]
            for k,param in enumerate(ov.local.parts):
                name=f'v82.overlay.{j}.{category}.{k}';expected.add(name)
                t=tensors.get(name)
                if (t is None or t.shape!=param.shape or t.dtype!=torch.float32 or
                        not bool(torch.isfinite(t).all())):
                    raise PatchError('LoHA anchor/remote factor mismatch')
                parts.append(t.to(device=param.device))
            return FrozenDelta(ov.local,parts).to(ov.local.parts[0].device)
        anchor=restore_frozen('anchor') if desc['anchor'] else None
        remotes=nn.ModuleDict({key:restore_frozen('remote:'+key) for key in desc['remotes']})
        overlays.append((ov,anchor,remotes))
    if set(tensors)!=expected:raise PatchError('unexpected LoHA snapshot tensor')
    return staged,overlays,base_update


@contextmanager
def temporarily_loaded(model,blob,*,base_resolver=None):
    """Atomic validation, then reversible installation of historical LoHA state."""
    staged,overlays,base_update=_validated(model,blob,base_resolver)
    originals=[(p,p.detach().clone()) for p,_ in staged]
    old_base=[(p,p.detach().clone()) for p,_ in base_update]
    old_overlays=[(ov,getattr(ov,'anchor',None),hasattr(ov,'anchor'),ov.remotes)
                  for ov,_,_ in overlays]
    # Preserve heterogeneous train/eval state (e.g. frozen dropout blocks).
    modes=[(m,m.training) for m in model.modules()]
    try:
        with torch.no_grad():
            for p,w in base_update:p.copy_(w)
            for p,w in staged:p.copy_(w)
            for ov,anchor,remotes in overlays:
                if anchor is None:
                    if hasattr(ov,'anchor'):del ov.anchor
                else:ov.anchor=anchor
                ov.remotes=remotes
        model.eval()
        yield
    finally:
        with torch.no_grad():
            for p,w in originals:p.copy_(w)
            for p,w in old_base:p.copy_(w)
            for ov,anchor,present,remotes in old_overlays:
                if present:ov.anchor=anchor
                elif hasattr(ov,'anchor'):del ov.anchor
                ov.remotes=remotes
        for m,mode in modes:m.training=mode


def replay_logps(model,blob,rollout,args,device,*,base_resolver=None):
    """Verify old mu with the LoHA state restored, not with today's peer state."""
    from torch.nn import functional as F
    from .mixed_learning_v65 import _logits
    prompt=rollout['prompt'];actions=rollout['tokens'];n=len(actions)
    ids=torch.tensor([prompt+actions],device=device,dtype=torch.long)
    with temporarily_loaded(model,blob,base_resolver=base_resolver),torch.no_grad():
        out=_logits(model,ids,args,start=len(prompt)-1,length=n)
        actual=F.log_softmax(out.float(),dim=-1).gather(
            -1,torch.tensor(actions,device=device).view(1,n,1)).view(n)
    claimed=torch.tensor(rollout['behavior_logp'],dtype=torch.float64,device=device)
    if (actual.numel()!=n or not bool(torch.isfinite(actual).all()) or
            not bool(torch.all(torch.abs(actual.double()-claimed)<=LOGP_ATOL))):
        raise PatchError('behavior probabilities disagree with LoHA checkpoint')
    return actual.detach().cpu().tolist()
