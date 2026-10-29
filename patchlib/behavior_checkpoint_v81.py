"""Byte-committed, reproducible behavior-policy adapter checkpoints.

Every trainable tensor that can affect the model forward is captured; all other
parameter and buffer bytes are committed as an immutable-base SHA-256. Replay
is permitted only on a recipient whose own frozen base is exactly identical.
This is an exact-value commitment, NOT a proof the sender sampled honestly.
"""
from __future__ import annotations
import hashlib
import json
import math
from contextlib import contextmanager

import torch
from safetensors.torch import save as safe_save, load as safe_load
from .core import PatchError, _canonical_json

MAX_CHECKPOINT_BYTES = 64 * 1024**2
MAX_CHECKPOINT_PAGES = 2048
PAGE_BYTES = 48 * 1024
LOGP_ATOL = 2e-3  # cached-vs-full forward and cross-device floating point tolerance


def _tensor_digest(items):
    h=hashlib.sha256(b'KAGAMI-V81-MODEL-TENSOR-COMMITMENT\0')
    for name, tensor in sorted(items, key=lambda x:x[0]):
        x=tensor.detach().to('cpu').contiguous()
        if x.layout != torch.strided:
            raise PatchError('non-dense actor tensor cannot be replayed')
        header=_canonical_json([name,str(x.dtype),list(x.shape)])
        h.update(len(header).to_bytes(4,'big'));h.update(header)
        h.update(memoryview(x.reshape(-1).view(torch.uint8).numpy()))
    return h.hexdigest()


def _parts(model):
    all_params=list(model.named_parameters())
    if len({id(p) for _,p in all_params})!=len(all_params):
        raise PatchError('tied/shared parameters require an explicit behavior replay contract')
    live=[(n,p) for n,p in all_params if p.requires_grad]
    frozen=[('parameter:'+n,p) for n,p in all_params if not p.requires_grad]
    frozen.extend(('buffer:'+n,b) for n,b in model.named_buffers())
    if not live:
        raise PatchError('no trainable Actor parameters to checkpoint')
    return live,frozen


def capture(model):
    """Capture under the SAME model/adapter lock, before the first rollout token."""
    live,frozen=_parts(model)
    tensors={n:p.detach().to('cpu').contiguous().clone() for n,p in live}
    if not all(bool(torch.isfinite(t).all()) for t in tensors.values()):
        raise PatchError('nonfinite sampled Actor weights')
    meta={'version':'actor-trainable-and-frozen-sha256-v81',
          'base_sha256':_tensor_digest(frozen),
          'weights':sorted(tensors)}
    blob=safe_save(tensors,metadata={'behavior':_canonical_json(meta).decode('utf8')})
    if len(blob)>MAX_CHECKPOINT_BYTES:
        raise PatchError('Actor behavior checkpoint exceeds 64MiB v81 bound')
    return blob


def digest(blob):
    if not isinstance(blob,bytes) or not 0<len(blob)<=MAX_CHECKPOINT_BYTES:
        raise PatchError('invalid behavior checkpoint size')
    return hashlib.sha256(blob).hexdigest()


def _validated(model,blob):
    if not isinstance(blob,bytes) or not 0<len(blob)<=MAX_CHECKPOINT_BYTES:
        raise PatchError('oversized behavior checkpoint')
    # safetensors.load does not expose file metadata; use the in-memory safe_open
    # equivalent via a tiny, strictly bounded header reader.
    if len(blob)<8:raise PatchError('truncated behavior checkpoint')
    size=int.from_bytes(blob[:8],'little')
    if not 0<size<=1024*1024 or size>len(blob)-8:
        raise PatchError('invalid behavior checkpoint header')
    try:
        hdr=json.loads(blob[8:8+size]);meta=json.loads(hdr['__metadata__']['behavior'])
        tensors=safe_load(blob)
    except (KeyError,ValueError,TypeError,RuntimeError) as exc:
        raise PatchError('corrupt behavior checkpoint') from exc
    if (not isinstance(meta,dict) or set(meta)!={'version','base_sha256','weights'} or
        meta['version']!='actor-trainable-and-frozen-sha256-v81' or
        not isinstance(meta['weights'],list) or meta['weights']!=sorted(tensors)):
        raise PatchError('behavior checkpoint layout mismatch')
    live,frozen=_parts(model)
    if set(tensors)!={n for n,_ in live}:
        raise PatchError('behavior Actor trainable layout differs from recipient')
    if meta['base_sha256']!=_tensor_digest(frozen):
        raise PatchError('behavior Actor frozen base differs from recipient; cannot replay')
    staged=[]
    for name,param in live:
        t=tensors[name]
        if t.dtype!=param.dtype or t.shape!=param.shape or not bool(torch.isfinite(t).all()):
            raise PatchError('behavior Actor weight dtype/shape/finite mismatch')
        staged.append((param,t.to(device=param.device)))
    return staged


@contextmanager
def temporarily_loaded(model,blob):
    """Restore ALL trainable params and model mode even when forward raises."""
    staged=_validated(model,blob)  # validate ALL before first model mutation
    original=[(p,p.detach().clone()) for p,_ in staged]
    was_training=model.training
    try:
        with torch.no_grad():
            for p,w in staged:p.copy_(w)
        model.eval()
        yield
    finally:
        with torch.no_grad():
            for p,w in original:p.copy_(w)
        model.train(was_training)


def replay_logps(model,blob,rollout,args,device):
    """Independently recompute mu(a_t | prompt,a_<t) from restored Actor."""
    from .mixed_learning_v65 import _logits
    from torch.nn import functional as F
    prompt=rollout['prompt'];actions=rollout['tokens'];n=len(actions)
    ids=torch.tensor([prompt+actions],device=device,dtype=torch.long)
    with temporarily_loaded(model,blob),torch.no_grad():
        out=_logits(model,ids,args,start=len(prompt)-1,length=n)
        actual=F.log_softmax(out.float(),dim=-1).gather(
            -1,torch.tensor(actions,device=device).view(1,n,1)).view(n)
    claimed=torch.tensor(rollout['behavior_logp'],dtype=torch.float64,device=device)
    if actual.numel()!=n or not bool(torch.isfinite(actual).all()):
        raise PatchError('nonfinite checkpoint Actor likelihoods')
    if not bool(torch.all(torch.abs(actual.double()-claimed)<=LOGP_ATOL)):
        raise PatchError('behavior probabilities disagree with reproducible Actor checkpoint')
    return actual.detach().to('cpu').tolist()
