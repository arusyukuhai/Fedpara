"""Data-only, crash-atomic HF value-head and AdamW state; pinned HF base is reloaded from Hub."""
from __future__ import annotations
import json
import os
import secrets
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file,load_file
from .hf_value_critic_v69 import HFValueError


def snapshot_file(root):return Path(root)/'critic-hf-v69.safetensors'


def _identity(critic,identity):
    return json.dumps({'spec':critic.spec,'head':critic.head_name,'room':str(identity)},sort_keys=True,separators=(',',':'))


def save(root,critic,optimizer,*,identity,steps):
    folder=Path(root).expanduser()
    if folder.is_symlink():raise HFValueError('Symlink Critic directory forbidden')
    folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    target=snapshot_file(folder)
    tensors={'h.'+k:v for k,v in critic.snapshot_weights().items()}
    params=critic.trainable_parameters()
    for i,p in enumerate(params):
        state=optimizer.state.get(p,{})
        for k in ('step','exp_avg','exp_avg_sq'):
            if isinstance(state.get(k),torch.Tensor):tensors[f'o.{i}.{k}']=state[k].detach().cpu().contiguous().clone()
    if not tensors or any(not torch.isfinite(v).all() for v in tensors.values()):
        raise HFValueError('Invalid HF Critic checkpoint tensors')
    if sum(x.numel()*x.element_size() for x in tensors.values())>128*1024**2:
        raise HFValueError('HF Critic head+optimizer exceeds 128MiB; reduce head size')
    tmp=folder/('.hf-critic-'+secrets.token_hex(10)+'.tmp')
    try:
        save_file(tensors,str(tmp),metadata={'format':'kagami-hf-value-head-v69','identity':_identity(critic,identity),'steps':str(steps)})
        with tmp.open('rb') as f:os.fsync(f.fileno())
        os.replace(tmp,target)
        d=os.open(folder,os.O_RDONLY)
        try:os.fsync(d)
        finally:os.close(d)
    finally:tmp.unlink(missing_ok=True)


def load(path,critic,optimizer,*,identity):
    path=Path(path)
    if not path.is_file() or path.stat().st_size>129*1024**2:
        raise HFValueError('HF Critic checkpoint missing or oversized')
    with safe_open(str(path),framework='pt',device='cpu') as f:meta=f.metadata() or {}
    if meta.get('format')!='kagami-hf-value-head-v69' or meta.get('identity')!=_identity(critic,identity):
        raise HFValueError('HF Critic checkpoint base revision/head/room mismatch')
    weights=load_file(str(path),device='cpu')
    expected={'h.'+k:v for k,v in critic.snapshot_weights().items()}
    params=critic.trainable_parameters()
    allowed=set(expected)
    for key,v in expected.items():
        if key not in weights or weights[key].shape!=v.shape or weights[key].dtype!=v.dtype:
            raise HFValueError('HF Critic head state mismatch')
    for i,p in enumerate(params):
        present={k for k in ('step','exp_avg','exp_avg_sq') if f'o.{i}.{k}' in weights}
        if present and present!={'step','exp_avg','exp_avg_sq'}:
            raise HFValueError('Incomplete HF Critic AdamW state')
        for k in present:
            key=f'o.{i}.{k}';v=weights[key];allowed.add(key)
            if k=='step':
                if v.numel()!=1 or float(v.item())<0:raise HFValueError('Invalid AdamW step')
            elif v.shape!=p.shape or v.dtype!=p.dtype:raise HFValueError('HF Critic optimizer layout mismatch')
    if allowed!=set(weights) or any(not torch.isfinite(v).all() for v in weights.values()):
        raise HFValueError('Unexpected or nonfinite HF Critic state')
    steps=int(meta['steps'])
    if not 0<=steps<2**63:raise HFValueError('Invalid HF Critic checkpoint steps')
    # Validation is complete; mutate only trainable head and optimizer.
    with torch.no_grad():
        for k,p in critic.head.named_parameters():p.copy_(weights['h.'+k].to(p.device))
    optimizer.state.clear()
    for i,p in enumerate(params):
        state={k:weights[f'o.{i}.{k}'].clone().to(p.device) for k in ('step','exp_avg','exp_avg_sq') if f'o.{i}.{k}' in weights}
        if state:optimizer.state[p]=state
    return steps
