"""Local PPO value-critic snapshot, data-only and atomically replaceable.

No torch.load/pickle, no model code execution. Critic optimizer and weights are
kept entirely outside signed peer adapter patches. This is a *local* checkpoint.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import secrets

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


class CriticStateError(ValueError):
    pass


def _valid_dir(root: Path):
    original=Path(root).expanduser()
    if original.is_symlink():
        raise CriticStateError('Critic state directory must not be a symlink')
    root=original.resolve()
    if not root.is_dir():
        root.mkdir(parents=True, mode=0o700, exist_ok=True)
    if root.is_symlink():
        raise CriticStateError('Critic state directory must not be a symlink')
    return root


def snapshot_file(root):
    return Path(root)/'critic-v66.safetensors'


def _metadata(kind, hidden, identity, steps):
    return {'format':'kagami-local-critic-v66', 'kind':str(kind),
            'hidden':str(hidden),'identity':str(identity),'steps':str(steps)}


def _tensors(critic, optimizer):
    tensors={'c.'+name:weight.detach().cpu().contiguous().clone()
             for name,weight in critic.state_dict().items()}
    for i,p in enumerate(critic.parameters()):
        state=optimizer.state.get(p,{})
        for key in ('step','exp_avg','exp_avg_sq'):
            value=state.get(key)
            if isinstance(value,torch.Tensor):
                tensors[f'o.{i}.{key}']=value.detach().cpu().contiguous().clone()
    if not tensors or any(not torch.isfinite(t).all() for t in tensors.values()):
        raise CriticStateError('Nonfinite or missing critic tensors')
    return tensors


def save(root,critic,optimizer,*,kind,hidden,identity,steps):
    destdir=_valid_dir(root)
    tmp=destdir/('.critic-'+secrets.token_hex(8)+'.tmp')
    dest=snapshot_file(destdir)
    tensors=_tensors(critic,optimizer)
    try:
        save_file(tensors,str(tmp),metadata=_metadata(kind,hidden,identity,steps))
        with tmp.open('rb') as f:os.fsync(f.fileno())
        os.replace(tmp,dest)
        fd=os.open(destdir,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def load(path,critic,optimizer,*,kind,hidden,identity,allow_initial=False):
    path=Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size>16*1024*1024:
        raise CriticStateError('Critic safetensors is missing or over 16MiB')
    try:
        with safe_open(str(path),framework='pt',device='cpu') as f:
            metadata=f.metadata() or {}
        weights=load_file(str(path),device='cpu')
        if metadata.get('format')!='kagami-local-critic-v66':
            raise CriticStateError('Not a Kagami v66 critic checkpoint')
        if metadata.get('kind')!=str(kind) or metadata.get('hidden')!=str(hidden):
            raise CriticStateError('Critic kind/hidden size differs from saved state')
        if not allow_initial and metadata.get('identity')!=str(identity):
            raise CriticStateError('Critic checkpoint belongs to another room/profile')
        # An explicitly chosen initial checkpoint may be copied between rooms,
        # but it can never replace an already existing local progress file.
        expected={'c.'+name:p for name,p in critic.state_dict().items()}
        if any(k not in weights or weights[k].shape!=p.shape or weights[k].dtype!=p.dtype
               for k,p in expected.items()):
            raise CriticStateError('Critic checkpoint tensor layout mismatch')
        if any(not torch.isfinite(x).all() for x in weights.values()):
            raise CriticStateError('Nonfinite critic checkpoint')
        parameters=list(critic.parameters())
        allowed=set(expected)
        for i,p in enumerate(parameters):
            for key in ('step','exp_avg','exp_avg_sq'):
                name=f'o.{i}.{key}'
                if name not in weights:continue
                allowed.add(name)
                item=weights[name]
                if key=='step':
                    if item.numel()!=1 or float(item.item())<0:
                        raise CriticStateError('Invalid optimizer step counter')
                elif item.shape!=p.shape or item.dtype!=p.dtype:
                    raise CriticStateError('Invalid optimizer tensor layout')
        if set(weights)!=allowed:
            raise CriticStateError('Unexpected critic checkpoint tensors')
        for i,p in enumerate(parameters):
            keys={key for key in ('step','exp_avg','exp_avg_sq') if f'o.{i}.{key}' in weights}
            if keys and keys!={'step','exp_avg','exp_avg_sq'}:
                raise CriticStateError('Incomplete AdamW critic state')
        # Finish all structural validation before mutating live state.
        critic.load_state_dict({name:weights['c.'+name] for name in critic.state_dict()},strict=True)
        optimizer.state.clear()
        for i,p in enumerate(parameters):
            state={}
            for key in ('step','exp_avg','exp_avg_sq'):
                name=f'o.{i}.{key}'
                if name in weights:
                    state[key]=weights[name].clone().to(p.device)
            if state:
                if set(state)!={'step','exp_avg','exp_avg_sq'}:
                    raise CriticStateError('Incomplete AdamW critic state')
                optimizer.state[p]=state
        return int(metadata.get('steps','0'))
    except (OSError,ValueError,KeyError,RuntimeError,OverflowError) as exc:
        if isinstance(exc,CriticStateError):raise
        raise CriticStateError('Invalid or damaged critic safetensors') from exc
