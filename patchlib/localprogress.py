"""Crash-recoverable, local-only LoRA/LoHA working state (no pickle).

This file does NOT contain the full model and is never sent to another peer.
Committed peer updates live in the SQLite journal; a working snapshot is valid
only when its recorded journal tip still matches. A crash cannot roll a newer
journal back to stale adapter factors.
"""
from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .core import PatchError


def _tip(journal):
    row = journal.db.execute('SELECT origin,seq,digest FROM events ORDER BY rowid DESC LIMIT 1').fetchone()
    return [journal.count(), list(row) if row else None]


def _parameters(adapter):
    return list(adapter.trainable_parameters())


def snapshot_path(journal):
    return Path(str(journal.path) + '.working.safetensors')


def save_working(journal, adapter, optimizer, steps: int, *, journal_tip=None):
    if steps < 0:
        raise ValueError('invalid local step')
    target = snapshot_path(journal)
    target.parent.mkdir(parents=True, exist_ok=True)
    tensors = {}
    with adapter.lock, torch.no_grad():
        params = _parameters(adapter)
        for i, param in enumerate(params):
            tensors[f'f.{i}'] = param.detach().to('cpu').contiguous().clone()
            for name in ('step','exp_avg','exp_avg_sq','max_exp_avg_sq'):
                if name in optimizer.state[param]:
                    value = optimizer.state[param][name]
                    if isinstance(value, torch.Tensor):
                        tensors[f'o.{i}.{name}'] = value.detach().to('cpu').contiguous().clone()
        meta = {'version':1,'genesis':journal.genesis,'tip':(journal_tip if journal_tip is not None else _tip(journal)),
                'steps':steps,'layout':adapter.layout,'kind':adapter.kind,
                'rank':adapter.rank,'alpha':adapter.alpha,'factor_count':len(params)}
    name = target.with_name(target.name + '.' + os.urandom(8).hex() + '.tmp')
    try:
        save_file(tensors,str(name),metadata={'fedpara':json.dumps(meta,sort_keys=True)})
        with name.open('rb') as fh: os.fsync(fh.fileno())
        os.replace(name,target)
    finally:
        name.unlink(missing_ok=True)


def restore_working(journal, adapter, optimizer):
    target = snapshot_path(journal)
    if not target.is_file():
        return 0
    with safe_open(str(target),framework='pt',device='cpu') as reader:
        try: meta = json.loads((reader.metadata() or {})['fedpara'])
        except (KeyError,ValueError,TypeError) as e:
            raise PatchError('corrupt local working snapshot') from e
        if meta.get('genesis')!=journal.genesis or meta.get('layout')!=adapter.layout or meta.get('kind')!=adapter.kind or meta.get('rank')!=adapter.rank or meta.get('alpha')!=adapter.alpha:
            raise PatchError('local working snapshot is from a different model or adapter')
        if meta.get('tip')!=_tip(journal):
            # A newer committed event supersedes this local factor snapshot.
            # Never reapply factors to a changed base.
            return 0
        params = _parameters(adapter)
        if meta.get('factor_count')!=len(params) or type(meta.get('steps')) is not int or meta['steps'] < 0:
            raise PatchError('working snapshot parameter count mismatch')
        expected = {f'f.{i}' for i in range(len(params))}
        keys = set(reader.keys())
        if not expected <= keys or any(k not in expected and not (k.startswith('o.') and k.rsplit('.',1)[-1] in ('step','exp_avg','exp_avg_sq','max_exp_avg_sq')) for k in keys):
            raise PatchError('unexpected local working snapshot keys')
        restored = []
        for i, p in enumerate(params):
            t = reader.get_tensor(f'f.{i}')
            if t.shape != p.shape or t.dtype!=p.dtype or not torch.isfinite(t).all():
                raise PatchError('working snapshot factor mismatch or nonfinite')
            record = {}
            for name in ('step','exp_avg','exp_avg_sq','max_exp_avg_sq'):
                key = f'o.{i}.{name}'
                if key not in keys: continue
                item = reader.get_tensor(key)
                if name == 'step':
                    if item.numel()!=1: raise PatchError('invalid AdamW step')
                    record[name] = item.to('cpu')
                else:
                    if item.shape!=p.shape or item.dtype!=p.dtype or not torch.isfinite(item).all():
                        raise PatchError('invalid AdamW moment')
                    record[name] = item.to(p.device)
            restored.append((p,t,record))
        with adapter.lock, torch.no_grad():
            for p,t,record in restored:
                p.copy_(t.to(p.device))
                p.grad = None
                optimizer.state[p].clear()
                optimizer.state[p].update(record)
        return meta['steps']
