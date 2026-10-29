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


def capture_working(journal, adapter, optimizer, steps: int, *, journal_tip=None):
    """Immutable CPU copy at a stable optimizer boundary; no disk access.

    The caller must coordinate journal changes so the captured tip describes
    exactly the base these LoRA/LoHA factors were trained against.
    """
    if steps < 0:
        raise ValueError('invalid local step')
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
        meta = {'version':1,'genesis':journal.genesis,
                'tip':(journal_tip if journal_tip is not None else _tip(journal)),
                'steps':steps,'layout':adapter.layout,'kind':adapter.kind,
                'rank':adapter.rank,'alpha':adapter.alpha,'factor_count':len(params)}
        if getattr(optimizer,'live_disk_optimizer_v59',False):
            meta['disk_optimizer'] = optimizer.snapshot_moments(snapshot_path(journal))
    return snapshot_path(journal), tensors, meta, (optimizer if getattr(optimizer,'live_disk_optimizer_v59',False) else None)


def write_working_capture(captured):
    """Slow safetensors serialization/fsync; safe to run in a worker thread."""
    target, tensors, meta, disk_optimizer = captured
    target.parent.mkdir(parents=True, exist_ok=True)
    name = target.with_name(target.name + '.' + os.urandom(8).hex() + '.tmp')
    published=False
    try:
        if disk_optimizer is not None:
            # Immutable reflinks are checksummed off the trainer/optimizer lock.
            disk_optimizer.finalize_moments(meta['disk_optimizer'],target.parent)
        save_file(tensors,str(name),metadata={'fedpara':json.dumps(meta,sort_keys=True)})
        with name.open('rb') as fh: os.fsync(fh.fileno())
        os.replace(name,target)
        published=True
        # Atomic rename alone is not a power-loss durable publication.
        dfd=os.open(target.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try: os.fsync(dfd)
        finally: os.close(dfd)
        if disk_optimizer is not None:
            # Old immutable reflink snapshots are no longer authoritative.
            import shutil
            keep = meta['disk_optimizer']['folder']
            for folder in target.parent.glob(target.name+'.moments-*'):
                if folder.is_dir() and not folder.is_symlink() and folder.name!=keep:
                    shutil.rmtree(folder)
    finally:
        name.unlink(missing_ok=True)
        if disk_optimizer is not None and not published:
            # Only the unpublished COW directory may be destroyed. The older
            # journal-bound working snapshot remains valid and untouched.
            import shutil
            shutil.rmtree(target.parent/meta['disk_optimizer']['folder'],ignore_errors=True)


def save_working(journal, adapter, optimizer, steps: int, *, journal_tip=None):
    write_working_capture(capture_working(journal,adapter,optimizer,steps,
                                           journal_tip=journal_tip))


class AsyncWorkingSaver:
    """Bounded one-at-a-time disk writer, never writes mutable tensors.

    Skips optional snapshots while a write is in flight. Committed signed
    events are never skipped and remain FULL durable in the SQLite WAL.
    """
    def __init__(self):
        self.task = None
        self.skipped = 0
        self.disk_ms = 0.0

    async def checkpoint(self,journal,adapter,optimizer,steps,*,journal_tip=None):
        import asyncio
        import time
        if self.task is not None:
            if not self.task.done():
                self.skipped += 1
                return False
            self.task.result()  # surface a failed fsync immediately
        # Capture is an atomic tensor snapshot at the caller's optimizer
        # boundary; only this (much shorter) stage gates further training.
        captured = await asyncio.to_thread(capture_working,journal,adapter,optimizer,
                                             steps,journal_tip=journal_tip)
        async def flush():
            started=time.monotonic()
            await asyncio.to_thread(write_working_capture,captured)
            self.disk_ms=(time.monotonic()-started)*1000
        self.task=asyncio.create_task(flush(),name='fedpara-working-snapshot-writer')
        return True

    def check(self):
        if self.task is not None and self.task.done():
            self.task.result()

    async def drain(self):
        if self.task is not None:
            await self.task


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
        disk_mode = getattr(optimizer,'live_disk_optimizer_v59',False)
        if disk_mode != ('disk_optimizer' in meta):
            raise PatchError('optimizer checkpoint format differs from selected live optimizer')
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
            if disk_mode:
                optimizer.restore_moments(meta['disk_optimizer'],target.parent)
            for p,t,record in restored:
                p.copy_(t.to(p.device))
                p.grad = None
                if not disk_mode:
                    optimizer.state[p].clear()
                    optimizer.state[p].update(record)
        return meta['steps']
