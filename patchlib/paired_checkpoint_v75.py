"""Crash-atomic *local* paired Actor/AdamW/Critic/AdamW safetensors snapshot.

The signed P2P journal is NOT part of the filesystem transaction. A paired
snapshot may only be restored at its EXACT durable journal tip and genesis.
Snapshots run on immutable CPU copies; never read live model tensors in writer.
Disk-backed actor optimizers and non-AdamW actor optimizers are deliberately
excluded until their moments can be captured with equivalent guarantees.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from .core import PatchError
from .localprogress import _tip

FORMAT='kagami-paired-actor-critic-v75'
MAX_BYTES=256 * 1024**2


def snapshot_path(journal):
    return Path(str(journal.path)+'.paired-v75.safetensors')


def _params(module):
    return list(module)


def _copy(value):
    if not isinstance(value,torch.Tensor):
        raise PatchError('unsupported non-tensor optimizer moment in paired checkpoint')
    out=value.detach().to('cpu').contiguous().clone()
    if not bool(torch.isfinite(out).all()):
        raise PatchError('nonfinite tensor in paired checkpoint')
    return out


def _record(prefix,params,optimizer,*,actor=False):
    out={}
    names=('step','exp_avg','exp_avg_sq')
    for i,p in enumerate(params):
        out[f'{prefix}.p.{i}']=_copy(p)
        state=optimizer.state.get(p,{})
        selected=set(state).intersection(names)
        if selected and selected!=set(names):
            raise PatchError('partial AdamW optimizer state in paired checkpoint')
        # Validate a complete moment set. Unknown moment formats fail closed.
        if set(state)-set(names):
            raise PatchError('unsupported actor or critic optimizer state; paired checkpoint needs AdamW')
        for key in names:
            if key in selected:
                val=_copy(state[key])
                if key=='step':
                    if val.numel()!=1 or float(val.item())<0:
                        raise PatchError('invalid AdamW step counter')
                elif val.shape!=p.shape or val.dtype!=p.dtype:
                    raise PatchError('AdamW moment has wrong shape or dtype')
                out[f'{prefix}.o.{i}.{key}']=val
    return out


def _critic_params(mixer):
    c=mixer.critic
    return list(c.trainable_parameters() if getattr(mixer.args,'critic_kind','mlp')=='hf' else c.parameters())


def _identity(mixer):
    kind=getattr(mixer.args,'critic_kind','mlp')
    spec=(mixer.critic.spec if kind=='hf' else
          {'kind':kind,'hidden':int(getattr(mixer.args,'critic_hidden',32))})
    return {'room':mixer._critic_identity,'critic':spec}


def capture(journal,adapter,actor_optimizer,mixer,steps,*,journal_tip=None):
    if mixer is None or mixer.critic is None or mixer.critic_optim is None:
        raise PatchError('PPO Critic required for paired checkpoint')
    if getattr(actor_optimizer,'live_disk_optimizer_v59',False):
        raise PatchError('disk optimizer requires external crash-atomic reflinks; paired v75 unsupported')
    if type(steps) is not int or steps<0:raise PatchError('invalid paired local step')
    # Called only at a safe optimizer boundary, by the awaited trainer loop.
    # adapter.lock prevents concurrent actor/critic changes during CPU copy.
    with adapter.lock, torch.no_grad():
        actor=_params(adapter.trainable_parameters())
        critic=_critic_params(mixer)
        tensors={**_record('a',actor,actor_optimizer,actor=True),
                 **_record('c',critic,mixer.critic_optim)}
        if sum(t.numel()*t.element_size() for t in tensors.values())>MAX_BYTES:
            raise PatchError('paired checkpoint exceeds 256MiB frozen tensor budget')
        metadata={'format':FORMAT,'genesis':journal.genesis,
                  'tip':journal_tip if journal_tip is not None else _tip(journal),
                  'steps':steps,'actor':{'layout':adapter.layout,'kind':adapter.kind,
                         'rank':adapter.rank,'alpha':adapter.alpha,'count':len(actor)},
                  'critic':_identity(mixer),'critic_steps':mixer.critic_steps,
                  'critic_count':len(critic)}
    return snapshot_path(journal),tensors,metadata


def _tensor_digest(tensors):
    """End-to-end integrity of inert tensor bytes, computed off the trainer lock."""
    h=hashlib.sha256()
    for name in sorted(tensors):
        item=tensors[name].contiguous().cpu()
        header=json.dumps([name,str(item.dtype),list(item.shape)],separators=(',',':')).encode()
        h.update(len(header).to_bytes(4,'big'))
        h.update(header)
        h.update(memoryview(item.reshape(-1).view(torch.uint8).numpy()))
    return h.hexdigest()


def write(captured):
    target,tensors,metadata=captured
    metadata=dict(metadata)
    metadata['tensor_sha256']=_tensor_digest(tensors)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink() or target.parent.is_symlink():
        raise PatchError('unsafe paired checkpoint symlink')
    tmp=target.with_name('.'+target.name+'.'+os.urandom(8).hex()+'.tmp')
    try:
        save_file(tensors,str(tmp),metadata={'paired':json.dumps(metadata,sort_keys=True,separators=(',',':'))})
        with tmp.open('rb') as f:os.fsync(f.fileno())
        os.replace(tmp,target)
        dfd=os.open(target.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(dfd)
        finally:os.close(dfd)
    finally:tmp.unlink(missing_ok=True)
    return target


def _validated_params(reader,keys,prefix,params):
    restored=[]
    valid=set()
    for i,p in enumerate(params):
        key=f'{prefix}.p.{i}';valid.add(key)
        if key not in keys:raise PatchError(f'missing paired weight {key}')
        weight=reader.get_tensor(key)
        if weight.shape!=p.shape or weight.dtype!=p.dtype or not bool(torch.isfinite(weight).all()):
            raise PatchError(f'paired weight {key} has invalid layout/values')
        saved={}
        present={k for k in ('step','exp_avg','exp_avg_sq') if f'{prefix}.o.{i}.{k}' in keys}
        if present and present!={'step','exp_avg','exp_avg_sq'}:
            raise PatchError('partial paired AdamW state')
        for name in present:
            key=f'{prefix}.o.{i}.{name}';valid.add(key)
            item=reader.get_tensor(key)
            if not bool(torch.isfinite(item).all()):raise PatchError('nonfinite paired AdamW state')
            if name=='step':
                if item.numel()!=1 or float(item.item())<0:raise PatchError('invalid paired optimizer step')
            elif item.shape!=p.shape or item.dtype!=p.dtype:
                raise PatchError('invalid paired optimizer moment layout')
            saved[name]=item
        restored.append((p,weight,saved))
    return restored,valid


def restore(journal,adapter,actor_optimizer,mixer,*,strict=True):
    """Validate journal + ALL tensors before mutating either model. Fail closed.

    Returns None when absent. Old/stale snapshots are never replayed over a
    changed signed base. Does not roll back the signed journal.
    """
    target=snapshot_path(journal)
    if not target.exists():return None
    if target.is_symlink() or target.stat().st_size>MAX_BYTES+1024**2:
        raise PatchError('oversized or unsafe paired checkpoint')
    with safe_open(str(target),framework='pt',device='cpu') as reader:
        try: meta=json.loads((reader.metadata() or {})['paired'])
        except (KeyError,TypeError,ValueError) as exc:raise PatchError('invalid paired metadata') from exc
        if meta.get('format')!=FORMAT or meta.get('genesis')!=journal.genesis or meta.get('tip')!=_tip(journal):
            raise PatchError('paired checkpoint does not match signed journal; refusing mixed actor/critic recovery')
        expected={'layout':adapter.layout,'kind':adapter.kind,'rank':adapter.rank,
                  'alpha':adapter.alpha,'count':len(list(adapter.trainable_parameters()))}
        if meta.get('actor')!=expected or meta.get('critic')!=_identity(mixer):
            raise PatchError('paired checkpoint room / adapter / critic configuration mismatch')
        # A v74 unsealed actor working snapshot might be newer than this
        # pair. Never silently roll its optimizer moments back to older steps.
        from .localprogress import snapshot_path as old_working_path
        legacy=old_working_path(journal)
        if legacy.is_file():
            with safe_open(str(legacy),framework='pt',device='cpu') as older:
                try:old_meta=json.loads((older.metadata() or {})['fedpara'])
                except (KeyError,TypeError,ValueError) as exc:
                    raise PatchError('invalid pre-existing actor working snapshot') from exc
                if (old_meta.get('genesis')==journal.genesis and old_meta.get('tip')==meta['tip']
                    and type(old_meta.get('steps')) is int and old_meta['steps']>meta.get('steps',-1)):
                    raise PatchError('newer unsealed actor working state than paired snapshot; refusing rollback')
        if type(meta.get('steps')) is not int or meta['steps']<0 or (
             type(meta.get('critic_steps')) is not int or meta['critic_steps']<0):
            raise PatchError('invalid paired step counters')
        ap=list(adapter.trainable_parameters());cp=_critic_params(mixer)
        if meta.get('critic_count')!=len(cp):raise PatchError('paired critic trainable layout mismatch')
        keys=set(reader.keys())
        # Verify the complete checkpoint before touching either optimizer.
        # A safetensors header alone does not checksum tensor payload bytes.
        if meta.get('tensor_sha256')!=_tensor_digest({key:reader.get_tensor(key) for key in keys}):
            raise PatchError('paired checkpoint tensor digest mismatch')
        a,ak=_validated_params(reader,keys,'a',ap)
        c,ck=_validated_params(reader,keys,'c',cp)
        if ak|ck!=keys:raise PatchError('unknown paired checkpoint tensor')
        # Stage *both* device copies before mutating either model: OOM/transfer
        # failure must not leave one model restored and the other unchanged.
        staged=[]
        for params,optim in ((a,actor_optimizer),(c,mixer.critic_optim)):
            part=[]
            for param,value,state in params:
                placed=value.to(param.device)
                moments={k:v.clone().to('cpu' if k=='step' else param.device)
                         for k,v in state.items()}
                part.append((param,placed,moments))
            staged.append((part,optim))
        with adapter.lock,torch.no_grad():
            for params,optim in staged:
                optim.state.clear()
                for param,value,state in params:
                    param.copy_(value)
                    param.grad=None
                    if state:optim.state[param]=state
            mixer.critic_steps=meta['critic_steps']
            mixer.cached=None
            mixer.reuses=0
    return meta['steps']


class AsyncPairedSaver:
    """At most one async writer, with immutable bounded CPU captures."""
    def __init__(self):
        self.task=None
        self.skipped=0
        self.disk_ms=0.0
    async def checkpoint(self,journal,adapter,actor_optimizer,mixer,steps,*,journal_tip=None,force=False):
        if self.task is not None:
            if force:await self.task
            elif not self.task.done():
                self.skipped+=1
                return False
            else:self.task.result()
        # SQLite connections are thread-affine. Pin the WAL tip on the event
        # loop BEFORE worker capture; the caller holds inbound_commit_lock.
        if journal_tip is None:
            journal_tip=_tip(journal)
        # Copy under lock off-loop; FS writes never hold adapter.lock.
        captured=await asyncio.to_thread(capture,journal,adapter,actor_optimizer,mixer,
                                         steps,journal_tip=journal_tip)
        async def flush():
            start=time.monotonic()
            await asyncio.to_thread(write,captured)
            self.disk_ms=(time.monotonic()-start)*1000
        self.task=asyncio.create_task(flush(),name='kagami-paired-v75-writer')
        return True
    def check(self):
        if self.task is not None and self.task.done():self.task.result()
    async def drain(self):
        if self.task is not None:await self.task
