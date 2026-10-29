"""Per-parameter disk-paged AdamW moments; no full-device optimizer state.

FP32 first/second moments are mmapped from files. On an accelerator, only one
parameter's moments are staged at a time. Crash-consistent full-step recovery
is NOT provided (call flush() for a best-effort durable boundary).
"""
from __future__ import annotations
import hashlib
import json
import os
import math
from pathlib import Path

import torch


class DiskAdamW(torch.optim.Optimizer):
    def __init__(self, named_parameters, directory, *, lr=1e-4, weight_decay=0.,
                 betas=(.9,.999), eps=1e-8, resume=False):
        named=list(named_parameters)
        if not named or len({name for name,_ in named})!=len(named):
            raise ValueError('named parameters must be nonempty and unique')
        if not 0<lr or not 0<=weight_decay or not all(0<=b<1 for b in betas) or eps<=0:
            raise ValueError('invalid AdamW settings')
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.entries=[]
        for name,p in named:
            if p.layout!=torch.strided or p.dtype not in (torch.float32,torch.float16,torch.bfloat16) or p.device.type=='meta':
                raise ValueError('unsupported optimizer parameter')
            namehash=hashlib.sha256(name.encode()).hexdigest()
            self.entries.append((name,p,namehash))
        super().__init__([p for _,p,_ in self.entries],dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay))
        self.iteration=0
        manifest_path=self.directory/'optimizer.json'
        desc=[{'name':name,'shape':list(p.shape),'dtype':str(p.dtype),'hash':key} for name,p,key in self.entries]
        if resume:
            manifest=json.loads(manifest_path.read_text())
            if manifest.get('params')!=desc:
                raise ValueError('optimizer state layout mismatch')
            self.iteration=int(manifest['iteration'])
            restored_steps=manifest.get('steps',{})
            if set(restored_steps)!=set(x[2] for x in self.entries):
                raise ValueError('disk optimizer steps mismatch')
        elif manifest_path.exists():
            raise FileExistsError('optimizer directory exists; use resume=True or choose a new directory')
        for name,p,key in self.entries:
            for suffix in ('m','v'):
                path=self.directory/f'{key}.{suffix}.f32'
                size=p.numel()*4
                if resume:
                    if not path.is_file() or path.stat().st_size!=size:
                        raise ValueError(f'missing/truncated disk moment for {name}')
                else:
                    with path.open('wb') as f:f.truncate(size)
        if resume:
            for _,p,key in self.entries:
                self.state[p]['step']=int(restored_steps[key])
        else:
            self._manifest(desc)
        self._description=desc

    def _manifest(self, description):
        staging=self.directory/'optimizer.json.tmp'
        data=json.dumps({'format':'disk-adamw-v1','iteration':self.iteration,'params':description,
                         'steps':{key:int(self.state[p].get('step',0)) for _,p,key in self.entries}})
        with staging.open('w') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(staging,self.directory/'optimizer.json')

    def flush(self):
        """Flush current mapped moment files and write current step counter."""
        for _,_,key in self.entries:
            for suffix in ('m','v'):
                with (self.directory/f'{key}.{suffix}.f32').open('rb') as f:os.fsync(f.fileno())
        self._manifest(self._description)

    @torch.no_grad()
    def step(self, closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        group=self.param_groups[0]
        lr=group['lr'];b1,b2=group['betas'];eps=group['eps'];wd=group['weight_decay']
        self.iteration+=1
        for _,p,key in self.entries:
            grad=p.grad
            if grad is None:continue
            if grad.is_sparse:raise RuntimeError('sparse gradients unsupported')
            state=self.state[p]
            state['step']=state.get('step',0)+1
            step=state['step']
            m_cpu=torch.from_file(str(self.directory/f'{key}.m.f32'),shared=True,size=p.numel(),dtype=torch.float32).view(p.shape)
            v_cpu=torch.from_file(str(self.directory/f'{key}.v.f32'),shared=True,size=p.numel(),dtype=torch.float32).view(p.shape)
            m=m_cpu.to(device=p.device);v=v_cpu.to(device=p.device)
            g=grad.detach().float()
            m.mul_(b1).add_(g,alpha=1-b1)
            v.mul_(b2).addcmul_(g,g,value=1-b2)
            param=p.data.float()
            param.mul_(1-lr*wd)
            denom=v.sqrt().div_(math.sqrt(1-b2**step)).add_(eps)
            param.addcdiv_(m,denom,value=-lr/(1-b1**step))
            p.copy_(param)
            m_cpu.copy_(m.to('cpu'));v_cpu.copy_(v.to('cpu'))
        return loss


def adapter_named_parameters(adapter):
    """Stable optimizer names from Fedpara canonical alias + local factor index."""
    return [(f'{group.canonical}::factor{i}',p)
            for group in adapter.groups for i,p in enumerate(group.overlay.local.parts)]
