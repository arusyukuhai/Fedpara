"""Conservative capability probes and *advisory* LoHa/LoRA/expert limits.

Hardware free memory is a transient observation, not a promise. Placement and
rank decisions must be frozen in the shared room manifest; never permit peers
with different devices to silently create different adapter layouts.
"""
from __future__ import annotations
import math
import os
import shutil
import subprocess
import re
import sys
from dataclasses import dataclass, asdict

import torch
from torch import nn


@dataclass(frozen=True)
class DeviceBudget:
    device: str
    available_ram_bytes: int
    available_accelerator_bytes: int
    free_disk_bytes: int
    usable_bytes: int
    memory_fraction: float
    source: str


def _ram_available():
    def capped(byte_count, source):
        # Host free RAM is misleading under Docker/systemd memory quotas.
        # Honor cgroup v2 and legacy v1 residual budgets when present.
        for limit_file,used_file in (
            ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
            ('/sys/fs/cgroup/memory/memory.limit_in_bytes',
             '/sys/fs/cgroup/memory/memory.usage_in_bytes')):
            try:
                raw=open(limit_file).read().strip()
                if raw=='max':continue
                limit=int(raw)
                used=int(open(used_file).read().strip())
                if 0<limit<1<<60:
                    return min(byte_count,max(0,limit-used)),source+'+cgroup'
            except (OSError,ValueError):pass
        return byte_count,source
    try:
        import psutil
        return capped(int(psutil.virtual_memory().available),'psutil')
    except ImportError:
        pass
    if os.path.exists('/proc/meminfo'):
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                return capped(int(line.split()[1])*1024,'proc-meminfo')
    if sys.platform=='darwin':
        try:
            output=subprocess.check_output(['vm_stat'],text=True,timeout=2)
            match=re.search(r'page size of (\d+) bytes',output)
            page_bytes=int(match.group(1)) if match else 4096
            counts={m.group(1):int(m.group(2)) for m in re.finditer(
                r'^(Pages free|Pages inactive|Pages speculative):\s*(\d+)\.',output,re.M)}
            free=(counts.get('Pages free',0)+counts.get('Pages inactive',0)
                  +counts.get('Pages speculative',0))*page_bytes
            if free>0:return capped(free,'macos-vm_stat-estimate')
        except (OSError,ValueError,subprocess.SubprocessError):pass
    try:
        pages=os.sysconf('SC_AVPHYS_PAGES');sz=os.sysconf('SC_PAGE_SIZE')
        if pages>0 and sz>0:return capped(int(pages*sz),'sysconf')
    except (ValueError,OSError,AttributeError):pass
    # No reliable free-memory probe => refuse to over-allocate.
    return 0,'unavailable'


def detect_device_budget(device='auto',directory='.',memory_fraction=.5):
    if not 0 < memory_fraction <= .85:
        raise ValueError('memory_fraction must be in (0,.85]')
    if device=='auto':
        device=('cuda' if torch.cuda.is_available() else
                'mps' if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available() else 'cpu')
    d=torch.device(device)
    ram,source=_ram_available()
    vram=0
    if d.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        vram=int(torch.cuda.mem_get_info(d)[0]);source+='+cuda-mem_get_info'
    elif d.type=='mps':
        if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
        # MPS unified memory: do not double-count RAM and GPU memory.
        usable=ram
        try:
            vram=max(0,int(torch.mps.recommended_max_memory())-int(torch.mps.current_allocated_memory()))
            usable=min(ram,vram)
            source+='+mps-recommended'
        except (AttributeError,RuntimeError):pass
    else:
        usable=ram
    if d.type=='cuda':usable=vram
    free_disk=shutil.disk_usage(directory).free
    return DeviceBudget(str(d),ram,vram,free_disk,int(usable*memory_fraction),
                        memory_fraction,source)


def estimate_adapter_memory(model: nn.Module, *, kind='loha', rank=16,
                            optimizer_moments=2):
    """Estimate adaptation factor+grad+optimizer cost, without model base/activations."""
    if kind not in ('lora','loha') or rank<=0 or optimizer_moments<0:
        raise ValueError('invalid adapter memory arguments')
    seen=set();factors=0;layers=0;max_factor_bytes=0
    for _,mod in model.named_modules():
        for pname,p in mod.named_parameters(recurse=False):
            # Covers nn.Linear/Embedding, Conv1D-like custom projections and
            # other named 2-D weights. Overcount is safer than silent OOM.
            if pname!='weight' or id(p) in seen or p.ndim!=2 or not p.is_floating_point():
                continue
            seen.add(id(p));m,n=p.shape
            r=min(rank,m,n)
            slots=2 if kind=='lora' else 4
            size=(m+n)*r*slots*4
            factors+=size;max_factor_bytes=max(max_factor_bytes,size);layers+=1
    total=factors*(2+optimizer_moments)  # fp32 factors+grads+moments
    return {'layer_count':layers,'factor_bytes':factors,'factor_grad_optimizer_bytes':total,
            'largest_factor_bytes':max_factor_bytes,'rank':rank,'kind':kind}


def suggest_rank_and_experts(model, *, device='auto',disk_dir='.',kind='loha',
                             candidates=(4,8,16,32,64),
                             memory_fraction=.5, expert_bytes=None,
                             other_reserved_bytes=0):
    budget=detect_device_budget(device,disk_dir,memory_fraction)
    if other_reserved_bytes<0:raise ValueError('negative memory reserve')
    available=max(0,budget.usable_bytes-other_reserved_bytes)
    # Conservative half reserved for activations, comm buffers and workspace.
    factor_budget=available//2
    profiles=[estimate_adapter_memory(model,kind=kind,rank=r) for r in sorted(set(candidates))]
    feasible=([p['rank'] for p in profiles if p['factor_grad_optimizer_bytes']<=factor_budget]
              if profiles[0]['layer_count'] else [])
    rank_default=(max((r for r in feasible if r<=16),default=min(feasible))
                  if feasible else None)
    rank_max_safe=max(feasible) if feasible else None
    # Report a hard conservative default, not an inference of exact OOM limit.
    if expert_bytes is not None:
        if not isinstance(expert_bytes,(list,tuple)) or any(type(b)!=int or b<=0 for b in expert_bytes):
            raise ValueError('expert_bytes must be positive integer byte sizes')
        # Reserve another half of available for runtime, not all free memory.
        rem=available//2
        sizes=sorted(expert_bytes)
        keep=0
        for s in sizes:
            if s>rem:break
            rem-=s;keep+=1
        expert_default=min(keep,8)
        expert_max_safe=keep
    else:expert_default=None;expert_max_safe=None
    return {'budget':asdict(budget),'rank_default':rank_default,
            'rank_max_safe':rank_max_safe,'rank_options_feasible':feasible,'rank_profiles':profiles,
            'expert_count_default':expert_default,
            'expert_count_max_safe':expert_max_safe,
            'expert_count_upper_bound':len(expert_bytes) if expert_bytes is not None else None,
            'warning':'Per-device advisory only: freeze rank and expert ownership in the signed shared genesis. '
                      'Activation peaks, allocator fragmentation, routing skew and network cost are not measured.'}


def automatic_expert_counts(groups, budget_bytes, *, default_per_group=8):
    """Safe local active-expert defaults from heterogeneous module sizes.

    Uses the largest Expert in each group as a conservative reservation;
    this describes local residency, not global number of routed experts.
    """
    if type(budget_bytes)!=int or budget_bytes<0 or default_per_group<1:
        raise ValueError('invalid expert budget')
    group_list=list(groups)
    if not group_list:return {}
    if any(not g.bytes_per_expert or min(g.bytes_per_expert)<=0 for g in group_list):
        raise ValueError('invalid expert group')
    reservations={g.path:max(g.bytes_per_expert) for g in group_list}
    min_cost=sum(reservations.values())
    if min_cost>budget_bytes:raise MemoryError('not enough memory for one expert per MoE layer')
    remaining=budget_bytes-min_cost
    selected={g.path:1 for g in group_list}
    # One additional slot per group per pass, favoring small experts.
    while True:
        candidates=sorted((reservations[g.path],g.path,g.count) for g in group_list
                          if selected[g.path]<min(g.count,default_per_group))
        if not candidates:break
        progressed=False
        for cost,path,_ in candidates:
            if remaining>=cost:
                remaining-=cost;selected[path]+=1;progressed=True
        if not progressed:break
    return selected
