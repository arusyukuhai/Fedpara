"""Shape-aware heterogeneous MoE inventory and deterministic distributed placement.

Recognizes unpacked ModuleList/ModuleDict, packed/fused expert tensors with
expert count in their leading dimension, and explicit architecture plugins.
Packed experts are INVENTORIED, not rewritten as ordinary expert(x) calls.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass,asdict

import torch
from torch import nn

from .expert_parallel import expert_owner, discover_moe


@dataclass(frozen=True)
class ExpertGroup:
    path: str
    representation: str
    count: int
    bytes_per_expert: tuple[int,...]
    executable: bool
    router: str


class ExpertAdapterRegistry:
    def __init__(self):self._adapters={}
    def register(self, model_type: str, detector):
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,120}',model_type) or not callable(detector):
            raise ValueError('invalid expert adapter')
        if model_type in self._adapters:raise ValueError('duplicate expert adapter')
        self._adapters[model_type]=detector
    def inspect(self, model, model_type):
        if model_type not in self._adapters:
            raise KeyError('no registered expert shape adapter')
        groups=self._adapters[model_type](model)
        if not isinstance(groups,list) or not all(isinstance(g,ExpertGroup) for g in groups):
            raise TypeError('detector must return list[ExpertGroup]')
        return groups


def inspect_expert_shapes(model: nn.Module, *, registry=None, model_type=None):
    """Inspect arbitrary expert dimensions; fail closed on ambiguous packed groups."""
    if registry is not None and model_type in registry._adapters:
        return registry.inspect(model,model_type)
    result=[];matched=set()
    known={row['experts']:row for row in discover_moe(model)}
    for path,mod in model.named_modules():
        name=path.rsplit('.',1)[-1].lower()
        if name not in ('experts','expert') and 'experts' not in name:continue
        if isinstance(mod,(nn.ModuleList,nn.ModuleDict)) and len(mod):
            costs=tuple(sum(p.numel()*p.element_size() for p in ex.parameters())
                        for ex in mod.values() if isinstance(mod,nn.ModuleDict)) if isinstance(mod,nn.ModuleDict) else tuple(
                        sum(p.numel()*p.element_size() for p in ex.parameters()) for ex in mod)
            if any(c==0 for c in costs):continue
            meta=known.get(path,{})
            result.append(ExpertGroup(path,'module',len(costs),costs,
                        bool(meta.get('supported',False)),meta.get('router','unverified')))
            matched.add(path)
        elif not isinstance(mod,(nn.ModuleList,nn.ModuleDict)):
            # Detect direct Parameter tensors with shared leading E axis.
            tensors={name:p for name,p in mod.named_parameters(recurse=False)
                     if p.ndim>=2 and p.is_floating_point()}
            candidates=[(name,p) for name,p in tensors.items() if
                        name.lower() in ('w1','w2','w3','gate_up_proj','down_proj','up_proj','gate_proj')
                        or 'expert' in name.lower()]
            if candidates:
                leading={int(p.shape[0]) for _,p in candidates}
                if len(leading)==1 and next(iter(leading))>=2:
                    count=next(iter(leading))
                    each=sum(p[0].numel()*p.element_size() for _,p in candidates)
                    result.append(ExpertGroup(path,'packed',count,tuple([each]*count),False,'shape-only'))
                    matched.add(path)
    return result


def plan_expert_placement(groups: list[ExpertGroup], room, peer_capacities: dict[str,int],
                          *, replicas=1, generation='genesis',peer_byte_budgets=None):
    """Assign entire experts via capacity-weighted consistent hashing.

    Capacity is an integer weight, not a literal byte limit. The caller must
    enforce memory/disk budgets and authenticate the roster on every node.
    """
    if not isinstance(room,str) or not room or not isinstance(generation,str) or not generation:
        raise ValueError('room and generation required')
    if len({g.path for g in groups})!=len(groups):raise ValueError('duplicate expert path')
    result={}
    if peer_byte_budgets is not None:
        if set(peer_byte_budgets)!=set(peer_capacities) or any(
                type(n)!=int or n<0 for n in peer_byte_budgets.values()):
            raise ValueError('invalid per-peer memory budgets')
        remaining=dict(peer_byte_budgets)
    else:
        remaining=None
    # Descending sizes avoids stranding the most demanding experts.
    flattened=sorted(((cost,g,index) for g in groups
                      for index,cost in enumerate(g.bytes_per_expert)),
                     key=lambda item:(-item[0],item[1].path,item[2]))
    for cost,group,index in flattened:
        if cost<=0 or group.count<1 or len(group.bytes_per_expert)!=group.count:
            raise ValueError('invalid expert size')
        key=f'{generation}:{group.path}:{index}'
        preferred=expert_owner(room,key,peer_capacities,len(peer_capacities))
        if remaining is None:
            owners=preferred[:min(replicas,len(preferred))]
        else:
            owners=tuple(peer for peer in preferred if remaining[peer]>=cost)[:replicas]
            if len(owners)<replicas:
                raise MemoryError(f'no expert placement with the given per-peer budgets: {key}')
            for peer in owners:remaining[peer]-=cost
        result[key]={'owners':list(owners), 'bytes':cost,
                     'representation':group.representation,
                     'rpc_executable':group.executable}
    return {'room':room,'generation':generation,'peer_capacity_weights':peer_capacities,
            'replicas':replicas,'peer_remaining_bytes':remaining,'experts':result,
            'warning':'Packed/fused experts require a model-specific execution adapter. '
                      'Placement does not automatically wire network RPC into the training loop.'}


class PackedExpertView(nn.Module):
    """A named packed-SwiGLU expert exposed as expert(x)->Tensor for RPC.

    Supports same-shape E-leading w1/w2/w3 arrays, including non-square
    feed-forward dimensions. Custom shapes must implement their own kernel;
    do not assume field order without validation.
    """
    def __init__(self, packed_module: nn.Module, expert_index: int):
        super().__init__()
        w1=getattr(packed_module,'w1',None)
        w2=getattr(packed_module,'w2',None)
        w3=getattr(packed_module,'w3',None)
        if not all(isinstance(x,torch.Tensor) and x.ndim==3 for x in (w1,w2,w3)):
            raise ValueError('packed SwiGLU requires 3-D w1/w2/w3 tensors')
        if w1.shape!=w3.shape or w1.shape[0]!=w2.shape[0] or w1.shape[1]!=w2.shape[2] or w1.shape[2]!=w2.shape[1]:
            raise ValueError('packed SwiGLU expert dimensions incompatible')
        if not 0<=expert_index<w1.shape[0]:raise ValueError('bad expert index')
        self.packed=packed_module
        self.index=expert_index

    def forward(self,x):
        import torch.nn.functional as F
        a=F.linear(x,self.packed.w1[self.index])
        b=F.linear(x,self.packed.w3[self.index])
        return F.linear(F.silu(a)*b,self.packed.w2[self.index])


def packed_swiglu_expert_views(module:nn.Module):
    """Create RPC-compatible views; no assumption about HF model names."""
    w1=getattr(module,'w1',None)
    if not isinstance(w1,torch.Tensor) or w1.ndim!=3:
        raise ValueError('not a packed w1/w2/w3 SwiGLU module')
    return {str(i):PackedExpertView(module,i) for i in range(w1.shape[0])}
