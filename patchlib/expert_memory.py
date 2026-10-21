"""Conservative 4-bit training blocks and tiered expert paging.

4-bit stores GROUPWISE symmetric int4 bases; optional NF4/bitsandbytes HF loading
is separate. QuantizedLoHaLinear is intentionally *not* a drop-in for the old
UniversalAdapter's on-wire layout: use its own adapter tensors / explicit sync.
"""
from __future__ import annotations
import math
import os
import re
from pathlib import Path
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file,load_file
from .core import FactorDelta


def pack_int4(weight:torch.Tensor,group_size:int=64):
    if weight.ndim != 2 or not weight.is_floating_point() or not 16<=group_size<=256 or group_size%2:
        raise ValueError('require floating 2D weights and even group size in 16..256')
    flat=weight.detach().float().reshape(-1)
    padding=(-flat.numel())%group_size
    if padding:flat=F.pad(flat,(0,padding))
    groups=flat.reshape(-1,group_size)
    scales=groups.abs().amax(dim=-1).clamp_min(1e-8)/7
    quant=torch.round(groups/scales[:,None]).clamp(-8,7).to(torch.int16).flatten()+8
    packed=((quant[0::2]|(quant[1::2]<<4))).to(torch.uint8).contiguous()
    return packed,scales.to(torch.float16).contiguous(),tuple(weight.shape),group_size


def unpack_int4(packed,scales,shape,group_size=64,device='cpu',dtype=torch.float32):
    count=math.prod(shape)
    if count<=0 or packed.dtype!=torch.uint8 or scales.ndim!=1 or scales.numel()!=(count+group_size-1)//group_size:
        raise ValueError('invalid packed 4bit tensor')
    packed=packed.to(device)
    q=torch.empty(packed.numel()*2,device=device,dtype=torch.int16)
    q[0::2]=(packed & 15).to(torch.int16)-8
    q[1::2]=(packed >> 4).to(torch.int16)-8
    return (q.reshape(-1,group_size).float()*scales.to(device).float()[:,None]).reshape(-1)[:count].reshape(shape).to(dtype)


class QuantizedLoHaLinear(nn.Module):
    """4-bit frozen linear base, fp32/fp16 factorized trainable update.

    Computes dense base transiently in forward; storage is genuinely 4-bit,
    but peak activation memory is NOT 4-bit and performance is not bnb NF4.
    """
    def __init__(self,linear:nn.Linear,rank:int=16,kind:str='loha',group_size:int=64):
        super().__init__()
        packed,scales,shape,self.group_size=pack_int4(linear.weight,group_size)
        self.register_buffer('packed_base',packed)
        self.register_buffer('scales',scales)
        self.shape=shape
        self.adapter=FactorDelta(shape,kind,rank,float(rank))
        if linear.bias is not None:self.bias=nn.Parameter(linear.bias.detach().clone(),requires_grad=False)
        else:self.register_parameter('bias',None)
    def forward(self,x):
        base=unpack_int4(self.packed_base,self.scales,self.shape,self.group_size,x.device,x.dtype)
        delta=self.adapter.delta().to(x.dtype)
        return F.linear(x,base+delta,self.bias.to(x.dtype) if self.bias is not None else None)
    @property
    def storage_bytes(self):return self.packed_base.numel()+self.scales.numel()*self.scales.element_size()


def quantize_expert_linears(module:nn.Module,rank=16,kind='loha')->int:
    """Convert ONLY expert-owned Linear submodules; never quantize router gates."""
    replaced=0
    for name,child in list(module.named_children()):
        if isinstance(child,nn.Linear):
            setattr(module,name,QuantizedLoHaLinear(child,rank,kind));replaced+=1
        else:replaced+=quantize_expert_linears(child,rank,kind)
    return replaced


class ExpertPager:
    """Explicit CPU/NVMe paging with no eviction during active forward/backward.

    NVMe page-out removes parameters (meta tensors) and consequently invalidates
    optimizer parameter references. Caller MUST finish the optimizer step,
    reinitialize optimizer after page-in; optim state is NOT serialized.
    """
    def __init__(self, experts:dict[str,nn.Module],root:Path):
        self.experts=dict(experts)
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self._active=set();self._disk={}
    def _path(self,key):
        import hashlib
        return self.root/(hashlib.sha256(key.encode()).hexdigest()+'.safetensors')
    @contextmanager
    def lease(self,key,device='cpu'):
        if key in self._active:raise RuntimeError('expert is already leased')
        self.page_in(key,device)
        self._active.add(key)
        try:yield self.experts[key]
        finally:self._active.remove(key)
    def page_out(self,key,tier='cpu'):
        if key in self._active:raise RuntimeError('cannot evict active expert graph')
        module=self.experts[key]
        if tier=='cpu':
            module.to('cpu')
        elif tier=='nvme':
            state={name:p.detach().cpu().contiguous() for name,p in module.state_dict().items()}
            if not state:raise ValueError('cannot save expert with empty state')
            path=self._path(key);temp=path.with_suffix('.tmp')
            save_file(state,str(temp));os.replace(temp,path)
            module.to_empty(device='meta')
            self._disk[key]=path
        else:raise ValueError('tier must be cpu or nvme')
    def page_in(self,key,device='cpu'):
        module=self.experts[key]
        if key in self._disk:
            state=load_file(str(self._disk.pop(key)),device='cpu')
            module.load_state_dict(state,assign=True,strict=True)
        module.to(device)
        return module
