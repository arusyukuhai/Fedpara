"""Opt-in disk-mapped frozen Linear base weights with bounded device residency.

Supports nn.Linear including parametrized LoRA/LoHa weights. Training gradients
flow to inputs and factor parameters. Base weights are immutable; every forward
and backward loads only the active layer's frozen weight from its disk file.
Supports Hugging Face GPT-2 style Conv1D (stored [in,out]) as well as
Linear ([out,in]); refuses unsupported tied or packed projections.
"""
from __future__ import annotations

import hashlib
import math
import os
import types
from collections import OrderedDict
from threading import RLock
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def _mapping(path, shape, dtype, device):
    if getattr(path,'_fedpara_is_paged',False):
        if torch.device(device).type!='cpu':
            raise RuntimeError('paged virtual tensor must execute through bounded tiling')
        return path
    if isinstance(path,torch.Tensor):
        value=path
    else:
        count=math.prod(shape)
        value=torch.from_file(str(path),shared=True,size=count,dtype=dtype).view(shape)
    return value if torch.device(device).type=='cpu' else value.to(device=device)


def _source_key(source):
    return f'tensor:{id(source)}' if isinstance(source,torch.Tensor) else str(source)


class DiskWeightCache:
    """Memory-aware accelerator LRU for frozen base weights.

    Hot weights stay on GPU up to the measured free-memory budget; large weights
    use two-dimensional tiling and never occupy a whole contiguous GPU tensor.
    The CPU path is OS mmap/page-cache backed, with no artificial RAM ceiling.
    """
    def __init__(self, device='cpu', *, fraction=.6, limit_mib=0, tile_mib=16):
        self.device=torch.device(device)
        if not 0 <= fraction <= .85 or limit_mib < 0 or not 1 <= tile_mib <= 256:
            raise ValueError('invalid disk-cache settings')
        self.fraction=fraction
        self.limit_bytes=int(limit_mib*1024**2)
        self.tile_bytes=int(tile_mib*1024**2)
        if self.device.type=='cpu':
            # Let large-RAM peers execute wider GEMM tiles while Linux/macOS
            # independently retain the file-backed model in the page cache.
            from .device_planner import _ram_available
            free,_=_ram_available()
            self.tile_bytes=max(self.tile_bytes,
                       min(256*1024**2,max(1*1024**2,int(free//64))))
        self.lock=RLock()
        self.items=OrderedDict()
        self.used=0
        self.hits=0;self.misses=0
        self.max_bytes=self._budget()

    def _budget(self):
        if self.device.type=='cpu':return 0  # let the OS page cache use free RAM
        free=0
        if self.device.type=='cuda':
            try:free=int(torch.cuda.mem_get_info(self.device)[0])
            except (RuntimeError,AssertionError):pass
        elif self.device.type=='mps':
            try:
                free=max(0,int(torch.mps.recommended_max_memory())-int(torch.mps.current_allocated_memory()))
            except (RuntimeError,AttributeError):pass
        return min(self.limit_bytes,int(free*self.fraction)) if self.limit_bytes else int(free*self.fraction)

    def invalidate(self,path=None):
        with self.lock:
            for key in list(self.items):
                if path is None or key[0]==_source_key(path):
                    value=self.items.pop(key)
                    self.used-=value.numel()*value.element_size()

    def weight(self,path,shape,dtype,device,*, compute_dtype=None):
        device=torch.device(device)
        if getattr(path,'_fedpara_is_paged',False):return None
        if device.type=='cpu':return None
        target_dtype=compute_dtype or dtype
        size=math.prod(shape)*target_dtype.itemsize
        with self.lock:
            if size>self.max_bytes or self.max_bytes==0:return None
            key=(_source_key(path),tuple(shape),dtype,str(device),target_dtype)
            if key in self.items:
                self.hits+=1;self.items.move_to_end(key)
                return self.items[key]
            while self.items and self.used+size>self.max_bytes:
                _,old=self.items.popitem(last=False)
                self.used-=old.numel()*old.element_size()
            # The caller's activations and optimizer still need headroom. If
            # allocation fails under concurrent demand, back off to tiling.
            try:
                value=_mapping(path,shape,dtype,device)
                if value.dtype!=target_dtype:value=value.to(target_dtype)
            except (torch.cuda.OutOfMemoryError,MemoryError):
                self.invalidate()
                return None
            self.items[key]=value;self.used+=size;self.misses+=1
            return value

    def stats(self):
        with self.lock:
            return dict(budget_bytes=self.max_bytes,resident_bytes=self.used,
                        hits=self.hits,misses=self.misses,entries=len(self.items),
                        tile_bytes=self.tile_bytes)


def _tiled_linear_forward(x,weight,bias,device,tile_bytes):
    in_features=weight.shape[1];out_features=weight.shape[0]
    x2=x.reshape(-1,in_features)
    y=x2.new_zeros((x2.shape[0],out_features))
    # Avoid allocating a tensor larger than the configured staging budget even
    # for a single *row* of a spectacularly rectangular matrix.
    elem=x2.element_size()
    ostep=max(1,min(out_features,tile_bytes//max(elem*min(in_features,4096),1)))
    for o in range(0,out_features,ostep):
        oe=min(out_features,o+ostep)
        istep=max(1,min(in_features,tile_bytes//max(elem*(oe-o),1)))
        for i in range(0,in_features,istep):
            ie=min(in_features,i+istep)
            block=weight[o:oe,i:ie].to(device=device,dtype=x.dtype)
            y[:,o:oe].add_(x2[:,i:ie] @ block.T)
        if bias is not None:
            y[:,o:oe].add_(bias[o:oe].to(device=device,dtype=y.dtype))
    return y.view(*x.shape[:-1],out_features)


def _tiled_linear_backward(grad,weight,device,tile_bytes):
    out_features,in_features=weight.shape
    gy=grad.reshape(-1,out_features)
    gx=gy.new_zeros((gy.shape[0],in_features))
    elem=gy.element_size()
    istep=max(1,min(in_features,tile_bytes//max(elem*min(out_features,4096),1)))
    for i in range(0,in_features,istep):
        ie=min(in_features,i+istep)
        ostep=max(1,min(out_features,tile_bytes//max(elem*(ie-i),1)))
        for o in range(0,out_features,ostep):
            oe=min(out_features,o+ostep)
            block=weight[o:oe,i:ie].to(device=device,dtype=gy.dtype)
            gx[:,i:ie].add_(gy[:,o:oe] @ block)
    return gx.view(*grad.shape[:-1],in_features)


class _FrozenDiskLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,bias,path,shape,dtype,cache):
        # A signed remote event may arrive between forward and backward.
        # Pin the virtual version to compute the correct input gradient and
        # keep all tiles within this one kernel on the same frozen revision.
        path=path.snapshot() if hasattr(path,'snapshot') else path
        ctx.path,ctx.shape,ctx.dtype,ctx.cache=path,shape,dtype,cache
        ctx.has_bias=bias is not None
        if cache is not None:
            w=cache.weight(path,shape,dtype,x.device,compute_dtype=x.dtype)
        else:w=None
        if w is not None:
            return F.linear(x,w,bias.to(x.device,dtype=x.dtype) if bias is not None else None)
        base=_mapping(path,shape,dtype,'cpu')
        tile=cache.tile_bytes if cache else 16*1024**2
        return _tiled_linear_forward(x,base,bias,x.device,tile)

    @staticmethod
    def backward(ctx,grad_out):
        grad_x=grad_bias=None
        if ctx.needs_input_grad[0]:
            cache=ctx.cache
            w=cache.weight(ctx.path,ctx.shape,ctx.dtype,grad_out.device,
                           compute_dtype=grad_out.dtype) if cache else None
            if w is not None:
                grad_x=grad_out @ w
            else:
                base=_mapping(ctx.path,ctx.shape,ctx.dtype,'cpu')
                grad_x=_tiled_linear_backward(grad_out,base,grad_out.device,
                           cache.tile_bytes if cache else 16*1024**2)
        if ctx.has_bias and ctx.needs_input_grad[1]:
            grad_bias=grad_out.reshape(-1,grad_out.shape[-1]).sum(0)
        return grad_x,grad_bias,None,None,None,None


def _factor_linear(x, spec):
    if spec.dense:
        return F.linear(x, spec.parts[0] if hasattr(spec,'parts') else spec.part0)
    parts=list(spec.parts) if hasattr(spec,'parts') else [getattr(spec,f'part{i}') for i in range(spec.nparts)]
    if len(spec.shape)!=2:
        raise ValueError('disk Linear offload requires 2-D weight factors')
    if spec.kind=='lora':
        a,b=parts
        return F.linear(F.linear(x,b.T),a) * (spec.alpha/spec.rank)
    if spec.kind=='loha':
        a,b,c,d=parts
        # Hadamard of two rank-r products is rank <= r*r, without creating
        # an out_features x in_features dense effective weight matrix.
        projected=torch.einsum('...i,ir,is->...rs',x,b,d)
        return torch.einsum('...rs,or,os->...o',projected,a,c) * (spec.alpha/spec.rank)
    raise ValueError('unsupported factor type')


def _embedding_residual(ids, spec):
    if spec.dense:
        return spec.parts[0][ids]
    parts=list(spec.parts) if hasattr(spec,'parts') else [getattr(spec,f'part{i}') for i in range(spec.nparts)]
    if len(spec.shape)!=2:raise ValueError('disk Embedding requires 2-D weight factors')
    if spec.kind=='lora':
        a,b=parts
        return (a[ids] @ b.T)*(spec.alpha/spec.rank)
    if spec.kind=='loha':
        a,b,c,d=parts
        return ((a[ids] @ b.T)*(c[ids] @ d.T))*(spec.alpha/spec.rank)
    raise ValueError('unsupported embedding factor type')


def _disk_embedding_forward(module, ids):
    if module.max_norm is not None:
        raise RuntimeError('max_norm requires in-place embedding renormalization and is unsafe with disk base')
    # Only the indexed rows are ever copied from the file-backed CPU tensor.
    source=module._fedpara_disk_source
    cpu_ids=ids.to('cpu') if ids.device.type!='cpu' else ids
    base=_mapping(source,module._fedpara_disk_shape,module._fedpara_disk_dtype,'cpu')
    if getattr(base,'_fedpara_is_paged',False):
        y=base.gather_rows(cpu_ids).to(ids.device)
    else:
        y=F.embedding(cpu_ids,base,padding_idx=module.padding_idx,
                      scale_grad_by_freq=False,sparse=False).to(ids.device)
    overlay=getattr(module,'_fedpara_disk_overlay',None)
    if overlay is not None:
        residual=_embedding_residual(ids,overlay.local)
        if hasattr(overlay,'anchor'):
            residual=residual-_embedding_residual(ids,overlay.anchor)
        for remote in overlay.remotes.values():
            residual=residual+_embedding_residual(ids,remote)
        residual=residual/(1+len(overlay.remotes))
        if module.padding_idx is not None:
            # Native nn.Embedding keeps the *value* of the padding row while
            # suppressing its gradient. Reproduce both semantics.
            residual=torch.where((ids==module.padding_idx).unsqueeze(-1),
                                 residual.detach(),residual)
        y=y+residual.to(dtype=y.dtype)
    return y


def _disk_forward(module, x):
    if x.dtype not in (torch.float32,torch.float64,torch.float16,torch.bfloat16):
        raise TypeError('unsupported input dtype for disk-backed Linear')
    y=_FrozenDiskLinear.apply(x,module.bias,module._fedpara_disk_source,
                              module._fedpara_disk_shape,module._fedpara_disk_dtype,
                              getattr(module,"_fedpara_disk_cache",None))
    overlay=getattr(module,'_fedpara_disk_overlay',None)
    if overlay is not None:
        residual=_factor_linear(x,overlay.local)
        if hasattr(overlay,'anchor'):
            residual=residual-_factor_linear(x,overlay.anchor)
        for remote in overlay.remotes.values():
            residual=residual+_factor_linear(x,remote)
        y=y+residual/(1+len(overlay.remotes))
    return y


def _is_conv1d(module):
    """Hugging Face GPT-2 Conv1D without materializing parametrized weights."""
    if not any(cls.__name__=='Conv1D' for cls in type(module).__mro__):return False
    plist=getattr(module,'parametrizations',None)
    base=(plist.weight.original if plist is not None and 'weight' in plist
          else module._parameters.get('weight'))
    bias=module._parameters.get('bias')
    return (isinstance(base,torch.Tensor) and isinstance(bias,torch.Tensor)
            and base.ndim==2 and bias.ndim==1
            and base.shape[1]==bias.numel())


class _FrozenDiskConv1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,bias,source,shape,dtype,cache):
        source=source.snapshot() if hasattr(source,'snapshot') else source
        ctx.source,ctx.shape,ctx.dtype,ctx.cache=source,shape,dtype,cache
        weight=(cache.weight(source,shape,dtype,x.device,compute_dtype=x.dtype)
                if cache is not None else None)
        if weight is None:
            weight=_mapping(source,shape,dtype,'cpu')
        if not getattr(weight,'_fedpara_is_paged',False) and weight.device==x.device and weight.dtype==x.dtype:
            return F.linear(x,weight.T,bias.to(x.device,dtype=x.dtype))
        tile=cache.tile_bytes if cache is not None else 16*1024**2
        return _tiled_linear_forward(x,weight.T,bias,x.device,tile)

    @staticmethod
    def backward(ctx,grad):
        gx=gb=None
        if ctx.needs_input_grad[0]:
            cache=ctx.cache
            weight=(cache.weight(ctx.source,ctx.shape,ctx.dtype,grad.device,
                    compute_dtype=grad.dtype) if cache is not None else None)
            if weight is None:weight=_mapping(ctx.source,ctx.shape,ctx.dtype,'cpu')
            if not getattr(weight,'_fedpara_is_paged',False) and weight.device==grad.device and weight.dtype==grad.dtype:
                gx=grad @ weight.T
            else:
                gx=_tiled_linear_backward(grad,weight.T,grad.device,
                     cache.tile_bytes if cache is not None else 16*1024**2)
        if ctx.needs_input_grad[1]:
            gb=grad.reshape(-1,grad.shape[-1]).sum(0)
        return gx,gb,None,None,None,None


def _factor_conv1d(x,spec):
    parts=(list(spec.parts) if hasattr(spec,'parts')
           else [getattr(spec,f'part{i}') for i in range(spec.nparts)])
    if spec.dense:return x @ parts[0]
    if spec.kind=='lora':
        a,b=parts
        return (x @ a) @ b.T * (spec.alpha/spec.rank)
    if spec.kind=='loha':
        a,b,c,d=parts
        projection=torch.einsum('...i,ir,is->...rs',x,a,c)
        return torch.einsum('...rs,or,os->...o',projection,b,d) * (spec.alpha/spec.rank)
    raise ValueError('unsupported Conv1D factor')


def _disk_conv1d_forward(module,x):
    result=_FrozenDiskConv1D.apply(x,module.bias,module._fedpara_disk_source,
            module._fedpara_disk_shape,module._fedpara_disk_dtype,
            getattr(module,'_fedpara_disk_cache',None))
    overlay=getattr(module,'_fedpara_disk_overlay',None)
    if overlay is not None:
        residual=_factor_conv1d(x,overlay.local)
        if hasattr(overlay,'anchor'):
            residual=residual-_factor_conv1d(x,overlay.anchor)
        for remote in overlay.remotes.values():
            residual=residual+_factor_conv1d(x,remote)
        result=result+residual/(1+len(overlay.remotes))
    return result


def install_disk_linear_offload(model: nn.Module, adapter, directory, *, min_weight_mib=0.,
                                allowed_paths=None,cache_fraction=.6,cache_mib=0,tile_mib=16,
                                include_embeddings=False,max_dir_bytes=None):
    """Map eligible *frozen* Linear weights from disk, replacing their forward.

    Call AFTER moving the model and adapter factors to the target device.
    Modules with tied weights are refused to prevent inconsistent aliases.
    Disk files are regular raw tensor binaries; no pickle is ever read.
    """
    if not math.isfinite(min_weight_mib) or min_weight_mib<0:
        raise ValueError('invalid minimum disk offload weight size')
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    allowed=set(allowed_paths) if allowed_paths is not None else None
    owners={}
    for owner_path,owner_module in model.named_modules():
        if isinstance(owner_module,nn.Linear) or _is_conv1d(owner_module):
            plist=getattr(owner_module,'parametrizations',None)
            weight=(plist.weight.original if plist is not None and 'weight' in plist
                    else owner_module.weight)
            owners.setdefault(id(weight),set()).add(owner_path)
        elif isinstance(owner_module,nn.Embedding):
            owners.setdefault(id(owner_module.weight),set()).add(owner_path)
    cache=DiskWeightCache(next(model.parameters()).device, fraction=cache_fraction,
                          limit_mib=cache_mib,tile_mib=tile_mib)
    count=0; bytes_total=0; skipped=[]
    for path,module in model.named_modules():
        if not (isinstance(module,nn.Linear) or _is_conv1d(module)) or (allowed is not None and path not in allowed):
            continue
        plist=getattr(module,'parametrizations',None)
        overlay=plist.weight[0] if plist is not None and 'weight' in plist else None
        if overlay is not None and (not hasattr(overlay,'local') or not hasattr(overlay,'remotes')):
            skipped.append((path,'unsupported parametrization'));continue
        base=plist.weight.original if overlay is not None else module.weight
        aliases=owners.get(id(base),set())
        tied_eligible=(include_embeddings and aliases and all(
            isinstance(model.get_submodule(ap), (nn.Linear,nn.Embedding)) for ap in aliases))
        if base.requires_grad or (len(aliases)!=1 and not tied_eligible):
            skipped.append((path,'trainable or unsupported tied base'));continue
        if not base.is_floating_point() or base.layout!=torch.strided or base.ndim!=2:
            skipped.append((path,'unsupported base weight'));continue
        size=base.numel()*base.element_size()
        if size < min_weight_mib*1024*1024:
            continue
        # Name includes shape and path; persistent dir belongs to a single
        # immutable model generation. Refuse reinstallation on the same module.
        if hasattr(module,'_fedpara_disk_path'):
            raise RuntimeError(f'disk offload already installed for {path}')
        if max_dir_bytes is not None and bytes_total+size>max_dir_bytes:
            raise RuntimeError('disk Linear base exceeds configured per-model capacity')
        inherited=getattr(base,'_fedpara_raw_path',None)
        if inherited:
            file=Path(inherited)
            if file.stat().st_size!=size:
                raise RuntimeError(f'offloaded backing file size changed: {path}')
            source=str(file)
        elif getattr(model,'_fedpara_streaming_base_info',{}).get('mode')=='read-only-mmap':
            # Reuse safetensors' existing file-backed mapping; do NOT write
            # a second copy of a 300B base just to offload its Linear layers.
            source=base.detach()
        else:
            fname=hashlib.sha256(path.encode()).hexdigest()+'.weight'
            file=directory/fname
            with file.open('wb') as f:
                for block in base.detach().reshape(-1).split(max(1,2**20//base.element_size())):
                    f.write(block.to('cpu').contiguous().view(torch.uint8).numpy().tobytes())
                f.flush();os.fsync(f.fileno())
            source=str(file)
        shape=tuple(base.shape);dtype=base.dtype
        if not isinstance(source,torch.Tensor):
            base.data=_mapping(source,shape,dtype,'cpu')
        module._fedpara_disk_path=str(source) if isinstance(source,str) else '<safetensors-mmap>'
        module._fedpara_disk_source=source
        module._fedpara_disk_shape=shape
        module._fedpara_disk_dtype=dtype
        module._fedpara_disk_cache=cache
        # Do not register overlay twice: nn.Linear's parametrization contains it.
        object.__setattr__(module,'_fedpara_disk_overlay',overlay)
        module.forward=types.MethodType(_disk_conv1d_forward if _is_conv1d(module) else _disk_forward,module)
        bytes_total+=size;count+=1
    embeddings=0
    if include_embeddings:
        for path,module in model.named_modules():
            if not isinstance(module,nn.Embedding):continue
            if module.max_norm is not None or module.scale_grad_by_freq or module.sparse:
                skipped.append((path,'embedding max_norm/frequency/sparse unsupported'));continue
            plist=getattr(module,'parametrizations',None)
            overlay=plist.weight[0] if plist is not None and 'weight' in plist else None
            if overlay is not None and not hasattr(overlay,'local'):
                skipped.append((path,'unsupported embedding parametrization'));continue
            original=plist.weight.original if overlay is not None else module.weight
            if original.requires_grad:continue
            source=getattr(original,'_fedpara_raw_path',None)
            if source is None and getattr(model,'_fedpara_streaming_base_info',{}).get('mode')=='read-only-mmap':
                source=original.detach()
            if source is None:
                skipped.append((path,'embedding needs streaming safetensors source'));continue
            object.__setattr__(module,'_fedpara_disk_overlay',overlay)
            module._fedpara_disk_path=source if isinstance(source,str) else '<safetensors-mmap>'
            module._fedpara_disk_source=source
            module._fedpara_disk_shape=tuple(original.shape)
            module._fedpara_disk_dtype=original.dtype
            module._fedpara_disk_cache=cache
            module.forward=types.MethodType(_disk_embedding_forward,module)
            embeddings+=1
    return {'layers':count,'embeddings':embeddings,
            'disk_weight_bytes':bytes_total,'skipped':skipped,'cache':cache.stats()}


def place_streaming_compute_device(model: nn.Module, adapter, device, *, cache=None):
    """Move small/non-offloaded state and adapter factors to accelerator only.

    Frozen mapped Linear/Embedding bases stay on CPU. Refuse any alias whose
    regular forward could touch an offloaded CPU base, instead of silently
    causing a CPU/GPU mismatch or materializing the base via parametrization.
    """
    device=torch.device(device)
    if device.type=='cpu':return {'resident_bytes':0,'offloaded_bases':0}
    frozen={}
    for prefix,mod in model.named_modules():
        if not hasattr(mod,'_fedpara_disk_source'):continue
        if not (isinstance(mod,(nn.Linear,nn.Embedding)) or _is_conv1d(mod)):
            raise RuntimeError(f'unsupported disk compute module {prefix}')
        plist=getattr(mod,'parametrizations',None)
        weight=(plist.weight.original if plist is not None and 'weight' in plist else mod.weight)
        frozen[id(weight)]=weight
    if not frozen:
        raise RuntimeError('no streaming kernels were installed before accelerator placement')
    owners={}
    for prefix,mod in model.named_modules():
        for name,p in mod._parameters.items():
            if p is not None and id(p) in frozen:
                # ParametrizationList.original is an alias of the same frozen
                # tensor, and is never called by the replaced module forward.
                if name!='original' and not hasattr(mod,'_fedpara_disk_source'):
                    raise RuntimeError(f'unsupported shared offloaded weight owner: {prefix}.{name}')
    moved=0;visited=set()
    for mod in model.modules():
        for name,p in mod._parameters.items():
            if p is None or id(p) in visited:continue
            visited.add(id(p))
            if id(p) in frozen:continue
            size=p.numel()*p.element_size()
            if device.type=='cuda':
                free=torch.cuda.mem_get_info(device)[0]
                if size>free*.8:
                    raise MemoryError(f'nonoffloaded parameter {size/1024**2:.1f}MiB exceeds remaining GPU budget; model-specific offload required')
            with torch.no_grad():p.data=p.data.to(device,non_blocking=False)
            moved+=size
        for name,b in mod._buffers.items():
            if b is None or b.device==device:continue
            size=b.numel()*b.element_size()
            if device.type=='cuda' and size>torch.cuda.mem_get_info(device)[0]*.8:
                raise MemoryError('nonoffloaded buffer too large for accelerator')
            mod._buffers[name]=b.to(device)
            moved+=size
    if cache:
        cache.device=device
        cache.max_bytes=cache._budget()
    return {'resident_bytes':moved,'offloaded_bases':len(frozen),
            'cache_budget_bytes':cache.max_bytes if cache else 0}
