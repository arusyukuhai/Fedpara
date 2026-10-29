"""On-demand disk-mapped, frozen experts with forward/backward recomputation.

Works for heterogeneous Expert(x)->Tensor modules, including arbitrary nested
linear/activation layouts, provided all expert parameters are FROZEN.
Model-specific MoE routing must call the expert module directly. Packed/fused
kernels cannot be replaced by this generic proxy.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call


class _ExpertCall(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,proxy):
        ctx.proxy=proxy
        ctx.save_for_backward(x.detach())
        with torch.no_grad():
            weights=proxy._load(x.device)
            y=functional_call(proxy.expert,weights,(x,),strict=False)
            if not isinstance(y,torch.Tensor) or y.shape[:-1]!=x.shape[:-1]:
                raise ValueError('expert(x) must return one Tensor and preserve leading axes')
            return y

    @staticmethod
    def backward(ctx,grad_output):
        if not ctx.needs_input_grad[0]:return None,None
        with torch.enable_grad():
            (saved,)=ctx.saved_tensors
            x=saved.to(grad_output.device).detach().requires_grad_(True)
            weights=ctx.proxy._load(x.device)
            y=functional_call(ctx.proxy.expert,weights,(x,),strict=False)
            dx=torch.autograd.grad(y,x,grad_output)[0]
        return dx,None


class DiskExpertProxy(nn.Module):
    """Frozen expert parameters remain CPU-backed and are paged per invocation.

    Distinct forwards keep their own detached activations for backward. Frozen
    base weights are reloaded only for the active expert on each pass.
    """
    def __init__(self, expert: nn.Module, directory, *, identifier: str):
        super().__init__()
        if any(p.requires_grad for p in expert.parameters()):
            raise ValueError('DiskExpertProxy only supports frozen experts')
        self.expert=expert
        self.path=Path(directory);self.path.mkdir(parents=True,exist_ok=True)
        self.identifier=identifier
        self._descriptors={}
        self._buffers={}
        for name, tensor in list(expert.named_parameters())+list(expert.named_buffers()):
            if not tensor.is_floating_point() or tensor.layout!=torch.strided:
                raise ValueError('unsupported expert tensor')
            key=hashlib.sha256((identifier+':'+name).encode()).hexdigest()
            path=self.path/(key+'.raw')
            with path.open('wb') as f:
                flat=tensor.detach().reshape(-1)
                for chunk in flat.split(max(1,2**20//tensor.element_size())):
                    f.write(chunk.to('cpu').contiguous().view(torch.uint8).numpy().tobytes())
                f.flush();os.fsync(f.fileno())
            desc=(str(path),tuple(tensor.shape),tensor.dtype)
            self._descriptors[name]=desc
            mapped=torch.from_file(str(path),shared=False,size=tensor.numel(),dtype=tensor.dtype).view(tensor.shape)
            if name in dict(expert.named_parameters()):
                dict(expert.named_parameters())[name].data=mapped
            else:
                # Buffered tensors: update the direct owning module.
                parent,_,leaf=name.rpartition('.')
                target=expert.get_submodule(parent) if parent else expert
                target._buffers[leaf]=mapped

    def _load(self,device):
        return {name:torch.from_file(path,shared=False,size=int(torch.Size(shape).numel()),dtype=dtype)
                      .view(shape).to(device=device)
                for name,(path,shape,dtype) in self._descriptors.items()}

    def forward(self,x):
        # The autograd function retains input activation but never saves expert
        # parameters in the forward graph; backward reloads this expert only.
        if torch.is_grad_enabled() and x.requires_grad:
            return _ExpertCall.apply(x,self)
        with torch.no_grad():
            y=functional_call(self.expert,self._load(x.device),(x,),strict=False)
            if not isinstance(y,torch.Tensor):raise TypeError('expert result must be tensor')
            return y


def install_disk_experts(model:nn.Module,expert_path, directory, *, indices=None):
    """Replace unpacked experts with lazy expert(x) wrappers.

    Do this before constructing adapters; this is not a packed-kernel hook.
    Frozen parameters are required for exact forward/input-gradient behavior.
    """
    experts=model.get_submodule(expert_path)
    if not isinstance(experts,(nn.ModuleList,nn.ModuleDict)):
        raise ValueError('only unpacked expert ModuleList/ModuleDict')
    keys=list(range(len(experts))) if isinstance(experts,nn.ModuleList) else list(experts.keys())
    if indices is not None:
        chosen=set(indices)
        if not chosen.issubset(set(keys)):raise ValueError('unknown expert index')
        keys=[k for k in keys if k in chosen]
    for key in keys:
        expert=experts[key]
        experts[key]=DiskExpertProxy(expert,directory,identifier=f'{expert_path}:{key}')
    return len(keys)


class StreamedFrozenExpert(nn.Module):
    """Per-call Expert loading from existing immutable CPU mappings.

    Keeps *no second expert archive*. Loading on GPU occurs for just the
    invoked expert, and backward recomputes it, including input gradients.
    Expert's own weights and factors are frozen; non-expert model/router can
    still train. This does not support packed MoE or training expert factors.
    """
    def __init__(self, expert:nn.Module):
        super().__init__()
        if any(p.requires_grad for p in expert.parameters()):
            raise ValueError('streamed expert contains trainable parameters; refuse silent gradient loss')
        if any(p.device.type!='cpu' for p in expert.parameters()):
            raise ValueError('streamed expert requires mapped CPU base')
        self.expert=expert

    def _load(self,device):
        # CPU branch leaves mmapped views intact. GPU branch loads the selected
        # expert only. Avoid retaining loaded device tensors between calls.
        values={}
        for n,t in list(self.expert.named_parameters())+list(self.expert.named_buffers()):
            if t.device.type!='cpu':raise RuntimeError('streamed expert mapping moved off CPU')
            values[n]=t.detach().to(device=device)
        return values

    def forward(self,x):
        # Checkpointed recomputation must reproduce forward exactly. Training
        # dropout would resample a mask; training BatchNorm mutates buffers.
        if any(m.training and isinstance(m,(nn.modules.dropout._DropoutNd,
                         nn.modules.batchnorm._BatchNorm))
               for m in self.expert.modules()):
            raise RuntimeError('stateful/stochastic frozen Expert training is unsupported')
        if torch.is_grad_enabled() and x.requires_grad:
            return _ExpertCall.apply(x,self)
        with torch.no_grad():
            result=functional_call(self.expert,self._load(x.device),(x,),strict=False)
        if not isinstance(result,torch.Tensor):raise TypeError('streamed expert must return tensor')
        return result


def install_streamed_experts(model:nn.Module, path:str, *, indices=None):
    """Explicit unpacked Expert modules only; no changes to signed model layout."""
    modules=model.get_submodule(path)
    if not isinstance(modules,(nn.ModuleList,nn.ModuleDict)):
        raise ValueError('streamed experts require ModuleList or ModuleDict')
    keys=list(range(len(modules))) if isinstance(modules,nn.ModuleList) else list(modules.keys())
    if indices is not None:
        wanted=set(indices)
        if not wanted.issubset(set(keys)):raise ValueError('unknown expert identifiers')
        keys=[k for k in keys if k in wanted]
    # Prevalidate everything so an incompatible expert cannot cause partial install.
    for key in keys:
        expert=modules[key]
        if any(p.requires_grad or p.device.type!='cpu' for p in expert.parameters()):
            raise ValueError('streamed expert must have immutable CPU tensors and no trainable factors')
        if any(b.device.type!='cpu' for b in expert.buffers()):
            raise ValueError('streamed expert buffers must be on CPU')
    for key in keys:
        modules[key]=StreamedFrozenExpert(modules[key])
    return len(keys)
