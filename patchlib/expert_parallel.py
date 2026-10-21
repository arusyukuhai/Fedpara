"""Expert-level distributed execution for unpacked ModuleList MoE.

Designed as a transport-neutral, backprop-capable component. The supplied RPC
transport MUST be a verified, bounded direct-peer data channel; no model code is
received or executed over the network. Existing model top-k routing remains in
charge: only selected expert modules are replaced with autograd RPC proxies.

This is NOT a universal implementation for packed/fused MoE kernels. Those need
an architecture-specific tensor splitting adapter.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable
import torch
from torch import nn
from safetensors.torch import save as tensor_save, load as tensor_load

MAX_RPC_BYTES=4*1024*1024
MAX_INFLIGHT=128
MAX_TOKENS=4096
MAX_HIDDEN=16384
PATH=re.compile(r'^[A-Za-z0-9_.-]{1,240}$')


def discover_moe(model: nn.Module) -> list[dict[str,Any]]:
    """Recognize conventional MoE top-k gates, with explicit unsupported cases.

    Handles unpacked Mixtral/Qwen2 legacy, DeepSeek MoEGate weight and Switch
    Transformers style router.classifier + ModuleDict experts. The newest fused
    HF kernels are reported as packed but are NOT silently rewritten.
    """
    found=[]
    for path, block in model.named_modules():
        experts=getattr(block,'experts',None)
        if experts is None:continue
        expert_count=len(experts) if isinstance(experts,(nn.ModuleList,nn.ModuleDict)) else None
        gate=getattr(block,'gate',None)
        gate_name='gate'
        if gate is None:gate=getattr(block,'router',None);gate_name='router'
        if gate is None:continue
        supported=False;gate_kind='unknown'
        if expert_count is not None:
            if isinstance(gate,nn.Linear) and gate.out_features==expert_count:
                supported=True;gate_kind='linear-topk'
            elif (isinstance(getattr(gate,'weight',None),nn.Parameter) and
                  gate.weight.ndim==2 and gate.weight.shape[0]==expert_count):
                # DeepSeek custom MoEGate: forward already computes top-k indices.
                supported=True;gate_kind='deepseek-weight-topk'
            elif isinstance(getattr(gate,'classifier',None),nn.Linear) and gate.classifier.out_features==expert_count:
                supported=True;gate_kind='switch-classifier'
            # ModuleDict indices must follow switch-transformers expert_N pattern.
            if isinstance(experts,nn.ModuleDict) and sorted(experts.keys())!=sorted('expert_'+str(i) for i in range(expert_count)):
                supported=False
        elif any(isinstance(getattr(experts,field,None),torch.Tensor) for field in
                 ('gate_up_proj','down_proj','w1','w2','w3')):
            gate_kind='packed/fused-unsupported'
        if supported or gate_kind=='packed/fused-unsupported':
            found.append({'path':path,'experts':path+'.experts' if path else 'experts',
                'gate':(path+'.' if path else '')+gate_name,
                'count':expert_count,'router':gate_kind,'supported':supported,
                'family_hint':('mixtral' if 'mixtral' in type(block).__module__.lower() or 'block_sparse_moe' in path
                      else 'qwen' if 'qwen' in type(block).__module__.lower()
                      else 'deepseek' if 'deepseek' in type(block).__module__.lower()
                      else 'switch' if 'switch' in type(block).__module__.lower() else 'generic')})
    return found


def expert_owner(room: str, expert: str, peers: dict[str,int], replicas: int=2) -> tuple[str,...]:
    """Stable weighted rendezvous placement, no automatic rebalance during a step.

    Call with the same *versioned roster* on every peer; fail closed on conflict.
    """
    if not peers or not 1<=replicas<=32:raise ValueError('empty roster or bad replica count')
    scores=[]
    for peer,weight in peers.items():
        if not PATH.fullmatch(peer) or not 1<=weight<=100:raise ValueError('bad peer/capacity')
        d=hashlib.sha256(f'fp-expert/v1/{room}/{expert}/{peer}'.encode()).digest()
        u=(int.from_bytes(d[:8],'big')+1)/(2**64+1)
        scores.append((-math.log(u)/weight,peer))
    return tuple(peer for _,peer in sorted(scores)[:min(replicas,len(scores))])


class ExpertRPCError(RuntimeError):pass


def _tensor_msg(x:torch.Tensor,max_bytes=MAX_RPC_BYTES)->bytes:
    if not isinstance(x,torch.Tensor) or not x.is_floating_point() or x.ndim not in (1,2,3):
        raise ExpertRPCError('unsupported activation tensor')
    if x.numel()>MAX_TOKENS*MAX_HIDDEN:raise ExpertRPCError('activation too large')
    payload=tensor_save({'x':x.detach().to('cpu').contiguous()})
    if len(payload)>max_bytes:raise ExpertRPCError('RPC tensor exceeds byte limit')
    return payload


def _read_tensor(payload:bytes,max_bytes=MAX_RPC_BYTES)->torch.Tensor:
    if not isinstance(payload,bytes) or len(payload)>max_bytes:raise ExpertRPCError('invalid RPC bytes')
    tensors=tensor_load(payload)
    if set(tensors)!={'x'}:raise ExpertRPCError('unexpected RPC tensor keys')
    x=tensors['x']
    if x.ndim not in (1,2,3) or not x.is_floating_point() or x.numel()>MAX_TOKENS*MAX_HIDDEN:
        raise ExpertRPCError('invalid activation shape/dtype')
    return x


class ExpertOwner:
    """Execute an expert forward and backward, keep graph only until backward/TTL.

    Caller is authenticated at the transport layer. Handles are scoped to peer,
    request id and base generation; repeat requests are rejected.
    """
    def __init__(self, experts:dict[str,nn.Module],generation:str,ttl:float=120.,max_pending:int=MAX_INFLIGHT):
        self.experts=dict(experts);self.generation=generation
        self.ttl=ttl;self.max_pending=max_pending
        self._pending={};self._lock=threading.RLock()
    def handle(self,peer:str,req:dict)->dict:
        if not isinstance(req,dict) or req.get('version')!=1 or req.get('generation')!=self.generation:
            raise ExpertRPCError('invalid expert RPC generation/version')
        expert=req.get('expert');rid=req.get('id');op=req.get('op')
        if not isinstance(expert,str) or expert not in self.experts or not isinstance(rid,str) or not re.fullmatch('[a-f0-9]{32}',rid):
            raise ExpertRPCError('invalid expert or request id')
        key=(peer,rid)
        with self._lock:
            t=time.monotonic()
            for k,v in list(self._pending.items()):
                if t-v[0]>self.ttl:del self._pending[k]
            if op=='forward':
                if key in self._pending or len(self._pending)>=self.max_pending:raise ExpertRPCError('duplicate/overloaded request')
                x=_read_tensor(req.get('tensor')).detach().requires_grad_(True)
                module=self.experts[expert]
                device=next(module.parameters(),torch.empty((),device=x.device)).device
                x=x.to(device).detach().requires_grad_(True)
                with torch.enable_grad(): y=module(x)
                if not isinstance(y,torch.Tensor) or y.ndim!=x.ndim:raise ExpertRPCError('expert forward must return a tensor')
                self._pending[key]=(t,expert,x,y)
                return {'version':1,'id':rid,'tensor':_tensor_msg(y)}
            if op=='backward':
                entry=self._pending.pop(key,None)
                if entry is None or entry[1]!=expert:raise ExpertRPCError('no matching pending forward')
                _,_,x,y=entry
                dy=_read_tensor(req.get('tensor')).to(y.device,y.dtype)
                if y.shape!=dy.shape:raise ExpertRPCError('gradient shape mismatch')
                y.backward(dy)
                return {'version':1,'id':rid,'tensor':_tensor_msg(x.grad)}
        raise ExpertRPCError('unsupported expert op')


class _RemoteExpertFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,endpoint,expert,generation):
        import secrets
        rid=secrets.token_hex(16)
        response=endpoint.request({'version':1,'generation':generation,'expert':expert,
            'op':'forward','id':rid,'tensor':_tensor_msg(x)})
        if response.get('id')!=rid:raise ExpertRPCError('mismatched forward response')
        ctx.endpoint,ctx.expert,ctx.generation,ctx.id=endpoint,expert,generation,rid
        return _read_tensor(response['tensor']).to(device=x.device,dtype=x.dtype)
    @staticmethod
    def backward(ctx,dy):
        response=ctx.endpoint.request({'version':1,'generation':ctx.generation,
            'expert':ctx.expert,'op':'backward','id':ctx.id,'tensor':_tensor_msg(dy)})
        if response.get('id')!=ctx.id:raise ExpertRPCError('mismatched backward response')
        dx=_read_tensor(response['tensor']).to(device=dy.device,dtype=dy.dtype)
        return dx,None,None,None


class RemoteExpert(nn.Module):
    """Drop-in for expert(x)->Tensor; network must progress on ANOTHER thread.

    No silent local fallback: if owner disappears mid-backward the step fails and
    must roll back before electing a verified replica with identical weights.
    """
    def __init__(self,endpoint,expert:str,generation:str):
        super().__init__()
        self.endpoint,self.expert,self.generation=endpoint,expert,generation
    def forward(self,x):return _RemoteExpertFn.apply(x,self.endpoint,self.expert,self.generation)


def install_remote_experts(model:nn.Module, experts_path:str, remote:dict[int,Any],generation:str):
    experts=model.get_submodule(experts_path)
    if not isinstance(experts,(nn.ModuleList,nn.ModuleDict)):
        raise ValueError('only unpacked ModuleList or Switch ModuleDict experts are supported')
    for i,endpoint in remote.items():
        if type(i) is not int or not 0<=i<len(experts):raise ValueError('bad expert index')
        name=str(i) if isinstance(experts,nn.ModuleList) else 'expert_'+str(i)
        if isinstance(experts,nn.ModuleDict) and name not in experts:
            raise ValueError('missing Switch expert key')
        experts[name if isinstance(experts,nn.ModuleDict) else i]=RemoteExpert(
            endpoint,f'{experts_path}.{name}',generation)
    return experts


class LocalEndpoint:
    """Loopback transport for correctness tests; swap for authenticated DataChannel RPC."""
    def __init__(self,owner:ExpertOwner,peer='test-peer'):self.owner,self.peer=owner,peer
    def request(self,msg):return self.owner.handle(self.peer,msg)
