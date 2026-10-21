"""Opt-in, fail-closed post-load pruning for *unpacked* ModuleList MoE.

Does not support packed fused experts (e.g. Qwen packed expert tensors), and
requires a verified gate module path. This mutates model architecture and must
be identical across peers sharing an elastic genesis / LoRA layout.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
import torch
from torch import nn

class UnavailableExpert(nn.Module):
    def forward(self, *a, **kw):
        raise RuntimeError('MoE selected a pruned expert: router mask integration failed')

@dataclass(frozen=True)
class ExpertSlice:
    path: str
    gate_path: str
    keep: tuple[int, ...]


def balanced_experts(n: int, count: int, seed: str = '') -> tuple[int,...]:
    if not 1 <= count <= n <= 100_000:raise ValueError('bad expert selection')
    # Same selection given same seed. Do not use Python hash() (per-process random).
    scored=sorted(range(n), key=lambda i:hashlib.sha256(f'{seed}:{i}'.encode()).digest())
    return tuple(sorted(scored[:count]))


def slice_moe_modulelist(model: nn.Module, expert_path: str, gate_path: str,
                         *, count: int, seed: str = '') -> ExpertSlice:
    """Reclaim non-selected expert weights *after initial model load*.

    Requires: explicit ModuleList of experts and a gate whose output tensor's
    last axis is logits over all those experts. A forward hook filters routing
    logits; unsupported gates refuse pruning, not silently corrupt weights.
    Initial checkpoint loading still needs memory for the full checkpoint.
    """
    experts=model.get_submodule(expert_path)
    gate=model.get_submodule(gate_path)
    if not isinstance(experts, nn.ModuleList) or not isinstance(gate, nn.Linear):
        raise ValueError('MoE pruning supports only ModuleList experts + Linear gate; packed layers need a separate backend')
    n=len(experts)
    if gate.out_features != n or not 1 <= count <= n:
        raise ValueError('gate output dimension does not match expert count')
    keep=balanced_experts(n,count,seed)
    keep_set=set(keep)
    # Ensure existing modules are not shared via tied parameters with another layer.
    to_free=[p for i,ex in enumerate(experts) if i not in keep_set for p in ex.parameters()]
    pruned_modules={id(m) for i,ex in enumerate(experts) if i not in keep_set for m in ex.modules()}
    retained={id(p) for module in model.modules() if id(module) not in pruned_modules for p in module.parameters(recurse=False)}
    if any(id(p) in retained for p in to_free):
        raise ValueError('refusing to delete experts with shared parameters')
    for i in range(n):
        if i not in keep_set: experts[i]=UnavailableExpert()
    mask=torch.tensor([i in keep_set for i in range(n)],dtype=torch.bool)
    def mask_gate(_module, _args, output):
        if not isinstance(output, torch.Tensor) or output.shape[-1]!=n:
            raise RuntimeError('unsupported MoE router output type')
        allowed=mask.to(output.device)
        return output.masked_fill(~allowed, torch.finfo(output.dtype).min)
    gate.register_forward_hook(mask_gate)
    return ExpertSlice(expert_path,gate_path,keep)


def pruned_model_estimate(model, paths: list[str]) -> dict:
    """Return named candidate ModuleLists; never automatically prune unknown architectures."""
    out=[]
    for path,m in model.named_modules():
        if isinstance(m,nn.ModuleList) and len(m)>=2 and ('expert' in path.lower()):
            out.append({'path':path,'experts':len(m),
                        'total_parameters':sum(p.numel() for p in m.parameters())})
    return {'candidates':out}
