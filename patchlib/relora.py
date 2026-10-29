"""Periodic ReLoRA-like refresh for FedSum.

A refresh changes trainable low-rank coordinates, never the learned base.
For the live protocol it is invoked AFTER the signed local event is durable
and its effective delta has been merged into the base. For offline central
training the local delta is merged directly before refreshing.
"""
from __future__ import annotations

import hashlib

import torch

from .core import FactorDelta, PatchError


def _refresh_spec(spec):
    """Keep Parameter identity stable; callers rebuild the optimizer separately."""
    fresh = FactorDelta(spec.shape, spec.kind, spec.rank, spec.alpha,
                        init_style=spec.init_style)
    if len(spec.parts) != len(fresh.parts) or any(
            tuple(param.shape) != tuple(value.shape)
            for param, value in zip(spec.parts, fresh.parts)):
        raise PatchError('cannot refresh incompatible mixed-rank adapter factors')
    for param, value in zip(spec.parts, fresh.parts):
        param.copy_(value.to(device=param.device, dtype=param.dtype))
        param.grad = None


def reset_committed_factors(adapter, *, seed: int):
    """Refresh live factor+anchor pairs after a *committed* signed event.

    This requires anchor == local (no pending gradients/unsent update). It can
    be repeated deterministically during journal replay, after a crash, and
    never mutates the base weights. Remote patches must not trigger a refresh.
    """
    with adapter.lock, torch.no_grad():
        for g in adapter.groups:
            if not hasattr(g.overlay, 'anchor'):
                raise PatchError('ReLoRA live refresh requires an anchor')
            if any(not torch.equal(p.detach(), getattr(g.overlay.anchor, f'part{i}'))
                   for i, p in enumerate(g.overlay.local.parts)):
                raise PatchError('cannot refresh while uncommitted local delta exists')
        # Seeding within fork_rng leaves sampling and unrelated training RNG
        # unchanged, and yields the same factors on signed-journal replay.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for g in adapter.groups:
                _refresh_spec(g.overlay.local)
                for i, p in enumerate(g.overlay.local.parts):
                    getattr(g.overlay.anchor, f'part{i}').copy_(p.detach())


def merge_and_reset_central(adapter, *, seed: int):
    """Merge the uncommitted central adapter exactly once then refresh it."""
    with adapter.lock, torch.no_grad():
        if any(hasattr(g.overlay, 'anchor') or len(g.overlay.remotes)
               for g in adapter.groups):
            raise PatchError('central merge only supports local unanchored adapters')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for g in adapter.groups:
                path, _, attr = g.aliases[0].rpartition('.')
                original = adapter.model.get_submodule(path).parametrizations[attr].original
                effective = original.float() + g.overlay.local.delta().detach().float()
                if not torch.isfinite(effective).all():
                    raise PatchError('nonfinite merged central weight')
                original.copy_(effective.to(original.dtype))
                _refresh_spec(g.overlay.local)


def reinit_seed(genesis: str, owner: str, event_seq: int) -> int:
    raw = f'{genesis}:{owner}:{event_seq}:reinit-v1'.encode('utf8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63)
