"""Bounded-factor, asynchronous effective-weight snapshots for Kagami.

Only call capture() at a *sealed* trainer step boundary while serialized with
all incoming signed model mutations. Frozen low-rank factors are independent
CPU copies; the immutable BASE tensor storage is shared, not cloned. The
original tensor versions/storage are checked before and after streaming.

This is NOT a generic concurrent checkpoint of arbitrary mutable PyTorch
models. Any base mutation, opaque .data write or uncoordinated module change
invalidates the immutable-base contract; the trusted trainer must serialize
such changes and reject an unrecognised mutation. Dense fallback LoHA factors
are subject to a strict capture byte budget.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path
import threading
import torch
from .core import PatchError
from .shard_runtime import effective_state_sources
from .epoch_export_v57 import _indexed_frontier


class _ImmutableFactor:
    def __init__(self, source, budget):
        for name in ('kind', 'rank', 'alpha', 'dense', 'shape'):
            setattr(self, name, getattr(source, name))
        parts = source.parts if hasattr(source, 'parts') else [getattr(source, f'part{i}') for i in range(source.nparts)]
        self.parts = []
        for part in parts:
            size = part.numel() * part.element_size()
            budget[0] += size
            if budget[0] > budget[1]:
                raise PatchError('frozen factor snapshot exceeds memory budget')
            # Offload the *low-rank* factors, not full model tensors.
            self.parts.append(part.detach().to('cpu', copy=True).contiguous())


class _ImmutableOverlay:
    def __init__(self, source, budget):
        self.local = _ImmutableFactor(source.local, budget)
        if hasattr(source, 'anchor'):
            self.anchor = _ImmutableFactor(source.anchor, budget)
        self.remotes = {name: _ImmutableFactor(value, budget)
                        for name, value in source.remotes.items()}


@dataclass(frozen=True)
class FrozenExport:
    adapter: object
    sources: tuple
    frontier: dict
    inventory: str
    event_count: int
    factor_bytes: int
    model: object
    frozen_base_sources: object = None

    def assert_immutable_base(self):
        """Fail closed on normal PyTorch in-place/storage/base replacements."""
        manager=getattr(self.model,'_fedpara_lazy_mutable_weights',None)
        with (manager.export_context() if getattr(manager,'_fedpara_is_factor_ledger',False)
              else nullcontext()):
            now = effective_state_sources(self.model, self.adapter)
        if set(now) != {x[0] for x in self.sources}:
            raise PatchError('model topology changed during frozen export')
        for name, ident, version, ptr, offset, shape, dtype in self.sources:
            tensor = now[name][0]
            if (id(tensor) != ident or tensor._version != version or
                    tensor.untyped_storage().data_ptr() != ptr or
                    tensor.storage_offset() != offset or
                    tuple(tensor.shape) != shape or str(tensor.dtype) != dtype):
                raise PatchError('base tensor changed during frozen export')


def capture(*, journal, model, adapter, max_factor_bytes=256 * 1024**2,
            defer_frontier=False):
    """Capture at one fenced optimizer step boundary, not from P2P callbacks.

    The caller owns `inbound_commit_lock`; optimizer and model installation may
    not run concurrently with this function. Factor copies and the indexed cut
    are made while the trainer is quiescent. Disk serialization happens later.
    """
    if type(max_factor_bytes) is not int or max_factor_bytes <= 0:
        raise ValueError('invalid frozen factor budget')
    manager=getattr(model,'_fedpara_lazy_mutable_weights',None)
    ledger=manager if getattr(manager,'_fedpara_is_factor_ledger',False) else None
    with adapter.lock, torch.no_grad(), (ledger.export_context() if ledger is not None else nullcontext()):
        sources = effective_state_sources(model, adapter)
        virtual=ledger.frozen_sources() if ledger is not None else None
        base_info = []
        for name, (tensor, _) in sorted(sources.items()):
            base_info.append((name, id(tensor), tensor._version,
                 tensor.untyped_storage().data_ptr(), tensor.storage_offset(),
                 tuple(tensor.shape), str(tensor.dtype)))
        budget = [0, max_factor_bytes]
        if ledger is not None:
            # Frozen snapshots pin the immutable event-factor tuples until the
            # worker finishes even if the live manager compacts meanwhile.
            budget[0] += sum(
                ledger.sources[g.canonical].stats()['factor_bytes']
                for g in adapter.groups if g.canonical in ledger.sources)
            if budget[0] > budget[1]:
                raise PatchError('frozen factor ledger exceeds snapshot byte budget')
        groups = [SimpleNamespace(aliases=tuple(group.aliases),
                   overlay=_ImmutableOverlay(group.overlay, budget))
                  for group in adapter.groups]
        frozen = SimpleNamespace(model_id=adapter.model_id,
            revision=adapter.revision, groups=groups, lock=threading.RLock())
        if defer_frontier:
            # The caller must have pinned a SQLite WAL reader snapshot under
            # the same optimizer+ingress fence BEFORE copying these factors.
            # Its indexed scan can then run after the fence is released.
            frontier, inventory = None, None
        else:
            frontier, inventory = _indexed_frontier(journal)
        result = FrozenExport(frozen, tuple(base_info), frontier, inventory,
                              sum(frontier.values()) if frontier is not None else None,
                              budget[0], model, virtual)
        result.assert_immutable_base()
        return result


def finish_deferred_frontier(frozen: FrozenExport, snapshot_db):
    """Complete a pinned WAL read snapshot *outside* the trainer writer fence.

    snapshot_db was BEGIN/read-pinned before factor capture, and must remain
    exclusively owned by this worker until this method returns. The caller
    closes it immediately after scanning to avoid prolonged WAL retention.
    """
    if frozen.frontier is not None or frozen.inventory is not None:
        raise PatchError('snapshot frontier was already finalized')
    frontier, inventory = _indexed_frontier(SimpleNamespace(db=snapshot_db))
    frozen.assert_immutable_base()
    return replace(frozen, frontier=frontier, inventory=inventory,
                   event_count=sum(frontier.values()))
