"""Continual elastic FedSum using signed before/after LoHa factors.

This is an asynchronous additive SUM protocol, NOT a barriered synchronous
FedSum round. It has a distinct genesis from the earlier averaging protocols.

All participant snapshots/journals contain the same additive effective updates.
Local factor and AdamW moments persist across publications; a frozen low-rank
anchor prevents the already-committed factor effect from being counted twice.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import time
from dataclasses import dataclass

import torch

from .core import FrozenDelta, PatchError, encode_patch
from .elastic import ElasticEngine, EventInfo, pack_event, unpack_event
from .livefed import _delta_block
from .numerics import squared_l2
from .lazy_mutable_weights_v60 import prepare_base_group
from .paged_mutable_v61 import paged_destination
from .factor_ledger_v62 import _factor_tile

PROTOCOL = 'fedpara-continual-elastic-fedsum-relora-mixed-async-v6'


@dataclass(frozen=True)
class PreparedContinualUpdate:
    """Untrusted signed factors plus a finite, bounded, precomputed delta.

    Constructed off the training lock. The dense float32 tensor exists only
    while this one event is being committed; it is not a persisted protocol.
    """
    now_parts: list
    old_parts: list
    deltas: list[torch.Tensor]



def enable_anchors(adapter):
    """Register a non-trainable, non-persistent factor-space reference."""
    with adapter.lock, torch.no_grad():
        for g in adapter.groups:
            if hasattr(g.overlay, 'anchor'):
                raise PatchError('continual anchors already enabled')
            spec = g.overlay.local
            g.overlay.anchor = FrozenDelta(spec, [x.detach() for x in spec.parts]).to(
                spec.parts[0].device)


def _invalidate_frozen_caches(model):
    """GPU LRU stores frozen base copies; writes to mmap make them stale."""
    seen=set()
    for module in model.modules():
        cache=getattr(module,'_fedpara_disk_cache',None)
        if cache is not None and id(cache) not in seen:
            seen.add(id(cache));cache.invalidate()


class ContinualElasticEngine(ElasticEngine):
    def __init__(self, adapter, self_id, genesis, journal, *, reinit_every=20, **kwargs):
        if type(reinit_every) is not int or not 0 <= reinit_every <= 100000:
            raise ValueError('reinit_every must be 0..100000 signed local events')
        self.reinit_every = reinit_every
        enable_anchors(adapter)
        super().__init__(adapter, self_id, genesis, journal, **kwargs)

    def should_reinitialize(self, seq):
        return type(seq) is int and self.reinit_every > 0 and seq > 0 and seq % self.reinit_every == 0

    def _check(self, wire, *, authenticated_source_genesis=None):
        # A certified generation handoff may replay the ORIGINAL signed v1
        # event against a new base.  Its signature and source genesis are still
        # verified; the caller may not forge a new epoch by rewriting metadata.
        expected_genesis = (self.genesis if authenticated_source_genesis is None
                            else authenticated_source_genesis)
        if (not isinstance(expected_genesis, str) or len(expected_genesis) != 64 or
                any(c not in '0123456789abcdef' for c in expected_genesis)):
            raise PatchError('invalid signed carry source genesis')
        origin, meta, weights, _ = unpack_event(wire, self.trusted, unsafe=self.unsafe)
        if (meta.get('elastic_protocol') != PROTOCOL or meta.get('genesis') != expected_genesis
            or type(meta.get('denominator')) is not int
            or meta['denominator'] != 1
            or meta.get('aggregation') != 'sum'
            or meta.get('reinit_every') != self.reinit_every
            or type(meta.get('reinit_after')) is not bool
            or meta['reinit_after'] != self.should_reinitialize(meta.get('seq'))
            or type(meta.get('seq')) is not int or not 1 <= meta['seq'] < 2**63):
            raise PatchError('invalid continual protocol/genesis/denominator/sequence')
        # Validate current factors and anchor factors against the same schema.
        current = {k:v for k,v in weights.items() if k.startswith('p') and '.f' in k}
        before = {k.replace('.b','.f',1):v for k,v in weights.items()
                  if k.startswith('p') and '.b' in k}
        if len(current) != len(before) or len(weights) != len(current) * 2:
            raise PatchError('invalid continual before/after factor inventory')
        now_parts = self.adapter._validate(meta, current)
        old_parts = self.adapter._validate(meta, before)
        # Crucial: validation is CPU-side and needs NO model/optimizer lock.
        # Group schemas and spec.kind/rank/alpha never change after attachment.
        # Previously we held adapter.lock while reconstructing each group, then
        # reconstructed every group AGAIN under adapter.lock during apply.
        # A connected peer could thus stall every optimizer step for seconds.
        deltas=[]
        with torch.no_grad():
            for group, a, b in zip(self.adapter.groups,now_parts,old_parts):
                spec = group.overlay.local
                tensors_a=[t.to(device='cpu',dtype=torch.float32) for t in a]
                tensors_b=[t.to(device='cpu',dtype=torch.float32) for t in b]
                n0 = group.shape[0] if group.shape else 1
                row_size = math.prod(group.shape[1:]) if len(group.shape)>1 else 1
                rows = max(1,self.block_elements//row_size)
                power=0.
                # Bounded temporary cache: avoid allocating a whole dense LLM.
                cache_group=(sum(t.numel()*4 for t in deltas if t is not None) +
                             math.prod(group.shape)*4 <= 64*1024*1024
                             and not (len(group.shape)==2 and row_size>self.block_elements))
                blocks=[]
                for lo in range(0,n0,rows):
                    hi=min(lo+rows,n0)
                    if len(group.shape)==2 and row_size>self.block_elements:
                        # A single extremely wide row may be larger than the
                        # entire staging budget. Bound both axes rather than
                        # materializing that full row during signature check.
                        parts=((_factor_tile(spec,tensors_a,slice(lo,hi),slice(cl,ce))-
                                _factor_tile(spec,tensors_b,slice(lo,hi),slice(cl,ce)))
                               for cl in range(0,row_size,self.block_elements)
                               for ce in [min(row_size,cl+self.block_elements)])
                    else:
                        parts=(_delta_block(spec,tensors_a,lo,hi)-
                               _delta_block(spec,tensors_b,lo,hi),)
                    for diff in parts:
                        if not bool(torch.isfinite(diff).all().item()):
                            raise PatchError('nonfinite continual effective update')
                        power += squared_l2(diff)
                        if cache_group:
                            blocks.append(diff.contiguous())
                rms=math.sqrt(power/math.prod(group.shape))
                if not math.isfinite(rms) or rms>self.max_delta_rms:
                    raise PatchError('excessive continual effective update RMS')
                deltas.append((blocks[0] if len(blocks)==1 else torch.cat(blocks,dim=0))
                              if cache_group else None)
        info=EventInfo(origin,meta['seq'],hashlib.sha256(wire).hexdigest(),meta['denominator'])
        return info, PreparedContinualUpdate(now_parts,old_parts,deltas)

    def _apply_checked(self, info, prepared):
        # Check reconstructed candidate bases in bounded batches.  Calling
        # .all().item() after EVERY small layer synchronizes CUDA/MPS hundreds
        # of times per event; one reduction per 8 MiB is much cheaper while
        # preserving nonfinite-base protection before each bounded write.
        # No model/optimizer tensors are modified until a batch passes.
        ledger=getattr(self.adapter.model,'_fedpara_lazy_mutable_weights',None)
        if ledger is not None and getattr(ledger,'_fedpara_is_factor_ledger',False):
            with self.adapter.lock,torch.no_grad():
                ledger.apply_verified(self,[(info,prepared)])
                if info.origin == self.self_id:
                    for group,parts in zip(self.adapter.groups,prepared.now_parts):
                        spec=group.overlay.local
                        for param,tensor in zip(spec.parts,parts):
                            param.copy_(tensor.to(param.device,dtype=param.dtype))
                            param.grad=None
                        for j,tensor in enumerate(parts):
                            getattr(group.overlay.anchor,f'part{j}').copy_(
                                tensor.to(spec.parts[j].device,dtype=torch.float32))
                    if self.should_reinitialize(info.seq):
                        from .relora import reset_committed_factors,reinit_seed
                        reset_committed_factors(self.adapter,seed=reinit_seed(
                            self.genesis,self.self_id,info.seq))
                _invalidate_frozen_caches(self.adapter.model)
                self.applied.add((info.origin,info.seq))
            return
        with self.adapter.lock, torch.no_grad():
            pending = []
            flags = []
            staged_bytes = 0
            pending_device = None
            local_factor_refresh = []

            def flush():
                nonlocal staged_bytes, pending_device
                if not pending:
                    return
                if not bool(torch.stack(flags).all().item()):
                    raise PatchError('nonfinite continual global weights')
                for dest, candidate in pending:
                    dest.copy_(candidate.to(dtype=dest.dtype))
                pending.clear()
                flags.clear()
                staged_bytes = 0
                pending_device = None

            for group, delta, a, b in zip(self.adapter.groups, prepared.deltas,
                                          prepared.now_parts, prepared.old_parts):
                prepare_base_group(self.adapter,group)
                path, _, attr = group.aliases[0].rpartition('.')
                original = self.adapter.model.get_submodule(path).parametrizations[attr].original
                spec = group.overlay.local
                n0 = group.shape[0] if group.shape else 1
                row_size = math.prod(group.shape[1:]) if len(group.shape) > 1 else 1
                rows = max(1, self.block_elements // row_size)
                # Copy the verified effective delta to the accelerator ONCE,
                # not separately for every 512K-element destination slice.
                # Very large groups retain the original chunked fallback.
                if delta is not None and delta.numel() * delta.element_size() <= 32*1024*1024:
                    device_delta = delta.to(device=original.device, dtype=torch.float32)
                else:
                    device_delta = None
                if delta is None:
                    current = [t.to(original.device, dtype=torch.float32) for t in a]
                    previous = [t.to(original.device, dtype=torch.float32) for t in b]
                for lo in range(0, n0, rows):
                    end = min(lo + rows, n0)
                    dest = paged_destination(self.adapter,group,lo,end,original)
                    if pending and dest.device != pending_device:
                        flush()
                    if device_delta is not None:
                        update = device_delta[lo:end] if group.shape else device_delta
                    elif delta is not None:
                        update = (delta[lo:end] if group.shape else delta).to(
                            device=original.device, dtype=torch.float32)
                    else:
                        update = (_delta_block(spec, current, lo, end) -
                                  _delta_block(spec, previous, lo, end))
                    value = dest.float() + update
                    pending.append((dest, value))
                    flags.append(torch.isfinite(value).all())
                    staged_bytes += value.numel() * value.element_size()
                    pending_device = dest.device
                    if staged_bytes >= 8 * 1024 * 1024:
                        flush()
                if info.origin == self.self_id:
                    # Defer local anchor/factor changes until every candidate
                    # base has been validated and written. Otherwise even our
                    # own frequent publication would synchronize once PER
                    # group, defeating the batched fast path.
                    local_factor_refresh.append((group, a))
            flush()
            for group, factor_parts in local_factor_refresh:
                spec = group.overlay.local
                for param, tensor in zip(spec.parts, factor_parts):
                    param.copy_(tensor.to(param.device, dtype=param.dtype))
                    param.grad = None
                for idx, tensor in enumerate(factor_parts):
                    getattr(group.overlay.anchor, f'part{idx}').copy_(
                        tensor.to(spec.parts[idx].device, dtype=torch.float32))
            if info.origin == self.self_id and self.should_reinitialize(info.seq):
                from .relora import reset_committed_factors, reinit_seed
                reset_committed_factors(self.adapter, seed=reinit_seed(
                    self.genesis, self.self_id, info.seq))
            _invalidate_frozen_caches(self.adapter.model)
            self.applied.add((info.origin, info.seq))

    def _apply_checked_many(self, updates):
        """One base application for a signed, de-duplicated multi-origin FedSum.

        Never add the same (origin,seq) twice; caller has already authenticated
        and durably journaled each individual contributor. Keep the previous
        single-event path unchanged for local event seals / ReLoRA anchors.
        """
        if not updates:
            return
        if len(updates) == 1:
            return self._apply_checked(*updates[0])
        ids=[(info.origin,info.seq) for info,_ in updates]
        if len(set(ids)) != len(ids) or any(info.origin == self.self_id for info,_ in updates):
            raise PatchError('aggregate must contain distinct remote signed events')
        ledger=getattr(self.adapter.model,'_fedpara_lazy_mutable_weights',None)
        if ledger is not None and getattr(ledger,'_fedpara_is_factor_ledger',False):
            with self.adapter.lock,torch.no_grad():
                ledger.apply_verified(self,updates)
                _invalidate_frozen_caches(self.adapter.model)
                self.applied.update(ids)
            return
        with self.adapter.lock, torch.no_grad():
            pending=[]
            flags=[]
            staged_bytes=0
            pending_device=None

            def flush():
                nonlocal staged_bytes,pending_device
                if not pending:
                    return
                if not bool(torch.stack(flags).all().item()):
                    raise PatchError('nonfinite aggregated continual global weights')
                for dest,candidate in pending:
                    dest.copy_(candidate.to(dtype=dest.dtype))
                pending.clear();flags.clear()
                staged_bytes=0;pending_device=None

            for index,group in enumerate(self.adapter.groups):
                prepare_base_group(self.adapter,group)
                path,_,attr=group.aliases[0].rpartition('.')
                original=self.adapter.model.get_submodule(path).parametrizations[attr].original
                spec=group.overlay.local
                n0=group.shape[0] if group.shape else 1
                row_size=math.prod(group.shape[1:]) if len(group.shape)>1 else 1
                rows=max(1,self.block_elements//row_size)
                sources=[]
                for _,prepared in updates:
                    delta=prepared.deltas[index]
                    if delta is not None:
                        sources.append(('dense',delta.to(original.device,dtype=torch.float32)))
                    else:
                        a=[t.to(original.device,dtype=torch.float32) for t in prepared.now_parts[index]]
                        b=[t.to(original.device,dtype=torch.float32) for t in prepared.old_parts[index]]
                        sources.append(('factor',(a,b)))
                for lo in range(0,n0,rows):
                    end=min(lo+rows,n0)
                    dest=paged_destination(self.adapter,group,lo,end,original)
                    if pending and dest.device!=pending_device:
                        flush()
                    combined=None
                    for kind,value in sources:
                        if kind=='dense':
                            delta=value[lo:end] if group.shape else value
                        else:
                            a,b=value
                            delta=_delta_block(spec,a,lo,end)-_delta_block(spec,b,lo,end)
                        combined=delta if combined is None else combined+delta
                    candidate=dest.float()+combined
                    pending.append((dest,candidate))
                    flags.append(torch.isfinite(candidate).all())
                    staged_bytes+=candidate.numel()*candidate.element_size()
                    pending_device=dest.device
                    if staged_bytes >= 8*1024*1024:
                        flush()
            flush()
            _invalidate_frozen_caches(self.adapter.model)
            self.applied.update(ids)

    def _make_wire(self, denominator, seq, dtype='float32', max_wire_bytes=None):
        if type(denominator) is not int or denominator != 1:
            raise PatchError('FedSum event must have denominator=1; old averaging peers are incompatible')
        if dtype!='float32':
            # Half-precision factor rounding changes the local anchor and can
            # create a spurious local update at every seal. Fail closed.
            raise PatchError('continual v2 requires float32 factor wire precision')
        with self.adapter.lock:
            weights={}
            for gi,g in enumerate(self.adapter.groups):
                for j,param in enumerate(g.overlay.local.parts):
                    weights[f'p{gi}.f{j}']=param.detach().to('cpu',dtype=torch.float32).contiguous()
                    weights[f'p{gi}.b{j}']=getattr(g.overlay.anchor,f'part{j}').detach().to('cpu').contiguous()
            meta=self.adapter._metadata(self.self_id,seq)
            meta.update({'elastic_protocol':PROTOCOL,'genesis':self.genesis,
                         'aggregation':'sum','denominator':1,
                         'reinit_every':self.reinit_every,
                         'reinit_after':self.should_reinitialize(seq)})
            patch=encode_patch(meta,weights)
        wire=pack_event(patch,self.self_id,self.signer,public_key=self.event_public)
        if max_wire_bytes is not None and len(wire)>max_wire_bytes:
            raise PatchError('outgoing continual update exceeds --max-patch-mb')
        return wire

    def seal_local(self, denominator, *, dtype='float32',max_wire_bytes=None):
        wire=self._make_wire(denominator,self.journal.next_seq(self.self_id),dtype,max_wire_bytes)
        if not self.ingest(wire):
            raise PatchError('new local continual update had duplicate sequence')
        return wire

    async def seal_local_responsive(self, denominator, *, dtype='float32',max_wire_bytes=None):
        # sqlite3 connections are thread-affine: obtain next sequence on the
        # owning event-loop thread, then serialize tensors off-loop.
        seq=self.journal.next_seq(self.self_id)
        started = time.monotonic()
        wire=await asyncio.to_thread(self._make_wire,denominator,seq,dtype,max_wire_bytes)
        export_ms=(time.monotonic()-started)*1000
        if not await self.ingest_responsive(wire):
            raise PatchError('new local continual update had duplicate sequence')
        self.last_local_seal_stages_ms = (export_ms, *self.last_ingest_stages_ms)
        return wire
