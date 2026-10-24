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

import torch

from .core import FrozenDelta, PatchError, encode_patch
from .elastic import ElasticEngine, EventInfo, pack_event, unpack_event
from .livefed import _delta_block
from .numerics import squared_l2

PROTOCOL = 'fedpara-continual-elastic-fedsum-relora-v4'


def enable_anchors(adapter):
    """Register a non-trainable, non-persistent factor-space reference."""
    with adapter.lock, torch.no_grad():
        for g in adapter.groups:
            if hasattr(g.overlay, 'anchor'):
                raise PatchError('continual anchors already enabled')
            spec = g.overlay.local
            g.overlay.anchor = FrozenDelta(spec, [x.detach() for x in spec.parts]).to(
                spec.parts[0].device)


class ContinualElasticEngine(ElasticEngine):
    def __init__(self, adapter, self_id, genesis, journal, *, reinit_every=20, **kwargs):
        if type(reinit_every) is not int or not 0 <= reinit_every <= 100000:
            raise ValueError('reinit_every must be 0..100000 signed local events')
        self.reinit_every = reinit_every
        enable_anchors(adapter)
        super().__init__(adapter, self_id, genesis, journal, **kwargs)

    def should_reinitialize(self, seq):
        return type(seq) is int and self.reinit_every > 0 and seq > 0 and seq % self.reinit_every == 0

    def _check(self, wire):
        origin, meta, weights, _ = unpack_event(wire, self.trusted, unsafe=self.unsafe)
        if (meta.get('elastic_protocol') != PROTOCOL or meta.get('genesis') != self.genesis
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
        with self.adapter.lock, torch.no_grad():
            for group, a, b in zip(self.adapter.groups,now_parts,old_parts):
                spec = group.overlay.local
                tensors_a=[t.to(spec.parts[0].device,dtype=torch.float32) for t in a]
                tensors_b=[t.to(spec.parts[0].device,dtype=torch.float32) for t in b]
                n0 = group.shape[0] if group.shape else 1
                row_size = math.prod(group.shape[1:]) if len(group.shape)>1 else 1
                rows = max(1,self.block_elements//row_size)
                power=0.
                for lo in range(0,n0,rows):
                    diff = _delta_block(spec,tensors_a,lo,min(lo+rows,n0)) - _delta_block(
                        spec,tensors_b,lo,min(lo+rows,n0))
                    if not torch.isfinite(diff).all().item():
                        raise PatchError('nonfinite continual effective update')
                    power += squared_l2(diff)
                rms=math.sqrt(power/math.prod(group.shape))
                if not math.isfinite(rms) or rms>self.max_delta_rms:
                    raise PatchError('excessive continual effective update RMS')
        return EventInfo(origin,meta['seq'],hashlib.sha256(wire).hexdigest(),meta['denominator']), (now_parts,old_parts)

    def _apply_checked(self, info, parts):
        now_parts,old_parts=parts
        with self.adapter.lock,torch.no_grad():
            for group,a,b in zip(self.adapter.groups,now_parts,old_parts):
                path,_,attr=group.aliases[0].rpartition('.')
                original=self.adapter.model.get_submodule(path).parametrizations[attr].original
                spec=group.overlay.local
                current=[x.to(original.device,dtype=torch.float32) for x in a]
                previous=[x.to(original.device,dtype=torch.float32) for x in b]
                n0=group.shape[0] if group.shape else 1
                row_size=math.prod(group.shape[1:]) if len(group.shape)>1 else 1
                rows=max(1,self.block_elements//row_size)
                for lo in range(0,n0,rows):
                    end=min(lo+rows,n0)
                    # FedSum: add the full effective delta ONCE per event.
                    # No 1/peers factor, irrespective of live membership count.
                    update=_delta_block(spec,current,lo,end)-_delta_block(spec,previous,lo,end)
                    dest=original[lo:end] if group.shape else original
                    value=dest.float()+update
                    if not torch.isfinite(value).all().item():
                        raise PatchError('nonfinite continual global weights')
                    dest.copy_(value.to(dest.dtype))
                if info.origin==self.self_id:
                    # During normal sealing, the factor values are already
                    # equal to the serialized 'after' values. During journal
                    # replay, restore them from the signed event; in either
                    # case anchor==local means no uncommitted local overlay.
                    for param,tensor in zip(spec.parts,current):
                        param.copy_(tensor.to(param.device,dtype=param.dtype))
                        param.grad=None
                    for idx,tensor in enumerate(current):
                        getattr(group.overlay.anchor,f'part{idx}').copy_(
                            tensor.to(spec.parts[idx].device,dtype=torch.float32))
            if info.origin == self.self_id and self.should_reinitialize(info.seq):
                # This signed event commits the current update to the base.
                # Refresh is deterministic on replay, even if a process crashes
                # before saving its local working checkpoint.
                from .relora import reset_committed_factors, reinit_seed
                reset_committed_factors(self.adapter, seed=reinit_seed(
                    self.genesis, self.self_id, info.seq))
            self.applied.add((info.origin,info.seq))

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
        wire=await asyncio.to_thread(self._make_wire,denominator,seq,dtype,max_wire_bytes)
        if not await self.ingest_responsive(wire):
            raise PatchError('new local continual update had duplicate sequence')
        return wire
