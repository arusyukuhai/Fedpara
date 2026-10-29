"""Fedpara Elastic v2 prototype: deterministic, epoch-anchored buffered aggregation.

This is a *separate* mathematical aggregation engine. It intentionally does NOT
replace the live v1 optimizer: a correct live v2 needs an agreed immutable epoch
base and a distributed epoch commit protocol. A v1 patch was optimized against
moving, peer-specific weights and must NEVER be called a real v2 training trace.

For a sealed cohort S, each participant contributes the sample-weighted mean of
its capped proposals, and participants are averaged with equal weight:
  W(S) = W_epoch + eta * sum_i(mean_j(delta_ij, samples_ij)) / (lambda + |S|).
Event IDs, epoch, checkpoint fingerprint and shapes are validated. Proposed
updates are only trusted after transport signature verification by the caller.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
from collections import defaultdict
from typing import Mapping

import torch

from .core import PatchError


@dataclass(frozen=True)
class EpochProposal:
    origin: str
    seq: int
    epoch: str  # immutable 64-character SHA-256 of the shared epoch checkpoint
    samples: int
    deltas: Mapping[str, torch.Tensor]
    digest: str  # verified signed event wire digest, if transported over P2P


class EpochBuffer:
    """Bounded, deterministic collection for ONE immutable base checkpoint.

    Duplicates are idempotent. No arrivals can alter a previously sealed cohort.
    The first `max_updates_per_peer` SEQUENCE IDs from each participant count,
    chosen in sorted order, so arrival order cannot affect the final result.
    """
    def __init__(self, epoch: str, shapes: Mapping[str, tuple[int, ...]], *,
                 max_updates_per_peer: int = 8, max_samples_per_update: int = 65536,
                 max_events: int = 4096, max_elements: int = 2_000_000):
        if len(epoch) != 64 or any(c not in '0123456789abcdef' for c in epoch):
            raise ValueError('epoch must be the checkpoint SHA-256 hex digest')
        if not shapes or max_updates_per_peer < 1 or max_samples_per_update < 1 or max_events < 1:
            raise ValueError('invalid epoch configuration')
        self.shapes = {str(k): tuple(int(x) for x in v) for k, v in shapes.items()}
        if any(not k or any(x <= 0 for x in v) for k,v in self.shapes.items()):
            raise ValueError('invalid parameter shapes')
        if sum(math.prod(v) for v in self.shapes.values()) > max_elements:
            raise ValueError('epoch prototype size cap exceeded; use tensor-by-tensor streaming for production')
        self.epoch = epoch
        self.max_updates_per_peer = max_updates_per_peer
        self.max_samples_per_update = max_samples_per_update
        self.max_events = max_events
        self.events: dict[tuple[str,int], EpochProposal] = {}
        self.sealed = False

    def add(self, proposal: EpochProposal) -> bool:
        if self.sealed:
            raise PatchError('epoch buffer sealed: start a new agreed epoch')
        if proposal.epoch != self.epoch:
            raise PatchError('stale / conflicting checkpoint epoch')
        if not proposal.origin or type(proposal.seq) is not int or proposal.seq < 1:
            raise PatchError('invalid event identity')
        if type(proposal.samples) is not int or not 1 <= proposal.samples <= self.max_samples_per_update:
            raise PatchError('invalid sample count')
        if not isinstance(proposal.digest,str) or len(proposal.digest)!=64 or any(c not in '0123456789abcdef' for c in proposal.digest):
            raise PatchError('invalid signed event fingerprint')
        if set(proposal.deltas) != set(self.shapes):
            raise PatchError('patch layout differs from agreed checkpoint')
        # Validate BEFORE retaining any part of the proposal.
        copied={}
        for name,shape in self.shapes.items():
            v=proposal.deltas[name]
            if not isinstance(v,torch.Tensor) or tuple(v.shape)!=shape or not v.is_floating_point():
                raise PatchError(f'invalid delta shape/dtype: {name}')
            t=v.detach().to('cpu',dtype=torch.float32).clone()
            if not bool(torch.isfinite(t).all()):
                raise PatchError('nonfinite epoch delta')
            copied[name]=t
        key=(proposal.origin,proposal.seq)
        old=self.events.get(key)
        if old is not None:
            if old.digest != proposal.digest:
                raise PatchError('equivocation: same peer/seq but different digest')
            return False
        if len(self.events)>=self.max_events:
            raise PatchError('epoch event cap reached; commit the epoch before accepting more')
        self.events[key]=EpochProposal(proposal.origin, proposal.seq, proposal.epoch,
                                       proposal.samples,copied,proposal.digest)
        return True

    def selected(self) -> dict[str,list[EpochProposal]]:
        result=defaultdict(list)
        for key in sorted(self.events):
            p=self.events[key]
            if len(result[p.origin]) < self.max_updates_per_peer:
                result[p.origin].append(p)
        return dict(result)

    def materialize(self, base: Mapping[str,torch.Tensor], *, eta: float = 1.,
                    ridge: float = 0., min_participants: int = 1,
                    seal: bool = False) -> dict[str,torch.Tensor]:
        if not math.isfinite(eta) or eta <= 0 or not math.isfinite(ridge) or ridge < 0:
            raise ValueError('invalid aggregation hyperparameters')
        if set(base) != set(self.shapes):
            raise PatchError('base has unexpected tensors')
        selected=self.selected()
        if len(selected) < min_participants:
            raise PatchError('insufficient epoch participants; no checkpoint agreement')
        out={}
        for name,shape in self.shapes.items():
            src=base[name]
            if tuple(src.shape)!=shape or not src.is_floating_point():
                raise PatchError('checkpoint tensor mismatch')
            # Fixed origin/seq iteration order ensures bitwise deterministic
            # floating point accumulation for identical event sets on CPU.
            acc=torch.zeros(shape,dtype=torch.float32)
            for origin in sorted(selected):
                entries=selected[origin]
                samples=sum(p.samples for p in entries)
                mean=torch.zeros_like(acc)
                for p in entries:
                    mean.add_(p.deltas[name],alpha=p.samples/samples)
                acc.add_(mean)
            value=src.detach().to('cpu',dtype=torch.float32) + acc * (eta/(ridge+len(selected)))
            if not bool(torch.isfinite(value).all()):
                raise PatchError('aggregated weight is not finite')
            out[name]=value.to(dtype=src.dtype,device='cpu')
        if seal:self.sealed=True
        return out

    def digest(self) -> str:
        h=hashlib.sha256()
        h.update(self.epoch.encode('ascii'))
        for origin in sorted(self.selected()):
            for item in self.selected()[origin]:
                h.update(f'{origin}/{item.seq}/{item.digest}/{item.samples}\n'.encode())
        return h.hexdigest()


def compare_legacy_swarm_trace(root, *, cap_per_peer: int = 8, eta: float = 1.0,
                               ridge: float = 0., max_parameters: int = 2_000_000):
    """Counterfactual v1 trace analysis, NOT a fair v2 training A/B trial.

    Uses real signed v1 patches and the same starting weights to show how the
    v2 aggregation formula *would* combine them. Actual v2 training would need
    fresh epoch-anchored local updates and epoch-finalization consensus.
    """
    import sqlite3
    from pathlib import Path
    from .elastic import unpack_event
    from .core import UniversalAdapter
    from .livefed import _delta_block
    from .state import apply_state
    root=Path(root)
    scenario=__import__('json').loads((root/'scenario.json').read_text())
    journal=root/'peer-00/events.sqlite3'
    if not journal.exists():raise FileNotFoundError(journal)
    from transformers import GPT2LMHeadModel
    model=GPT2LMHeadModel.from_pretrained(root/'fixture/model',local_files_only=True)
    model.eval()
    apply_state(model,root/'fixture/genesis.fstate',model_id='fixture/model',revision='main')
    adapter=UniversalAdapter(model,'fixture/model','fstate-sha256:PLACEHOLDER',
                             kind=scenario['kind'],rank=scenario['rank'])
    # The adapter revision must match the metadata of the actual signed events.
    import hashlib
    from .state import semantic_state_sha256
    adapter.revision='fstate-sha256:'+semantic_state_sha256(root/'fixture/genesis.fstate')
    base={g.canonical: model.get_submodule(g.aliases[0].rpartition('.')[0]).parametrizations[g.aliases[0].rpartition('.')[2]].original.detach().cpu().clone()
          for g in adapter.groups}
    shapes={k:tuple(x.shape) for k,x in base.items()}
    epoch=semantic_state_sha256(root/'fixture/genesis.fstate')
    buffer=EpochBuffer(epoch,shapes,max_updates_per_peer=cap_per_peer,
                       max_events=256,max_elements=max_parameters)
    db=sqlite3.connect(f'file:{journal.resolve()}?mode=ro',uri=True)
    try:
        owners=[r[0] for r in db.execute('SELECT DISTINCT origin FROM events ORDER BY origin')]
        if len(owners)*cap_per_peer>256:
            raise ValueError('counterfactual event budget exceeded: reduce --cap-per-peer')
        for owner in owners:
            from .event_archive import read_stored_wire
            columns={r[1] for r in db.execute('PRAGMA table_info(events)')}
            if 'log_offset' in columns:
                rows=db.execute("SELECT wire,log_offset,log_size,wire_len,digest FROM events WHERE origin=? ORDER BY seq LIMIT ?",
                                (owner,cap_per_peer)).fetchall()
                signed=(read_stored_wire(journal,*r) for r in rows)
            else:
                rows=db.execute('SELECT wire FROM events WHERE origin=? ORDER BY seq LIMIT ?',
                                (owner,cap_per_peer)).fetchall()
                signed=(bytes(r[0]) for r in rows)
            for wire in signed:
                origin,meta,weights,_=unpack_event(bytes(wire),{},unsafe=False)
                parts=adapter._validate(meta,weights)
                deltas={}
                for i,g in enumerate(adapter.groups):
                    # Materialize ONLY capped tiny-fixture deltas. Never use this
                    # unboundedly for production-scale LLM tensors.
                    fs=g.overlay.local
                    moved=[x.to(dtype=torch.float32,device='cpu') for x in parts[i]]
                    n0=g.shape[0] if g.shape else 1
                    deltas[g.canonical]=_delta_block(fs,moved,0,n0).reshape(g.shape).detach()
                proposal=EpochProposal(origin,int(meta['seq']),epoch,1,deltas,hashlib.sha256(bytes(wire)).hexdigest())
                buffer.add(proposal)
    finally:db.close()
    result=buffer.materialize(base,eta=eta,ridge=ridge,seal=True)
    with torch.no_grad():
        for g in adapter.groups:
            path,_,name=g.aliases[0].rpartition('.')
            original=adapter.model.get_submodule(path).parametrizations[name].original
            original.copy_(result[g.canonical].to(dtype=original.dtype,device=original.device))
            # Ensure local trainable overlays contribute no extra delta.
            for factor in g.overlay.local.parts:factor.zero_()
    # Fixed independent synthetic held-out samples, not the swarm's training batches.
    gen=torch.Generator().manual_seed(909123)
    losses=[]
    with torch.inference_mode():
        for _ in range(12):
            ids=torch.randint(3,64,(1,12),generator=gen,dtype=torch.long)
            losses.append(float(model(input_ids=ids,labels=ids).loss))
    # Compare against the actual final V1 checkpoint using identical held-out
    # samples; this compares *checkpoint states*, not a fair training algorithm
    # A/B trial, since V2 did not generate the logged proposals.
    baseline=GPT2LMHeadModel.from_pretrained(root/'fixture/model',local_files_only=True)
    from .state import verify_state
    v1_path=root/'peer-00/final.fstate'
    v1_meta=verify_state(v1_path)
    apply_state(baseline,v1_path,model_id=v1_meta['model_id'],revision=v1_meta['revision'])
    baseline.eval()
    baseline_losses=[]
    gen=torch.Generator().manual_seed(909123)
    with torch.inference_mode():
        for _ in range(12):
            ids=torch.randint(3,64,(1,12),generator=gen,dtype=torch.long)
            baseline_losses.append(float(baseline(input_ids=ids,labels=ids).loss))
    return {'status':'COUNTERFACTUAL_ONLY','v1_final_heldout_loss':sum(baseline_losses)/len(baseline_losses),'selected_events':sum(map(len,buffer.selected().values())),
            'participants':len(buffer.selected()),'cap_per_peer':cap_per_peer,'eta':eta,'ridge':ridge,
            'heldout_loss':sum(losses)/len(losses),'digest':buffer.digest(),
            'warning':'V1 patches used moving peer-specific bases: this is NOT a true V2 training run or fair accuracy benchmark.'}

PROTOCOL_V2 = 'fedpara-epoch-buffer-v2'


def pack_epoch_proposal(adapter, origin: str, seq: int, epoch: str,
                        samples: int, signer, public_key: str, *, dtype='float16') -> bytes:
    """Sign an epoch-tagged adapter update, preserving existing data-only patches.

    Caller must have trained against the immutable epoch base, rather than a
    separately evolved local model; otherwise the optimizer semantics differ.
    """
    if type(samples) is not int or not 1 <= samples <= 65536:
        raise PatchError('invalid epoch sample count')
    if not isinstance(epoch,str) or len(epoch)!=64 or any(c not in '0123456789abcdef' for c in epoch):
        raise PatchError('invalid epoch checkpoint digest')
    if signer is None or not public_key:
        raise PatchError('v2 epoch proposal requires a signing identity')
    from .elastic import pack_event
    patch=adapter.export(origin,seq=seq,dtype=dtype,
                         extra_meta={'v2_protocol':PROTOCOL_V2,'base_epoch':epoch,
                                     'v2_samples':samples})
    return pack_event(patch,origin,signer=signer,public_key=public_key)


def ingest_signed_epoch_proposal(buffer: EpochBuffer, adapter, wire: bytes,
                                 trusted: dict | None = None) -> bool:
    """Verify sender signature, model schema, epoch and bounded dense deltas.

    The network ingress path should enforce a separate wire-size quota. This
    method never executes remote code or deserializes pickle objects.
    """
    from .elastic import unpack_event
    from .livefed import _delta_block
    if not isinstance(wire,bytes) or len(wire) > 512*1024*1024:
        raise PatchError('oversized signed v2 proposal')
    origin,meta,weights,_=unpack_event(wire,trusted or {},unsafe=False)
    if (meta.get('v2_protocol')!=PROTOCOL_V2 or meta.get('base_epoch')!=buffer.epoch
        or type(meta.get('v2_samples')) is not int):
        raise PatchError('incompatible v2 epoch protocol / stale base')
    parts=adapter._validate(meta,weights)
    if set(buffer.shapes) != {g.canonical for g in adapter.groups}:
        raise PatchError('local layout differs from v2 cohort')
    deltas={}
    with adapter.lock,torch.no_grad():
        for i,g in enumerate(adapter.groups):
            spec=g.overlay.local
            factors=[p.to(device=spec.parts[0].device,dtype=torch.float32) for p in parts[i]]
            n0=g.shape[0] if g.shape else 1
            delta=_delta_block(spec,factors,0,n0).reshape(g.shape)
            deltas[g.canonical]=delta
    return buffer.add(EpochProposal(origin,int(meta['seq']),buffer.epoch,
                                    int(meta['v2_samples']),deltas,hashlib.sha256(wire).hexdigest()))
