"""Round-safe streaming FedAvg for arbitrary tensor LoRA/LoHA factors.

Each round starts with exactly the same frozen base for every participant.
All participants export ONE adapter before its factors are reset. After
receiving every distinct, authenticated participant's patch, apply
    W <- W + sum(delta(peer) / number_of_participants)
into the original weights, reset factors + AdamW, then continue training.
Never average factors. Never integrate an update from another round.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from .core import (FactorDelta, PatchError, UniversalAdapter, _canonical_json,
                   _cp, decode_patch, PEER_ID)
from .identity import signer_from_pem, verifier_from_pem
from .p2p import PatchNode

HEX64 = re.compile(r'^[a-f0-9]{64}$')


def round_anchor(previous: str, round_num: int, patches: dict[str, bytes]) -> str:
    """Deterministic logical revision; identical accepted wires imply identical anchor."""
    hashes = {peer: hashlib.sha256(wire).hexdigest() for peer, wire in sorted(patches.items())}
    return hashlib.sha256(_canonical_json({
        'protocol': 'fedpara-live-fedavg-v1', 'previous': previous,
        'round': round_num, 'patches': hashes,
    })).hexdigest()


def _delta_block(spec: FactorDelta, parts: list[torch.Tensor], start: int, stop: int) -> torch.Tensor:
    """Build an output slice without materializing an entire huge dense matrix."""
    if spec.dense:
        return parts[0][start:stop] if len(spec.shape) else parts[0]
    ndim = len(spec.shape)
    a = _cp([parts[0][start:stop], *parts[1:ndim]])
    if spec.kind == 'loha':
        a = a * _cp([parts[ndim][start:stop], *parts[ndim+1:]])
    return a * (spec.alpha / spec.rank)


@dataclass(frozen=True)
class RoundResult:
    round: int
    anchor: str
    contributors: tuple[str, ...]
    update_hashes: dict[str, str]


class FedAvgRound:
    """Small, testable state machine; network authentication happens at transport.

    Patches are validated before commit. Incremental WebRTC arrival is allowed,
    but the final base is changed ONLY after all members have supplied one
    contribution, avoiding a base that differs across peers mid-round.
    """
    def __init__(self, adapter: UniversalAdapter, self_id: str, participants: list[str],
                 initial_anchor: str, *, max_delta_rms: float = 10.0,
                 block_elements: int = 512 * 1024):
        participants = sorted(participants)
        if len(participants) < 2 or len(participants) > 32 or len(set(participants)) != len(participants):
            raise ValueError('2..32 unique participants required')
        if self_id not in participants or not all(PEER_ID.fullmatch(x) for x in participants):
            raise ValueError('invalid local / remote peer IDs')
        if not HEX64.fullmatch(initial_anchor):
            raise ValueError('base anchor must be 64 lowercase hex characters')
        if max_delta_rms <= 0 or not math.isfinite(max_delta_rms) or block_elements <= 0:
            raise ValueError('invalid update bounds')
        self.adapter, self.self_id = adapter, self_id
        self.participants = tuple(participants)
        self.anchor = initial_anchor
        self.round = 0
        self.patches: dict[str, bytes] = {}
        self.max_delta_rms = max_delta_rms
        self.block_elements = block_elements

    @property
    def complete(self):
        return set(self.patches) == set(self.participants)

    @property
    def local_sealed(self):
        return self.self_id in self.patches

    def _check_wire(self, peer: str, wire: bytes):
        if peer not in self.participants:
            raise PatchError('sender is not in the fixed peer roster')
        meta, weights = decode_patch(wire)
        if meta.get('peer_id') != peer:
            raise PatchError('sender peer identity does not match pinned connection')
        if meta.get('live_protocol') != 'fedpara-live-fedavg-v1':
            raise PatchError('not a live FedAvg patch')
        if type(meta.get('round')) is not int or meta['round'] != self.round:
            raise PatchError('stale/future FedAvg round')
        if meta.get('base_anchor') != self.anchor or meta.get('participants') != list(self.participants):
            raise PatchError('base/participant roster mismatch: resynchronize full model')
        if meta.get('seq') != self.round:
            raise PatchError('noncanonical live patch sequence')
        # _validate enforces exact model layout, finite factors, and shapes.
        parts = self.adapter._validate(meta, weights)
        return meta, parts

    def receive(self, peer: str, wire: bytes) -> bool:
        """Return False for exact retransmission, reject equivocation/replay."""
        with self.adapter.lock:
            self._check_wire(peer, wire)
            previous = self.patches.get(peer)
            if previous is not None:
                if previous != wire:
                    raise PatchError('peer sent conflicting update in one round')
                return False
            self.patches[peer] = bytes(wire)
            return True

    def seal_local(self, *, wire_dtype: str = 'float32') -> bytes:
        """Export trained local factors as exactly one immutable contribution."""
        with self.adapter.lock:
            if self.local_sealed:
                return self.patches[self.self_id]
            wire = self.adapter.export(
                self.self_id, seq=self.round, dtype=wire_dtype,
                extra_meta={'live_protocol':'fedpara-live-fedavg-v1',
                            'round':self.round, 'base_anchor':self.anchor,
                            'participants':list(self.participants)},
            )
            self.receive(self.self_id, wire)
            return wire

    def _prepare(self):
        """Untrusted factors pass a finite/size preflight before modifying weights."""
        if not self.complete:
            raise PatchError('FedAvg needs one contribution per configured peer')
        prepared = {}
        for peer in self.participants:
            _, parts = self._check_wire(peer, self.patches[peer])
            prepared[peer] = parts
        # Validate the *resulting* updates: finite factors can overflow when
        # multiplied (especially LoHA) or be adversarially enormous.
        for gi, group in enumerate(self.adapter.groups):
            spec = group.overlay.local
            n0 = group.shape[0] if group.shape else 1
            row_elems = math.prod(group.shape[1:]) if len(group.shape) > 1 else 1
            rows = max(1, self.block_elements // row_elems)
            for peer in self.participants:
                group_parts = [t.to(spec.parts[0].device, dtype=torch.float32)
                               for t in prepared[peer][gi]]
                power = 0.0
                for lo in range(0,n0,rows):
                    delta = _delta_block(spec,group_parts,lo,min(lo+rows,n0))
                    if not torch.isfinite(delta).all().item():
                        raise PatchError(f'nonfinite reconstructed delta from {peer}')
                    power += delta.double().square().sum().item()
                rms = math.sqrt(power / math.prod(group.shape))
                if not math.isfinite(rms) or rms > self.max_delta_rms:
                    raise PatchError(f'excessive update RMS from {peer}: {rms}')
        return prepared

    def commit(self) -> RoundResult:
        """Exactly Σ_p Delta_p/P, deterministic peer order; new adapter from zero.

        The caller must recreate its optimizer after commit. Checkpoint the
        base periodically for recovery. Keep peers synchronized per round.
        """
        with self.adapter.lock, torch.no_grad():
            prepared = self._prepare()  # No weight mutation before checking all.
            for gi, group in enumerate(self.adapter.groups):
                # Original is used exactly once even when tied to other modules.
                alias = group.aliases[0]
                module_path, _, attr = alias.rpartition('.')
                module = self.adapter.model.get_submodule(module_path)
                original = module.parametrizations[attr].original
                spec = group.overlay.local
                n0 = group.shape[0] if group.shape else 1
                row_elems = math.prod(group.shape[1:]) if len(group.shape)>1 else 1
                rows = max(1, self.block_elements // row_elems)
                device = original.device
                per_peer = {p: [t.to(device,dtype=torch.float32) for t in prepared[p][gi]]
                            for p in self.participants}
                for lo in range(0,n0,rows):
                    dest = original[lo:min(lo+rows,n0)] if group.shape else original
                    accum = dest.float().clone()
                    for peer in self.participants:
                        accum.add_(_delta_block(spec,per_peer[peer],lo,min(lo+rows,n0)),
                                   alpha=1.0/len(self.participants))
                    if not torch.isfinite(accum).all().item():
                        raise PatchError('averaged model weights became nonfinite')
                    dest.copy_(accum.to(dest.dtype))
                # Reset factor values in-place; no changes to graph structure.
                fresh = FactorDelta(group.shape,self.adapter.kind,
                                    self.adapter.rank,self.adapter.alpha)
                for old,new in zip(spec.parts,fresh.parts):
                    old.copy_(new.to(device=old.device,dtype=old.dtype))
                    old.grad = None
            result = RoundResult(
                round=self.round,
                anchor=round_anchor(self.anchor,self.round,self.patches),
                contributors=self.participants,
                update_hashes={p:hashlib.sha256(self.patches[p]).hexdigest()
                               for p in self.participants},
            )
            self.anchor = result.anchor
            self.round += 1
            self.patches.clear()
            return result


def parse_links(raw_links: list[str], my_id: str) -> dict[str, str]:
    out = {}
    for raw in raw_links:
        peer, sep, room = raw.partition('=')
        if not sep or not PEER_ID.fullmatch(peer) or peer == my_id:
            raise ValueError('--link must be OTHER_PEER=ROOM and cannot name self')
        if peer in out or not (16 <= len(room) <= 64) or not re.fullmatch(r'[A-Za-z0-9_-]+',room):
            raise ValueError('duplicate peer or invalid WebRTC room')
        out[peer] = room
    if not out:
        raise ValueError('at least one --link OTHER_PEER=ROOM required')
    return out


def parse_peer_keys(raw_keys: list[str]) -> dict[str, Path]:
    out = {}
    for spec in raw_keys:
        peer, sep, path = spec.partition('=')
        if not sep or not PEER_ID.fullmatch(peer) or not path or peer in out:
            raise ValueError('--trusted-peer-key must be OTHER_PEER=PATH')
        out[peer] = Path(path)
    return out


async def train_live(args, *, load_model: Callable, training_batches: Callable,
                     run_step: Callable):
    """Run one PatchNode per peer, train until a send or incoming contribution.

    State synchronization: all participants must use the same frozen model
    (.fstate loaded with --state-base), roster, rank, and initial base anchor.
    The networked process must not load untrusted Python model code.
    """
    if args.trust_remote_code or args.factory_file or args.allow_legacy_bin:
        raise PermissionError('Networked live training forbids executable custom model code and legacy pickle; use a trusted packaged model, offline training or reviewed native code')
    if args.resume:
        raise ValueError('--resume is not supported in live rounds; restart every peer from same full-state checkpoint')
    if args.grad_accum < 1 or args.local_steps < 1 or args.rounds < 1:
        raise ValueError('invalid gradient accumulation / local steps / rounds')
    if not args.state_base:
        raise ValueError('Live FedAvg requires --state-base from completed initial shard exchange')
    if not args.signing_key and not args.unsafe_unsigned:
        raise PermissionError('--signing-key and pinned --trusted-peer-key required (use --unsafe-unsigned for local tests)')
    if args.ws.startswith('ws://') and not args.unsafe_unsigned:
        hostname = args.ws.split('/')[2].split(':')[0].lower()
        if hostname not in ('localhost','127.0.0.1','[::1]'):
            raise PermissionError('Use WSS for remote signaling (or explicit --unsafe-unsigned)')
    links = parse_links(args.link,args.peer_id)
    roster = sorted([args.peer_id,*links])
    if args.peers is not None and args.peers != len(roster):
        raise ValueError('FedAvg denominator must match 1 + number of linked peers')
    keys = parse_peer_keys(args.trusted_peer_key)
    if not args.unsafe_unsigned and set(keys) != set(links):
        raise PermissionError('You must pin a distinct --trusted-peer-key for every linked peer')
    if not args.unsafe_unsigned and len({p.read_bytes() for p in keys.values()}) != len(keys):
        raise PermissionError('Different peer IDs must not share the same verification key')
    if args.unsafe_unsigned:
        print('WARNING: unsigned development mode, peer identities are untrusted',file=sys.stderr)
    model, adapter = load_model(args)
    if args.kind not in ('lora','loha'):
        raise ValueError('invalid kind')
    # Apply model state before adapter creation via load_model(). The resolved
    # revision is fstate-sha256:<digest> and binds base bytes unambiguously.
    base_ref = _canonical_json({'model_id':adapter.model_id, 'revision':adapter.revision,
                                'layout':adapter.layout, 'kind':adapter.kind,
                                'rank':adapter.rank, 'alpha':adapter.alpha})
    anchor = hashlib.sha256(base_ref).hexdigest()
    engine = FedAvgRound(adapter,args.peer_id,roster,anchor,
                         max_delta_rms=args.max_delta_rms)
    signer = signer_from_pem(Path(args.signing_key)) if args.signing_key else None
    nodes: dict[str,PatchNode] = {}
    tasks = {}
    for peer,room in links.items():
        verifier = verifier_from_pem(keys[peer]) if peer in keys else None
        nodes[peer] = PatchNode(args.ws,room,args.token,
                               max_bytes=args.max_patch_mb*1024**2,
                               signer=signer,verifier=verifier)
        tasks[peer] = asyncio.create_task(nodes[peer].run(),name='connection-'+peer)
    def check_connections():
        for peer,task in tasks.items():
            if task.done():
                exc = task.exception() if not task.cancelled() else None
                raise ConnectionError(f'Peer connection ended: {peer}: {exc}')
    def new_optimizer():
        return torch.optim.AdamW(list(adapter.trainable_parameters()),
                                 lr=args.lr,weight_decay=args.weight_decay)
    iterator = training_batches(args)
    optimizer = new_optimizer()
    try:
        loop = asyncio.get_running_loop()
        end = loop.time()+args.connect_timeout
        while not all(n.connected.is_set() for n in nodes.values()):
            check_connections()
            if loop.time() > end:
                raise TimeoutError('Not all peer DataChannels connected; verify --link pairwise rooms')
            await asyncio.sleep(.03)
        print(f'live roster={roster}, anchor={engine.anchor}, links={len(nodes)}',flush=True)
        for round_id in range(args.rounds):
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            step = 0
            last_loss = float('nan')
            deadline = loop.time()+args.round_timeout
            # Train locally while peers train and exchange in background.
            # An early arriving patch ends this local window immediately.
            while step < args.local_steps and not engine.local_sealed:
                check_connections()
                if loop.time() > deadline:
                    raise TimeoutError('Timed out training this FedAvg round')
                if any(not n.incoming.empty() for n in nodes.values()):
                    break
                last_loss = await asyncio.to_thread(run_step,model,adapter,optimizer,next(iterator),args)
                step += 1; accumulated += 1
                if accumulated == args.grad_accum:
                    with adapter.lock:
                        optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    accumulated = 0
                if step == 1 or step % args.log_steps == 0:
                    print(f'round {round_id} step={step}/{args.local_steps} loss={last_loss:.6g}',flush=True)
            if accumulated:
                with adapter.lock:
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
            # Seal and fan out the exact same immutable bytes on every link.
            wire = engine.seal_local(wire_dtype=args.wire_dtype)
            await asyncio.gather(*(node.outgoing.put(wire) for node in nodes.values()))
            print(f'round={round_id}: local sealed at step={step}, distributing {len(wire)} bytes',flush=True)
            # Remote snapshots are accepted only from their authenticated link.
            while not engine.complete:
                check_connections()
                if loop.time()>deadline:
                    raise TimeoutError(f'Round {round_id} missing peers: {set(roster)-set(engine.patches)}')
                progressed = False
                for peer,node in nodes.items():
                    while not node.incoming.empty():
                        incoming = node.incoming.get_nowait()
                        if not isinstance(incoming,bytes):
                            raise PatchError('live exchange received a non-patch payload')
                        accepted = engine.receive(peer,incoming)
                        if accepted:
                            progressed=True
                            print(f'round={round_id}: accepted {peer}, remaining={len(roster)-len(engine.patches)}',flush=True)
                if not progressed and not engine.complete:
                    await asyncio.sleep(.02)
            result = engine.commit()
            # Discard all stale optimizer moments associated with old factors.
            optimizer = new_optimizer()
            print(f'FedAvg committed round={result.round}, peers={len(roster)}, next_anchor={result.anchor}',flush=True)
            if args.checkpoint_every and (result.round+1)%args.checkpoint_every==0:
                from .shards import export_state_sharded
                p=Path(args.checkpoint_dir); p.mkdir(parents=True,exist_ok=True)
                # The frozen base is now the whole model (all adapter deltas zero).
                # A crash-safe staged checkpoint can be manually assembled for
                # an all-peer restart from the same resulting version.
                path=p/f'round-{result.round+1:06d}.fmanifest'
                export_state_sharded(model,p/'objects-store',path,
                    model_id=adapter.model_id,revision=adapter.revision,
                    peer_id=args.peer_id,seq=result.round+1,adapter=adapter,
                    max_state_bytes=args.max_state_gb*1024**3)
                print('wrote round checkpoint:',path,flush=True)
        if args.shard_state_manifest and args.shard_store:
            from .shards import export_state_sharded
            export_state_sharded(model,args.shard_store,args.shard_state_manifest,
                model_id=adapter.model_id,revision=adapter.revision,
                peer_id=args.peer_id,seq=engine.round,adapter=adapter,
                shard_bytes=args.shard_mb*1024**2,
                max_state_bytes=args.max_state_gb*1024**3)
            print('final sharded state:',args.shard_state_manifest,flush=True)
        if args.full_state_output:
            from .state import export_state
            export_state(model,args.full_state_output,
                         model_id=adapter.model_id,revision=adapter.revision,
                         peer_id=args.peer_id,seq=engine.round,adapter=adapter,
                         max_bytes=args.max_state_gb*1024**3)
            print('final full state:',args.full_state_output,flush=True)
    finally:
        # Flush previously queued data before closing transports when possible.
        for node in nodes.values():
            if node._channel is not None and node._channel.readyState == 'open':
                end = asyncio.get_running_loop().time() + 2
                while node._channel.bufferedAmount and asyncio.get_running_loop().time() < end:
                    await asyncio.sleep(.03)
        for task in tasks.values(): task.cancel()
        await asyncio.gather(*tasks.values(),return_exceptions=True)
