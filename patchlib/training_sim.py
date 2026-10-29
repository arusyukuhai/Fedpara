"""Real-gradient P2P training simulator (not a WebRTC benchmark).

Each node runs an independent UniversalAdapter + AdamW over disjoint training
partitions. Effective LoHa/LoRA weight differences are transferred through a
bounded-degree, delay/bandwidth/loss/churn network. Delivery is exactly-once,
includes batched cross-origin FedSum, and is repaired by peer inventories.

Networking is a *discrete-event abstraction*; no actual crypto, WebRTC or
network byte serialization occurs. Virtual time is distinct from measured CPU
training time. Unlike an ideal synchronous benchmark, nodes do not wait for
one another before local optimization steps.
"""
from __future__ import annotations

import csv
import heapq
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F

from .core import UniversalAdapter
from .continual import enable_anchors
from .relora import reset_committed_factors, merge_and_reset_central, reinit_seed
from .scale_sim import bounded_neighbors, choose_targets, publication_interval
from .effectiveness_benchmark import (_data, _local_text_data, _tinystories_data,
    _builtin_story_data, _batch, _logits, _train_step,
    _create_pretrained_model, TINYSTORIES_1M_ID, TINYSTORIES_1M_REVISION)


@dataclass
class Event:
    origin: int
    sequence: int
    deltas: dict[str, torch.Tensor]
    bytes_estimate: int
    created_at: float

    @property
    def key(self):
        return self.origin, self.sequence


@dataclass
class Peer:
    model: object
    adapter: UniversalAdapter
    optimizer: object
    seen: set[tuple[int, int]] = field(default_factory=set)
    emitted: int = 0
    compute_s: float = 0.
    last_loss: float = float('nan')


def _original(model, group):
    module_name, _, pname = group.aliases[0].rpartition('.')
    module = model.get_submodule(module_name)
    return module.parametrizations[pname].original


@torch.no_grad()
def _commit_local(peer: Peer, now: float) -> Event:
    """Merge the uncommitted part once, then advance that peer's anchor."""
    deltas = {}
    factor_bytes = 0
    for group in peer.adapter.groups:
        overlay = group.overlay
        if not hasattr(overlay, 'anchor'):
            raise RuntimeError('continual anchor missing')
        change = (overlay.local.delta() - overlay.anchor.delta()).detach()
        if not torch.isfinite(change).all().item():
            raise FloatingPointError('nonfinite local update')
        deltas[group.canonical] = change.cpu().clone()
        factor_bytes += 2 * sum(p.numel() * p.element_size() for p in overlay.local.parts)
        _original(peer.model, group).add_(change)
        for j, param in enumerate(overlay.local.parts):
            getattr(overlay.anchor, f'part{j}').copy_(param)
    peer.emitted += 1
    event = Event(origin=-1, sequence=peer.emitted, deltas=deltas,
                  bytes_estimate=factor_bytes + 512, created_at=now)
    return event


@torch.no_grad()
def _apply_events(peer: Peer, incoming: list[Event]) -> int:
    """Aggregate only NEW events and apply the dense effective FedSum once."""
    unique = []
    pending = set()
    for event in incoming:
        if event.key in peer.seen or event.key in pending:
            continue
        if set(event.deltas) != {g.canonical for g in peer.adapter.groups}:
            raise ValueError('P2P adapter layout mismatch')
        unique.append(event)
        pending.add(event.key)
    if not unique:
        return 0
    for group in peer.adapter.groups:
        target = _original(peer.model, group)
        total = torch.zeros_like(target)
        for event in unique:
            delta = event.deltas[group.canonical]
            if delta.shape != total.shape or not torch.isfinite(delta).all().item():
                raise ValueError('invalid update in simulated network')
            total.add_(delta.to(device=target.device, dtype=target.dtype))
        target.add_(total)
    peer.seen.update(pending)
    return len(unique)


class GossipTransport:
    """Deterministic queued network with push fanout, repair and finite uplinks."""
    def __init__(self, n, *, fanout, bundle_size, bandwidth_mbps, latency_ms,
                 loss_probability, compression_ratio, repair_every, seed,
                 online, max_pending_packets=200000, max_outstanding_mib=512.):
        self.n, self.fanout, self.bundle_size = n, fanout, bundle_size
        self.bandwidth_mbps, self.latency_ms = bandwidth_mbps, latency_ms
        self.loss_probability, self.compression_ratio = loss_probability, compression_ratio
        self.repair_every, self.online = repair_every, online
        self.rng = random.Random(seed + 200)
        self.neighbors = [bounded_neighbors(i, n) for i in range(n)]
        self.events: dict[tuple[int,int], Event] = {}
        self.pending_push: dict[tuple[int,int], set[tuple[int,int]]] = {}
        self.max_outstanding_bytes=int(max_outstanding_mib*2**20)
        self.outstanding_dense_bytes=0
        self.peers: list[Peer] = []
        self.queue = []
        self.tx_free = [0.] * n
        self.counter = 0
        self.inflight: set[tuple[int,int,tuple[int,int]]] = set()
        self.sent_bytes, self.sent_packets, self.dropped_packets = 0, 0, 0
        self.repair_bytes, self.bundles, self.duplicates, self.applied = 0, 0, 0, 0
        self.max_queue, self.max_events_cached = 0, 0
        self.max_pending_packets = max_pending_packets
        self.last_repair = float('-inf')

    def _send(self, src, dst, keys, now, repair=False):
        if not self.online(src, now) or not self.online(dst, now) or not keys:
            return
        selected = [key for key in keys if key in self.events and
                    (src, dst, key) not in self.inflight and key not in self.peers[dst].seen]
        for start in range(0, len(selected), self.bundle_size):
            chunk = tuple(selected[start:start+self.bundle_size])
            if not chunk:
                continue
            raw = sum(self.events[key].bytes_estimate for key in chunk)
            # Compression is an assumed ratio, not an actual codec measurement.
            size = raw if len(chunk) == 1 else round(raw * self.compression_ratio) + 128
            begin = max(now, self.tx_free[src]); end = begin + 8*size/(self.bandwidth_mbps*1e6)
            self.tx_free[src] = end
            for key in chunk:
                self.inflight.add((src,dst,key))
            if len(self.queue) >= self.max_pending_packets:
                raise RuntimeError('network queue capacity exceeded: lower publish cadence or raise link bandwidth')
            self.counter += 1
            heapq.heappush(self.queue, (end+self.latency_ms/1000, self.counter,
                                        src, dst, chunk, repair))
            self.sent_bytes += size; self.sent_packets += 1
            self.repair_bytes += size if repair else 0
            self.bundles += (len(chunk) > 1)
            self.max_queue=max(self.max_queue,len(self.queue))

    def _queue_push(self, src, key, excluded=None):
        event=self.events.get(key)
        if event is None:
            return
        event_id=event.origin*10000000+event.sequence
        for dst in choose_targets(src,self.neighbors[src],event_id,self.fanout,
                                   excluded=excluded,origin=event.origin):
            self.pending_push.setdefault((src,dst),set()).add(key)

    def _flush_push(self, now):
        pending, self.pending_push=self.pending_push, {}
        for (src,dst),keys in sorted(pending.items()):
            self._send(src,dst,sorted(keys),now)

    def publish(self, origin: int, event: Event, now: float):
        event.origin = origin
        if event.key in self.events:
            raise RuntimeError('duplicate local event key')
        dense_bytes=sum(v.numel()*v.element_size() for v in event.deltas.values())
        if self.outstanding_dense_bytes+dense_bytes > self.max_outstanding_bytes:
            raise RuntimeError('Outstanding P2P effective deltas exceeded memory limit. '
                'Increase bandwidth, reduce peers, or increase --max-outstanding-mib.')
        self.outstanding_dense_bytes+=dense_bytes
        self.events[event.key] = event
        self.peers[origin].seen.add(event.key)
        self._queue_push(origin,event.key)
        self.max_events_cached = max(self.max_events_cached,len(self.events))

    def tick(self, now):
        while self.queue and self.queue[0][0] <= now:
            arrival, _, src, dst, keys, repair = heapq.heappop(self.queue)
            for key in keys:
                self.inflight.discard((src,dst,key))
            if (not self.online(src,arrival) or not self.online(dst,arrival)
                    or self.rng.random() < self.loss_probability):
                self.dropped_packets += 1
                continue
            unseen = [self.events[k] for k in keys
                      if k in self.events and k not in self.peers[dst].seen]
            self.duplicates += len(keys) - len(unseen)
            count = _apply_events(self.peers[dst], unseen)
            self.applied += count
            for event in unseen:
                self._queue_push(dst,event.key,excluded=src)
        if now - self.last_repair >= self.repair_every:
            self.last_repair=now
            for dst in range(self.n):
                if not self.online(dst,now):
                    continue
                # One neighbor per repair interval; no omniscient inventory.
                src = self.neighbors[dst][int(now / self.repair_every + dst) % len(self.neighbors[dst])]
                if not self.online(src,now):
                    continue
                missing = sorted(self.peers[src].seen - self.peers[dst].seen)
                self._send(src,dst,missing[:self.bundle_size*4],now,repair=True)
        self._flush_push(now)
        # Once every peer knows an update, never keep its full dense tensors in
        # memory indefinitely. Duplicates still use per-peer seen-ID sets.
        removable=[k for k in self.events if all(k in p.seen for p in self.peers)]
        for k in removable:
            self.outstanding_dense_bytes-=sum(x.numel()*x.element_size()
                                                for x in self.events[k].deltas.values())
            del self.events[k]

    def catchup(self, now, max_seconds):
        end=now+max_seconds
        while self.events and now < end:
            now += min(self.repair_every, max(0.005,self.latency_ms/1000+0.005))
            self.tick(now)
        return now

    def stats(self):
        return {'sent_mib':round(self.sent_bytes/2**20,3),
            'repair_mib':round(self.repair_bytes/2**20,3),
            'sent_packets':self.sent_packets,'bundled_packets':self.bundles,
            'dropped_packets':self.dropped_packets,'duplicates_ignored':self.duplicates,
            'remote_updates_applied':self.applied,'max_queue_packets':self.max_queue,
            'max_cached_events':self.max_events_cached,
            'outstanding_events':len(self.events),
            'outstanding_dense_mib':round(self.outstanding_dense_bytes/2**20,3)}


def _eval(model, data, batch_size=16):
    model.eval(); total=0.; count=0
    device=next(model.parameters()).device
    with torch.no_grad():
        for chunk in data.split(batch_size):
            x=chunk[:,:-1].to(device); y=chunk[:,1:].to(device)
            logits=_logits(model(x))
            total += F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='sum').item()
            count += y.numel()
    model.train()
    return total/count


def _sync_compute_device(device):
    """Ensure asynchronous CUDA/MPS kernels finish inside timed training steps."""
    dev = torch.device(device)
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)
    elif dev.type == 'mps':
        torch.mps.synchronize()


def _measured_train(model, optim, batch, clip):
    """Measure *only* an optimizer step, including completed accelerator work."""
    device = next(model.parameters()).device
    _sync_compute_device(device)
    start = time.perf_counter()
    loss = _train(model, optim, batch, clip)
    _sync_compute_device(device)
    return loss, time.perf_counter() - start


def _train(model, optim, batch, clip):
    device=next(model.parameters()).device
    x,y=batch
    return _train_step(model,optim,(x.to(device),y.to(device)),grad_clip=clip)


def _make_problem(*,model,peers,seed,context,width,vocab,sequences,
                  validation_sequences,dataset,text_file):
    torch.manual_seed(seed)
    if model=='toy':
        from .effectiveness_benchmark import TinyCausalLM
        initial=TinyCausalLM(vocab,width,context)
        base={k:v.detach().clone() for k,v in initial.state_dict().items()}
        if text_file:
            train,valid,meta=_local_text_data(Path(text_file),peers,sequences,context,
                                              seed,validation_sequences)
        else:
            train,valid=_data(peers,sequences,context,vocab,seed,validation_sequences)
            meta={'type':'synthetic-non-iid-markov'}
        return initial,train,valid,meta,base,None
    try:
        from transformers import AutoModelForCausalLM
        from .hf_config_v83 import load_hf_config
        from .hf_load_v85 import load_hf_tokenizer
    except ImportError as exc:
        raise RuntimeError('TinyStories requires transformers, datasets and tokenizers; install requirements-patches.txt') from exc
    config=load_hf_config(TINYSTORIES_1M_ID,revision=TINYSTORIES_1M_REVISION,
                                      trust_remote_code=False)
    config._attn_implementation='eager'; config.use_cache=False
    initial=AutoModelForCausalLM.from_config(config,trust_remote_code=False)
    for mod in initial.modules():
        if isinstance(mod,torch.nn.Dropout):mod.p=0.
    initial.eval()
    tokenizer=load_hf_tokenizer(TINYSTORIES_1M_ID,
              revision=TINYSTORIES_1M_REVISION,trust_remote_code=False)
    if text_file:
        from .effectiveness_benchmark import _tokenized_file_data
        train,valid,meta=_tokenized_file_data(text_file,tokenizer,peers,sequences,
                                               context,seed,validation_sequences)
    elif dataset=='tinystories':
        train,valid,meta=_tinystories_data(tokenizer,peers,sequences,context,
                                           seed,validation_sequences)
    elif dataset=='builtin':
        train,valid,meta=_builtin_story_data(tokenizer,peers,sequences,context,
                                              seed,validation_sequences)
    else:
        raise ValueError('TinyStories requires dataset=tinystories/builtin or --text-file')
    base={k:v.detach().clone() for k,v in initial.state_dict().items()}
    return initial,train,valid,meta,base,initial


def run_training_sim(*, model='toy', peers=5, rounds=10, local_steps=4,
                     reinit_every=20, lr=1e-4, rank=16, wide_lora_rank=4,
                     batch_size=4, context=12, width=24, vocab=32, sequences=96,
                     validation_sequences=64, dataset='builtin', text_file=None,
                     seed=20261025, threads=1, fanout=2, bundle_size=3,
                     uplink_mbps=600., latency_ms=20., loss_probability=0.,
                     churn_fraction=0., repair_every=1., sim_steps_per_second=5.,
                     compression_ratio=.75, adaptive=True, catchup_seconds=30.,
                     eval_every=1, output_dir=None, device='cpu',
                     max_outstanding_mib=512.):
    """Run actual LM optimizer steps with delayed exactly-once P2P FedSum.

    A central trainer sees the same sample IDs as each online peer, thus
    total TRAINING tokens are identical. Peers train sequentially on this host;
    network and ideal parallel times are modeled, not observed WebRTC time.
    """
    if (not 2<=peers<=(64 if model=='toy' else 16) or not 1<=rounds<=200 or not 1<=local_steps<=40
       or not 0<=reinit_every<=200 or not 1<=rank<=32 or not 1<=wide_lora_rank<=16
       or not 1<=batch_size<=64 or not 1<=threads<=64 or not 4<=context<=2048
       or not 0<lr<=.1 or not 1<=fanout<=8 or not 1<=bundle_size<=8
       or not 0<=churn_fraction<1 or not 0<=loss_probability<1
       or not 0<compression_ratio<=1 or min(uplink_mbps,repair_every,sim_steps_per_second,catchup_seconds)<=0
       or latency_ms<0 or not 1<=eval_every<=rounds or max_outstanding_mib<=0):
        raise ValueError('simulation settings outside supported range')
    if model not in ('toy','tinystories-1m') or device not in ('cpu','cuda','mps'):
        raise ValueError('invalid model or device')
    if model=='toy' and (not 16<=vocab<=256 or not 16<=width<=128 or context>64 or rank>width
                          or (text_file is not None and vocab!=256)):
        raise ValueError('invalid toy dimensions')
    if model=='tinystories-1m' and context>512:
        raise ValueError('context too large for simulation memory budget')
    if device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    if device=='mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable')
    old_threads=torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        initial,train,valid,dataset_info,base,source_model=_make_problem(
            model=model,peers=peers,seed=seed,context=context,width=width,
            vocab=vocab,sequences=sequences,validation_sequences=validation_sequences,
            dataset=dataset,text_file=text_file)
        if model=='toy':
            from .effectiveness_benchmark import TinyCausalLM
            def make():
                m=TinyCausalLM(vocab,width,context);m.load_state_dict(base)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed+12345)
                    adapter=UniversalAdapter(m,'simulation-toy','v1',kind='loha',rank=rank,
                          target_scope='auto',wide_lora_rank=wide_lora_rank)
                return m.to(device),adapter
        else:
            def make():
                m,ad=_create_pretrained_model(source_model,kind='loha',rank=rank,
                    init_seed=seed+12345,target_scope='auto')
                # wide_lora_rank is a per-group part of the target layout.
                if wide_lora_rank!=4:
                    raise ValueError('TinyStories currently requires wide_lora_rank=4')
                return m.to(device),ad
        from .lora_muon import make_adapter_optimizer
        central,ca=make()
        peers_state=[]
        for idx in range(peers):
            m,ad=make();enable_anchors(ad)
            optim=make_adapter_optimizer(ad,optimizer='adamw',lr=lr)
            peers_state.append(Peer(m,ad,optim))
        central_opt=make_adapter_optimizer(ca,optimizer='adamw',lr=lr)
        initial_loss=_eval(initial.to(device),valid,batch_size=8)
        del initial
        rng=random.Random(seed+91)
        offline_ids=set(rng.sample(range(peers),round(peers*churn_fraction)))
        def online(peer,t):
            # Fixed, observable outage during training. Everyone rejoins for catch-up.
            a=(rounds*.3)*local_steps/sim_steps_per_second
            b=(rounds*.55)*local_steps/sim_steps_per_second
            return peer not in offline_ids or not (a<=t<b)
        net=GossipTransport(peers,fanout=fanout,bundle_size=bundle_size,
                bandwidth_mbps=uplink_mbps,latency_ms=latency_ms,
                loss_probability=loss_probability,compression_ratio=compression_ratio,
                repair_every=repair_every,seed=seed,online=online,
                max_outstanding_mib=max_outstanding_mib)
        net.peers=peers_state
        publish_steps=publication_interval(local_steps,peers) if adaptive else local_steps
        # Adjust publish cadence only; every round still represents local_steps steps.
        uncommitted_steps=[0]*peers
        result_rows=[]; simulated_time=0.; central_seconds=0.; serial_peer_seconds=0.
        train_tokens=0

        def measured_times():
            # P2P is trained serially in this simulation: division by the
            # configured peer count is an IDEAL-parallel compute-time estimate.
            # Keep the largest single-peer compute time as an imbalance bound.
            return {
                'central_compute_s':round(central_seconds, 6),
                'p2p_compute_sum_s':round(serial_peer_seconds, 6),
                'p2p_compute_div_peers_s':round(serial_peer_seconds / peers, 6),
                'p2p_compute_max_peer_s':round(max(p.compute_s for p in peers_state), 6),
            }

        def record_round(round_no, round_losses, active):
            per_peer=[_eval(p.model, valid, batch_size=8) for p in peers_state]
            c_loss=_eval(central, valid, batch_size=8)
            row={'round':round_no,'tokens_seen':train_tokens,
                 'simulated_time_s':round(simulated_time,4),
                 **measured_times(),
                 'central_valid_loss':c_loss,
                 'p2p_mean_valid_loss':sum(per_peer)/peers,
                 'p2p_worst_valid_loss':max(per_peer),
                 'p2p_best_valid_loss':min(per_peer),
                 'p2p_train_loss':sum(round_losses)/len(round_losses) if round_losses else float('nan'),
                 'active_peers':len(active),
                 'updates_published':sum(p.emitted for p in peers_state),
                 'applied_remote_updates':net.applied,
                 'min_seen_events':min(len(p.seen) for p in peers_state),
                 'max_seen_events':max(len(p.seen) for p in peers_state),
                 'network_sent_mib':round(net.sent_bytes/2**20,3),
                 'peer_valid_losses':per_peer}
            result_rows.append(row)
            print(f'ROUND {round_no}/{rounds} tokens={train_tokens} '
                  f'central={c_loss:.4f} P2Pmean={row["p2p_mean_valid_loss"]:.4f} '
                  f'compute_s: central={row["central_compute_s"]:.2f} '
                  f'P2P_sum/peers={row["p2p_compute_div_peers_s"]:.2f} '
                  f'coverage={row["min_seen_events"]}/{row["max_seen_events"]}',flush=True)

        # Baseline point at elapsed training compute = 0 on both curves.
        record_round(0,[],list(range(peers)))
        for round_no in range(1,rounds+1):
            net.tick(simulated_time)
            active=[i for i in range(peers) if online(i,simulated_time)]
            round_losses=[]
            for step in range(local_steps):
                for i in active:
                    sample=_batch(train[i],round_no-1,step,batch_size,seed,i,local_steps)
                    _,dt_central=_measured_train(central,central_opt,sample,1.)
                    central_seconds+=dt_central
                    local_loss,dt=_measured_train(peers_state[i].model,
                                                 peers_state[i].optimizer,sample,1.)
                    peers_state[i].compute_s+=dt; serial_peer_seconds+=dt
                    peers_state[i].last_loss=local_loss
                    round_losses.append(local_loss)
                    uncommitted_steps[i]+=1
                    train_tokens+=batch_size*context
            simulated_time+=local_steps/sim_steps_per_second
            for i in active:
                if uncommitted_steps[i]>=publish_steps:
                    event=_commit_local(peers_state[i],simulated_time)
                    net.publish(i,event,simulated_time)
                    uncommitted_steps[i]=0
                    if reinit_every and peers_state[i].emitted%reinit_every==0:
                        reset_committed_factors(peers_state[i].adapter,
                            seed=reinit_seed('p2p-training-sim',f'{seed}-{i}',peers_state[i].emitted))
                        peers_state[i].optimizer=make_adapter_optimizer(
                            peers_state[i].adapter,optimizer='adamw',lr=lr)
            if reinit_every and round_no%reinit_every==0:
                merge_and_reset_central(ca,
                    seed=reinit_seed('sim-central',str(seed),round_no))
                central_opt=make_adapter_optimizer(ca,optimizer='adamw',lr=lr)
            net.tick(simulated_time)
            if round_no%eval_every==0 or round_no==rounds:
                record_round(round_no, round_losses, active)
        # Final publication of any pending factor changes, allowing convergence.
        for i,p in enumerate(peers_state):
            if uncommitted_steps[i]:
                net.publish(i,_commit_local(p,simulated_time),simulated_time)
        simulated_time=net.catchup(simulated_time,catchup_seconds)
        published=sum(p.emitted for p in peers_state)
        seen=[len(p.seen) for p in peers_state]
        post_losses=[_eval(p.model,valid,batch_size=8) for p in peers_state]
        output={
            'kind':'REAL_GRADIENT_P2P_TRAINING_SIMULATION',
            'limitations':['Actual PyTorch LM forward/backward and AdamW optimization; no real WebRTC sockets',
                'Sparse Gossip delivery and compression are simulated with assumed transport parameters',
                'Peer optimizers run sequentially on one machine; per-peer parallel wall time is a projection',
                'Simulated events carry effective dense deltas in memory; byte estimates come from factor pairs',
                'Events originate within a trusted simulator, so cryptographic signature checks are not emulated',
                'This does NOT establish that real asynchronous Fedpara achieves the simulated convergence'],
            'config':{'model':model,'dataset':dataset_info,'peers':peers,'rounds':rounds,
                 'local_steps':local_steps,'publish_steps':publish_steps,'reinit_every':reinit_every,
                 'rank':rank,'wide_lora_rank':wide_lora_rank,'lr':lr,'batch_size':batch_size,
                 'context':context,'seed':seed,'fanout':fanout,'bundle_size':bundle_size,
                 'uplink_mbps':uplink_mbps,'latency_ms':latency_ms,
                 'loss_probability':loss_probability,'churn_fraction':churn_fraction,
                 'compression_ratio_assumption':compression_ratio,'device':device,
                 'max_outstanding_mib':max_outstanding_mib},
            'fairness':{'same_genesis':True,'same_batch_ids_for_each_active_peer':True,
                'same_total_tokens_in_central_and_p2p':True,
                'same_optimizer_family':'AdamW','central_reinit_unit':'rounds',
                'p2p_reinit_unit':'local publish events'},
            'result':{'initial_loss':initial_loss,'central_loss':_eval(central,valid,batch_size=8),
                 'p2p_mean_loss':sum(post_losses)/peers,
                 'p2p_worst_loss':max(post_losses),'p2p_losses':post_losses,
                 'total_tokens_each_arm':train_tokens,
                 'total_published_events':published,'event_counts_received':seen,
                 'all_events_received':all(count==published for count in seen),
                 'central_training_wall_s':round(central_seconds,4),
                 'peer_training_serial_wall_s':round(serial_peer_seconds,4),
                 'peer_training_ideal_parallel_wall_s':round(serial_peer_seconds/peers,4),
                 'peer_training_max_single_peer_wall_s':round(max(p.compute_s for p in peers_state),4),
                 'timing_definition':'central: measured synchronized optimizer compute; '
                     'P2P graph: sum of synchronized optimizer compute times / configured peers; '
                     'no validation, event creation, merges, transfer, or real concurrent device contention',
                 'simulated_network_elapsed_s':round(simulated_time,4),
                 **net.stats()},'trace':result_rows}
        if output_dir:
            dest=Path(output_dir);dest.mkdir(parents=True,exist_ok=True)
            (dest/'p2p-training-result.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
            keys=[k for k in result_rows[0] if k!='peer_valid_losses']
            with (dest/'p2p-training-trace.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(
                    {k:r[k] for k in keys} for r in result_rows)
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig,ax=plt.subplots(figsize=(9,5.4))
                xcentral=[r['central_compute_s'] for r in result_rows]
                xp2p=[r['p2p_compute_div_peers_s'] for r in result_rows]
                ax.plot(xcentral,[r['central_valid_loss'] for r in result_rows],
                        marker='.',markersize=2.5,label='Centralized: measured compute')
                ax.plot(xp2p,[r['p2p_mean_valid_loss'] for r in result_rows],
                        marker='.',markersize=2.5,label=f'P2P mean: sum(compute)/{peers}')
                ax.plot(xp2p,[r['p2p_worst_valid_loss'] for r in result_rows],
                        ls=':',label=f'P2P worst: sum(compute)/{peers}')
                ax.axhline(initial_loss,ls='--',label='Initial loss')
                ax.set(xlabel=f'Training compute seconds (P2P sum / {peers}; network/eval excluded)',
                       ylabel='Heldout cross-entropy',
                       title='Actual training / ideal-parallel P2P (NOT WebRTC or wall time)')
                ax.legend();ax.grid(alpha=.25);fig.tight_layout()
                fig.savefig(dest/'p2p-training-loss.png',dpi=150)
                plt.close(fig)
                # A separate token-axis view is retained for quality/sample efficiency.
                fig,ax=plt.subplots(figsize=(9,5))
                xtokens=[r['tokens_seen'] for r in result_rows]
                ax.plot(xtokens,[r['central_valid_loss'] for r in result_rows],label='Centralized')
                ax.plot(xtokens,[r['p2p_mean_valid_loss'] for r in result_rows],label='P2P mean')
                ax.plot(xtokens,[r['p2p_worst_valid_loss'] for r in result_rows],ls=':',label='P2P worst')
                ax.axhline(initial_loss,ls='--',label='Initial loss')
                ax.set(xlabel='Training tokens (same total)',ylabel='Heldout cross-entropy',
                       title='Sample efficiency: same total training tokens')
                ax.legend();ax.grid(alpha=.25);fig.tight_layout()
                fig.savefig(dest/'p2p-training-loss-tokens.png',dpi=150)
                plt.close(fig)
            except ImportError:
                pass
        return output
    finally:
        torch.set_num_threads(old_threads)
