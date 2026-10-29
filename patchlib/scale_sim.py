"""Deterministic, dependency-free *network* model of Fedpara sparse signed Gossip.

This is NOT PyTorch training, aiortc, Nim signaling, or a measured physical
network. It models bytes, serialization, bounded direct neighbors, drop/churn,
per-(origin,seq) deduplication, signed-aggregate-sized packets, periodic repair
and adaptive local publication. See README_NETWORK_SIM_V20.md for assumptions.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


def publication_interval(base: int, peers: int, cap: int = 64) -> int:
    if min(base, peers, cap) < 1:
        raise ValueError('publication settings must be positive')
    if base > cap:
        return base
    mul, boundary = 1, 20
    while peers >= boundary and base * mul < cap:
        mul *= 2
        boundary *= 4
    return min(cap, base * mul)


def bounded_neighbors(index: int, count: int, degree: int = 8) -> list[int]:
    if not 1 <= degree <= 8 or count < 2:
        raise ValueError('need at least two peers and degree 1..8')
    neighbors = set()
    # Sparse undirected ring and shortcuts; each node's degree remains <=8.
    for distance in (1, 2, 4, 8):
        for direction in (1, -1):
            other = (index + direction * distance) % count
            if other != index:
                neighbors.add(other)
            if len(neighbors) >= degree:
                break
        if len(neighbors) >= degree:
            break
    return sorted(neighbors)[:degree]


def choose_targets(index: int, peers: list[int], event_id: int, fanout: int,
                   excluded: int | None = None, origin: int | None = None) -> list[int]:
    choices = [x for x in sorted(peers) if x != index and x != excluded and x != origin]
    if len(choices) <= fanout:
        return choices
    clockwise = next((x for x in choices if x > index), choices[0])
    rest = [x for x in choices if x != clockwise]
    seed = event_id.to_bytes(8, 'big')
    rest.sort(key=lambda x: hashlib.blake2b(seed + x.to_bytes(4, 'big'),
                                            digest_size=8).digest(), reverse=True)
    return [clockwise, *rest[:fanout - 1]]


@dataclass(frozen=True)
class Parameters:
    peers: int = 20
    events_per_peer: int = 6
    duration_s: float = 16.
    catchup_s: float = 45.
    steps_per_s: float = 5.
    baseline_steps: int = 4
    adaptive: bool = True
    fanout: int = 2
    bundle_size: int = 3
    compression_ratio: float = .75
    event_kib: int = 1024
    uplink_mbps: float = 100.
    latency_ms: float = 20.
    loss_probability: float = 0.
    churn_fraction: float = 0.
    repair_period_s: float = 1.0
    max_backlog_s: float = 30.
    seed: int = 42

    def validate(self):
        if (not 2 <= self.peers <= 4096 or not 1 <= self.events_per_peer <= 100000
            or not 1 <= self.fanout <= 8 or not 1 <= self.bundle_size <= 8
            or self.event_kib < 1 or self.baseline_steps < 1 or
            min(self.duration_s, self.catchup_s, self.steps_per_s,
                self.uplink_mbps, self.repair_period_s, self.max_backlog_s) <= 0 or
            not 0 <= self.loss_probability < 1 or not 0 <= self.churn_fraction <= 1 or
            not 0 < self.compression_ratio <= 1 or self.latency_ms < 0):
            raise ValueError('invalid simulation parameters')


class NetworkSimulation:
    def __init__(self, config: Parameters):
        config.validate()
        self.c = config
        self.rng = random.Random(config.seed)
        self.neighbors = [bounded_neighbors(i, config.peers) for i in range(config.peers)]
        self.known = [set() for _ in range(config.peers)]
        self.originated = [0] * config.peers
        self.next_publish = []
        self.tx_free = [0.] * config.peers
        self.pending_push: dict[tuple[int, int], set[int]] = {}
        self.arrivals = []
        self.counter = 0
        self.events: dict[int, tuple[int, int, int, float]] = {}
        self.sent_bytes = 0
        self.sent_packets = 0
        self.push_bytes = 0
        self.repair_bytes = 0
        self.repair_packets = 0
        self.dropped_packets = 0
        self.duplicate_events = 0
        self.delivered_events = 0
        self.lags = []
        self.repair_in_flight = {}  # (recipient,event) -> simulated request retry deadline
        self.rows = []
        self.turn = [0]*config.peers
        self.churn = set(self.rng.sample(range(config.peers),round(config.peers*config.churn_fraction)))
        interval = publication_interval(config.baseline_steps,config.peers) if config.adaptive else config.baseline_steps
        self.publish_steps = interval
        spacing=interval/config.steps_per_s
        self.next_publish = [self.rng.uniform(0, spacing) + spacing for _ in range(config.peers)]
        self.online_at_end = [True]*config.peers

    def online(self, peer: int, t: float) -> bool:
        if peer not in self.churn:
            return True
        d=self.c.duration_s
        # Two independent temporary outages in the training window, each
        # followed by recovery, including the final catch-up period.
        return not (d*.22 <= t < d*.38 or d*.66 <= t < d*.79)

    def _send(self, src: int, dst: int, ids: list[int], now: float, kind: str):
        if not ids or not self.online(src,now):
            return None
        # All events have independent signatures + header. Compression is a
        # configurable *assumption*, not a measured zlib compression ratio.
        raw=len(ids)*self.c.event_kib*1024
        wire_bytes=(raw if len(ids)==1 else min(raw, 100+round(raw*self.c.compression_ratio)))
        if kind=='repair':
            wire_bytes += 128  # model one NEED control message; small approximation
        start=max(now,self.tx_free[src])
        if start-now > self.c.max_backlog_s:
            self.dropped_packets+=1
            return None
        finish=start+8*wire_bytes/(self.c.uplink_mbps*1e6)
        self.tx_free[src]=finish
        arrival=finish + self.c.latency_ms/1000.
        self.counter+=1
        heapq.heappush(self.arrivals,(arrival,self.counter,src,dst,tuple(ids),kind,wire_bytes))
        self.sent_bytes+=wire_bytes
        self.sent_packets+=1
        if kind=='push':
            self.push_bytes+=wire_bytes
        else:
            self.repair_bytes+=wire_bytes
            self.repair_packets+=1
        return arrival

    def _queue_push(self, src: int, item: int, excluded: int | None):
        origin=self.events[item][0]
        for dest in choose_targets(src,self.neighbors[src],item,self.c.fanout,
                                   excluded=excluded,origin=origin):
            self.pending_push.setdefault((src,dest),set()).add(item)

    def _flush_push(self, t: float):
        work,self.pending_push=self.pending_push,{}
        for (src,dst),pending in sorted(work.items()):
            ids=sorted(pending)
            for offset in range(0,len(ids),self.c.bundle_size):
                self._send(src,dst,ids[offset:offset+self.c.bundle_size],t,'push')

    def _receive(self, t: float):
        while self.arrivals and self.arrivals[0][0] <= t:
            timestamp,_,src,dst,ids,kind,_=heapq.heappop(self.arrivals)
            if (not self.online(src,timestamp) or not self.online(dst,timestamp)
                    or self.rng.random() < self.c.loss_probability):
                self.dropped_packets += 1
                continue
            for event in ids:
                if event in self.known[dst]:
                    self.duplicate_events += 1
                    continue
                self.known[dst].add(event)
                self.repair_in_flight.pop((dst,event),None)
                self.delivered_events+=1
                self.lags.append(max(0., timestamp-self.events[event][3]))
                self._queue_push(dst,event,src)

    def _repair(self,t:float):
        # Optimistic inventory model: peers compare their durable event-ID sets
        # after a frontier exchange. Real WebRTC NEED can take extra rounds and
        # might have unavailable intermediates; time/byte numbers are estimates.
        for dst in range(self.c.peers):
            if not self.online(dst,t):
                continue
            links=self.neighbors[dst]
            src=links[self.turn[dst]%len(links)]
            self.turn[dst]+=1
            if not self.online(src,t):
                continue
            missing={item for item in self.known[src]-self.known[dst]
                     if self.repair_in_flight.get((dst,item),float('-inf')) <= t}
            if missing:
                # Incremental request with strict bound, and one signed reply
                # for a few events. Full queues may be retried on a future tick.
                selected=sorted(missing)[:min(self.c.bundle_size*4,24)]
                for offset in range(0,len(selected),self.c.bundle_size):
                    chunk=selected[offset:offset+self.c.bundle_size]
                    arrival=self._send(src,dst,chunk,t,'repair')
                    if arrival is not None:
                        for item in chunk:
                            # Avoid duplicate NEED traffic while the packet
                            # is expected in flight. Lost packets retry later.
                            self.repair_in_flight[(dst,item)]=arrival+3.

    def _generate(self,t:float):
        period=self.publish_steps/self.c.steps_per_s
        for peer in range(self.c.peers):
            if not self.online(peer,t):
                self.next_publish[peer]=max(self.next_publish[peer],t+period)
                continue
            if self.originated[peer]>=self.c.events_per_peer:
                continue
            if t+1e-9 < self.next_publish[peer]:
                continue
            seq=self.originated[peer]+1
            item=peer*self.c.events_per_peer+seq
            # Integer additive delta: exactly-once sum is checked without
            # floating point/optimizer effects; no fabricated training loss.
            delta=self.rng.randint(-1000,1000)
            self.events[item]=(peer,seq,delta,t)
            self.originated[peer]=seq
            self.known[peer].add(item)
            self._queue_push(peer,item,None)
            self.next_publish[peer]+=period

    def _row(self,t:float):
        total=len(self.events)
        fractions=sum(len(s) for s in self.known)
        self.rows.append({'time_s':round(t,2),'generated':total,
            'coverage':round(fractions/(total*self.c.peers),6) if total else 1.,
            'missing_copies':total*self.c.peers-fractions,
            'packets':self.sent_packets,'sent_mib':round(self.sent_bytes/1048576,5),
            'dropped_packets':self.dropped_packets})

    def run(self):
        dt=.2
        t=0.
        last_repair=-1e20
        last_row=-1e20
        end=self.c.duration_s+self.c.catchup_s
        # Bound simulated runtime and memory; no real allocations of payload MB.
        while t<=end+1e-7:
            self._receive(t)
            if t <= self.c.duration_s:
                self._generate(t)
            self._flush_push(t)
            if t-last_repair >= self.c.repair_period_s - 1e-9:
                self._repair(t)
                last_repair=t
            if t-last_row >= 1-1e-9:
                self._row(t)
                last_row=t
            t=round(t+dt,10)
        self._receive(end)
        self._row(end)
        universal=set(self.events)
        gaps=sum(len(universal-self.known[i]) for i in range(self.c.peers))
        expected=sum(item[2] for item in self.events.values())
        exact=sum(sum(self.events[i][2] for i in known)==expected
                  for known in self.known)
        sorted_lags=sorted(self.lags)
        p95=(sorted_lags[min(len(sorted_lags)-1, math.ceil(len(sorted_lags)*.95)-1)]
             if sorted_lags else None)
        return {'configuration':vars(self.c),'publish_steps':self.publish_steps,
            'published_events':len(self.events),'peers':self.c.peers,
            'event_copies_missing':gaps,'coverage':1-gaps/max(1,len(universal)*self.c.peers),
            'exact_fedsum_peers':exact,'converged':gaps==0,
            'sent_mib':round(self.sent_bytes/1048576,3),
            'push_mib':round(self.push_bytes/1048576,3),
            'repair_mib':round(self.repair_bytes/1048576,3),
            'sent_packets':self.sent_packets,'repair_packets':self.repair_packets,
            'duplicate_events':self.duplicate_events,'dropped_packets':self.dropped_packets,
            'p95_arrival_lag_s':round(p95,3) if p95 is not None else None,
            'simulated_time_s':end,'rows':self.rows}


def write_report(result: dict, directory: Path, *, prefix='network'):
    directory.mkdir(parents=True,exist_ok=True)
    rows=result['rows']
    with (directory/f'{prefix}-timeseries.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    summary={k:v for k,v in result.items() if k!='rows'}
    (directory/f'{prefix}-summary.json').write_text(json.dumps(summary,ensure_ascii=False,
                                                   indent=2)+'\n',encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4))
        ax.plot([row['time_s'] for row in rows],[row['coverage'] for row in rows],
                label='Replica coverage')
        ax.set(xlabel='Simulated seconds',ylabel='Coverage',ylim=(0,1.02))
        ax.grid(alpha=.25);fig.tight_layout()
        fig.savefig(directory/f'{prefix}-coverage.png',dpi=140)
        plt.close(fig)
    except ImportError:
        pass
    return summary
