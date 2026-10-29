"""Signed elastic anti-entropy over Mummy signaling + aiortc DataChannels."""
from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from .core import PEER_ID, PatchError, _canonical_json
from .elastic import ElasticEngine, EventJournal
from .aggregate_wire import adaptive_publish_steps, pack_aggregate
from .snapshot_boundary import (SnapshotBoundaryAssembler, boundary_pages,
                                validate_cutoffs)
from .identity import signer_from_pem, verifier_from_pem
from .livefed import parse_links, parse_peer_keys
from .p2p import FrameReceiver, PatchNode, WINDOW, transmit

log = logging.getLogger('fedpara.elastic')
CTRL_VERSION = 3
CTRL_MAX = 16*1024
INVENTORY_PAGE = 64


class HandshakeError(PatchError):
    """Unrecoverable peer identity or model-generation mismatch."""
    pass
REQUEST_PAGE = 32
REQUEST_BACKLOG_LIMIT = 4096  # identities, not patch bodies; bounded at each neighbor
FRONTIER_WINDOW = 256
# Signed control messages are capped at 16 KiB. Large rooms MUST page the
# origin frontier; never try to transmit thousands of origins in one frame.
FRONTIER_ORIGIN_LIMIT = 96  # accept v17-sized signed pages for compatibility
FRONTIER_PAGE_SIZE = 48  # v18 sends small rotating pages
FRONTIER_KNOWN_ORIGINS_LIMIT = 16384
FRONTIER_PENDING_LIMIT = 4096
PUSH_FANOUT_DEFAULT = 2
NEED_IN_FLIGHT_RETRY_SECONDS = 6.0  # avoid repeatedly requesting large events still in flight


def frontier_advertisement_delay(room_members: int, *, catchup: bool = False) -> float:
    """Pace redundant frontier pages without delaying initial hole discovery.

    A peer with missing events receives the fast control plane until its
    inventory/frontier repair has caught up. Large stable rooms amortize the
    per-neighbor vector clock cycle; full-event NEED stays demand-driven.
    """
    if catchup or room_members <= 64:
        return .3
    if room_members <= 512:
        return .75
    return 1.5


def inventory_cycle_delay(configured: float, room_members: int, *,
                          peer_synced: bool, has_frontier: bool) -> float:
    """A safety audit of event identities remains, but not every tick.

    The authenticated frontier/NEED path remains active between audits.
    Without v1 signed frontiers, keep the original interval unchanged.
    """
    if not peer_synced or not has_frontier:
        return configured
    return max(configured, 120.0 if room_members > 512 else 30.0)


def frontier_advertisement_page(heads, cursor: int, *, limit=FRONTIER_PAGE_SIZE):
    """A bounded, rotating page of the per-origin vector clock.

    The cursor advances only after a successful signed send. A new origin
    arriving or disappearing from the ordered list cannot cause starvation:
    every finite, stable list is fully advertised within ceil(N/limit) sends.
    """
    if not 1 <= limit <= FRONTIER_PAGE_SIZE:
        raise ValueError('invalid frontier page size')
    if not heads:
        return [], 0
    start = cursor % len(heads)
    # For a huge list use a fixed slice and wrap on the next send. Avoid
    # an oversized control by never carrying more than `limit` heads.
    page = heads[start:start + limit]
    return page, 0 if start + len(page) >= len(heads) else start + len(page)


def selective_push_targets(local_peer: str, candidate_peers, *,
                           exclude_peer=None, origin_peer=None,
                           event_seed: bytes = b'', fanout=PUSH_FANOUT_DEFAULT):
    """Bound unsolicited large-event pushes per hop, not per room size.

    Prefer the next clockwise connected ring neighbor (a deterministic
    connected backbone in bounded-v2). Choose any remaining shortcuts using
    a stable event-keyed rendezvous score; NEED/frontier repair remains the
    durable fallback under churn, network partitions and queue overflow.
    """
    if not 1 <= fanout <= 8:
        raise ValueError('invalid gossip push fanout')
    candidates = sorted({p for p in candidate_peers
                         if p not in (local_peer, exclude_peer, origin_peer)})
    if len(candidates) <= fanout:
        return candidates
    clockwise = next((p for p in candidates if p > local_peer), candidates[0])
    remaining = [p for p in candidates if p != clockwise]
    scored = sorted(remaining, key=lambda p: hashlib.blake2b(
        event_seed + p.encode('ascii'), digest_size=8).digest(), reverse=True)
    return [clockwise, *scored[:fanout-1]]



def validated_need(entries):
    """Validate an entire signed NEED before servicing any of its IDs."""
    if not isinstance(entries, list) or not 1 <= len(entries) <= REQUEST_PAGE:
        raise PatchError('bad need request')
    result=[]
    for ident in entries:
        if not (isinstance(ident,list) and len(ident)==2 and
                isinstance(ident[0],str) and PEER_ID.fullmatch(ident[0]) and
                type(ident[1]) is int and 1<=ident[1]<2**63):
            raise PatchError('bad requested event ID')
        result.append((ident[0],ident[1]))
    return result


def validated_frontiers(entries):
    """Only authenticated peers may advertise bounded per-origin maxima."""
    if not isinstance(entries,list) or len(entries)>FRONTIER_ORIGIN_LIMIT:
        raise PatchError('bad frontier advertisement')
    maxima={}
    for entry in entries:
        if not (isinstance(entry,list) and len(entry)==2 and
                isinstance(entry[0],str) and PEER_ID.fullmatch(entry[0]) and
                type(entry[1]) is int and 1<=entry[1]<2**63):
            raise PatchError('bad frontier entry')
        if entry[0] in maxima:
            raise PatchError('duplicate frontier origin')
        maxima[entry[0]]=entry[1]
    return maxima



def inventory_missing_bounded(node, journal, entries):
    """Index untrusted-but-signed inventory without unbounded pending growth.

    The sender will advertise the same identities again in periodic inventory
    and frontier messages, so it is safe to stop admitting IDs at the bound.
    Never mark an incomplete peer 'ready' while the page overflowed.
    """
    if not isinstance(entries, list) or len(entries) > INVENTORY_PAGE:
        raise PatchError('malformed inventory')
    missing=[]
    for item in entries:
        if not (isinstance(item,list) and len(item)==3 and
                isinstance(item[0],str) and PEER_ID.fullmatch(item[0]) and
                type(item[1]) is int and 1 <= item[1] < 2**63 and
                isinstance(item[2],str) and len(item[2])==64):
            raise PatchError('malformed event identity in inventory')
        key=(item[0],item[1])
        original=journal.has(*key)
        if original and original[0] != item[2]:
            raise PatchError('conflicting event in inventory')
        if original is None and key not in node.pending:
            if len(node.pending) >= FRONTIER_PENDING_LIMIT:
                node.ready=False
                node.control_overflow_drops+=1
                continue
            missing.append([*key])
            node.pending.add(key)
    if missing:
        node.ready=False
    return missing



def eligible_snapshot_sources(slots, local_peer: str):
    """Prefer verified, available remote sources; never fan out full states.

    A per-source retry counter provides automatic failover to less-failed
    sources. A stable hash spreads first-time requests across room holders.
    """
    candidates=[slot for slot in slots.values()
                if not slot.disabled and slot.node is not None and
                slot.node.connected.is_set() and slot.node.verified_at > 0 and
                slot.node.remote_can_serve_snapshot]
    return sorted(candidates, key=lambda s:(s.retries,hashlib.sha256(
        (local_peer+'/'+s.peer).encode('ascii')).digest()))


@dataclass
class SnapshotProviderPool:
    """Leaderless full-state bootstrap: prefer the active verified provider.

    No peer owns an administrator role. Once any peer has installed a verified
    snapshot and its durable journal, it can provide a snapshot to a newcomer.
    The receiver switches providers if the chosen peer leaves or stalls, while
    never requesting multi-GB exports from multiple providers at once.
    """
    current: str | None = None
    cooldown_until: dict[str, float] = field(default_factory=dict)

    def select(self, slots, local_peer: str, now: float):
        candidates = eligible_snapshot_sources(slots, local_peer)
        # Keep memory bounded as peers churn in long-lived rooms.
        present = {s.peer for s in candidates}
        self.cooldown_until = {k: v for k, v in self.cooldown_until.items()
                               if k in present and v > now}
        available = [s for s in candidates if s.peer not in self.cooldown_until]
        # A cooldown must not force an unsafe unverified peer into service.
        if not available:
            self.current = None
            return None
        for slot in available:
            if slot.peer == self.current:
                return slot
        self.current = available[0].peer
        return available[0]

    def failed(self, peer: str, now: float, delay: float = 30.0):
        self.cooldown_until[peer] = now + delay
        if self.current == peer:
            self.current = None


def snapshot_history_still_unchanged(last_mismatch, *, manifest_sha: str,
                                    inventory_sha: str, event_count: int,
                                    journal_count: int) -> bool:
    """Skip repeated O(history) inventory scans until durable history changes.

    The signed boundary and the local append-only journal count together
    identify a previously failed catch-up attempt. Missing events can only
    become available when a new durable event is appended.
    """
    return last_mismatch == (manifest_sha, inventory_sha, event_count, journal_count)


def can_supply_snapshot(*, room_only: bool, bootstrap) -> bool:
    """Verified state, never creator identity, confers snapshot-serving rights."""
    return not room_only or (bootstrap.snapshot_installed and not bootstrap.fresh_join)


def encode_control(msg: dict, signer=None) -> str:
    body = {'v':CTRL_VERSION, **msg}
    raw = _canonical_json(body)
    digest = hashlib.sha256(raw).hexdigest()
    header = {'size':len(raw),'sha256':digest,
              'sig':signer(len(raw),digest) if signer else None}
    wire = json.dumps({**body, '_auth':header},separators=(',',':'))
    if len(wire) > CTRL_MAX:
        raise PatchError('oversized elastic control')
    return wire


def decode_control(wire: str, verifier=None, unsafe=False) -> dict:
    if not isinstance(wire,str) or len(wire)>CTRL_MAX:
        raise PatchError('invalid elastic control size')
    try: msg=json.loads(wire)
    except ValueError as e: raise PatchError('bad elastic control JSON') from e
    if not isinstance(msg,dict) or msg.get('v')!=CTRL_VERSION or '_auth' not in msg:
        raise PatchError('not an elastic control frame')
    auth=msg.pop('_auth')
    if not isinstance(auth,dict) or set(auth)!={'size','sha256','sig'}:
        raise PatchError('invalid control auth')
    payload = _canonical_json(msg)
    if type(auth['size']) is not int or auth['size']!=len(payload) or hashlib.sha256(payload).hexdigest()!=auth['sha256']:
        raise PatchError('corrupt elastic control frame')
    if not unsafe:
        if verifier is None: raise PatchError('peer public key missing')
        verifier(auth)
    return msg


async def pump_heartbeats(slots, stop: asyncio.Event, *, interval: float, peer_timeout: float):
    """Transport keepalive must progress during async disk/GPU checkpoint work.

    This coroutine owns no model weights, and never awaits remote peers.
    """
    period=max(0.2, min(float(interval), float(peer_timeout)/3))
    while not stop.is_set():
        for slot in tuple(slots.values()):
            node=slot.node
            if node and not slot.disabled and node.connected.is_set():
                if hasattr(node, 'can_serve_snapshot'):
                    node.control('heartbeat',serves_snapshot=node.can_serve_snapshot)
                else:
                    node.control('heartbeat')
        try:
            await asyncio.wait_for(stop.wait(), timeout=period)
        except asyncio.TimeoutError:
            pass


class ElasticNode(PatchNode):
    def __init__(self, *args, origin: str, remote: str, genesis: str, unsafe=False,
                 public_key: str | None = None, on_authenticated=None, snapshot_store=None,
                 enable_generation=False, generation_root=None,
                 snapshot_recv=None, snapshot_limit=32*1024**3, **kwargs):
        super().__init__(*args,**kwargs)
        self.origin,self.remote,self.genesis,self.unsafe=origin,remote,genesis,unsafe
        self.expert_bridge=None
        self.public_key=public_key
        self.on_authenticated=on_authenticated
        self.hello_sent=False
        self.hello_received=False
        # Small bounded batch buffer; never accumulate unbounded large updates.
        self.outgoing = asyncio.Queue(maxsize=3)
        self.aggregate_sent_batches = 0
        self.aggregate_sent_events = 0
        self.aggregate_bytes_avoided = 0
        self.controls = asyncio.Queue(maxsize=128)
        self.generation_controls = asyncio.Queue(maxsize=64)
        self.enable_generation = bool(enable_generation)
        # An immutable room-root identity distinguishes a legitimate
        # different certified epoch from another unrelated model. Merely
        # announcing generation_v1 must never bypass genesis validation.
        if generation_root is not None and (not isinstance(generation_root,str)
                or not re.fullmatch('[0-9a-f]{64}',generation_root)
                or not self.enable_generation):
            raise ValueError('generation root requires authenticated generation transport')
        self.generation_root = generation_root
        self.remote_genesis = None
        self.bridge_only = False
        self.remote_supports_generation = False
        self.remote_supports_critic = False
        self.remote_supports_rollout = False
        self.last_rollout_frontier = 0.0
        self.rollout_frontier_offset = 0
        self.last_critic_frontier = 0.0
        self.critic_frontier_offset = 0
        self.counter = 0
        self.remote_counter = -1
        self.ready = False
        self.verified_at = 0.0
        self.pending: set[tuple[str,int]] = set()
        self.need_cursor = 0  # prevent an undeliverable low-ID prefix starving the rest
        self.backpressure_drops = 0  # only retryable traffic is discarded
        self.push_sent_wire_bytes = 0
        self.repair_sent_wire_bytes = 0
        self.push_sent_events = 0
        self.repair_sent_events = 0
        self.inventory_done = False
        self.inventory_cursor = 0
        self.last_inventory = 0.0
        self.next_inventory_at = 0.0
        self.next_need_retry_at = 0.0
        self.need_in_flight: dict[tuple[str, int], float] = {}
        self.next_frontier_at = 0.0
        self.remote_frontiers: dict[str, int] = {}
        self.frontier_cursors: dict[str, int] = {}
        self.frontier_origin_cursor = 0
        self.frontier_advertisement_cursor = 0
        self.next_frontier_scan_at = 0.0
        self.frontier_scanned = 0
        self.need_received = 0
        # Demand-driven anti-entropy responses have precedence over redundant
        # broadcast fanout. Keep only IDs here; payloads remain in SQLite.
        self.requested_events = OrderedDict()
        self.lookup_requested_event = None
        self.lookup_requested_event_async = None
        self.request_backlog_overflow = 0
        self.request_responses_sent = 0
        self.connected_since = 0.0
        # Signed availability advertisement: a cold joiner must not request
        # a full model from ANOTHER cold joiner, which has nothing to serve.
        self.can_serve_snapshot = False
        self.remote_can_serve_snapshot = False
        self.remote_supports_frontier = False
        self.remote_supports_aggregate = False
        self.snapshot_link = None
        self.snapshot_tip = None
        # At most one signed full-state offer is in flight per RTC link.
        # A critical manifest boundary must never be discarded merely because
        # the slow model-training reader filled the normal control queue.
        self.snapshot_tip_latch = None
        self.snapshot_boundary_assembler = SnapshotBoundaryAssembler()
        self.remote_supports_snapshot_pages = False
        # Signed snapshot progress is only relevant to the current request.
        # A bounded latch prevents a busy trainer from dropping the status
        # and incorrectly treating a live source as stalled.
        self.snapshot_status_latch = None
        self.control_overflow_drops = 0
        self.latest_snapshot_tip_counter = -1
        self.snapshot_requested_at = 0.0
        # A signed snapshot-request may arrive before the independent state
        # DataChannel opens. Never discard that request.
        self.snapshot_pending = False
        self.snapshot_pending_id = ''
        self.snapshot_tx_stage = ''
        self.snapshot_tx_last_report = 0.0
        self.snapshot_request_id = ''
        self.snapshot_remote_stage = ''
        self.snapshot_remote_stage_at = 0.0
        if snapshot_store is not None:
            from .snapshot_live import LiveSnapshotLink
            self.snapshot_link=LiveSnapshotLink(Path(snapshot_store),Path(snapshot_recv),
                snapshot_limit,lambda: self.verifier if self.verified_at else None)

    def _attach_extra_channel(self,ch):
        if self.snapshot_link is None:
            ch.close()
        else:
            self.snapshot_link.attach(ch)

    def control(self, kind, **data) -> bool:
        ch=self._channel
        if ch is None or ch.readyState!='open' or ch.bufferedAmount > WINDOW:
            return False
        # SCTP delivery is ordered, but a failed first send must never be
        # followed by a heartbeat or inventory before the signed hello.
        if self.bridge_only and kind not in ('hello','heartbeat',
                'generation-head','generation-need','generation-wire',
                'generation-have','generation-snapshot-request',
                'generation-compact-have','generation-compact-snapshot-request',
                'generation-compact-head','generation-compact-need',
                'generation-carry-offer','generation-carry-need','generation-carry-chunk'):
            return False  # A signed v1 update must never cross epochs.
        if kind != 'hello' and not self.hello_sent:
            if not self.control('hello',serves_snapshot=self.can_serve_snapshot, frontier_v1=True, aggregate_v1=True, snapshot_tip_pages_v1=True, generation_v1=self.enable_generation, critic_v78=True, trajectory_v82=True,
                    **({'generation_root':self.generation_root} if self.generation_root else {}),
                    **({'pub':self.public_key} if self.public_key else {})):
                return False
        self.counter+=1
        try:
            ch.send(encode_control({'type':kind,'peer':self.origin,'genesis':self.genesis,
                                    'counter':self.counter,**data},self.signer))
            if kind == 'hello':
                self.hello_sent=True
        except (ConnectionError, OSError, RuntimeError) as exc:
            self.last_error=exc
            # A transient SCTP backpressure error is not proof that ICE died.
            # Keep the link eligible for heartbeat-driven hello retry as long
            # as the DataChannel itself remains open.
            if ch.readyState!='open':
                self.connected.clear()
            return False
        return True

    def accept_snapshot_request(self, request_id: str) -> None:
        """Record a signed, authenticated request even if state SCTP is not ready."""
        if (not isinstance(request_id,str) or len(request_id)>32 or
            (request_id and not re.fullmatch(r'[a-f0-9]{16}',request_id))):
            raise PatchError('invalid snapshot request id')
        if self.snapshot_pending_id != request_id:
            self.snapshot_tx_stage=''
        self.snapshot_pending=True
        self.snapshot_pending_id=request_id
        print(f'SNAPSHOT RX REQUEST peer={self.remote} '
              f'state_channel={bool(self.snapshot_link and self.snapshot_link.connected.is_set())}',
              flush=True)
        self.control('snapshot-status', request_id=request_id,stage='accepted')

    def retry_missing(self, now: float) -> bool:
        """Round-robin signed NEED requests without retrying in-flight IDs at 3 Hz.

        An ACK-less NEED response might take seconds to serialize or cross a
        congested link. Sending the same 32-item page every 300 ms amplifies
        backpressure. Re-request only after a bounded cooldown; missing IDs
        still rotate fairly across thousands of origins. Failed control sends
        never mark IDs in flight.
        """
        if not self.connected.is_set() or not self.pending or now < self.next_need_retry_at:
            return False
        # Bound request metadata by the existing bounded pending set, and
        # remove bookkeeping for events already durably applied.
        for key in list(self.need_in_flight):
            if key not in self.pending:
                self.need_in_flight.pop(key, None)
        ordered=sorted(self.pending)
        offset=self.need_cursor % len(ordered)
        rotated=ordered[offset:]+ordered[:offset]
        page=[key for key in rotated
              if now-self.need_in_flight.get(key,float('-inf'))>=NEED_IN_FLIGHT_RETRY_SECONDS][:REQUEST_PAGE]
        if not page:
            self.next_need_retry_at=now+0.3
            return False
        success=self.control('need',events=[list(item) for item in page])
        if success:
            for key in page:
                self.need_in_flight[key]=now
            self.need_cursor=(offset+len(page))%len(ordered)
        self.next_need_retry_at=now+(0.3 if success else 0.2)
        return success

    def reconcile_frontiers(self, journal, *, budget: int = 512) -> int:
        """Detect holes (including old gaps below the highest local seq).

        A scan continues across ticks and wraps, so missing events cannot
        starve behind a long history or a frequently reconnected channel.
        Only an authenticated signed advertisement supplies the upper bound.
        """
        if not self.remote_frontiers or len(self.pending) >= FRONTIER_PENDING_LIMIT:
            return 0
        origins=sorted(self.remote_frontiers)
        # Cycle origins as well as sequences. Otherwise a 256-entry budget
        # repeatedly scans the first origin and STARVES all other origins.
        start=self.frontier_origin_cursor%len(origins)
        found=0
        for i in range(min(4,len(origins))):
            if budget <= 0 or len(self.pending) >= FRONTIER_PENDING_LIMIT:
                break
            origin=origins[(start+i)%len(origins)]
            high=self.remote_frontiers[origin]
            first=self.frontier_cursors.get(origin,1)
            if first>high:first=1
            last=min(high,first+63,first+budget-1)
            missing=journal.missing_range(origin,first,last)
            for key in missing:
                if len(self.pending)>=FRONTIER_PENDING_LIMIT:break
                if key not in self.pending:
                    self.pending.add(key)
                    found+=1
            budget-=last-first+1
            self.frontier_scanned+=last-first+1
            self.frontier_cursors[origin]=1 if last==high else last+1
        self.frontier_origin_cursor=(start+min(4,len(origins)))%len(origins)
        if found:
            self.ready=False
            # Retry right now when a signed frontier first reveals gaps.
            self.next_need_retry_at=0.0
        return found

    def queue_requested_events(self, identities) -> None:
        """Remember authenticated, bounded NEED requests until delivered.

        Never discard the suffix of a 32-event request merely because the
        ordinary three-slot broadcast queue was full when the control arrived.
        An old request is safe to retry: every event is content addressed.
        """
        for origin, seq in identities:
            key=(origin,seq)
            if key in self.requested_events:
                continue
            if len(self.requested_events) >= REQUEST_BACKLOG_LIMIT:
                self.request_backlog_overflow += 1
                break  # sender will retry via periodic inventory/need
            self.requested_events[key] = None

    async def _pump(self):
        """Service durable anti-entropy demands before speculative fanout.

        Bound per-peer memory to IDs, fetch signed blobs from the durable
        journal only as needed, and send one normal gossip message after each
        eight requested messages so live fanout never starves completely.
        """
        priority_streak=0
        deferred=deque()
        try:
            while True:
                # A mixed-generation link transports ONLY signed certificate
                # controls. Keep v1 events in the durable journal for same-epoch
                # peers; never send them to a differently keyed optimizer.
                if not self.hello_received or self.bridge_only:
                    await asyncio.sleep(.05)
                    continue
                data=None
                demand_selected=False
                demanded=[]
                aggregate_stats=None  # count only successfully transmitted bundles
                if (self.requested_events and self.lookup_requested_event is not None
                        and (priority_streak < 8 or
                             (self.outgoing.empty() and not deferred))):
                    ident,_=self.requested_events.popitem(last=False)
                    if self.lookup_requested_event_async is not None:
                        data=await self.lookup_requested_event_async(*ident)
                    else:
                        data=self.lookup_requested_event(*ident)
                    if data is None:
                        continue  # upstream has not received it: requester retries
                    demanded=[ident]
                    priority_streak += 1
                    demand_selected=True
                    # Cross-origin NEED repair also benefits from aggregation.
                    # Preserve independently signed constituent events, and
                    # restore every unsent ID on fallback/cancellation.
                    if (self.remote_supports_aggregate and
                            data.startswith(b'FPEL1\0') and self.requested_events):
                        collected=[data]
                        for _ in range(2):
                            if not self.requested_events:
                                break
                            next_ident,_=self.requested_events.popitem(last=False)
                            if self.lookup_requested_event_async is not None:
                                nxt=await self.lookup_requested_event_async(*next_ident)
                            else:
                                nxt=self.lookup_requested_event(*next_ident)
                            if nxt is None:
                                continue
                            if not nxt.startswith(b'FPEL1\0'):
                                self.requested_events[next_ident]=None
                                self.requested_events.move_to_end(next_ident,last=False)
                                break
                            collected.append(nxt)
                            demanded.append(next_ident)
                        if len(collected)>1:
                            try:
                                combined=await asyncio.to_thread(
                                    pack_aggregate,collected,max_bytes=self.max_bytes)
                            except (ValueError,PatchError):
                                combined=None
                            if combined is not None and len(combined)<sum(map(len,collected)):
                                data=combined
                                aggregate_stats=(len(collected),sum(map(len,collected))-len(combined))
                            else:
                                for excess in reversed(demanded[1:]):
                                    self.requested_events[excess]=None
                                    self.requested_events.move_to_end(excess,last=False)
                                demanded=demanded[:1]
                else:
                    try:
                        data=(deferred.popleft() if deferred else
                              await asyncio.wait_for(self.outgoing.get(),timeout=.10))
                        priority_streak=0
                    except asyncio.TimeoutError:
                        continue
                # Coalesce original independently signed events across ANY
                # origins. Relay adds no authority: receivers authenticate
                # every constituent and only apply previously unseen IDs.
                event_count=1
                if (not demand_selected and self.remote_supports_aggregate
                        and data.startswith(b'FPEL1\0')):
                    collected=[data]
                    while len(collected)<3 and not self.outgoing.empty():
                        nxt=self.outgoing.get_nowait()
                        if not nxt.startswith(b'FPEL1\0'):
                            deferred.append(nxt)
                            break
                        collected.append(nxt)
                    if len(collected)>1:
                        try:
                            aggregate=await asyncio.to_thread(
                                pack_aggregate,collected,max_bytes=self.max_bytes)
                        except (ValueError,PatchError):
                            aggregate=None
                        if aggregate is not None and len(aggregate)<sum(map(len,collected)):
                            data=aggregate
                            event_count=len(collected)
                            aggregate_stats=(event_count,sum(map(len,collected))-len(data))
                        else:
                            deferred.extendleft(reversed(collected[1:]))
                self.sent.clear()
                # A permanently wedged SCTP send buffer must not keep this
                # authenticated-looking link unable to respond indefinitely.
                timeout=max(60.0, min(300.0, 60.0 + len(data)/(256*1024)))
                try:
                    await asyncio.wait_for(
                        transmit(self._channel,data,max_bytes=self.max_bytes,signer=self.signer),
                        timeout=timeout)
                except BaseException:
                    # The response has been removed from the NEED backlog,
                    # but SCTP might not have delivered even its first frame.
                    # Restore its ID so a recovered channel on the same node
                    # can retry; a replacement node is repaired by frontiers.
                    if demand_selected:
                        for excess in reversed(demanded):
                            self.requested_events[excess]=None
                            self.requested_events.move_to_end(excess,last=False)
                    raise
                if aggregate_stats is not None:
                    self.aggregate_sent_batches+=1
                    self.aggregate_sent_events+=aggregate_stats[0]
                    self.aggregate_bytes_avoided+=aggregate_stats[1]
                if demand_selected:
                    self.request_responses_sent += len(demanded)
                    self.repair_sent_wire_bytes += len(data)
                    self.repair_sent_events += len(demanded)
                else:
                    self.push_sent_wire_bytes += len(data)
                    self.push_sent_events += event_count
                self.sent.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # An uncaught pump exception used to leave a seemingly connected
            # channel unable to send another update indefinitely.
            self.last_error=exc
            log.warning('signed gossip transmit failed peer=%s error=%s: %s',
                        self.remote,type(exc).__name__,exc)
            self.protocol_failure.set()
            self.link_closed.set()

    def _attach_channel(self, ch):
        if self._channel is not ch:
            self.connected.clear()
            if self._send_task is not None:
                self._send_task.cancel()
                self._send_task=None
        self._channel=ch
        self.ready=False
        self.inventory_done=False
        self.link_closed.clear()
        opened_once=False
        # aiortc may deliver an incoming signed hello before the local open
        # callback fires. Initialize per-channel auth HERE, never in on_open.
        self.hello_sent=False
        self.hello_received=False
        self.verified_at=0.0
        self.snapshot_tip_latch=None
        self.snapshot_boundary_assembler.clear()
        self.remote_supports_snapshot_pages=False
        self.snapshot_status_latch=None
        self.latest_snapshot_tip_counter=-1
        # A newly attached SCTP generation is a fresh signed control stream.
        # Never process a delayed hello/status from the previous transport.
        while not self.controls.empty():
            self.controls.get_nowait()
        self.snapshot_pending=False
        self.snapshot_pending_id=''
        self.snapshot_tx_stage=''
        self.snapshot_remote_stage=''
        self.snapshot_remote_stage_at=0.0
        self.remote_can_serve_snapshot=False
        self.remote_supports_frontier=False
        self.remote_supports_aggregate=False
        self.remote_supports_generation=False
        self.remote_genesis=None
        self.bridge_only=False
        while not self.generation_controls.empty():
            self.generation_controls.get_nowait()
        # A new transport generation must be allowed to re-request missing
        # events immediately; the previous channel's NEEDs may be lost.
        self.need_in_flight.clear()
        self.remote_counter=-1
        framer=FrameReceiver(self.max_bytes,self.verifier)

        @ch.on('open')
        def on_open():
            nonlocal opened_once
            if self._channel is not ch or opened_once:
                return
            opened_once=True
            self.connected.set()
            self.set_phase('datachannel-open')
            self.connected_since=time.monotonic()
            self._send_task=asyncio.create_task(self._pump())
            if not self.control('hello',serves_snapshot=self.can_serve_snapshot, frontier_v1=True, aggregate_v1=True, snapshot_tip_pages_v1=True, generation_v1=self.enable_generation, **({'generation_root':self.generation_root} if self.generation_root else {}), **({'pub':self.public_key} if self.public_key else {})):
                log.warning('signed hello initial send deferred peer=%s',self.remote)

        @ch.on('close')
        def on_close():
            if self._channel is not ch: return
            self.connected.clear()
            self.link_closed.set()
            if self.last_error is None:
                self.last_error=ConnectionError('signed patch DataChannel closed')
            self.set_phase('datachannel-closed')
            self.ready=False
            self.hello_sent=False
            self.hello_received=False
            self.verified_at=0.0
            if self._send_task: self._send_task.cancel()

        @ch.on('message')
        def on_message(message):
            if self._channel is not ch: return
            try:
                if isinstance(message,str) and message.startswith('FP-EXPERT1:'):
                    if self.verifier is None or not self.connected.is_set():
                        raise PatchError('expert frame before authenticated hello')
                    if self.expert_bridge is None:
                        raise PatchError('no expert handler configured')
                    self.expert_bridge.accept(message)
                elif isinstance(message,str) and '"_auth"' in message:
                    # Auto mode permits exactly one unauthenticated *claim*:
                    # a hello containing a key which must verify its own
                    # key-derived peer ID and the signature over this hello.
                    # All peers MUST authenticate their first control frame.
                    # Even a previously pinned verifier cannot turn a later
                    # heartbeat into a substitute for a signed hello.
                    proposed=json.loads(message)
                    new_verifier=None
                    if self.public_key is not None and self.verifier is None:
                        if proposed.get('type')!='hello' or proposed.get('peer')!=self.remote:
                            raise HandshakeError('signed hello required before other control traffic; update BOTH peers')
                        from .roomid import identity_from_public_b64
                        new_verifier=identity_from_public_b64(proposed.get('pub'),self.remote)
                        decoded=decode_control(message,new_verifier,False)
                    else:
                        decoded=decode_control(message,self.verifier,self.unsafe)
                    if not self.hello_received and decoded.get('type') != 'hello':
                        raise HandshakeError('signed hello required before '+str(decoded.get('type'))+'; update BOTH peers')
                    if decoded.get('peer') != self.remote:
                        raise HandshakeError('HELLO PEER ID MISMATCH (room peer identity differs from signed frame)')
                    if not self.hello_received:
                        if decoded.get('genesis') != self.genesis:
                            if (decoded.get('type')!='hello' or not self.enable_generation
                                    or self.generation_root is None
                                    or decoded.get('generation_v1') is not True
                                    or decoded.get('generation_root') != self.generation_root):
                                raise HandshakeError('GENESIS MISMATCH: different model or unverified generation root')
                            self.bridge_only=True
                        self.remote_genesis=decoded.get('genesis')
                    elif decoded.get('genesis') != self.remote_genesis:
                        raise HandshakeError('peer changed genesis inside an authenticated connection')
                    if self.bridge_only and decoded.get('type') not in (
                            'hello','heartbeat','generation-head','generation-need',
                            'generation-wire','generation-have','generation-snapshot-request',
                            'generation-compact-head','generation-compact-need',
                            'generation-compact-have','generation-compact-snapshot-request',
                            'generation-carry-offer','generation-carry-need',
                            'generation-carry-chunk'):
                        raise PatchError('cross-generation v1 traffic forbidden')
                    if type(decoded.get('counter')) is not int or decoded['counter']<=self.remote_counter:
                        raise PatchError('replayed/out-of-order control counter')
                    if decoded.get('type')=='hello':
                        if self.public_key is not None:
                            # Authenticate the self-certified key even if an
                            # old verifier for this peer was loaded from disk.
                            from .roomid import identity_from_public_b64
                            pub_verifier=identity_from_public_b64(decoded.get('pub'),self.remote)
                            pub_verifier(proposed['_auth'])
                            if self.on_authenticated is not None:
                                self.on_authenticated(self.remote,decoded['pub'],pub_verifier)
                        self.hello_received=True
                        # Capability negotiation preserves compatibility with
                        # older v3 peers that do not understand 'frontier'.
                        self.remote_supports_critic = decoded.get('critic_v78') is True
                        self.remote_supports_rollout = decoded.get('trajectory_v82') is True
                        self.remote_supports_frontier = decoded.get('frontier_v1') is True
                        self.remote_supports_aggregate = decoded.get('aggregate_v1') is True
                        self.remote_supports_snapshot_pages = decoded.get('snapshot_tip_pages_v1') is True
                        self.remote_supports_generation = self.enable_generation and decoded.get('generation_v1') is True
                        if self.bridge_only and (not self.remote_supports_generation
                                or decoded.get('generation_root')!=self.generation_root):
                            raise HandshakeError('unauthenticated generation bridge')
                        log.info('SIGNED HELLO OK peer=%s',self.remote)
                    if new_verifier is not None:
                        self.verifier=new_verifier
                        framer.verifier=new_verifier
                    self.remote_counter=decoded['counter']
                    self.verified_at=time.monotonic()
                    if decoded.get('type') in ('hello','heartbeat'):
                        # Trusted only AFTER signature and key-derived identity
                        # verification. Ignore untrusted/missing legacy flags.
                        self.remote_can_serve_snapshot = decoded.get('serves_snapshot') is True
                    if self.phase != 'authenticated':
                        self.set_phase('authenticated')
                    # NEED does not depend on model/GPU operations. Servicing
                    # it via the bounded trainer control queue used to silently
                    # drop legitimate retransmission requests under pressure.
                    # Accept the entire authenticated request immediately.
                    if decoded.get('type')=='need':
                        self.queue_requested_events(validated_need(decoded.get('events')))
                        self.need_received+=1
                        return
                    if decoded.get('type')=='frontier':
                        maxima=validated_frontiers(decoded.get('heads'))
                        for origin,seq in maxima.items():
                            if (origin not in self.remote_frontiers and
                                len(self.remote_frontiers) >= FRONTIER_KNOWN_ORIGINS_LIMIT):
                                raise PatchError('remote frontier origin budget exceeded')
                            self.remote_frontiers[origin]=max(
                                seq,self.remote_frontiers.get(origin,0))
                        return
                    if decoded.get('type') in ('generation-head','generation-need','generation-wire','generation-have','generation-snapshot-request','generation-compact-head','generation-compact-need',
                           'generation-carry-offer','generation-carry-need',
                           'generation-carry-chunk'):
                        if not self.remote_supports_generation:
                            raise PatchError('generation frame without mutual negotiated capability')
                        fields={k:v for k,v in decoded.items() if k not in ('type','peer','genesis','counter','v')}
                        if decoded['type'].startswith('generation-carry-'):
                            from .epoch_carry_gossip_v53 import validate_carry_control
                            validate_carry_control(decoded['type'],fields)
                        else:
                            from .live_generation_transport import validate_control
                            validate_control(decoded['type'],fields)
                        # Critical state changes bypass the ordinary queue; never
                        # silently drop certified generation evidence. Retries via
                        # anti-entropy recover overflow after reconnect.
                        self.generation_controls.put_nowait((decoded['type'],fields))
                        return
                    if decoded.get('type')=='snapshot-tip-page':
                        if (self.snapshot_request_id and decoded.get('request_id') != self.snapshot_request_id):
                            return  # Obsolete signed offer from an earlier request.
                        assembled=self.snapshot_boundary_assembler.accept(decoded)
                        if assembled is not None:
                            self.snapshot_tip_latch=assembled
                        return
                    if decoded.get('type')=='snapshot-tip' and self.controls.full():
                        # Snapshot boundaries are non-droppable: when the
                        # trainer falls behind, coalesce the latest signed
                        # tip separately rather than tearing down good RTC.
                        # The ordinary non-full path remains FIFO-compatible.
                        self.snapshot_tip_latch=decoded
                        return
                    if decoded.get('type') == 'snapshot-status' and self.controls.full():
                        # Only the latest status for OUR outstanding request
                        # matters. A paused trainer must not cause an otherwise
                        # valid snapshot provider to reconnect repeatedly.
                        if decoded.get('request_id') == self.snapshot_request_id:
                            self.snapshot_status_latch=decoded
                        else:
                            self.control_overflow_drops+=1
                        return
                    # Control-channel ACK and durable in-memory request latch
                    # must not wait for a GPU training step to finish. Handling
                    # the request here does NOT read or mutate model weights;
                    # snapshot export is scheduled later in the training loop.
                    if decoded.get('type')=='snapshot-request':
                        self.accept_snapshot_request(decoded.get('request_id',''))
                    # The signature has already been checked and verified_at
                    # refreshed. During a long GPU step, the application-level
                    # queue can fill with redundant keepalives. Dropping a
                    # redundant heartbeat is safer than closing a valid SCTP
                    # connection due to QueueFull; never drop state-changing
                    # controls (inventory, need, snapshot, hello).
                    if self.controls.full():
                        # These frames are periodically regenerated (or need
                        # requests retried). Closing an authenticated DTLS
                        # link because its small control queue is temporarily
                        # full creates a reconnect storm under 8+ peers.
                        if decoded.get('type') in ('heartbeat','inventory','hello-ack','hello','snapshot-request','snapshot-status','epoch-v2-wire','critic-frontier','trajectory-frontier'):
                            self.backpressure_drops+=1
                            return
                        # Snapshot controls are NOT silently dropped. A fresh
                        # snapshot request / status must be retried instead.
                        raise PatchError('critical control queue saturated')
                    self.controls.put_nowait(decoded)
                else:
                    if not self.hello_received:
                        raise PatchError('data received before authenticated hello')
                    if self.bridge_only:
                        raise PatchError('cross-generation signed data frame forbidden')
                    event=framer.feed(message)
                    if event is not None:
                        if self.incoming.full():
                            # Complete, signed journal events are replayable via
                            # anti-entropy. Never kill the good connection for
                            # a transiently saturated application queue.
                            self.backpressure_drops+=1
                        else:
                            self.incoming.put_nowait(event)
            except (PatchError,ValueError,asyncio.QueueFull,TypeError,RuntimeError,OSError,KeyError) as e:
                if not self.hello_received and not isinstance(e,HandshakeError):
                    e=HandshakeError('SIGNED HELLO FAILED: '+str(e))
                self.last_error=e
                log.warning('SIGNED HELLO / peer frame rejected peer=%s phase=%s reason=%s',self.remote,self.phase,e)
                self.protocol_failure.set()  # wake signaling receive loop now, not in 120s
                ch.close()

        # An incoming aiortc DataChannel may ALREADY be open when our
        # RTCPeerConnection datachannel callback receives it. Its 'open' event
        # will not fire again. Initialize authenticated handshake immediately.
        if ch.readyState == 'open':
            on_open()


class PresenceHysteresis:
    """Debounce disappearing group members, without delaying new joins.

    Presence is a signaling hint, NOT evidence that an authenticated SCTP
    channel has died. Suppress transient `group-peer-left` / `group-peer-joined`
    flapping while keeping the *same* membership set used for bounded overlay
    calculations. That preserves its maximum-degree guarantee.
    """

    def __init__(self, grace_seconds: float = 20.0):
        self.grace_seconds = float(grace_seconds)
        self.last_seen: dict[str, float] = {}

    def observe(self, current: set[str], now: float, *, online: bool) -> set[str]:
        if online:
            for peer in current:
                self.last_seen[peer] = now
            for peer, seen in list(self.last_seen.items()):
                if peer not in current and now - seen >= self.grace_seconds:
                    del self.last_seen[peer]
        # During a signaling outage last known peers are kept for direct
        # WebRTC; they are dropped when an authoritative roster is stable.
        return set(self.last_seen)


@dataclass
class PeerSlot:
    peer: str
    room: str
    key_path: Path|None
    node: ElasticNode|None = None
    task: asyncio.Task|None = None
    retry_after: float = 0.0
    started_at: float = 0.0
    retries: int = 0
    disabled: bool = False
    stable_since: float = 0.0
    blocked_reason: str = ''
    heartbeat_warned: bool = False
    retiring: asyncio.Task | None = None


async def send_snapshot_boundary(node, manifest_sha, inventory_sha, count, cutoffs, *, timeout=60.0):
    """Offer an authenticated boundary, in signed pages if necessary.

    Serializes pages over the existing ordered SCTP control DataChannel, never
    enqueues all pages into a bounded control queue. On backpressure retry the
    *same* page with fresh signed counter. Abort on connection/request turnover.
    """
    validate_cutoffs(cutoffs)
    # Preserve the v25 one-frame format for old peers when it actually fits.
    if len(cutoffs) <= 32:
        try:
            if node.control('snapshot-tip',manifest_sha256=manifest_sha,
                            inventory_sha256=inventory_sha,events=count,cutoffs=cutoffs):
                return True
        except PatchError as exc:
            if 'oversized' not in str(exc):
                raise
    if not node.remote_supports_snapshot_pages:
        raise PatchError('snapshot-tip paging requires an upgraded peer; update both participants')
    request_id = node.snapshot_pending_id
    transport = node._channel
    deadline = time.monotonic() + timeout
    for part in boundary_pages(manifest_sha, inventory_sha, count, cutoffs, request_id):
        while True:
            if (node._channel is not transport or transport is None or
                transport.readyState != 'open' or node.snapshot_pending_id != request_id):
                return False
            if node.control('snapshot-tip-page', **part):
                break
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(.025)
    return True


async def service_snapshot_requests(slots, *, source_allowed: bool, send_snapshot, now: float):
    """Drain authenticated snapshot requests, including ones received before state open.

    An ACK is not proof of delivery. A request remains pending until the
    manifest has been queued for transfer, so a channel-opening race or
    transient backpressure cannot permanently strand a cold joiner.
    """
    for slot in list(slots.values()):
        node=slot.node
        if node is None or not node.snapshot_pending or slot.disabled:
            continue
        channel_ready=bool(node.snapshot_link and node.snapshot_link.connected.is_set())
        if source_allowed and channel_ready and node.verified_at > 0:
            try:
                if await send_snapshot(slot):
                    node.snapshot_pending=False
            except Exception as exc:
                node.last_error=exc
                node.protocol_failure.set()
                print(f'SNAPSHOT SEND FAILED peer={slot.peer}: '
                      f'{type(exc).__name__}: {exc}',flush=True)
        elif now-node.snapshot_tx_last_report > 5:
            stage='source-not-ready' if not source_allowed else 'waiting-state-channel'
            node.control('snapshot-status',request_id=node.snapshot_pending_id,stage=stage)
            print(f'SNAPSHOT SEND PENDING peer={slot.peer} stage={stage}',flush=True)
            node.snapshot_tx_last_report=now


def read_roster(path: Path, myself: str) -> dict[str, tuple[str,Path|None]]:
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data,dict) or len(data)>255:
        raise ValueError('roster file must be object of <=255 peers')
    result={}
    for peer,rec in data.items():
        if not PEER_ID.fullmatch(peer) or peer==myself or not isinstance(rec,dict):
            raise ValueError('invalid roster peer')
        if set(rec)-{'room','public_key'}:
            raise ValueError('unknown roster peer key')
        room=rec.get('room')
        if not isinstance(room,str) or not 16<=len(room)<=64 or not re.fullmatch(r'[A-Za-z0-9_-]+',room):
            raise ValueError('room must be 16-64 ASCII characters')
        key=rec.get('public_key')
        if key is not None and not isinstance(key,str):
            raise ValueError('public_key must be path or omitted in unsafe test mode')
        result[peer]=(room, Path(key) if key else None)
    return result


async def run_step_offloop(run_step, model, adapter, optimizer, batch_iter, args, *, supplied_batch=None):
    """Run both dataset fetch and optimization off the WebRTC event loop.

    Remote Parquet streaming legitimately issues repeated HTTP 302/206 requests.
    Report *which* operation is taking time, rather than treating individual
    HTTP requests as an infinite reconnect loop. Never abandon a still-running
    ``next(batch_iter)`` and start a second one: Python cannot cancel that
    blocking worker thread, and concurrent iteration would corrupt the stream.
    """
    stage={'name':'data-fetch', 'since':time.monotonic()}
    started=stage['since']
    def worker():
        try:
            batch=next(batch_iter) if supplied_batch is None else supplied_batch
        except StopIteration as exc:
            raise RuntimeError('training dataset exhausted; configure repetition or finite rounds') from exc
        fetched=time.monotonic()
        if fetched-started >= 5:
            print(f'DATA BATCH READY fetch_s={fetched-started:.1f}',flush=True)
        stage.update(name='model-step',since=fetched)
        return run_step(model,adapter,optimizer,batch,args)

    # One in-flight iterator operation, even while the application reports
    # slow network requests. wait() does not cancel the underlying task.
    task=asyncio.create_task(asyncio.to_thread(worker),name='fedpara-data-and-model-step')
    report_seconds=float(getattr(args,'data_wait_report_seconds',15.0))
    if report_seconds <= 0:
        return await task
    while True:
        done,_=await asyncio.wait({task},timeout=report_seconds)
        if done:
            return task.result()
        elapsed=time.monotonic()-stage['since']
        print(f'DATA WAIT stage={stage["name"]} seconds={elapsed:.1f} '
              f'network_stream={stage["name"] == "data-fetch"} '
              f'peer={getattr(args,"peer_id",None) or "standalone"}',flush=True)


async def train_elastic(args, *, load_model:Callable, training_batches:Callable, run_step:Callable):
    """Train regardless of absent peers; signed event gossip eventually reconciles.

    This is asynchronous additive model-delta SUM, not synchronized FedSum.
    No consensus under network partition: peers can temporarily diverge.
    """
    if args.trust_remote_code or args.factory_file or args.allow_legacy_bin:
        raise PermissionError('Network-enabled training forbids untrusted Python/pickle. Train in offline sandbox.')
    if args.resume: raise ValueError('resume local .fpatch not compatible with elastic updates')
    if not args.state_base and not getattr(args,'streaming_base',False) and not getattr(args,'sharded_base_manifest',None):
        raise ValueError('elastic needs --state-base, --sharded-base-manifest, or --streaming-base as immutable common genesis')
    if args.grad_accum<1 or args.local_steps<1 or args.rounds<0 or getattr(args,'durable_every',0)<0 or not 1<=getattr(args,'working_save_every',10)<=10000 or not 1<=getattr(args,'inbound_queue',32)<=256:
        raise ValueError('bad training step settings')
    if (args.max_journal_gb < 0 or args.peer_timeout <= args.heartbeat_interval or
        args.sync_timeout < 0 or args.inventory_interval <= 0 or
        args.inventory_cycle <= 0 or args.checkpoint_every < 0):
        raise ValueError('invalid elastic liveness, journal or checkpoint settings')
    if not 2 <= getattr(args,'expected_peers',2) <= 4096:
        raise ValueError('--expected-peers must be 2..4096')
    if getattr(args,'max_sync_steps',64) < 1:
        raise ValueError('--max-sync-steps must be positive')
    if not 1 <= getattr(args,'push_fanout',PUSH_FANOUT_DEFAULT) <= 8:
        raise ValueError('--push-fanout must be 1..8')
    if args.peers is not None:
        raise ValueError('--peers is only supported with --mode synchronous; elastic FedSum uses each signed delta once, with no peer divisor')
    room_only=bool(getattr(args,'room_id',None))
    if room_only:
        from .roomid import validate_room, auto_identity, RoomDiscovery, pair_room
        validate_room(args.room_id)
        peer_id, room_signer, room_pub, room_verify = auto_identity(
            Path(str(args.journal)+'.identity.pem'))
        if args.peer_id is not None and args.peer_id!=peer_id:
            raise ValueError('room mode uses a generated cryptographic peer ID; omit --peer-id')
        args.peer_id=peer_id
        if args.unsafe_unsigned or args.signing_key or args.self_public_key or args.link or args.roster_file or args.trusted_peer_key:
            raise ValueError('room mode is automatic; remove legacy roster/key options')
    if not room_only and not args.unsafe_unsigned and not args.signing_key:
        raise PermissionError('legacy elastic mode requires signed events and pinned public keys')
    for signal_url in [args.ws,*getattr(args,'ws_backup',[])]:
        if signal_url.startswith('ws://') and not args.unsafe_unsigned:
            host=signal_url.split('/')[2].split(':')[0]
            if host not in ('127.0.0.1','localhost','[::1]'):
                raise PermissionError('internet signaling must use WSS')
    fixed_links=parse_links(args.link,args.peer_id) if args.link else {}
    fixed_keys=parse_peer_keys(args.trusted_peer_key)
    roster={p:(room,fixed_keys.get(p)) for p,room in fixed_links.items()}
    if args.roster_file:
        roster.update(read_roster(args.roster_file,args.peer_id))
    if not args.unsafe_unsigned:
        if any(key is None or not key.is_file() for _,key in roster.values()):
            raise PermissionError('each remote peer needs pinned Ed25519 public key')
    if args.unsafe_unsigned: log.warning('UNSIGNED elastic mode is INSECURE; local testing only')
    # Device-specific decisions are finalized BEFORE allocating the model.
    # Architecture, LoHA rank and initial model layout remain shared invariants.
    controller=None
    if getattr(args,'auto_tune',False):
        from .adaptive import detect_device, initial_policy, AdaptiveController
        profile=detect_device()
        policy=initial_policy(profile,requested_local_steps=args.local_steps,
                              max_batch=max(1,args.batch_size))
        controller=AdaptiveController(profile,policy)
        if args.device == 'cpu' and policy.device != 'cpu':args.device=policy.device
        args.batch_size=policy.batch_size
        args.grad_accum=max(args.grad_accum,policy.grad_accum)
        args.gradient_checkpointing |= policy.gradient_checkpointing
        args.wire_dtype=policy.wire_dtype
        args.local_steps=min(getattr(args,'max_sync_steps',64),policy.local_steps)
        print('adaptive local policy:',json.dumps(policy.json()),flush=True)

    room_manifest=None
    if room_only:
        from .room_manifest import load_manifest, make_manifest
        has_local_data=(any(getattr(args,field,None) for field in ('batches_dir','text_file','jsonl'))
                        or not hasattr(args,'batches_dir'))
        if not getattr(args,'room_manifest',None) and not has_local_data:
            from .roomid import fetch_room_manifest
            room_manifest=await fetch_room_manifest(args.ws,args.room_id,args.token)
            # Keep a pinned local copy: restart must not silently accept a changed room.
            local_pin=Path(str(args.journal)+'.room-manifest.json')
            if local_pin.exists():
                from .room_manifest import manifest_digest
                if manifest_digest(load_manifest(local_pin)) != manifest_digest(room_manifest):
                    raise PatchError('Room descriptor changed since last session; inspect before replacing pinned copy')
            else:
                local_pin.parent.mkdir(parents=True,exist_ok=True)
                local_pin.write_text(json.dumps(room_manifest,ensure_ascii=False,indent=2),encoding='utf8')
            args.room_manifest=local_pin
        if room_manifest is None:
            room_manifest=(load_manifest(args.room_manifest) if getattr(args,'room_manifest',None)
                           else make_manifest(getattr(args,'model',getattr(args,'model_id',None) or 'unknown'),
                                              getattr(args,'revision','main'),'private-local',
                                              architecture=getattr(args,'hf_task','causal-lm'),
                                              dataset_description='Private federated samples are kept on peers'))
        if ((getattr(args,'model',None) is not None and room_manifest['model']['id'] != args.model)
                or (getattr(args,'revision',None) is not None and room_manifest['model']['revision']!=args.revision)):
            raise PatchError('room manifest model ID/revision differs from local selected model')
        if room_manifest['code']['sha256'] and not getattr(args,'room_manifest',None):
            raise PatchError('custom code must be pinned by local manifest')
    signer=room_signer if room_only else (signer_from_pem(Path(args.signing_key)) if args.signing_key else None)
    # Trust keys retained for peers which permanently left: their signed history
    # must still authenticate when gossiped by another connected peer.
    trusted={}
    if room_only:
        trusted[args.peer_id]=room_verify  # local Paxos vote must also verify by self-certified identity
    # Persist historical peer pins incrementally. v31 rewrote the ENTIRE JSON
    # registry for EVERY new member: quadratic I/O under thousands of joins.
    # Keep the old JSON only as a one-way migration source.
    import sqlite3
    key_bytes={}
    legacy_registry=Path(str(args.journal)+'.trusted.json')
    trust_db=Path(str(args.journal)+'.trusted.sqlite3')
    trust_db.parent.mkdir(parents=True,exist_ok=True)
    if trust_db.is_symlink():raise PatchError('trusted identity database must not be a symlink')
    reg=sqlite3.connect(trust_db,timeout=20)
    reg.execute('PRAGMA journal_mode=WAL')
    reg.execute('PRAGMA synchronous=FULL')
    reg.execute('CREATE TABLE IF NOT EXISTS trusted(peer TEXT PRIMARY KEY, pem BLOB NOT NULL)')
    if legacy_registry.is_file():
        legacy=json.loads(legacy_registry.read_text(encoding='utf-8'))
        if not isinstance(legacy,dict) or len(legacy)>65536:
            raise PatchError('corrupt legacy trusted registry')
        with reg:
            for peer,encoded in legacy.items():
                if not PEER_ID.fullmatch(peer) or not isinstance(encoded,str):
                    raise PatchError('invalid legacy identity')
                data=base64.b64decode(encoded,validate=True)
                old=reg.execute('SELECT pem FROM trusted WHERE peer=?',(peer,)).fetchone()
                if old and old[0]!=data:
                    raise PatchError('legacy identity conflicts with pinned SQLite key')
                reg.execute('INSERT OR IGNORE INTO trusted(peer,pem) VALUES(?,?)',(peer,data))
        # Avoid parsing the entire historical JSON on every future restart.
        # Mark migration in the database, then rename legacy file for audit.
        archived=legacy_registry.with_suffix('.json.migrated')
        if not archived.exists():legacy_registry.replace(archived)
    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.exceptions import InvalidSignature
    from .identity import _message
    def make_verifier(k):
        def verify(msg):
            try:
                signature=base64.b64decode(msg['sig'],validate=True)
                k.verify(signature,_message(msg['size'],msg['sha256']))
            except (ValueError,TypeError,KeyError,InvalidSignature) as exc:
                raise PatchError('invalid archived origin signature') from exc
        return verify
    for peer,data in reg.execute('SELECT peer,pem FROM trusted'):
        if (not PEER_ID.fullmatch(peer) or not isinstance(data,bytes)
            or len(data)>4096 or len(key_bytes)>=65536):
            raise PatchError('corrupt pinned identity registry')
        pub=load_pem_public_key(data)
        if not isinstance(pub,Ed25519PublicKey):raise PatchError('non-Ed25519 archived key')
        trusted[peer]=make_verifier(pub)
        key_bytes[peer]=data

    def store_pin(peer, data, verifier):
        if peer in key_bytes and key_bytes[peer]!=data:
            raise PatchError(f'public key for {peer} changed; manual key migration required')
        if peer not in key_bytes:
            if len(key_bytes)>=65536:
                raise PatchError('archived peer identity budget exceeded (65536)')
            with reg:
                reg.execute('INSERT INTO trusted(peer,pem) VALUES(?,?)',(peer,data))
            key_bytes[peer]=data
            trusted[peer]=verifier

    for peer,(_,key) in roster.items():
        if key: store_pin(peer,key.read_bytes(),verifier_from_pem(key))
    if signer and not room_only:
        if not args.self_public_key:
            raise PermissionError('required --self-public-key for relay/replay verification')
        path=Path(args.self_public_key)
        store_pin(args.peer_id,path.read_bytes(),verifier_from_pem(path))
    model,adapter=load_model(args)
    genesis_input={'model_id':adapter.model_id,'revision':adapter.revision,
                   'layout':adapter.layout,'kind':adapter.kind,'rank':adapter.rank,
                   'alpha':adapter.alpha}
    if getattr(args,'moe_plan',None):
        plan=json.loads(Path(args.moe_plan).read_text(encoding='utf-8'))
        genesis_input['moe_plan_sha256']=hashlib.sha256(_canonical_json(plan)).hexdigest()
    if room_only:
        from .room_manifest import manifest_digest
        genesis_input['room_manifest_sha256']=manifest_digest(room_manifest)
        # Prevent joining a same-name room with different genesis checkpoint.
        # Archive timestamps/peer names are not the model identity.
        from .state import semantic_state_sha256
        if args.state_base:
            genesis_input['checkpoint_sha256']=semantic_state_sha256(Path(args.state_base))
        elif getattr(args,'sharded_base_manifest',None):
            from .shards import read_manifest, semantic_shard_sha256
            genesis_input['checkpoint_sha256']=semantic_shard_sha256(
                read_manifest(Path(args.sharded_base_manifest),max_state_bytes=args.max_state_gb*1024**3))
        else:
            digest=getattr(model,'_fedpara_streaming_base_info',{}).get('genesis_digest')
            if not digest:
                raise PatchError('streaming room requires a content-hashed immutable safetensors genesis')
            genesis_input['checkpoint_sha256']='streaming-safetensors-v1:'+digest
        genesis_input['room_fingerprint']=hashlib.sha256(args.room_id.encode('ascii')).hexdigest()
    # Live elastic uses ONLY signed before/after effective delta events.
    # Legacy reset-v1 peers/journals must never join this new genesis.
    genesis_input['optimizer_protocol']='continual-elastic-fedsum-relora-mixed-async-v6'
    genesis_input['reinit_every']=getattr(args,'reinit_every',20)
    from .mixed_learning_v65 import spec as mixed_learning_spec
    mix_protocol=mixed_learning_spec(args)
    if mix_protocol is not None:
        genesis_input['mixed_learning_protocol']=mix_protocol

    # Preserve existing AdamW genesis compatibility. LoRA-Muon journals must
    # not collide with AdamW journals because optimizer moments are different.
    if getattr(args,'optimizer','adamw')=='lora-muon':
        genesis_input['optimizer_name']='lora-muon'
        genesis_input['muon_momentum']=getattr(args,'muon_momentum',.9)
        genesis_input['muon_weight_decay']=getattr(args,'muon_weight_decay',0.)
    genesis=hashlib.sha256(_canonical_json(genesis_input)).hexdigest()
    # The root identity is immutable even after a verified hot epoch switch.
    # The active event genesis, by contrast, changes with each certificate.
    # NEVER open the genesis journal after a durable successor pointer exists.
    root_genesis=genesis
    epoch_dir=Path(str(args.journal)+'.live-epoch')
    epoch_boot=None
    if (getattr(args,'generation_v2_trainer',False) or
            getattr(args,'generation_v2_propose_every',0)) and not getattr(args,'generation_v1_committee',None):
        raise PatchError('compact trainer requires a pinned generation committee')
    if getattr(args,'generation_v2_propose_every',0)<0:
        raise PatchError('invalid automatic epoch export cadence')
    if getattr(args,'generation_v2_propose_every',0) and not getattr(args,'generation_v2_trainer',False):
        raise PatchError('automatic compact export requires --generation-v2-trainer')
    if ((epoch_dir/'active-epoch.json').exists() or
            (epoch_dir/'active-epoch.json').is_symlink()):
        if not getattr(args,'generation_v1_committee',None) or not room_only:
            raise PatchError('certified active epoch requires the pinned generation committee')
        from .generation_committee_v43 import read_generation_committee
        from .online_generation import GenerationLedger
        from .epoch_trainer_boot_v52 import resolve_certified_boot
        boot_committee,boot_keys=read_generation_committee(Path(args.generation_v1_committee))
        for boot_peer,boot_pem in boot_keys.items():
            store_pin(boot_peer,boot_pem,make_verifier(load_pem_public_key(boot_pem)))
        boot_ledger=GenerationLedger(Path(str(args.journal)+'.generation.sqlite3'),
            room=args.room_id,peer=args.peer_id,committee=boot_committee,
            signer=signer,trusted=trusted,genesis=root_genesis)
        try:
            if getattr(args,'generation_v2_trainer',False):
                from .epoch_certificate_v46 import CompactEpochLedger
                from .epoch_autonomous_v56 import AutonomousPagedEpoch
                from .epoch_compact_trainer_v57 import CertifiedCompactTrainerLedger
                compact_boot=CompactEpochLedger(
                    Path(str(args.journal)+'.generation-compact.sqlite3'),
                    room=args.room_id,peer=args.peer_id,committee=boot_committee,
                    signer=signer,trusted=trusted,genesis=root_genesis)
                store_boot=AutonomousPagedEpoch(compact_boot,
                    Path(str(args.journal)+'.generation-compact-stage.sqlite3'),
                    local_checkpoint=lambda _p:None,
                    local_export_verified=lambda _p:False)
                try:
                    v2_boot_ledger=CertifiedCompactTrainerLedger(compact_boot,store_boot)
                    epoch_boot=resolve_certified_boot(root=epoch_dir,journal_base=Path(args.journal),
                        ledger=v2_boot_ledger,root_genesis=root_genesis,
                        max_state_bytes=args.max_state_gb*1024**3)
                finally:
                    store_boot.close();compact_boot.close()
            else:
                epoch_boot=resolve_certified_boot(root=epoch_dir,journal_base=Path(args.journal),
                    ledger=boot_ledger,root_genesis=root_genesis,
                    max_state_bytes=args.max_state_gb*1024**3)
        finally:
            boot_ledger.close()
        if epoch_boot is None:
            raise PatchError('committed epoch pointer disappeared during safe recovery')
        from .shards import apply_sharded_effective_state
        await asyncio.to_thread(apply_sharded_effective_state,
            model,adapter,epoch_boot.manifest,epoch_boot.store,
            model_id=adapter.model_id,revision=adapter.revision,
            max_state_bytes=args.max_state_gb*1024**3)
        genesis=epoch_boot.current_genesis
        print(f'CERTIFIED EPOCH RECOVERY epoch={epoch_boot.epoch} '
              f'manifest={epoch_boot.manifest_hash[:16]} ',flush=True)
    journal_path=epoch_boot.journal_path if epoch_boot else Path(args.journal)
    from .snapshot_live import journal_tip, snapshot_inventory
    snapshot_root=Path(str(args.journal)+'.snapshots')
    snapshot_store=snapshot_root/'objects'
    snapshot_recv=snapshot_root/'incoming'
    if epoch_boot is not None:
        # The new-genesis writer must have been fsynced BEFORE the pointer.
        # Inventing an empty journal here could lose locally signed steps.
        if journal_path.is_symlink() or not journal_path.is_file():
            raise PatchError('certified epoch journal is missing after pointer publication')
        if epoch_boot.source_path.is_symlink() or not epoch_boot.source_path.is_file():
            raise PatchError('old certified source journal is missing; no local handoff proof')
    journal=EventJournal(journal_path,genesis,max_bytes=args.max_journal_gb*1024**3,
                         sync=getattr(args,'journal_sync','full').upper())
    print(f'journal durability={journal.sync}; FULL=power-loss-safe, NORMAL=process-crash-safe only', flush=True)
    from .continual import ContinualElasticEngine
    # factor roundoff would leave nonzero residual between anchor and local.
    args.wire_dtype='float32'
    engine=ContinualElasticEngine(adapter,args.peer_id,genesis,journal,
                                  reinit_every=getattr(args,'reinit_every',20),signer=signer,trusted=trusted,
                                  unsafe=args.unsafe_unsigned,max_delta_rms=args.max_delta_rms,
                                  event_public=room_pub if room_only else None)
    print(f'elastic optimization=FedSum+ReLoRA-mixed-async-v6 reinit_every={engine.reinit_every} local events',flush=True)
    carry_archive=None
    if epoch_boot is not None:
        # Recovery order is strict: immutable certified base, durable old signed
        # suffix, THEN events from the new-generation journal. All old bodies
        # remain signed with their ORIGINAL genesis, never forged/relabelled.
        from .epoch_carry_v46 import CrossEpochCarryArchive
        from .epoch_trainer_boot_v52 import verify_boot_carry
        old_source=EventJournal(epoch_boot.source_path,epoch_boot.previous_genesis,
            max_bytes=0,sync=getattr(args,'journal_sync','full').upper())
        try:
            carry_archive=CrossEpochCarryArchive(epoch_boot.carry_path,
                source_genesis=epoch_boot.previous_genesis,
                target_genesis=genesis,
                certified_manifest=epoch_boot.manifest_hash,
                epoch=epoch_boot.epoch,cutoffs=epoch_boot.proposal['cutoffs'])
            verify_boot_carry(epoch_boot,carry_archive,old_journal=old_source)
            print(f'CERTIFIED CARRY RECOVERY count={carry_archive.replay(engine)}',flush=True)
        finally:
            old_source.close()
    engine.replay()
    if epoch_boot is not None:
        with journal.db:
            journal.db.execute("INSERT OR REPLACE INTO meta(name,value) VALUES('initial_ready','1')")
    print(f'elastic peer={args.peer_id} genesis={genesis} recovered={journal.count()} events; log={journal.path}',flush=True)
    # Epoch-v2 signed crash-fault agreement PROBE. This is deliberately
    # isolated from v1 optimization: no v1 delta may be reinterpreted as an
    # epoch-anchored v2 proposal. A genuine v2 trainer requires pinned local
    # optimization against each newly committed base checkpoint.
    epoch_paxos=None
    epoch_probe_value=None
    epoch_probe_started=False
    epoch_probe_reported=False
    epoch_peer_cursors={}
    epoch_last_tx=0.0
    if getattr(args,'epoch_v2_probe_committee',None):
        from .epoch_consensus import EpochPaxos, make_value
        if not room_only or not signer or args.unsafe_unsigned:
            raise PermissionError('epoch-v2 probe requires authenticated room mode')
        committee=json.loads(Path(args.epoch_v2_probe_committee).read_text())
        if not isinstance(committee,list) or len(committee)!=len(set(committee)) or not 2<=len(committee)<=16 or any(not isinstance(p,str) or not PEER_ID.fullmatch(p) for p in committee) or args.peer_id not in committee:
            raise PatchError('epoch-v2 probe requires a fixed pinned 2..16 peer committee')
        from .state import semantic_state_sha256
        base=(semantic_state_sha256(Path(args.state_base)) if args.state_base else
              getattr(model,'_fedpara_streaming_base_info',{}).get('genesis_digest'))
        epoch_id=hashlib.sha256(_canonical_json(['fedpara-epoch-probe-v1',genesis,base,sorted(committee)])).hexdigest()
        epoch_probe_value=make_value(epoch_id,base,tuple(sorted(committee)),[])
        epoch_paxos=EpochPaxos(Path(str(args.journal)+'.epoch-v2.sqlite3'),
                               peer=args.peer_id,committee=committee,epoch=epoch_id,base=base,
                               signer=signer,trusted=trusted)
        print(f'EPOCH V2 PROBE ARMED peer={args.peer_id} committee={len(committee)} '
              f'quorum={epoch_paxos.majority} epoch={epoch_id[:16]} '
              f'(no model mutation)',flush=True)
    # V42 authenticated certificate/anti-entropy traffic is opt-in until the
    # asynchronous model/optimizer can safely rebase post-cut updates.
    generation_committee=None
    generation_bridge=None
    generation_ledger=None
    generation_compact_ledger=None
    generation_compact_control=None
    generation_trainer_ledger=None
    generation_gc_runtime=None
    generation_gc_next=0.0
    generation_gc_task=None
    generation_compact_vote_task=None
    generation_compact_vote_next=0.0
    if getattr(args,'generation_v1_committee',None):
        if not room_only or args.unsafe_unsigned or signer is None:
            raise PermissionError('generation control requires authenticated room mode')
        from .generation_committee_v43 import read_generation_committee
        generation_committee,committee_keys=read_generation_committee(Path(args.generation_v1_committee))
        # Stable PUBLIC keys can be pre-pinned. An offline validator need not
        # join WebRTC just to make its past signatures verifiable. Identity
        # derives from Ed25519, never from a user-supplied display label.
        from .identity import verifier_from_pem as _verifier_from_file
        for committee_peer,pem in committee_keys.items():
            pub=load_pem_public_key(pem)
            store_pin(committee_peer,pem,make_verifier(pub))
    def maybe_start_generation_bridge():
        nonlocal generation_bridge,generation_ledger,generation_gc_runtime
        nonlocal generation_compact_ledger,generation_compact_control,generation_trainer_ledger
        if generation_bridge is not None or generation_committee is None:
            return
        if any(k not in trusted for k in generation_committee):
            return  # learn and pin public keys through authenticated Hello first
        from .online_generation import GenerationLedger,AutomaticGenerationControl
        from .live_generation_transport import GenerationWebRTCBridge
        from .snapshot_live import snapshot_inventory
        from .shards import read_manifest, verify_shards, semantic_shard_sha256
        def locally_verified(p):
            # A matching safetensors manifest and separately matching log cut
            # do NOT prove the weights were exported at *that* cut. Until a
            # signed, atomic live boundary receipt is produced by the trainer,
            # casting a vote could certify a silently stale checkpoint.
            # Therefore this transport-only integration MUST NOT cast votes.
            return False
        def locally_covered(p):
            return False
        generation_ledger=GenerationLedger(Path(str(args.journal)+'.generation.sqlite3'),
            room=args.room_id,peer=args.peer_id,committee=generation_committee,
            signer=signer,trusted=trusted,genesis=root_genesis)
        def has_snapshot(manifest_sha):
            # A verified content-addressed download survives reconnects.
            path=snapshot_recv/(manifest_sha+'.fmanifest')
            if not path.is_file():return False
            with path.open('rb') as fp:
                return hashlib.file_digest(fp,'sha256').hexdigest()==manifest_sha
        from .epoch_certificate_v46 import CompactEpochLedger
        from .epoch_autonomous_v56 import AutonomousPagedEpoch
        generation_compact_ledger=CompactEpochLedger(
            Path(str(args.journal)+'.generation-compact.sqlite3'),
            room=args.room_id,peer=args.peer_id,committee=generation_committee,
            signer=signer,trusted=trusted,genesis=root_genesis)
        def compact_locally_verified(p):
            if not getattr(args,'generation_v2_trainer',False):return False
            # A committee member may only attest an epoch following the
            # model/JOURNAL generation it is actually running. In particular,
            # an empty signed frontier must not permit a stale trainer to
            # vote for the next generation's unrelated checkpoint.
            if (p['epoch']!=trainer_router.current.epoch+1 or
                    journal.genesis!=trainer_router.current.genesis):
                return False
            return generation_compact_control.has_export_receipt(p)
        def compact_local_files(p):
            path=snapshot_recv/(p['manifest']+'.fmanifest')
            if path.is_symlink() or not path.is_file():return None
            return path,snapshot_store,journal.db
        generation_compact_control=AutonomousPagedEpoch(generation_compact_ledger,
            Path(str(args.journal)+'.generation-compact-stage.sqlite3'),
            local_checkpoint=compact_local_files,
            local_export_verified=compact_locally_verified,
            defer_votes=bool(getattr(args,'generation_v2_trainer',False)))
        if getattr(args,'generation_v2_trainer',False):
            from .epoch_compact_trainer_v57 import CertifiedCompactTrainerLedger
            generation_trainer_ledger=CertifiedCompactTrainerLedger(
                generation_compact_ledger,generation_compact_control)
        generation_bridge=GenerationWebRTCBridge(
            AutomaticGenerationControl(generation_ledger,locally_verified,locally_covered),
            snapshot_available=has_snapshot,
            compact_ledger=generation_compact_ledger,
            compact_control=generation_compact_control)
        # Existing verified local handoffs can finish their *own* GC while
        # the new trainer remains active. The v43 runtime checks signed
        # certificates, verified new journal, old-owner flock and uncovered
        # suffix again before any deletion. This does not authorize GC for
        # as-yet unimplemented live v1 model transitions.
        from .epoch_runtime_v43 import EpochRuntimeV43
        generation_gc_runtime=EpochRuntimeV43(
            Path(str(args.journal)+'.generation-state'),generation_ledger)
        print(f'GENERATION CONTROL WebRTC enabled committee={len(generation_committee)} '
              f'(certificate transport only: unsafely active-log GC DISABLED)',flush=True)
    def generation_gc_background():
        # Open ALL SQLite connections in this dedicated worker thread.
        # Never run a potentially multi-million-row old-log verification on
        # the event loop that must service heartbeats and live training.
        from .online_generation import GenerationLedger
        from .epoch_runtime_v43 import EpochRuntimeV43
        local_ledger=GenerationLedger(Path(str(args.journal)+'.generation.sqlite3'),
            room=args.room_id,peer=args.peer_id,committee=generation_committee,
            signer=signer,trusted=trusted,genesis=root_genesis)
        try:
            runtime=EpochRuntimeV43(Path(str(args.journal)+'.generation-state'),local_ledger)
            try:return runtime.poll()
            finally:runtime.close()
        finally:local_ledger.close()
    slots:dict[str,PeerSlot]={}
    room_stop=asyncio.Event() if room_only else None
    lobby=RoomDiscovery(args.ws,args.room_id,args.peer_id,args.token,
                         manifest=room_manifest,backup_urls=getattr(args,'ws_backup',[]),
                         publish=getattr(args,'room_public',False),
                         max_degree=getattr(args,'max_neighbors',8)) if room_only else None
    lobby_task=asyncio.create_task(lobby.run(room_stop)) if room_only else None
    def new_optimizer(target_adapter=None, target_genesis=None):
        selected=target_adapter if target_adapter is not None else adapter
        if getattr(args,'disk_optimizer_dir',None):
            if getattr(args,'optimizer','adamw')!='adamw':
                raise PatchError('live disk optimizer supports AdamW only')
            from .live_disk_optimizer_v59 import create_live_disk_optimizer
            # Distinct scratch allocations for every ReLoRA reset and epoch.
            # No crash can accidentally reuse partially modified mmap moments.
            scope=(target_genesis or genesis)[:32]
            return create_live_disk_optimizer(selected,
                Path(args.disk_optimizer_dir)/scope,lr=args.lr,
                weight_decay=args.weight_decay,
                max_gb=getattr(args,'disk_optimizer_max_gb',32))
        from .lora_muon import make_adapter_optimizer
        return make_adapter_optimizer(selected,optimizer=getattr(args,'optimizer','adamw'),
            lr=args.lr,weight_decay=(getattr(args,'muon_weight_decay',0.)
              if getattr(args,'optimizer','adamw')=='lora-muon' else args.weight_decay),
            muon_momentum=getattr(args,'muon_momentum',.9))
    optimizer=new_optimizer()
    optimizer.zero_grad(set_to_none=True)
    from .localprogress import restore_working, save_working, snapshot_path, AsyncWorkingSaver
    from .paired_checkpoint_v75 import AsyncPairedSaver, snapshot_path as paired_path
    from .paired_recovery_v76 import recover as recover_paired_v76
    from .hf_mixed_datasets_v68 import source_selected as _v75_has_source
    # The paired v75 format is local-only. Non-AdamW and disk-backed actor
    # optimizers need a separately designed atomic moment protocol.
    _v75_paired_active=bool(_v75_has_source(args,'ppo') and
        getattr(args,'optimizer','adamw')=='adamw' and
        not getattr(args,'disk_optimizer_dir',None))
    if getattr(args,'critic_rollout_gossip',False):
        # The v75 paired file represents a LOCAL value state; v80 maintains
        # a separate atomic Critic+receipt snapshot under its own signed lane.
        _v75_paired_active=False
    if _v75_paired_active and not getattr(args,'critic_state_dir',None):
        # Durable local value state should not depend on the GUI having
        # provided this optional path; CLI elastic rooms get a private one.
        args.critic_state_dir=Path(str(journal.path)+'.critic')
    args._v75_paired_active=_v75_paired_active
    # v77: every Critic participant starts from an identical room-pinned
    # value-head baseline; the separate signed Critic journal is authoritative.
    # The original v75 paired file may restore local optimizer state, but the
    # Critic head is reconstructed from its OWN journal after that restore.
    critic_gossip=None
    trajectory_gossip=None
    critic_baseline=None
    distributed_critic=bool(getattr(args,'distributed_critic',False))
    critic_journal_path=Path(str(args.journal)+'.critic-v78.sqlite3')
    if distributed_critic and Path(str(args.journal)+'.critic-v77.sqlite3').exists():
        raise PatchError('v77 Critic journal requires explicit migration; v78 has a distinct signed root lineage, start a new room')
    if distributed_critic:
        from .mixed_learning_v65 import build as _v77_build_mixer
        from .distributed_critic_v78 import critic_params as _v78_params
        import torch as _v77_torch
        args._v75_skip_legacy_critic=True
        try:
            with _v77_torch.random.fork_rng(devices=[]):
                _v77_torch.manual_seed(int(root_genesis[:16],16) % (2**63-1))
                initial_mixer=_v77_build_mixer(args)
        finally:
            args._v75_skip_legacy_critic=False
        if initial_mixer is None or initial_mixer.critic is None:
            raise PatchError('distributed Critic requires an active PPO dataset')
        critic_baseline=[p.detach().cpu().clone() for p in _v78_params(initial_mixer)]
        # Disallow silently upgrading a local-only v76 pair: unpublished
        # local value steps have no signed journal to reconstruct from.
        if paired_path(journal).exists() and not critic_journal_path.exists():
            initial_mixer.close()
            raise PatchError('cannot adopt an old local-only Critic checkpoint in place; start a new v78 room')
        args._v65_mixer=initial_mixer
        args._v65_mixer_adapter=adapter
    paired_saver=AsyncPairedSaver()
    paired_exists=paired_path(journal).exists()
    if paired_exists and not _v75_paired_active:
        raise PatchError('paired v75 checkpoint exists but PPO critic or AdamW paired recovery is disabled')
    if paired_exists:
        # All actor + critic tensors are validated together before any apply.
        # Never fall back to a separately saved Critic on a mismatched WAL tip.
        from .mixed_learning_v65 import build as _v75_build_mixer
        args._v75_skip_legacy_critic=True
        try:
            pair_mixer=(initial_mixer if distributed_critic else _v75_build_mixer(args))
            if pair_mixer is None or pair_mixer.critic is None:
                raise PatchError('paired v75 checkpoint requires a local PPO Critic')
            pair_mixer._v75_paired=True
            try:
                restored_steps,paired_recovery= recover_paired_v76(
                    journal,adapter,optimizer,pair_mixer,verified_replay=True)
                if paired_recovery is not None and paired_recovery.mode=='reset':
                    print('PAIRED AUTO-RECOVERY: signed history advanced '
                          f'{paired_recovery.previous_event_count} -> '
                          f'{paired_recovery.current_event_count}; '
                          'resetting uncommitted Actor optimizer and local Critic; '
                          f'old_checkpoint_local_steps={paired_recovery.previous_checkpoint_local_steps}',
                          flush=True)
            except BaseException:
                # Even on a rejected mismatched WAL tip, release HF readers
                # without publishing an unrelated v66/v69 legacy snapshot.
                pair_mixer.close()
                raise
            args._v65_mixer=pair_mixer
            args._v65_mixer_adapter=adapter
        finally:
            args._v75_skip_legacy_critic=False
    else:
        restored_steps=restore_working(journal,adapter,optimizer)
    if distributed_critic:
        from .distributed_critic_v78 import CriticGossip
        # Actor replay has already verified the signed genesis and events.
        # The Critic baseline is the same pinned HF value head or deterministic
        # seeded MLP head on every participant.
        # Actor lineage is independently certificate-verified. The Critic lane
        # always follows the immutable ROOT genesis, across certified epochs.
        critic_known_epochs={root_genesis}
        if epoch_boot is not None:
            from .critic_epochs_v78 import verified_epochs
            if generation_committee is None:
                raise PatchError('Critic cannot resume epochs without a pinned committee')
            if getattr(args,'generation_v2_trainer',False):
                from .epoch_certificate_v46 import CompactEpochLedger
                certified_ledger=CompactEpochLedger(
                    Path(str(args.journal)+'.generation-compact.sqlite3'),
                    room=args.room_id,peer=args.peer_id,committee=generation_committee,
                    signer=signer,trusted=trusted,genesis=root_genesis)
            else:
                from .online_generation import GenerationLedger
                certified_ledger=GenerationLedger(
                    Path(str(args.journal)+'.generation.sqlite3'),
                    room=args.room_id,peer=args.peer_id,committee=generation_committee,
                    signer=signer,trusted=trusted,genesis=root_genesis)
            try:
                critic_known_epochs=verified_epochs(certified_ledger,root_genesis,
                    epoch_boot.epoch,epoch_boot.current_genesis)
            finally:
                certified_ledger.close()
        critic_gossip=CriticGossip(critic_journal_path,root_genesis,args._v65_mixer,
            args.peer_id,signer=signer,public_key=room_pub if room_only else None,
            trusted=trusted,unsafe=False,max_bytes=args.max_journal_gb*1024**3,
            baseline=critic_baseline,lock=adapter.lock,
            allowed_actor_epochs=critic_known_epochs,actor_epoch=genesis)
        args._v65_mixer._v75_paired=_v75_paired_active
        if getattr(args,'critic_rollout_gossip',False):
            from .trajectory_gossip_v82 import LohaVerifiedTrajectoryGossip
            trajectory_gossip=LohaVerifiedTrajectoryGossip(
                Path(str(args.journal)+'.trajectory-v82.sqlite3'),critic_gossip.genesis,args._v65_mixer,
                args.peer_id,signer,pub=room_pub if room_only else None,
                trusted=trusted,allowed_epochs=critic_gossip.known_actor_epochs,
                actor_epoch=genesis,max_bytes=min(args.max_journal_gb*1024**3,512*1024**2))
            print('V82 LOHA CHECKPOINT-VERIFIED TRAJECTORY REPLAY active; only fully reproduced behavior probabilities update Critic',flush=True)
        print('DISTRIBUTED CRITIC active signed_events='
              f'{critic_gossip.journal.count()} lane={critic_gossip.genesis[:16]}',flush=True)
    critic_step_lock=asyncio.Lock()
    working_saver=AsyncWorkingSaver()
    if restored_steps:
        print(f'elastic restored unfinished local block: {restored_steps} steps and AdamW state',flush=True)
    batch_iter=training_batches(args)
    loop=asyncio.get_running_loop()
    local_steps=restored_steps
    emitted=journal.next_seq(args.peer_id)-1  # do not display a false zero after restart
    recent_loss=float('nan')
    from .live_telemetry import LocalTelemetry
    telemetry=LocalTelemetry(getattr(args,'telemetry_file',None),append=True)
    total_steps=0
    last_local_seal_ms=0.0
    last_local_lock_ms=0.0
    last_local_stages_ms=(0.0,0.0,0.0,0.0)
    last_working_save_ms=0.0
    def record_telemetry(state='training', force=False):
        telemetry.record(steps=total_steps,loss=recent_loss,emitted=emitted,
            inbound_applied=inbound_pipeline.applied,
            inbox=inbound_pipeline.queue.qsize(),
            channels=sum(bool(s.node and s.node.connected.is_set()) for s in slots.values()),
            state=state,force=force,
            verify_ms=inbound_pipeline.last_check_ms,
            apply_ms=inbound_pipeline.last_commit_ms,
            inbound_duplicates=inbound_pipeline.duplicates,
            push_mib=sum(s.node.push_sent_wire_bytes for s in slots.values() if s.node)/(1024*1024),
            repair_mib=sum(s.node.repair_sent_wire_bytes for s in slots.values() if s.node)/(1024*1024),
            lock_wait_ms=inbound_pipeline.last_lock_wait_ms,
            journal_ms=inbound_pipeline.last_journal_ms,
            weight_apply_ms=inbound_pipeline.last_weight_apply_ms,
            local_seal_ms=last_local_seal_ms,
            local_lock_ms=last_local_lock_ms,
            local_export_ms=last_local_stages_ms[0],
            local_verify_ms=last_local_stages_ms[1],
            local_journal_ms=last_local_stages_ms[2],
            local_apply_ms=last_local_stages_ms[3],
            working_save_ms=working_saver.disk_ms,
            publish_steps=(adaptive_publish_steps(args.local_steps,living_count(),
                           max_steps=getattr(args,'max_sync_steps',64))
                           if getattr(args,'adaptive_sync',True) else args.local_steps),
            aggregation_events=inbound_pipeline.aggregate_events,
            aggregation_batches=inbound_pipeline.aggregate_batches,
            aggregation_saved_mib=sum(s.node.aggregate_bytes_avoided for s in slots.values()
                                      if s.node)/(1024*1024),
            accuracy=getattr(args,'_latest_accuracy',None), peer_id=args.peer_id,
            membership_count=living_count(),
            connected_peers=[slot.peer for slot in slots.values()
                             if slot.node and slot.node.connected.is_set()][:8])
    shutdown=asyncio.Event()
    previous_signals={}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, shutdown.set)
        except (NotImplementedError, RuntimeError):
            previous_signals[sig]=signal.getsignal(sig)
            signal.signal(sig,lambda _signum,_frame: shutdown.set())
    last_control=0.0; last_config=0.0; first_boot=loop.time()
    # A cold join may NEVER optimize from genesis before the current network
    # state has been fetched. A recovering peer with durable history can keep
    # working in a genuine signaling outage.
    from .join_guard import initial_join_decision
    initial_complete=False
    # Restore evidence of a previously successful room handshake. This is
    # durable metadata, not a claim that the room is currently reachable.
    signaling_seen = journal.db.execute(
        "SELECT 1 FROM meta WHERE name='room_confirmed' AND value='1'").fetchone() is not None
    last_join_report=0.0
    next_checkpoint=(journal.count()//args.checkpoint_every+1)*args.checkpoint_every if args.checkpoint_every else None
    # Full-state catch-up is mandatory for fresh room members. Durable peers
    # continue from their local replayable journal without re-downloading.
    persisted_ready = journal.db.execute(
        "SELECT 1 FROM meta WHERE name='initial_ready' AND value='1'").fetchone() is not None
    # Initial genesis only: thereafter every verified member has equal rights.
    # Once founder training starts, another participant joining must not
    # reactivate the cold-join guard (this was the old-window deadlock).
    from .join_guard import BootstrapState
    bootstrap=BootstrapState.restored(persisted_ready if room_only else True)
    snapshot_tips={}
    received_snapshots={}
    received_hashes={}
    history_reported_at={}
    snapshot_history_mismatches={}
    last_snapshot_report=0.0
    snapshot_pool=SnapshotProviderPool()

    def update_roster():
        desired=dict(roster)
        if room_only:
            # Signaling sends exactly the bounded direct-neighbor view.
            # Preserve it during outages; do not infer a full room roster or
            # union stale+new neighborhoods (which would exceed the cap).
            close_peers=set(lobby.members)
            desired={peer:(pair_room(args.room_id,args.peer_id,peer),None) for peer in close_peers}

        if args.roster_file:
            desired.update(read_roster(args.roster_file,args.peer_id))
        for peer, (room,key) in desired.items():
            raw=key.read_bytes() if key else None
            if peer in key_bytes and raw is not None and key_bytes[peer]!=raw:
                raise PatchError('peer public key rotation requires explicit, audited migration')
            if key: store_pin(peer,raw,verifier_from_pem(key))
            if not room_only and not args.unsafe_unsigned and not key:
                raise PermissionError(f'peer {peer} lacks pinned key')
            old=slots.get(peer)
            if old and (old.room!=room or old.key_path!=key):
                raise PatchError('roster entry room/key changed while running; restart safely')
            if old is None:
                slots[peer]=PeerSlot(peer,room,key)
                print(f'peer registered: {peer}',flush=True)
        for peer,slot in list(slots.items()):
            slot.disabled=peer not in desired
            # The lobby sends explicit departure notifications. Remove inert
            # PeerSlot objects after their link has stopped, so months of
            # joins/leaves cannot grow the in-memory roster without bound.
            # Authenticated event history remains DURABLE in the SQLite journal.
            if (room_only and lobby.online and slot.disabled and slot.task is None
                    and (slot.retiring is None or slot.retiring.done())):
                del slots[peer]

    def active_count():
        now=time.monotonic()
        # active = authenticated DataChannels, independent of slow model-sync.
        # Report "synced" separately to avoid masking successful WebRTC links.
        return 1+sum(1 for s in slots.values() if not s.disabled and s.node and
                     s.node.connected.is_set() and s.node.verified_at>0 and
                     now-s.node.verified_at < args.peer_timeout)

    def synced_count():
        now=time.monotonic()
        return 1+sum(1 for s in slots.values() if not s.disabled and s.node and
                     s.node.connected.is_set() and s.node.ready and
                     now-s.node.verified_at < args.peer_timeout)

    def living_count():
        if room_only:
            # Group membership is maintained by the signaling service.
            # In the bounded four-ring topology at most eight neighbors
            # connect directly, so neighbor count is NOT a FedAvg divisor.
            if lobby.online:return lobby.online_count
            # Outage: use ONLY recent authenticated DataChannel heartbeats.
            now=time.monotonic()
            return 1+sum(1 for s in slots.values() if not s.disabled and s.node and
                         s.node.connected.is_set() and s.node.ready and
                         now-s.node.verified_at < args.peer_timeout)
        now=time.monotonic()
        return 1+sum(1 for s in slots.values() if not s.disabled and s.node is not None
                     and s.node.connected.is_set() and s.node.ready
                     and now-s.node.verified_at < args.peer_timeout)

    async def fanout(wire: bytes, *, exclude_peer: str | None = None,
                     origin_peer: str | None = None):
        # V18: do NOT push every signed, multi-MiB update to every neighbor.
        # The ring-forward route and a deterministic shortcut carry live
        # updates; signed frontier/NEED inventory performs lossless repair.
        # Exact replicas still eventually require each original event, so this
        # avoids redundant uploads rather than claiming sublinear information.
        eligible={slot.peer:slot for slot in slots.values()
                  if slot.node and slot.node.connected.is_set() and
                  not getattr(slot.node,'bridge_only',False) and not slot.disabled}
        fanout_limit=getattr(args,'push_fanout',PUSH_FANOUT_DEFAULT)
        seed=hashlib.blake2b(wire[:1024],digest_size=16).digest()
        chosen=selective_push_targets(args.peer_id,eligible,
                        exclude_peer=exclude_peer,origin_peer=origin_peer,
                        event_seed=seed,fanout=fanout_limit)
        for peer in chosen:
            node=eligible[peer].node
            try:
                node.outgoing.put_nowait(wire)
            except asyncio.QueueFull:
                # Journal frontier/NEED repair discovers this omitted event;
                # never drop it from the durable history.
                node.backpressure_drops+=1

    inbound_commit_lock=asyncio.Lock()

    async def seal_if_needed():
        nonlocal local_steps,emitted,optimizer,last_local_seal_ms,last_local_lock_ms,last_local_stages_ms,last_working_save_ms
        if not local_steps: return
        # Only unfinished accumulation needs one final optimizer step.
        if pending[0]:
            def finish_optimizer_step():
                with adapter.lock:
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
            await asyncio.to_thread(finish_optimizer_step)
            pending[0]=0
        seal_started=time.monotonic()
        async with inbound_commit_lock:
            last_local_lock_ms=(time.monotonic()-seal_started)*1000
            wire=await engine.seal_local_responsive(1,dtype=args.wire_dtype,
                                                    max_wire_bytes=args.max_patch_mb*1024**2)
        last_local_seal_ms=(time.monotonic()-seal_started)*1000
        last_local_stages_ms=getattr(engine,'last_local_seal_stages_ms',(0.0,0.0,0.0,0.0))
        emitted+=1;local_steps=0
        if engine.should_reinitialize(emitted):
            # _apply_checked committed delta and deterministically refreshed
            # local factors+anchors as part of this durable signed event.
            # AdamW moments refer to the OLD factor coordinates: drop them.
            old_optimizer=optimizer
            optimizer=new_optimizer()
            optimizer.zero_grad(set_to_none=True)
            # Snapshot writer reads only immutable reflinks after capture.
            if hasattr(old_optimizer,'close'):old_optimizer.close()
            print(f'RELORA REFRESH event={emitted} factors=reset optimizer=reset '
                  f'base=preserved',flush=True)
        nonlocal_optimizer[0]=optimizer
        # The signed event is durable independently. Optimizer moments are
        # local-only; persist them less frequently than large signed updates.
        # A crash between snapshots replays ALL signed updates; only the
        # uncheckpointed AdamW moments are lost (explicit durability tradeoff).
        if emitted % getattr(args,'working_save_every',10) == 0:
            from .localprogress import _tip
            async with inbound_commit_lock:
                tip=_tip(journal)
                mixer=getattr(args,'_v65_mixer',None)
                if _v75_paired_active and mixer is not None and mixer.critic is not None:
                    await paired_saver.checkpoint(journal,adapter,optimizer,mixer,0,journal_tip=tip)
                    last_working_save_ms=paired_saver.disk_ms
                else:
                    await working_saver.checkpoint(journal,adapter,optimizer,0,journal_tip=tip)
                    last_working_save_ms=working_saver.disk_ms
        await fanout(wire)
        record_telemetry(force=True)
        print(f'elastic local event={emitted}, living={living_count()}, active={active_count()}, bytes={len(wire)}',flush=True)

    pending=[0]

    async def loop_stall_watchdog(stop, *, period=1.0, warn_after=4.0):
        """Expose real asyncio starvation instead of blaming peer churn."""
        before=time.monotonic()
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(),timeout=period)
                return
            except asyncio.TimeoutError:
                pass
            current=time.monotonic()
            lag=current-before-period
            if lag>warn_after:
                log.warning('TRAINER EVENT LOOP STALL %.2fs; WebRTC/signaling timers delayed',lag)
            before=current

    async def heartbeat_daemon():
        await pump_heartbeats(slots, shutdown,
                              interval=args.heartbeat_interval,
                              peer_timeout=args.peer_timeout)

    async def stop_slot(slot:PeerSlot):
        """Retire an RTC generation without blocking every other live peer.

        aiortc.close() can itself await ICE/SCTP shutdown. Keep the retirement
        tracked; do not start a new generation until the previous one has
        closed (or its bounded cleanup deadline has elapsed). A stop/start
        racing with a callback cannot revive a stale slot.node.
        """
        previous=slot.task
        slot.task=None
        slot.node=None
        if previous is None:
            return
        previous.cancel()
        if slot.retiring is not None and not slot.retiring.done():
            # Normally impossible: a slot cannot reconnect while retiring.
            # Do not let two stale connections outlive a new generation.
            await asyncio.wait({slot.retiring},timeout=.1)
        async def retire():
            try:
                done,pending=await asyncio.wait({previous},timeout=15.)
                if pending:
                    log.warning('RTC peer %s teardown exceeded 15s; abandoning stuck generation',slot.peer)
                    previous.cancel()
                else:
                    await asyncio.gather(previous,return_exceptions=True)
            except asyncio.CancelledError:
                previous.cancel()
                raise
        slot.retiring=asyncio.create_task(retire(),name='rtc-retire-'+slot.peer)
        # Fast close finishes inline. A slow one is supervised, not awaited by
        # the entire training/gossip loop.
        await asyncio.wait({slot.retiring},timeout=.05)

    def retry_delay(slot:PeerSlot) -> float:
        # Per-direction deterministic jitter prevents symmetric peer resets.
        seed=f'{args.peer_id}/{slot.peer}/{slot.retries}'.encode('ascii')
        fraction=int.from_bytes(hashlib.sha256(seed).digest()[:2],'big') / 65535
        return min(60.,2.**min(slot.retries,6)) + 0.25 + fraction

    def accept_authenticated_peer(peer: str, pub64: str, verifier):
        """Persist first verified hello key for future and relayed events.

        Never trust the advertised peer ID alone: ElasticNode has already
        verified the Ed25519 signature and key-derived identity.
        """
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        raw=base64.b64decode(pub64,validate=True)
        pem=Ed25519PublicKey.from_public_bytes(raw).public_bytes(Encoding.PEM,PublicFormat.SubjectPublicKeyInfo)
        store_pin(peer,pem,verifier)

    async def start_slots(now):
        for slot in list(slots.values()):
            if slot.retiring is not None:
                if not slot.retiring.done():
                    continue
                await asyncio.gather(slot.retiring,return_exceptions=True)
                slot.retiring=None
            if slot.blocked_reason:
                continue  # explicit intervention required; never infinite auth retry
            if slot.disabled:
                if slot.task: await stop_slot(slot)
                continue
            node=slot.node
            # A closed DataChannel is terminal even when the Mummy signaling
            # socket is still open. Do not wait for the initial 120s ICE
            # negotiation timeout before releasing this stale peer task.
            if (slot.task and node is not None and node.link_closed.is_set()
                    and not isinstance(node.last_error, HandshakeError)):
                log.warning('peer %s SCTP DataChannel closed; scheduling reconnect: %s',
                            slot.peer, node.last_error or 'remote closed channel')
                await stop_slot(slot)
                slot.retries=min(slot.retries+1,6)
                slot.retry_after=now+retry_delay(slot)
                slot.stable_since=0
                slot.heartbeat_warned=False
                continue
            if slot.task and node is not None and node.connected.is_set():
                if node.verified_at:
                    age=now-node.verified_at
                    if age>args.peer_timeout and not slot.heartbeat_warned:
                        log.warning('peer %s signed heartbeat delayed %.1fs; '
                                    'excluding from active count but keeping SCTP open',
                                    slot.peer,age)
                        slot.heartbeat_warned=True
                    elif age<=args.peer_timeout and slot.heartbeat_warned:
                        log.info('peer %s signed heartbeat recovered; connection retained',
                                 slot.peer)
                        slot.heartbeat_warned=False
                # Give a healthy authenticated connection time to stabilize.
                if node.verified_at and not slot.stable_since:
                    slot.stable_since=now
                if slot.stable_since and now-slot.stable_since>=max(args.peer_timeout,10):
                    slot.retries=0
                # Losing a few signed heartbeats is not proof that an open
                # SCTP connection is dead (slow MPS/CPU training, congestion,
                # large state transfers). Keep the transport for a longer
                # probe window; liveness counts still expire conservatively.
                if now-max(node.verified_at,node.connected_since)>max(args.peer_timeout*4,120):
                    log.warning('peer %s signed heartbeat stalled for %.1fs; reconnecting phase=%s ice=%s rtc=%s',
                                slot.peer,now-node.verified_at,node.phase,node.ice_state,node.rtc_state)
                    await stop_slot(slot)
                    slot.retries=min(slot.retries+1,6)
                    slot.retry_after=now+retry_delay(slot)
                    slot.stable_since=0
                    continue
            # ICE negotiation frequently takes >30 seconds for restrictive NATs.
            # Do NOT close both WebSockets just because gather is still running.
            if (slot.task and node is not None and
                not node.connected.is_set() and now-slot.started_at>120.0):
                log.warning('peer %s negotiation timeout phase=%s role=%s ice=%s rtc=%s',
                            slot.peer,node.phase,node.role,node.ice_state,node.rtc_state)
                await stop_slot(slot)
                slot.retries=min(slot.retries+1,6)
                slot.retry_after=now+retry_delay(slot)
                slot.stable_since=0
                continue
            if slot.task and slot.task.done():
                if not slot.task.cancelled():
                    error=slot.task.exception()
                    if error:log.warning('peer %s disconnected: %s',slot.peer,error)
                fatal=slot.node.last_error if slot.node is not None else None
                await stop_slot(slot)
                if isinstance(fatal,HandshakeError):
                    slot.blocked_reason=str(fatal)
                    print(f'PEER BLOCKED peer={slot.peer} error={slot.blocked_reason}',flush=True)
                    continue
                slot.retries=min(slot.retries+1,6)
                slot.retry_after=now+retry_delay(slot)
                slot.stable_since=0
                slot.heartbeat_warned=False
            if slot.task is None and now>=slot.retry_after:
                if room_only:
                    import importlib.util
                    if importlib.util.find_spec('aiortc') is None:
                        raise RuntimeError('Detected another room peer, but aiortc is missing '
                                           'from this training Python environment. '
                                           'Install with: python -m pip install aiortc')
                node=ElasticNode((lobby.active_url if room_only and lobby.online else args.ws),slot.room,args.token,origin=args.peer_id,
                    remote=slot.peer,genesis=genesis,unsafe=args.unsafe_unsigned,
                    max_bytes=args.max_patch_mb*1024**2, signer=signer,
                    verifier=trusted.get(slot.peer),
                    public_key=room_pub if room_only else None,
                    on_authenticated=accept_authenticated_peer if room_only else None,
                    snapshot_store=snapshot_store if room_only else None,
                    enable_generation=generation_committee is not None,
                    generation_root=root_genesis if generation_committee is not None else None,
                    snapshot_recv=snapshot_recv if room_only else None,
                    snapshot_limit=args.max_state_gb*1024**3)
                node.lookup_requested_event=journal.get
                node.lookup_requested_event_async=journal.get_async
                slot.node=node
                slot.started_at=now
                slot.task=asyncio.create_task(node.run(),name='elastic-'+slot.peer)
                log.info('connecting peer=%s pair=%s retry=%d',slot.peer,slot.room[:12],slot.retries)

    async def send_full_snapshot(slot) -> bool:
        nonlocal optimizer
        node=slot.node
        if node is None or node.snapshot_link is None or not node.snapshot_link.connected.is_set():
            return False
        if node.verified_at == 0:
            return False
        link=node.snapshot_link
        if link.session.local_data is not None and not link.session.acked_local:
            # Never spin forever ignoring retry requests. The receiver also
            # checks progress, then reconnects to resume cached shards.
            if link.stalled(60.0):
                node.last_error=TimeoutError('snapshot sender stalled without ACK for 60s')
                node.protocol_failure.set()
                print(f'SNAPSHOT ERROR peer={slot.peer}: sender stalled; reconnecting',flush=True)
            return False  # Existing snapshot is still being sent; no overlap.
        # A potentially multi-GB snapshot export is slow; tell the requester
        # what is happening instead of leaving it guessing whether the request
        # was delivered. Control and state channels are independent SCTP streams.
        node.control('snapshot-status', request_id=node.snapshot_pending_id,
                     stage='exporting')
        node.snapshot_tx_stage='exporting'
        print(f'SNAPSHOT EXPORT START peer={slot.peer}',flush=True)
        # Seal every local step before capturing the full effective model.
        await seal_if_needed()
        async with inbound_commit_lock:
            tip,count,cutoffs=snapshot_inventory(journal)
            from .shards import export_state_sharded
            from .shards import manifest_bytes
            path=snapshot_root/f'offer-{args.peer_id}-{count}-{tip[:16]}.fmanifest'
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                # Keep the event loop responsive to SCTP heartbeats while streaming
                # tensors to disk. The training coroutine is paused at this await.
                await asyncio.to_thread(export_state_sharded,model,snapshot_store,path,
                     model_id=adapter.model_id,revision=adapter.revision,
                     peer_id=args.peer_id,seq=count,adapter=adapter,
                     shard_bytes=args.shard_mb*1024**2,max_state_bytes=args.max_state_gb*1024**3)
            # Do not allocate a potentially 96 MiB manifest merely to hash it.
            with path.open('rb') as manifest_stream:
                digest=hashlib.file_digest(manifest_stream,'sha256').hexdigest()
        if not await send_snapshot_boundary(node, digest, tip, count, cutoffs):
            print(f'SNAPSHOT TIP BACKPRESSURE peer={slot.peer}: retry pending',flush=True)
            return False
        node.snapshot_link.offer(path,signer)
        node.control('snapshot-status', request_id=node.snapshot_pending_id,
                     stage='offered')
        node.snapshot_tx_stage='offered'
        # Training can now resume immediately: the signed vector clock allows
        # recipients to apply post-snapshot events as a suffix without duplicates.
        print(f'SNAPSHOT OFFER peer={slot.peer} events={count} manifest={digest[:16]}',flush=True)
        return True

    async def read_peer_events():
        maybe_start_generation_bridge()
        if critic_gossip is not None:
            now=loop.time()
            for ps in slots.values():
                nd=ps.node
                if (nd is None or nd.bridge_only or not nd.verified_at or
                        not nd.remote_supports_critic or now<nd.last_critic_frontier+2.0):
                    continue
                heads=critic_gossip.journal.frontiers()
                # Bound controls even in a room with thousands of origins.
                if heads:
                    k=nd.critic_frontier_offset%len(heads)
                    page=(heads[k:]+heads[:k])[:32]
                    if nd.control('critic-frontier',heads=page):
                        nd.critic_frontier_offset=(k+len(page))%len(heads)
                        nd.last_critic_frontier=now
                else:
                    nd.last_critic_frontier=now
        if trajectory_gossip is not None:
            now=loop.time()
            heads=trajectory_gossip.frontiers()
            for ps in slots.values():
                nd=ps.node
                if (nd is None or nd.bridge_only or not nd.verified_at or
                        not nd.remote_supports_rollout or now<nd.last_rollout_frontier+2.0):
                    continue
                if heads:
                    k=nd.rollout_frontier_offset%len(heads)
                    page=(heads[k:]+heads[:k])[:32]
                    if nd.control('trajectory-frontier',heads=page):
                        nd.rollout_frontier_offset=(k+len(page))%len(heads)
                        nd.last_rollout_frontier=now
                else:
                    nd.last_rollout_frontier=now
        for slot in list(slots.values()):
            node=slot.node
            if node is None or slot.disabled: continue
            if generation_bridge is not None:
                for _ in range(16):
                    if node.generation_controls.empty():break
                    kind, fields=node.generation_controls.get_nowait()
                    try:
                        if kind=='generation-compact-snapshot-request':
                            # The cert is a pinned CFT quorum. A bare hash or
                            # peer hint NEVER authorizes a large file transfer.
                            cert=(generation_compact_ledger.certified_epoch(fields['epoch'])
                                  if generation_compact_ledger is not None else None)
                            p=cert['proposal'] if cert else None
                            if (p is not None and p['manifest']==fields['manifest']
                                    and node.snapshot_link is not None):
                                path=snapshot_recv/(p['manifest']+'.fmanifest')
                                if not path.is_symlink() and path.is_file():
                                    from .shards import read_manifest,verify_shards,semantic_shard_sha256
                                    with path.open('rb') as fp:
                                        digest=hashlib.file_digest(fp,'sha256').hexdigest()
                                    if digest!=p['manifest']:
                                        raise PatchError('compact checkpoint manifest corrupted')
                                    meta=read_manifest(path,max_state_bytes=args.max_state_gb*1024**3)
                                    await asyncio.to_thread(verify_shards,meta,snapshot_store)
                                    if semantic_shard_sha256(meta)!=p['semantic']:
                                        raise PatchError('compact checkpoint semantic mismatch')
                                    node.snapshot_link.offer(path,signer)
                        elif kind=='generation-snapshot-request':
                            # Exact certified content only; never serve an
                            # arbitrary file path named by the requesting peer.
                            cert=generation_ledger.latest()
                            p=cert['proposal'] if cert else None
                            if (p and p['epoch']==fields['epoch'] and
                                p['manifest']==fields['manifest'] and node.snapshot_link):
                                # Once hash-verified, a downloaded certified
                                # checkpoint may be re-served by ANY full holder,
                                # not only the initial committee replicas.
                                candidates=[snapshot_recv/(p['manifest']+'.fmanifest')]
                                if getattr(args,'shard_state_manifest',None):
                                    candidates.append(Path(args.shard_state_manifest))
                                for path in candidates:
                                    if not path.is_file() or path.is_symlink():continue
                                    with path.open('rb') as fp:
                                        digest=hashlib.file_digest(fp,'sha256').hexdigest()
                                    if digest!=p['manifest']:continue
                                    from .shards import read_manifest,verify_shards,semantic_shard_sha256
                                    meta=read_manifest(path,max_state_bytes=args.max_state_gb*1024**3)
                                    # The sender's shard store is the same
                                    # content-addressed cache as the receiver's.
                                    await asyncio.to_thread(verify_shards,meta,snapshot_store)
                                    if semantic_shard_sha256(meta)!=p['semantic']:
                                        raise PatchError('certified snapshot semantic mismatch')
                                    node.snapshot_link.offer(path,signer)
                                    break
                        elif kind.startswith('generation-carry-'):
                            if carry_relay is not None:
                                carry_relay.receive(slot.peer,kind,fields)
                        else:
                            generation_bridge.receive(slot.peer,kind,fields)
                    except (PatchError, ValueError, TypeError, KeyError, OSError, asyncio.QueueFull) as ex:
                        log.warning('reject generation control from %s: %s',slot.peer,ex)
            for _ in range(32):
                if node.snapshot_tip_latch is not None:
                    msg=node.snapshot_tip_latch
                    node.snapshot_tip_latch=None
                elif node.snapshot_status_latch is not None:
                    msg=node.snapshot_status_latch
                    node.snapshot_status_latch=None
                elif not node.controls.empty():
                    msg=node.controls.get_nowait()
                else:
                    break
                typ=msg.get('type')
                if typ == 'hello':
                    node.control('hello-ack')
                    node.inventory_cursor=0
                elif typ == 'inventory':
                    entries=msg.get('events')
                    missing=inventory_missing_bounded(node,journal,entries)
                    # A failed send (SCTP backpressure) MUST NOT cause us to
                    # silently regard the missing events as synchronized.
                    if missing: node.ready=False
                    if msg.get('done') is True: node.inventory_done=True
                    if node.inventory_done and not node.pending:node.ready=True
                elif typ == 'need':
                    # Compatibility for previously queued frames; current
                    # traffic bypasses this queue at authenticated receipt.
                    node.queue_requested_events(validated_need(msg.get('events')))
                elif typ == 'epoch-v2-wire':
                    if epoch_paxos is not None:
                        payload=msg.get('payload')
                        if not isinstance(payload,str) or len(payload)>16000:
                            raise PatchError('invalid epoch proof envelope')
                        try:
                            epoch_paxos.receive(payload)  # verify the ORIGINAL signer's inner signature
                        except PatchError as ex:
                            # Peers that have not yet received this signer's signed
                            # Hello cannot validate relayed packets. Periodic signed
                            # retransmission will retry after the key is pinned.
                            if 'not pinned' not in str(ex):
                                log.warning('invalid epoch evidence from %s: %s',slot.peer,ex)
                elif typ == 'snapshot-request':
                    # Already authenticated, ACKed and latched in the aiortc
                    # receive callback. The slow training loop must not be
                    # required for control-channel request acknowledgement.
                    pass
                elif typ == 'snapshot-status':
                    request_id=msg.get('request_id','')
                    stage=msg.get('stage')
                    if (not isinstance(request_id,str) or len(request_id)>32 or
                        stage not in ('accepted','waiting-state-channel',
                                      'source-not-ready','exporting','offered')):
                        raise PatchError('invalid snapshot status')
                    if request_id == node.snapshot_request_id:
                        if stage != node.snapshot_remote_stage:
                            print(f'SNAPSHOT SOURCE peer={slot.peer} stage={stage}',flush=True)
                        node.snapshot_remote_stage=stage
                        node.snapshot_remote_stage_at=time.monotonic()
                elif typ == 'snapshot-tip':
                    # The overflow latch is processed before the FIFO queue.
                    # Never let an older queued offer overwrite the newest
                    # signed boundary, or the receiver can wait forever for
                    # a manifest digest which was superseded long ago.
                    if msg['counter'] <= node.latest_snapshot_tip_counter:
                        continue
                    tip=msg.get('inventory_sha256')
                    digest=msg.get('manifest_sha256')
                    count=msg.get('events')
                    cutoffs=msg.get('cutoffs')
                    if (not isinstance(tip,str) or len(tip)!=64 or
                        not isinstance(digest,str) or len(digest)!=64 or
                        type(count) is not int or not 0<=count<2**63):
                        raise PatchError('invalid signed snapshot tip')
                    # Already signed and reassembled; reject invalid ranges
                    # before expensive journal comparison.
                    validate_cutoffs(cutoffs)
                    snapshot_tips[slot.peer]=(digest,tip,count,cutoffs)
                    node.latest_snapshot_tip_counter=msg['counter']
                    print(f'SNAPSHOT TIP peer={slot.peer} events={count}',flush=True)
                elif typ == 'critic-frontier':
                    if critic_gossip is not None and node.remote_supports_critic:
                        needed=critic_gossip.need(msg.get('heads'),limit=16)
                        if needed:
                            node.control('critic-need',events=[list(x) for x in needed])
                elif typ == 'critic-need':
                    if critic_gossip is not None and node.remote_supports_critic:
                        requests=msg.get('events')
                        if not isinstance(requests,list) or len(requests)>16:
                            raise PatchError('invalid critic need list')
                        from .core import PEER_ID as _v77_PEER_ID
                        for entry in requests:
                            if (not isinstance(entry,list) or len(entry)!=2 or
                                not isinstance(entry[0],str) or not _v77_PEER_ID.fullmatch(entry[0]) or
                                type(entry[1]) is not int or not 1<=entry[1]<2**63):
                                raise PatchError('invalid critic need identity')
                            wire=critic_gossip.journal.get(*entry)
                            if wire is not None and not node.outgoing.full():
                                node.outgoing.put_nowait(wire)
                elif typ == 'trajectory-frontier':
                    if trajectory_gossip is not None and node.remote_supports_rollout:
                        needed=trajectory_gossip.need(msg.get('heads'),limit=16)
                        if needed:node.control('trajectory-need',events=[list(x) for x in needed])
                elif typ == 'trajectory-need':
                    if trajectory_gossip is not None and node.remote_supports_rollout:
                        requests=msg.get('events')
                        if not isinstance(requests,list) or len(requests)>16:
                            raise PatchError('invalid trajectory-need list')
                        from .core import PEER_ID as _v80_PEER_ID
                        for entry in requests:
                            if (not isinstance(entry,list) or len(entry)!=2 or
                                not isinstance(entry[0],str) or not _v80_PEER_ID.fullmatch(entry[0]) or
                                type(entry[1]) is not int or not 1<=entry[1]<2**63):
                                raise PatchError('invalid trajectory-need identity')
                            wire=trajectory_gossip.journal.get(*entry)
                            if wire is not None and not node.outgoing.full():
                                node.outgoing.put_nowait(wire)
                elif typ in ('hello-ack','heartbeat'):
                    if controller is not None:
                        controller.observe(slot.peer,msg.get('cap'))
                else:
                    raise PatchError('unsupported elastic control')
            # Never await signature verification / tensor application / disk IO
            # here: keep network and local training responsive to incoming data.
            for _ in range(16):
                if node.incoming.empty() or inbound_pipeline.queue.full():
                    break
                wire=node.incoming.get_nowait()
                if not isinstance(wire,bytes):
                    raise PatchError('nonbinary elastic event')
                if wire.startswith((b'KTR82\0',b'KCP82\0')):
                    if trajectory_gossip is None or not node.remote_supports_rollout:
                        raise PatchError('unexpected v81 trajectory/checkpoint frame')
                    async with critic_step_lock:
                        with adapter.lock:
                            fresh=trajectory_gossip.ingest(wire)
                    if fresh:
                        for ps in slots.values():
                            other=ps.node
                            if (ps.peer!=slot.peer and other is not None and
                                    other.remote_supports_rollout and other.verified_at and
                                    not other.outgoing.full() and not other.bridge_only):
                                other.outgoing.put_nowait(wire)
                    continue
                if wire.startswith(b'KTR80\0') and trajectory_gossip is not None:
                    raise PatchError('v81 room rejects unsigned-checkpoint v80 trajectory')
                if wire.startswith(b'KVC78\0'):
                    if trajectory_gossip is not None:
                        raise PatchError('v81 rollout room must not ingest old Critic parameter deltas')
                    if critic_gossip is None or not node.remote_supports_critic:
                        raise PatchError('unexpected signed Critic frame in a non-v78 room')
                    async with critic_step_lock:
                        with adapter.lock:
                            applied=critic_gossip.ingest(wire)
                    if applied:
                        # Relay on the same bounded RTC queues. A missed
                        # broadcast is recovered via critic-frontier/need.
                        for ps in slots.values():
                            other=ps.node
                            if (ps.peer!=slot.peer and other is not None and
                                    other.remote_supports_critic and other.verified_at and
                                    not other.outgoing.full() and not other.bridge_only):
                                other.outgoing.put_nowait(wire)
                    continue
                if not inbound_pipeline.submit(slot.peer,wire):
                    # The byte budget can reject an item even while the
                    # queue has free *slots*. Retain its exact signed bytes
                    # in the source's bounded receive queue for a later tick.
                    # No await occurs between get and put, so the freed queue
                    # slot cannot be stolen by another event-loop callback.
                    node.incoming.put_nowait(wire)
                    break

    async def inbound_committed(info,wire,source_peer,was_applied):
        # An event may arrive from any relay. A verified receipt satisfies all
        # outstanding need requests; no early local seal is required.
        for ps in list(slots.values()):
            if ps.node is not None:
                ps.node.pending.discard((info.origin,info.seq))
                if ps.node.inventory_done and not ps.node.pending and not ps.node.bridge_only:
                    ps.node.ready=True
        if was_applied:
            await fanout(wire,exclude_peer=source_peer,origin_peer=info.origin)

    # Build the adjacent-generation carry relay only when authenticated
    # generation control is configured. The relay is a DISK source, not a
    # second model writer; every incoming update still passes AsyncInboundPipeline.
    from .epoch_carry_gossip_v53 import AdjacentEpochCarryGossip
    carry_relay=None
    if generation_committee is not None:
        if carry_archive is not None:
            # Stored carry bodies are old-genesis signed, but the carry journal
            # itself uses a domain-separated archive identity.
            class _OldCarryView:
                genesis=carry_archive.source_genesis
                db=carry_archive.journal.db
                get=carry_archive.journal.get
                get_async=carry_archive.journal.get_async
                has=carry_archive.journal.has
            relay_source=_OldCarryView()
            relay_genesis=carry_archive.source_genesis
        else:
            relay_source=journal
            relay_genesis=genesis
        carry_relay=AdjacentEpochCarryGossip(source_genesis=relay_genesis,
            source_journal=relay_source,spool=epoch_dir/'carry-spool',
            carry_archive=carry_archive,
            max_event=min(args.max_patch_mb*1024**2,16*1024**2))

    from .async_ingest import AsyncInboundPipeline
    engine.max_wire_bytes=args.max_patch_mb*1024**2
    inbound_pipeline=AsyncInboundPipeline(engine,inbound_committed,
        max_queue=getattr(args,'inbound_queue',32),commit_lock=inbound_commit_lock,
        carry_archive=carry_archive)

    # v53: certificate-triggered, locally safe auto-adoption. Preparation uses
    # a second model and independent optimizer. It NEVER rewires production
    # state from the asynchronous prewarm task: only the main trainer loop may
    # perform the final step-boundary publication. Missing certificates or
    # checkpoint shards simply defer adoption, without a fabricated vote.
    from .epoch_hot_swap_v45 import TrainerGeneration,TrainerStepRouter
    from .live_epoch_switch_v48 import (LiveEpochSwitchService,PreparedEpoch,
                                        certified_genesis,verify_candidate_files)
    from .epoch_local_stage_v53 import stage_verified_local_view
    from .epoch_live_rebase_v52 import StreamingCertifiedRebase
    from .epoch_carry_v46 import CrossEpochCarryArchive
    from .online_generation import _digest as _epoch_proposal_digest
    auto_epoch_service=None
    auto_epoch_catchup=None
    auto_epoch_prepare=None
    auto_epoch_next=0.0
    auto_epoch_candidate={}
    auto_epoch_prepared={}
    epoch_writer_frozen=False
    original_epoch=epoch_boot.epoch if epoch_boot is not None else 0
    original_checkpoint=epoch_boot.manifest_hash if epoch_boot is not None else root_genesis
    trainer_router=TrainerStepRouter(TrainerGeneration(original_epoch,genesis,
        original_checkpoint,model,adapter,optimizer,journal))

    async def build_epoch_shadow(proposal):
        # Resume an unchanged, already loaded shadow when a certified cut was
        # merely incomplete during its first preparation attempt.
        already=auto_epoch_prepared.get(certified_genesis(proposal))
        if already is not None:return already
        # Existing snapshot receiver delivers all objects by content digest.
        # Stage with hardlinks; no new remote API or untrusted file path.
        candidate=await asyncio.to_thread(stage_verified_local_view,proposal,
            root=epoch_dir,incoming=snapshot_recv,objects=snapshot_store,
            max_state_bytes=args.max_state_gb*1024**3)
        fresh_model,fresh_adapter=await asyncio.to_thread(load_model,args)
        await asyncio.to_thread(__import__('patchlib.shards',fromlist=['apply_sharded_effective_state']).apply_sharded_effective_state,
            fresh_model,fresh_adapter,candidate.manifest,candidate.store,
            model_id=fresh_adapter.model_id,revision=fresh_adapter.revision,
            max_state_bytes=args.max_state_gb*1024**3)
        next_genesis=certified_genesis(proposal)
        next_journal_path=epoch_dir/'journals'/f'epoch-{proposal["epoch"]:08d}.sqlite3'
        if next_journal_path.is_symlink():raise PatchError('unsafe successor journal')
        next_journal=EventJournal(next_journal_path,next_genesis,max_bytes=args.max_journal_gb*1024**3,
                                  sync=getattr(args,'journal_sync','full').upper())
        archive=None
        try:
            if next_journal.count():
                raise PatchError('uncommitted new epoch journal already contains updates')
            next_engine=ContinualElasticEngine(fresh_adapter,args.peer_id,next_genesis,next_journal,
                reinit_every=getattr(args,'reinit_every',20),signer=signer,trusted=trusted,
                unsafe=False,max_delta_rms=args.max_delta_rms,event_public=room_pub)
            next_engine.max_wire_bytes=args.max_patch_mb*1024**2
            archive=CrossEpochCarryArchive(epoch_dir/'carries'/f'epoch-{proposal["epoch"]:08d}.sqlite3',
                source_genesis=journal.genesis,target_genesis=next_genesis,
                certified_manifest=proposal['manifest'],epoch=proposal['epoch'],
                cutoffs=proposal['cutoffs'])
            archive.replay(next_engine) # always on the NEW independently loaded base
            rebaser=StreamingCertifiedRebase(journal,archive,next_engine,proposal)
            next_optimizer=new_optimizer(fresh_adapter,next_genesis)
            next_optimizer.zero_grad(set_to_none=True)
            result=PreparedEpoch(proposal['epoch'],_epoch_proposal_digest(proposal),
                proposal['manifest'],TrainerGeneration(proposal['epoch'],next_genesis,
                    proposal['manifest'],fresh_model,fresh_adapter,next_optimizer,next_journal),
                candidate.manifest,candidate.store,rebaser)
            auto_epoch_candidate[next_genesis]=(archive,next_engine,rebaser)
            auto_epoch_prepared[next_genesis]=result
            return result
        except BaseException:
            if archive is not None:archive.close()
            next_journal.close()
            raise

    def verify_epoch_local(prepared,proposal):
        return (prepared.generation.genesis==certified_genesis(proposal) and
            prepared.generation.genesis in auto_epoch_candidate and
            prepared.generation.journal.genesis==prepared.generation.genesis)

    async def freeze_old_epoch(_old):
        nonlocal epoch_writer_frozen,network_receiver_task
        # The main optimizer task calls activate_next ONLY between batches.
        # Seal every partial gradient before preventing future old ingress.
        await seal_if_needed()
        # Mark the fence as soon as irreversible source shutdown begins.
        # Any exception before pointer publication now terminates this trainer
        # instead of resuming with a half-closed old inbound pipeline.
        epoch_writer_frozen=True
        if network_receiver_task is not None:
            network_receiver_task.cancel()
            await asyncio.gather(network_receiver_task,return_exceptions=True)
            network_receiver_task=None
        # Preserve already-received frames before dropping the old session.
        # stop_slot detaches slot.node, so save the bounded receive queues.
        received_nodes=[(slot.peer,slot.node) for slot in slots.values()
                        if slot.node is not None]
        await asyncio.gather(*(stop_slot(s) for s in tuple(slots.values())))
        retire=[s.retiring for s in slots.values() if s.retiring is not None]
        if retire:await asyncio.gather(*retire)
        if any(node._channel is not None and node._channel.readyState=='open'
               for _,node in received_nodes):
            raise PatchError('old generation WebRTC channel did not close')
        # Deliver the bounded pending DataChannel queue through the ORIGINAL
        # signed pipeline; future retransmissions will use carry anti-entropy.
        for peer,node in received_nodes:
            while not node.incoming.empty():
                wire=node.incoming.get_nowait()
                if not isinstance(wire,bytes):
                    raise PatchError('malformed old DataChannel event at handoff')
                if wire.startswith((b'KTR82\0',b'KCP82\0')):
                    if trajectory_gossip is None:
                        raise PatchError('unexpected v81 trajectory/checkpoint at epoch handoff')
                    async with critic_step_lock:
                        with adapter.lock:
                            trajectory_gossip.ingest(wire)
                    continue
                if wire.startswith(b'KTR80\0') and trajectory_gossip is not None:
                    raise PatchError('v80 trajectory forbidden in v81 epoch handoff')
                if wire.startswith(b'KVC78\0'):
                    if trajectory_gossip is not None:
                        raise PatchError('old-generation Critic delta in a v81 rollout room')
                    if critic_gossip is None:
                        raise PatchError('unexpected Critic frame at epoch handoff')
                    async with critic_step_lock:
                        with adapter.lock:
                            critic_gossip.ingest(wire)
                    continue
                while not inbound_pipeline.submit(peer,wire):
                    await inbound_pipeline.drain()
        await inbound_pipeline.drain()
        await inbound_pipeline.stop()
        await working_saver.drain()
        await paired_saver.drain()

    def attest_new_epoch(new,proposal):
        bundle=auto_epoch_candidate.get(new.genesis)
        if bundle is None:return False
        archive,new_engine,rebaser=bundle
        receipt=rebaser.verify_handoff_receipt()
        if (receipt.get('source_genesis')!=journal.genesis or
                receipt.get('source_rowid')!=journal.db.execute(
                    'SELECT COALESCE(MAX(rowid),0) FROM events').fetchone()[0] or
                new_engine.journal is not new.journal or new.journal.count()!=0):
            return False
        with new.journal.db:
            new.journal.db.execute(
                "INSERT OR REPLACE INTO meta(name,value) VALUES('initial_ready','1')")
        return True

    async def publish_new_epoch(old,new,proposal):
        nonlocal model,adapter,optimizer,journal,engine,genesis
        nonlocal inbound_pipeline,carry_archive,carry_relay,local_steps,emitted
        nonlocal network_receiver_task,next_checkpoint,epoch_writer_frozen
        if not epoch_writer_frozen:
            raise PatchError('cannot publish without frozen old epoch writers')
        archive,new_engine,_=auto_epoch_candidate.pop(new.genesis)
        auto_epoch_prepared.pop(new.genesis,None)
        old_archive=carry_archive
        old_relay=carry_relay
        if critic_gossip is not None:
            if archive.source_genesis not in critic_gossip.known_actor_epochs:
                raise PatchError('unknown predecessor for Critic epoch transition')
            # publish_new_epoch executes only after the verified quorum
            # certificate, complete carry handoff, and frozen source writer.
            critic_gossip.register_certified_epoch(new.genesis,archive.source_genesis)
            if trajectory_gossip is not None:
                trajectory_gossip.actor_epoch=new.genesis
                # Keep the shared value head while resetting any optimizer
                # history calculated for a previous Actor epoch.
                trajectory_gossip.persist()
        model,adapter,optimizer,journal=(new.model,new.adapter,new.optimizer,new.journal)
        if hasattr(old.optimizer,'close'):old.optimizer.close()
        engine=new_engine;genesis=new.genesis;carry_archive=archive
        if critic_gossip is not None:
            # Keep the independent Critic and its signed history across the
            # verified Actor generation switch; update PPO's Actor reference.
            args._v65_mixer_adapter=adapter
            critic_gossip.lock=adapter.lock
        local_steps=0;pending[0]=0;emitted=journal.next_seq(args.peer_id)-1
        nonlocal_optimizer[0]=optimizer
        next_checkpoint=(journal.count()//args.checkpoint_every+1)*args.checkpoint_every if args.checkpoint_every else None
        if old_relay is not None:old_relay.close()
        if old_archive is not None:old_archive.close()
        if carry_relay is not None:
            class _OldCarryView53:
                genesis=archive.source_genesis
                db=archive.journal.db
                get=archive.journal.get
                get_async=archive.journal.get_async
                has=archive.journal.has
            carry_relay=AdjacentEpochCarryGossip(source_genesis=archive.source_genesis,
                source_journal=_OldCarryView53(),spool=epoch_dir/'carry-spool',
                carry_archive=archive,max_event=min(args.max_patch_mb*1024**2,16*1024**2))
        inbound_pipeline=AsyncInboundPipeline(engine,inbound_committed,
            max_queue=getattr(args,'inbound_queue',32),commit_lock=inbound_commit_lock,
            carry_archive=carry_archive)
        inbound_pipeline.start()
        if generation_bridge is not None:generation_bridge.neighbors.clear()
        network_receiver_task=asyncio.create_task(network_receiver_loop(),
            name='fedpara-network-receiver')
        epoch_writer_frozen=False
        old.journal.close()
        print(f'LIVE EPOCH ACTIVATED epoch={new.epoch} genesis={genesis[:16]} '
              f'carried={archive.journal.count()} (new optimizer, journal, WebRTC)',flush=True)
        return True

    def make_auto_epoch_service():
        selected_ledger=(generation_trainer_ledger if getattr(args,'generation_v2_trainer',False)
                         else generation_ledger)
        if selected_ledger is None:return None
        # Reconstruct the active pointer FIRST at startup (v52). Only this
        # verified trainer is allowed to open the v53 publishing service.
        async def old_frozen(_old):return epoch_writer_frozen
        async def carry_complete(old,new,proposal):
            bundle=auto_epoch_candidate.get(new.genesis)
            if not epoch_writer_frozen or bundle is None:
                raise PatchError('attempted carry without old ingress barrier')
            bundle[2].verify_handoff_receipt()
        return LiveEpochSwitchService(ledger=selected_ledger,router=trainer_router,
            root=epoch_dir,build_shadow=build_epoch_shadow,
            verify_local=verify_epoch_local,
            flush_unpublished=freeze_old_epoch,carry_suffix=carry_complete,
            transport_handoff=publish_new_epoch,attest_journal=attest_new_epoch,
            old_ingress_frozen=old_frozen)

    async def network_receiver_loop():
        # The trainer awaits model forward/backward in a worker thread.  This
        # coroutine continues to drain signed DataChannels throughout that
        # time, rather than processing arrivals only between minibatches.
        next_carry_send=0.0
        while not shutdown.is_set():
            await read_peer_events()
            if carry_relay is not None and generation_bridge is not None:
                if loop.time()>=next_carry_send:
                    next_carry_send=loop.time()+.05
                    carry_relay.tick(((p,s.node) for p,s in slots.items()
                        if not s.disabled and s.node is not None),now=loop.time())
                await carry_relay.flush_completed(inbound_pipeline.submit)
            await asyncio.sleep(.003)

    async def maybe_install_snapshot():
        nonlocal optimizer, initial_complete
        if bootstrap.snapshot_installed or not bootstrap.fresh_join:
            return
        from .shards import assemble_state,read_manifest
        from .state import apply_effective_state
        for peer,manifest in list(received_snapshots.items()):
            tip=snapshot_tips.get(peer)
            if tip is None:continue
            expected_sha,events_digest,event_count,cutoffs=tip
            if received_hashes.get(peer)!=expected_sha:
                continue  # Tip may describe a newer snapshot; request anew.
            meta=read_manifest(manifest,max_state_bytes=args.max_state_gb*1024**3)
            if meta['peer_id']!=peer or meta['model_id']!=adapter.model_id or meta['revision']!=adapter.revision:
                raise PatchError('received snapshot has wrong peer or model identity')
            # Do not rescan the ENTIRE signed journal several times per
            # second while a cold joiner waits for one missing event. The
            # journal is append-only; the same boundary and row count cannot
            # produce a new result. Recheck immediately after any durable
            # incoming event or changed snapshot tip.
            journal_count = journal.count()
            boundary = (expected_sha,events_digest,event_count,journal_count)
            if snapshot_history_still_unchanged(snapshot_history_mismatches.get(peer),
                    manifest_sha=expected_sha,inventory_sha=events_digest,
                    event_count=event_count,journal_count=journal_count):
                continue
            current_digest,current_count,_=snapshot_inventory(journal,cutoffs)
            if (current_digest,current_count)!=(events_digest,event_count):
                snapshot_history_mismatches[peer]=boundary
                # Must not replay the same updates on top of a complete snapshot.
                # The transfer is complete: wait for the missing journal events,
                # rather than downloading the same multi-GB model repeatedly.
                stamp=time.monotonic()
                if stamp-history_reported_at.get(peer,0)>10:
                    print(f'SNAPSHOT HISTORY WAIT peer={peer} '
                          f'verified_model=yes events={current_count}/{event_count} '
                          f'boundary_match={current_digest==events_digest}',flush=True)
                    history_reported_at[peer]=stamp
                continue
            # Any durable, unsealed LoHA + AdamW state must be committed before
            # installing a peer baseline so crash recovery cannot lose it.
            if local_steps:
                await seal_if_needed()
            # Install verified shards directly: no giant intermediate .fstate
            # and no second whole-model allocation. This runs off the event
            # loop, at the quiescent training boundary established above.
            from .shards import apply_sharded_effective_state
            await asyncio.to_thread(apply_sharded_effective_state,
                model,adapter,manifest,snapshot_store,
                model_id=adapter.model_id,revision=adapter.revision,
                max_state_bytes=args.max_state_gb*1024**3)
            # Later updates absent from the source's snapshot must be applied
            # AFTER the snapshot. Covered events must not be double-applied.
            from .elastic import unpack_event
            suffix=0
            for wire in journal.events():
                origin,meta,_,_=unpack_event(wire,trusted,unsafe=args.unsafe_unsigned)
                if meta['seq']<=cutoffs.get(origin,0):continue
                info,parts=engine._check(wire)
                engine._apply_checked(info,parts)
                suffix+=1
            # Journal remains the complete durable restart authority. On
            # restart, genesis+all signed events reconstruct identical state.
            # A fresh join receives a fully verified GLOBAL model. The importer
            # resets local factors; match its anchor, and restart moments ONLY
            # here (not after ordinary signed gossip updates).
            with adapter.lock, torch.no_grad():
                for g in adapter.groups:
                    for j,part in enumerate(g.overlay.local.parts):
                        getattr(g.overlay.anchor,f'part{j}').copy_(part.detach())
            old_optimizer=optimizer
            optimizer=new_optimizer();optimizer.zero_grad(set_to_none=True)
            if hasattr(old_optimizer,'close'):old_optimizer.close()
            nonlocal_optimizer[0]=optimizer
            from .localprogress import _tip
            await working_saver.drain()
            await paired_saver.drain()
            await asyncio.to_thread(save_working,journal,adapter,
                optimizer,0,journal_tip=_tip(journal))
            # The verified model already covers the signed journal cut-off.
            # Inventory-page readiness is *not* a second mandatory initial
            # barrier: the neighbor may be concurrently announcing newer
            # events forever. Elastic anti-entropy handles those asynchronously.
            bootstrap.confirm_verified_snapshot()
            with journal.db:
                journal.db.execute(
                    "INSERT OR REPLACE INTO meta(name,value) VALUES('initial_ready','1')")
            initial_complete=True
            print(f'SNAPSHOT READY: verified full model from {peer}; '
                  f'events={event_count}, suffix_events={suffix}, manifest={manifest}',flush=True)
            print(f'JOIN READY: verified-full-model-snapshot; '
                  f'events={journal.count()} peer={args.peer_id}',flush=True)
            return

    async def compact_vote_worker(p,pages):
        """Verify full checkpoint bytes and journal cut off the asyncio loop."""
        from .epoch_certificate_v46 import CompactEpochLedger
        import sqlite3
        def worker():
            ledger=CompactEpochLedger(Path(str(args.journal)+'.generation-compact.sqlite3'),
                room=args.room_id,peer=args.peer_id,committee=generation_committee,
                signer=signer,trusted=trusted,genesis=root_genesis)
            try:
                with sqlite3.connect(journal.path,timeout=30) as cut_db:
                    manifest=snapshot_recv/(p['manifest']+'.fmanifest')
                    return ledger.vote(p,pages=pages,journal_db=cut_db,
                                       manifest_path=manifest,shard_store=snapshot_store)
            finally:ledger.close()
        return await asyncio.to_thread(worker)

    compact_last_export=0
    compact_export_task=None
    async def maybe_propose_compact_epoch():
        nonlocal compact_last_export,compact_export_task
        import sqlite3
        frequency=getattr(args,'generation_v2_propose_every',0)
        if not frequency or not initial_complete:
            return
        # The expensive model+shard I/O runs concurrently with new training
        # steps. Only the short low-rank factor+cut capture needs ingress and
        # optimizer admission locked at a sealed step boundary.
        if compact_export_task is not None:
            if not compact_export_task.done():
                return
            completed=compact_export_task
            compact_export_task=None
            try:
                epoch_at_capture,parent_at_capture,p,pages,receipt=completed.result()
                if (trainer_router.current.epoch!=epoch_at_capture-1 or
                        generation_compact_ledger.head()!=parent_at_capture):
                    log.warning('obsolete compact snapshot ignored after epoch switch')
                else:
                    generation_compact_control.receive_export_receipt(receipt,p)
                    generation_compact_control.offer(p,pages)
                    print(f'COMPACT EPOCH CANDIDATE epoch={p["epoch"]} '
                          f'origins={p["frontier_header"]["total"]} '
                          f'pages={len(pages)} manifest={p["manifest"][:16]}',flush=True)
            except (PatchError,OSError,sqlite3.Error,ValueError) as exc:
                log.warning('background compact checkpoint export failed: %s',exc)
            return
        if (generation_compact_control is None or generation_trainer_ledger is None or
                args.peer_id!=min(generation_committee) or
                emitted-compact_last_export<frequency):
            return
        latest=generation_compact_ledger.latest()
        if latest is not None and latest['proposal']['epoch']>trainer_router.current.epoch:
            return
        target_epoch=trainer_router.current.epoch+1
        parent=generation_compact_ledger.head()
        if generation_compact_control.db.execute(
                'SELECT 1 FROM proposals WHERE epoch=? LIMIT 1',
                (target_epoch,)).fetchone():
            return
        from types import SimpleNamespace
        from .epoch_export_v57 import export_at_step_boundary
        from .epoch_frozen_export_v58 import (capture as freeze_export,
                                                     finish_deferred_frontier)
        compact_last_export=emitted
        try:
            await seal_if_needed()
            async with inbound_commit_lock:
                def capture_worker():
                    # Read-only WAL snapshot is pinned BEFORE factors are
                    # captured, while the trainer and ingress are fenced.
                    # The O(history) prefix scan runs only after release.
                    db=sqlite3.connect(journal.path,timeout=30,
                                       check_same_thread=False)
                    try:
                        db.execute('PRAGMA query_only=ON')
                        db.execute('BEGIN')
                        db.execute('SELECT COALESCE(MAX(rowid),0) FROM events').fetchone()
                        frozen=freeze_export(journal=SimpleNamespace(db=db),
                            model=model,adapter=adapter,defer_frontier=True,
                            max_factor_bytes=min(256*1024**2,
                                                 args.max_state_gb*1024**3))
                        return frozen,db
                    except BaseException:
                        db.close()
                        raise
                frozen,snapshot_db=await asyncio.to_thread(capture_worker)
            captured_model=model
            captured_adapter=adapter
            captured_journal_path=journal.path
            def export_worker():
                try:
                    completed=finish_deferred_frontier(frozen,snapshot_db)
                finally:
                    # Never hold a WAL read transaction while writing a large
                    # checkpoint: it would indefinitely pin the growing WAL.
                    snapshot_db.close()
                with sqlite3.connect(captured_journal_path,timeout=30) as db:
                    db.execute('PRAGMA query_only=ON')
                    view=SimpleNamespace(db=db,count=lambda:db.execute(
                        'SELECT COUNT(*) FROM events').fetchone()[0])
                    p,pages,receipt=export_at_step_boundary(
                        journal=view,model=captured_model,adapter=captured_adapter,
                        store=snapshot_store,incoming=snapshot_recv,
                        room=args.room_id,epoch=target_epoch,parent=parent,
                        replicas=list(generation_compact_ledger.committee),
                        signer=signer,peer=args.peer_id,
                        max_state_bytes=args.max_state_gb*1024**3,
                        shard_bytes=args.shard_mb*1024**2,frozen=completed)
                    return target_epoch,parent,p,pages,receipt
            compact_export_task=asyncio.create_task(
                asyncio.to_thread(export_worker),name='kagami-v58-cow-shard-export')
        except (PatchError,OSError,sqlite3.Error,ValueError) as exc:
            log.warning('frozen compact checkpoint capture deferred: %s',exc)

    async def maybe_checkpoint():
        nonlocal next_checkpoint,optimizer
        if next_checkpoint is None or journal.count()<next_checkpoint:return
        # Make a bounded, durable recovery point of the *committed* base.
        await seal_if_needed()
        from .shards import export_state_sharded
        root=Path(args.checkpoint_dir)
        root.mkdir(parents=True,exist_ok=True)
        count=journal.count()
        manifest=root/f'elastic-{count:09d}.fmanifest'
        await asyncio.to_thread(export_state_sharded,model,root/'objects-store',manifest,
            model_id=adapter.model_id,revision=adapter.revision,
            peer_id=args.peer_id,seq=count,adapter=adapter,
            shard_bytes=args.shard_mb*1024**2,max_state_bytes=args.max_state_gb*1024**3)
        # Signed checkpoint descriptor binds manifest+genesis to a complete
        # log inventory. It does not automatically authorize an epoch reset.
        history_hash=hashlib.sha256();offset=0
        while True:
            page,done=journal.inventory(offset,96)
            for ident in page:
                history_hash.update(_canonical_json(ident)+b'\n')
            offset+=len(page)
            if done:break
        payload={'protocol':'fedpara-elastic-checkpoint-v1',
                 'origin':args.peer_id,'genesis':genesis,'event_count':offset,
                 'event_inventory_sha256':history_hash.hexdigest(),
                 'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
        contents=_canonical_json(payload)
        if signer:
            payload['signature']=signer(len(contents),hashlib.sha256(contents).hexdigest())
        target=manifest.with_suffix('.checkpoint.json')
        tmp=target.with_suffix('.part')
        with tmp.open('wb') as handle:
            handle.write(_canonical_json(payload));handle.flush();os.fsync(handle.fileno())
        os.replace(tmp,target)
        print(f'elastic recovery checkpoint: {manifest} ({count} journal events)',flush=True)
        next_checkpoint=(journal.count()//args.checkpoint_every+1)*args.checkpoint_every

    async def wal_maintenance():
        # WAL frames are already durable before acknowledgement. This merely
        # transfers them to the main DB outside the latency-critical INSERT.
        while not shutdown.is_set():
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=3.0)
                break
            except asyncio.TimeoutError:
                pass
            try:
                if Path(str(journal.path)+'-wal').stat().st_size >= 32*1024*1024:
                    await asyncio.to_thread(journal.passive_checkpoint)
            except FileNotFoundError:
                pass

    # Use one mutable reference when updating optimizer in nested receive handler.
    nonlocal_optimizer=[optimizer]
    heartbeat_task=None
    watchdog_task=None
    network_receiver_task=None
    wal_maintenance_task=None
    last_swarm_report=0.0
    try:
        update_roster()
        heartbeat_task=asyncio.create_task(heartbeat_daemon(),name='fedpara-heartbeat')
        watchdog_task=asyncio.create_task(loop_stall_watchdog(shutdown),name='fedpara-loop-watchdog')
        inbound_pipeline.start()
        network_receiver_task=asyncio.create_task(network_receiver_loop(),name='fedpara-network-receiver')
        wal_maintenance_task=asyncio.create_task(wal_maintenance(),name='fedpara-wal-checkpoint')
        while not shutdown.is_set() and (args.rounds == 0 or emitted < args.rounds):
            if generation_compact_vote_task is not None and generation_compact_vote_task.done():
                try:
                    vote= generation_compact_vote_task.result()
                    if generation_compact_control is not None:
                        generation_compact_control.receive('local',{'kind':'compact-vote','data':vote})
                except (PatchError,OSError,ValueError,sqlite3.Error) as exc:
                    log.warning('compact committee verification deferred: %s',exc)
                generation_compact_vote_task=None
            if (generation_compact_vote_task is None and generation_compact_control is not None
                    and getattr(args,'generation_v2_trainer',False)
                    and loop.time()>=generation_compact_vote_next):
                generation_compact_vote_next=loop.time()+.5
                ready=generation_compact_control.ready_for_vote()
                if ready is not None:
                    generation_compact_vote_task=asyncio.create_task(
                        compact_vote_worker(*ready),name='kagami-compact-vote')
            working_saver.check()
            paired_saver.check()
            inbound_pipeline.check()
            if (generation_trainer_ledger if getattr(args,'generation_v2_trainer',False)
                    else generation_ledger) is not None and auto_epoch_service is None:
                auto_epoch_service=make_auto_epoch_service()
                from .epoch_multihop_v56 import MultiEpochCatchup
                auto_epoch_catchup=MultiEpochCatchup(auto_epoch_service,
                    max_hops_per_poll=1,require_prewarmed=True)
            if auto_epoch_service is not None and initial_complete:
                # Prewarm runs in a task without changing ANY active pointers.
                # Main loop alone handles short destructive publication at a
                # safe optimizer boundary (no forward/backward is running).
                if auto_epoch_prepare is not None and auto_epoch_prepare.done():
                    try:auto_epoch_prepare.result()
                    except (FileNotFoundError,ConnectionError,PatchError,RuntimeError,OSError) as exc:
                        log.warning('certified epoch prewarm deferred: %s',exc)
                        auto_epoch_next=loop.time()+15.0
                    auto_epoch_prepare=None
                if (auto_epoch_prepare is None and auto_epoch_service.shadow_ready):
                    try:
                        await auto_epoch_catchup.poll()
                    except BaseException:
                        # If source receivers were fenced already, an uncertain
                        # pointer/transport outcome must stop the trainer.
                        if epoch_writer_frozen or trainer_router.pending_publication:
                            raise
                        log.exception('certified epoch activation deferred')
                        auto_epoch_next=loop.time()+15.0
                if (auto_epoch_prepare is None and not auto_epoch_service.shadow_ready and
                        loop.time()>=auto_epoch_next and
                        auto_epoch_service.ledger.certified_epoch(trainer_router.current.epoch+1) is not None):
                    auto_epoch_next=loop.time()+2.0
                    auto_epoch_prepare=asyncio.create_task(
                        auto_epoch_service.activate_next(prepare_only=True),
                        name='kagami-epoch-prewarm')
            # Update local queue/connection gauges even during network catch-up.
            record_telemetry('training' if initial_complete else 'waiting')
            if network_receiver_task.done(): network_receiver_task.result()
            if room_only and lobby.fatal_error is not None:
                raise PatchError('permanent room signaling rejection: '+str(lobby.fatal_error))
            now=loop.time()
            if now-last_config >= 2.0:
                update_roster();last_config=now
            await start_slots(now)
            frontier_cache=None
            for slot in slots.values():
                node=slot.node
                if node is None or slot.disabled or not node.connected.is_set():continue
                # Signed compact high-water marks allow all old holes to be
                # discovered without waiting for 60+ inventory pages to cycle.
                paused=bool(getattr(args,'swarm_pause_file',None) and
                            Path(args.swarm_pause_file).exists())
                if (now>=node.next_frontier_at and node.verified_at>0 and
                        node.remote_supports_frontier):
                    if frontier_cache is None:
                        frontier_cache=journal.frontiers()
                    page,next_cursor=frontier_advertisement_page(
                        frontier_cache,node.frontier_advertisement_cursor)
                    if page and node.control('frontier',heads=page):
                        node.frontier_advertisement_cursor=next_cursor
                        # At 4096 origins, 48/page and 0.3s/page advertise
                        # the full vector clock in ~26 seconds per neighbor.
                        node.next_frontier_at=now+(
                            .15 if paused else frontier_advertisement_delay(
                                living_count(),catchup=not node.ready))
                    elif not page:
                        node.next_frontier_at=now+2.5
                    else:
                        node.next_frontier_at=now+0.2
                if now >= node.next_inventory_at and now-node.last_inventory>=(
                    min(args.inventory_interval,0.25) if (
                        getattr(args,'swarm_pause_file',None) and
                        Path(args.swarm_pause_file).exists()) else args.inventory_interval):
                    ids,done=journal.inventory(node.inventory_cursor,INVENTORY_PAGE)
                    if node.control('inventory',events=ids,done=done):
                        node.inventory_cursor+=len(ids)
                        node.last_inventory=now
                        if done:
                            node.inventory_cursor=0
                            node.next_inventory_at=now+(
                                min(args.inventory_cycle,.3) if (
                                    getattr(args,'swarm_pause_file',None) and
                                    Path(args.swarm_pause_file).exists()) else
                                inventory_cycle_delay(args.inventory_cycle,living_count(),
                                    peer_synced=node.ready,
                                    has_frontier=node.remote_supports_frontier))
            # Background network_receiver_loop runs even DURING model steps.
            # No receive-side barrier or forced early local seal.
            if epoch_paxos is not None:
                # This probe checks the full production signed control path,
                # including reconnect and durable Paxos promises. It does NOT
                # apply v2 weights or reinterpret v1 updates as v2.
                pause_path=(Path(args.swarm_pause_file) if
                    getattr(args,'swarm_pause_file',None) else None)
                paused=bool(pause_path and pause_path.exists())
                # A pre-join bootstrap barrier is NOT an epoch commit request.
                # The post-training quiesce sentinel is a separate, explicit
                # phase even though it shares the training-pause mechanism.
                epoch_quiesce=False
                if paused and not epoch_probe_started:
                    try:
                        epoch_quiesce=pause_path.read_text(
                            encoding='utf-8').startswith('stop optimization;')
                    except OSError:
                        # The simulator may remove the bootstrap flag between
                        # exists() and read_text(). That is normal, not fatal.
                        pass
                if epoch_quiesce and not epoch_probe_started and initial_complete:
                    epoch_probe_started=True
                    if epoch_paxos.committee[0]==args.peer_id and not epoch_paxos.committed:
                        epoch_paxos.propose(epoch_probe_value,view=0)
                # Self-delivery is mandatory for leader's durable promise/accept.
                for _ in range(32):
                    if not epoch_paxos.outbox:break
                    packet=epoch_paxos.outbox.pop(0)
                    epoch_paxos.receive(packet)
                # Re-transmit immutable, individually signed envelopes over
                # actual WebRTC P2P DataChannels. At most two per link/tick;
                # relays never forge a sender's signature or deliver unchecked
                # model bytes. Every packet is persisted in WAL before relay.
                if now-epoch_last_tx>=0.12:
                    packets=epoch_paxos.messages()
                    for slot in slots.values():
                        node=slot.node
                        if (slot.disabled or node is None or not node.connected.is_set() or
                            node.verified_at==0 or not packets):continue
                        cursor=epoch_peer_cursors.get(slot.peer,0)
                        for _ in range(min(2,len(packets))):
                            packet=packets[cursor%len(packets)]
                            if not node.control('epoch-v2-wire',payload=packet):break
                            cursor+=1
                        epoch_peer_cursors[slot.peer]=cursor
                    epoch_last_tx=now
                if epoch_paxos.committed and not epoch_probe_reported:
                    epoch_probe_reported=True
                    print(f'EPOCH V2 CERTIFIED peer={args.peer_id} '
                          f'digest={epoch_paxos.committed["digest"][:16]} '
                          f'quorum={epoch_paxos.majority} '
                          f'ballot={epoch_paxos.committed["ballot"]}',flush=True)

            if generation_bridge is not None:
                try:
                    generation_bridge.tick(((p,s.node) for p,s in slots.items()
                           if not s.disabled and s.node is not None),now=now)
                except (PatchError,ValueError,OSError) as ex:
                    # A malformed/missing generation packet must never stop
                    # the existing signed v1 learning and gossip loop.
                    log.warning('generation gossip deferred: %s',ex)
            if generation_gc_task is not None and generation_gc_task.done():
                try:
                    removed=generation_gc_task.result()
                    if removed:print(f'VERIFIED EPOCH GC completed={removed}',flush=True)
                except (PatchError,OSError,sqlite3.Error) as ex:
                    log.warning('verified epoch GC deferred: %s',ex)
                generation_gc_task=None
            if (generation_gc_runtime is not None and now >= generation_gc_next
                    and generation_gc_task is None):
                generation_gc_next=now+30.0
                # Constant-cost index lookup when no verified transition is
                # installed: normal learning never needs an archive scan.
                if generation_gc_runtime.db.execute(
                    "SELECT 1 FROM handoffs WHERE state='installed' LIMIT 1").fetchone():
                    generation_gc_task=asyncio.create_task(
                        asyncio.to_thread(generation_gc_background),
                        name='kagami-verified-epoch-gc')
            # Every durably verified participant is an equal snapshot source:
            # the first peer has no special authority after genesis. Broadcast
            # a signed availability transition immediately, so if the original
            # peer leaves just after handing off state, other joiners can use
            # its successor without waiting for a periodic heartbeat.
            source_ready=can_supply_snapshot(room_only=room_only,bootstrap=bootstrap)
            for slot in slots.values():
                node=slot.node
                if node is not None and node.can_serve_snapshot != source_ready:
                    node.can_serve_snapshot=source_ready
                    if source_ready and node.connected.is_set():
                        node.control('heartbeat',serves_snapshot=True)
            # Any verified holder can answer an authenticated snapshot request.
            await service_snapshot_requests(
                slots, source_allowed=source_ready,
                send_snapshot=send_full_snapshot,now=now)
            # Pull completed, authenticated shard snapshots from the separate
            # fedpara-state DataChannel without buffering model tensors in RAM.
            for slot in slots.values():
                link=slot.node.snapshot_link if slot.node is not None else None
                if link is None:continue
                if link.error is not None:
                    # Fail the peer link and reconnect rather than silently
                    # waiting forever after an exception in an aiortc callback.
                    if slot.node is not None and not slot.node.protocol_failure.is_set():
                        slot.node.last_error=ConnectionError(
                            f'snapshot channel error: {type(link.error).__name__}: {link.error}')
                        slot.node.protocol_failure.set()
                        print(f'SNAPSHOT ERROR peer={slot.peer}: '
                              f'{type(link.error).__name__}: {link.error}; reconnecting',flush=True)
                    continue
                while not link.completed.empty():
                    received_snapshots[slot.peer]=link.completed.get_nowait()
                    # Hash verified manifests incrementally, independent of model size.
                    with received_snapshots[slot.peer].open('rb') as manifest_stream:
                        received_hashes[slot.peer]=hashlib.file_digest(
                            manifest_stream,'sha256').hexdigest()
                    print(f'SNAPSHOT DOWNLOADED peer={slot.peer}',flush=True)
            # Snapshot installation and asynchronous applied updates are
            # mutually exclusive. Signed suffix events are replayed once.
            if bootstrap.fresh_join and not bootstrap.snapshot_installed and local_steps:
                await seal_if_needed()
            async with inbound_commit_lock:
                await maybe_install_snapshot()
            # Request/re-request until a verified *full* state is installed.
            # An authenticated DataChannel can outlive Mummy signaling. A
            # cold joiner must still request state over that live channel.
            has_snapshot_source=any(not s.disabled and s.node is not None and
                s.node.connected.is_set() and s.node.verified_at>0
                for s in slots.values())
            from .join_guard import should_request_snapshot
            if should_request_snapshot(
                    fresh_join=bootstrap.fresh_join,
                    snapshot_installed=bootstrap.snapshot_installed,
                    signaling_has_members=bool(lobby is not None and lobby.online and lobby.members),
                    authenticated_transport=has_snapshot_source):
                # One source per joining peer, rather than simultaneously
                # exporting a multi-GB checkpoint from all eight neighbors.
                # Signed source readiness propagates outward along the overlay
                # after each successful bootstrap.
                previous=snapshot_pool.current
                selected=snapshot_pool.select(slots,args.peer_id,now)
                if selected is not None and selected.peer != previous:
                    print(f'SNAPSHOT SOURCE HANDOFF: {previous or "none"} -> '
                          f'{selected.peer} (authenticated, verified holder)',flush=True)
                # One verified source at a time. Abandoned peers cannot block
                # later joiners merely because they once founded the room.
                for slot in ([selected] if selected is not None else []):
                    node=slot.node
                    if node.snapshot_link is None or not node.snapshot_link.connected.is_set():
                        if now-node.connected_since > 25 and not node.protocol_failure.is_set():
                            node.last_error=TimeoutError('full-model DataChannel never opened')
                            node.protocol_failure.set()
                            snapshot_pool.failed(slot.peer,now)
                            print(f'SNAPSHOT ERROR peer={slot.peer}: state channel unavailable; '
                                  f'rotating to another verified holder',flush=True)
                        continue
                    link=node.snapshot_link
                    # A COMPLETED verified transfer may still await old signed
                    # journal events. Do not reset/download the same full model
                    # every 30s while its vector-clock boundary catches up.
                    downloaded=received_snapshots.get(slot.peer)
                    signed_tip=snapshot_tips.get(slot.peer)
                    if (link.session.remote_complete and downloaded is not None and
                        signed_tip is not None and
                        received_hashes.get(slot.peer)==signed_tip[0]):
                        continue
                    busy=link.session.remote_meta is not None and not link.session.remote_complete
                    # This applies to both a stalled manifest and a partially
                    # downloaded shard. Disconnect the broken transport; the
                    # next peer link will reuse already verified disk shards.
                    if node.snapshot_requested_at and not link.session.remote_complete and link.stalled(60.0):
                        node.last_error=TimeoutError(
                            'full-model snapshot transfer had no progress for 60s')
                        node.protocol_failure.set()
                        snapshot_pool.failed(slot.peer,now)
                        print(f'SNAPSHOT STALLED peer={slot.peer}: '
                              f'stage={"shards" if busy else "manifest"} '
                              f'missing={len(link.session.outstanding)}; '
                              f'rotating to another verified holder',flush=True)
                        continue
                    if node.snapshot_remote_stage=='source-not-ready':
                        snapshot_pool.failed(slot.peer,now)
                        print(f'SNAPSHOT SOURCE UNAVAILABLE peer={slot.peer}: '
                              'rotating to another verified holder',flush=True)
                        continue
                    # Retransmit requests if the sender has not acknowledged
                    # them. A 30-second blind retry hid dropped requests.
                    # Acknowledged exporting requests wait longer, but continue
                    # displaying the source stage to distinguish slow export.
                    acked = node.snapshot_remote_stage in ('accepted','exporting',
                                                           'waiting-state-channel','offered')
                    interval = 20.0 if acked else 5.0
                    if not busy and now-node.snapshot_requested_at>=interval:
                        if link.session.remote_complete:
                            link.reset_receiver()
                        if not node.snapshot_request_id:
                            node.snapshot_request_id=secrets.token_hex(8)
                        if node.control('snapshot-request',request_id=node.snapshot_request_id):
                            node.snapshot_requested_at=now
                            print(f'SNAPSHOT REQUEST peer={slot.peer} '
                                  f'id={node.snapshot_request_id} '
                                  f'source_stage={node.snapshot_remote_stage or "no-ack"}',flush=True)
            # Independent compact frontier reconciliation: detect gaps below
            # a peer's advertised highest sequence, not only missing entries
            # seen on an eventually completed inventory page.
            for slot in slots.values():
                node=slot.node
                if node is not None and node.connected.is_set() and not slot.disabled:
                    if now >= node.next_frontier_scan_at:
                        node.reconcile_frontiers(journal, budget=FRONTIER_WINDOW)
                        node.next_frontier_scan_at=now+0.15
            # Repeated, bounded requests recover from lost backpressured need
            # frames and DataChannel reconnects. Missing items remain pending
            # until their SHA-256 verified signed event has been ingested.
            for slot in slots.values():
                node=slot.node
                if node is not None:
                    node.retry_missing(now)
            if controller is not None:
                # Only mutate safe runtime pacing. Patch rank/layout never change.
                args.local_steps=min(getattr(args,'max_sync_steps',64),controller.tune(timeout=args.peer_timeout).local_steps)
            await maybe_checkpoint()
            # Explicit EVAL request: snapshot the CURRENT effective, converged-local
            # weights at a safe step boundary. A snapshot is data-only; external
            # Python source is never imported from another peer. Keep WebRTC up.
            eval_request=getattr(args,'eval_snapshot_request_file',None)
            if eval_request is not None and Path(eval_request).exists():
                request=Path(eval_request)
                status_path=request.with_suffix('.status.json')
                def write_snapshot_status(state, **data):
                    payload={'state':state,'checkpoint':str(args.shard_state_manifest or ''),
                             'timestamp':time.time(),**data}
                    tmp=status_path.with_name(status_path.name+'.tmp')
                    with tmp.open('w',encoding='utf8') as stream:
                        json.dump(payload,stream,ensure_ascii=False)
                        stream.flush();os.fsync(stream.fileno())
                    os.replace(tmp,status_path)
                # The marker is consumed only after the export is tracked.
                # Capture failures independently from network worker health.
                try:
                    write_snapshot_status('saving')
                    request.unlink(missing_ok=True)
                    if not (args.shard_state_manifest and args.shard_store):
                        raise PatchError('no sharded checkpoint output configured')
                    await seal_if_needed()
                    nonlocal_optimizer[0]=optimizer
                    from .shards import export_state_sharded
                    print('EVAL SNAPSHOT START:',args.shard_state_manifest,flush=True)
                    saved=await asyncio.to_thread(export_state_sharded,model,args.shard_store,
                        args.shard_state_manifest,model_id=adapter.model_id,
                        revision=adapter.revision,peer_id=args.peer_id,
                        seq=journal.count(),adapter=adapter,
                        shard_bytes=args.shard_mb*1024**2,
                        max_state_bytes=args.max_state_gb*1024**3)
                    write_snapshot_status('complete',bytes=saved['total_tensor_bytes'],
                                          seq=saved['seq'])
                    print('EVAL SNAPSHOT COMPLETE:',args.shard_state_manifest,flush=True)
                except Exception as exc:
                    # A full disk can prevent the error-status write too;
                    # never crash P2P relay just because EVAL export failed.
                    try:
                        write_snapshot_status('error',error=str(exc)[:1500],
                                              error_type=type(exc).__name__)
                    except OSError:
                        pass
                    print(f'EVAL SNAPSHOT ERROR: {type(exc).__name__}: {exc}',flush=True)
            optimizer=nonlocal_optimizer[0]
            # Wait briefly for catch-up when a known peer is reachable; do NOT
            # wait forever for offline peers (permanent departures).
            catchup=any(s.node and s.node.connected.is_set() and not s.node.ready
                        and time.monotonic()-s.node.connected_since < args.sync_timeout
                        for s in slots.values() if not s.disabled)
            if room_only:
                if lobby.online and not signaling_seen:
                    with journal.db:
                        journal.db.execute(
                            "INSERT OR REPLACE INTO meta(name,value) VALUES('room_confirmed','1')")
                    signaling_seen=True
                live_neighbors = {
                    peer: bool(s.node is not None and not s.disabled and
                               s.node.connected.is_set() and s.node.ready)
                    for peer,s in slots.items() if not s.disabled
                }
                decision = initial_join_decision(
                    initial_complete=initial_complete,
                    lobby_online=lobby.online, lobby_members=set(lobby.members),
                    neighbors=live_neighbors,
                    durable_history=journal.count()>0 or restored_steps>0,
                    signaling_seen=signaling_seen)
                # Crash recovery is NOT a cold join.  A previously verified
                # member reconstructs its committed base by replaying its
                # durable journal (and its local unfinished adapter checkpoint)
                # before entering this loop.  Waiting for *all* neighbors to
                # announce inventory-ready while they keep producing events
                # can deadlock a restarted member of an active swarm.
                # Its missing remote events are reconciled by anti-entropy;
                # the final swarm barrier checks identical inventories and
                # effective weights separately.
                from .join_guard import established_member_decision
                recovered_decision=established_member_decision(
                    persisted_ready=persisted_ready,
                    fresh_join=bootstrap.fresh_join,
                    lobby_online=lobby.online,
                    signaling_seen=signaling_seen)
                if recovered_decision is not None:
                    decision=recovered_decision
                # Preserve the existing emergency-mode guarantee: a known,
                # locally verified peer can keep training without Mummy.
                elif bootstrap.snapshot_installed and not bootstrap.fresh_join and not lobby.online:
                    from .join_guard import JoinDecision
                    decision=JoinDecision(True,'verified-model-emergency-offline')
                # One-time empty-room genesis initialization MUST happen before
                # evaluating the snapshot wait guard. Persist first so an
                # immediate crash/restart does not turn the founder into a
                # newcomer. This is a one-way transition for this process.
                became_founder=bootstrap.confirm_genesis_founder(
                    decision=decision, lobby_online=lobby.online,
                    lobby_members=set(lobby.members),
                    first_roster_was_empty=(lobby.first_roster_was_empty is True and not lobby.room_existing))
                if became_founder:
                    # A peer may have joined after our first authoritative
                    # group-members=[] but before the trainer's first loop.
                    # This is the verified genesis source, not a permanent admin.
                    from .join_guard import JoinDecision
                    decision=JoinDecision(True,'room-founder-from-first-roster')
                    with journal.db:
                        journal.db.execute(
                            "INSERT OR REPLACE INTO meta(name,value) VALUES('initial_ready','1')")
                    print('ROOM GENESIS READY (no permanent admin): baseline is verified; '
                          'new arrivals will download OUR verified snapshot',flush=True)
                if bootstrap.fresh_join and not bootstrap.snapshot_installed:
                    from .join_guard import JoinDecision
                    if lobby.online and lobby.room_existing and not lobby.members:
                        decision=JoinDecision(False,
                            'room-has-persisted-metadata-but-no-verified-model-source; '
                            'wait for a former participant to return')
                    if lobby.first_roster_was_empty is False and not lobby.members:
                        # The only model sources went offline before we got a
                        # verified snapshot. An empty CURRENT roster does not
                        # make this new participant the genesis owner.
                        decision=JoinDecision(False,
                            'waiting-for-original-model-source: prior peers disconnected; '
                            'do not restart from genesis')
                    elif lobby.members:
                        decision=JoinDecision(False,'waiting-for-verified-full-model-snapshot')
                if decision.can_train and lobby.online and not initial_complete:
                    initial_complete=True
                    if not bootstrap.fresh_join or not lobby.members or bootstrap.snapshot_installed:
                        with journal.db:
                            journal.db.execute(
                                "INSERT OR REPLACE INTO meta(name,value) VALUES('initial_ready','1')")
                    marker=('REJOIN READY' if persisted_ready else 'JOIN READY')
                    print(f'{marker}: {decision.reason}; synced_events={journal.count()} '
                          f'living={living_count()} active={active_count()} '
                          f'peer={args.peer_id}',flush=True)
                if not decision.can_train:
                    if now-last_join_report>3.0:
                        failures={peer:{'error':str(slot.node.last_error)[:120] if slot.node.last_error is not None else '',
                                        'phase':slot.node.phase,'role':slot.node.role,
                                        'ice':slot.node.ice_state,'rtc':slot.node.rtc_state,
                                        'since_s':round(now-slot.started_at,1)}
                                  for peer,slot in slots.items()
                                  if slot.node is not None}
                        # Make a stuck snapshot transfer actionable without
                        # printing private keys, room credentials or weights.
                        transfers={}
                        if bootstrap.fresh_join:
                            for peer,slot in slots.items():
                                node=slot.node
                                link=node.snapshot_link if node else None
                                if link is None: continue
                                transfers[peer]={
                                    'state_channel':link.connected.is_set(),
                                    'tip_received':peer in snapshot_tips,
                                    'manifest_received':link.session.remote_meta is not None,
                                    'snapshot_downloaded':peer in received_snapshots,
                                    'requested':node.snapshot_requested_at>0,
                                    'source_stage':node.snapshot_remote_stage or 'no-ack',
                                    'request_age_s':round(max(0,now-node.snapshot_requested_at),1) if node.snapshot_requested_at else None,
                                    'missing_shards':len(link.session.outstanding),
                                    'error':str(link.error)[:160] if link.error else ''}
                        print(f'JOIN WAIT: {decision.reason}; '
                              f'lobby_online={lobby.online} '
                              f'living={living_count()} active={active_count()} '
                              f'journal_events={journal.count()} '
                              f'neighbors={len(live_neighbors)} '
                              f'peer_errors={failures} '
                              f'snapshot_progress={transfers}',flush=True)
                        last_join_report=now
                    record_telemetry('waiting')
                    await asyncio.sleep(.1)
                    continue
            # Swarm E2E barrier: seal in-flight local updates once, but keep
            # signed WebRTC heartbeats, gossip, snapshot service and inventory
            # reconciliation running. This is NOT a change to normal training.
            pause_file=getattr(args,'swarm_pause_file',None)
            if pause_file is not None and Path(pause_file).exists():
                if local_steps:
                    await seal_if_needed()
                    nonlocal_optimizer[0]=optimizer
                if now-last_swarm_report>=5.0:
                    pending_total=sum(len(s.node.pending) for s in slots.values() if s.node)
                    congested=sum(s.node.backpressure_drops for s in slots.values() if s.node)
                    queued=sum(s.node.outgoing.qsize() for s in slots.values() if s.node)
                    demands=sum(len(s.node.requested_events) for s in slots.values() if s.node)
                    responses=sum(s.node.request_responses_sent for s in slots.values() if s.node)
                    frontiers=sum(s.node.frontier_scanned for s in slots.values() if s.node)
                    needs=sum(s.node.need_received for s in slots.values() if s.node)
                    print(f'SWARM CATCHUP peer={args.peer_id} events={journal.count()} '
                          f'pending={pending_total} queued={queued} demand={demands} '
                          f'demand_sent={responses} need_rx={needs} frontier_scanned={frontiers} '
                          f'backpressure_drops={congested} active={active_count()}',flush=True)
                    last_swarm_report=now
                # The swarm convergence phase is NOT a passive wait: increase
                # opportunity to process incoming signed updates and inventory.
                record_telemetry('paused')
                await asyncio.sleep(.02)
                continue
            # Bound generated load only in the explicit swarm test. The
            # production train-live path has no cap by default.
            swarm_cap=getattr(args,'swarm_max_events',0)
            if swarm_cap > 0 and emitted >= swarm_cap:
                if local_steps:
                    await seal_if_needed()
                    nonlocal_optimizer[0]=optimizer
                await asyncio.sleep(.02)
                continue
            if now-first_boot<args.boot_grace or catchup:
                await asyncio.sleep(.03)
                continue
            # Snapshot/housekeeping may have sealed a local block during this
            # iteration. Never run another step after reaching the event cap.
            if args.rounds and emitted >= args.rounds:
                break
            if trajectory_gossip is not None:
                # Each receiver chooses its own CURRENT target Actor and
                # independently evaluates the signed sample. No sender Critic
                # gradient or parameter delta is accepted in this mode.
                async with critic_step_lock:
                    with adapter.lock:
                        from .cli import input_device
                        trajectory_gossip.apply_pending(model,input_device(model,args.device),limit=1)
            if critic_gossip is not None:
                # Prevent network ingestion between the local Critic optimizer
                # step and durable commit of its signed delta.
                # Fetch slow HF streaming rows OUTSIDE the Critic publication
                # fence; authenticated value traffic must remain responsive
                # while the remote data provider is slow.
                def _fetch_v77_batch():
                    try:
                        return next(batch_iter)
                    except StopIteration as exc:
                        raise RuntimeError('training dataset exhausted before distributed Critic step') from exc
                batch=await asyncio.to_thread(_fetch_v77_batch)
                async with critic_step_lock:
                    loss=await run_step_offloop(run_step,model,adapter,optimizer,batch_iter,args,
                                                supplied_batch=batch)
                    before=getattr(args,'_v77_critic_before',None)
                    args._v77_critic_before=None
                    if trajectory_gossip is not None:
                        sample=getattr(args,'_v80_rollout_pending',None)
                        args._v80_rollout_pending=None
                        with adapter.lock:
                            if sample is not None:
                                from .localprogress import _tip
                                rollout,checkpoint=sample
                                # Fail locally before making a signed claim:
                                # full forward on the frozen source snapshot must
                                # reproduce all observed sampling log-probs.
                                from .behavior_checkpoint_v82 import replay_logps
                                from .cli import input_device
                                replay_logps(model,checkpoint,rollout,args,input_device(model,args.device))
                                signed=trajectory_gossip.local(rollout,checkpoint=checkpoint,actor_tip=_tip(journal))
                            else:
                                trajectory_gossip.persist()  # repeated PPO step
                        if sample is not None:
                            for ps in slots.values():
                                nd=ps.node
                                if nd is not None and nd.remote_supports_rollout and nd.verified_at and not nd.bridge_only:
                                    for frame in signed:
                                        if nd.outgoing.full():break
                                        nd.outgoing.put_nowait(frame)
                            # Remaining pages are recovered via durable v81
                            # trajectory frontier/need instead of unbounded queues.
                        # v81 substitutes recipient-side Critic training for
                        # weight delta gossip (avoid duplicated experience).
                        before=None
                    if before is not None:
                        from .localprogress import _tip
                        with adapter.lock:
                            critic_wire=critic_gossip.record_local(before,actor_tip=_tip(journal))
                        for ps in slots.values():
                            nd=ps.node
                            if (nd is not None and nd.remote_supports_critic and
                                    nd.verified_at and not nd.outgoing.full() and not nd.bridge_only):
                                nd.outgoing.put_nowait(critic_wire)
            else:
                loss=await run_step_offloop(run_step,model,adapter,optimizer,batch_iter,args)
            local_steps+=1; pending[0]+=1; recent_loss=loss
            total_steps+=1
            record_telemetry()
            if pending[0]>=args.grad_accum:
                def advance_optimizer():
                    with adapter.lock:
                        optimizer.step();optimizer.zero_grad(set_to_none=True)
                await asyncio.to_thread(advance_optimizer)
                pending[0]=0
            # Durable local-only adapter+AdamW state: an abrupt process death
            # must not reset every unsealed optimizer step to genesis.
            publish_target=(adaptive_publish_steps(args.local_steps,living_count(),
                            max_steps=getattr(args,'max_sync_steps',64))
                            if getattr(args,'adaptive_sync',True) else args.local_steps)
            if (pending[0] == 0 and local_steps < publish_target
                and getattr(args,'durable_every',4)
                and local_steps % getattr(args,'durable_every',4) == 0):
                # Preserve sqlite3's same-thread contract: capture the journal
                # tip on-loop and serialize safetensors/AdamW buffers off-loop.
                from .localprogress import _tip
                save_started=time.monotonic()
                async with inbound_commit_lock:
                    tip=_tip(journal)
                    mixer=getattr(args,'_v65_mixer',None)
                    if _v75_paired_active and mixer is not None and mixer.critic is not None:
                        await paired_saver.checkpoint(journal,adapter,optimizer,mixer,
                                                     local_steps,journal_tip=tip)
                    else:
                        await working_saver.checkpoint(journal,adapter,optimizer,
                                                       local_steps,journal_tip=tip)
                last_working_save_ms=(time.monotonic()-save_started)*1000
            if local_steps>=publish_target:
                await seal_if_needed()
                nonlocal_optimizer[0]=optimizer
                await maybe_checkpoint()
                await maybe_propose_compact_epoch()
            if local_steps==1 or local_steps%args.log_steps==0:
                channels=sum(bool(s.node and s.node.connected.is_set())
                             for s in slots.values() if not s.disabled)
                authed=sum(bool(s.node and s.node.verified_at)
                           for s in slots.values() if not s.disabled)
                print(f'elastic steps={local_steps} emitted={emitted} loss={recent_loss:.5g} '
                      f'active={active_count()} living={living_count()} '
                      f'async_inbound={inbound_pipeline.queue.qsize()} applied={inbound_pipeline.applied} '
                      f'channels={channels} authed={authed} synced={synced_count()} '
                      f'peer={args.peer_id}',flush=True)
            await asyncio.sleep(0)
        if local_steps:
            await seal_if_needed()
        record_telemetry('done',force=True)
        # Stop network submissions before draining pending signed updates,
        # otherwise busy live peers can refill the queue forever.
        network_receiver_task.cancel()
        await asyncio.gather(network_receiver_task,return_exceptions=True)
        await inbound_pipeline.drain()
        await working_saver.drain()
        # Clean shutdown: leave a current optimizer snapshot even if the
        # performance-oriented periodic working-save interval did not hit.
        from .localprogress import _tip
        async with inbound_commit_lock:
            final_tip=_tip(journal)
            mixer=getattr(args,'_v65_mixer',None)
            if _v75_paired_active and mixer is not None and mixer.critic is not None:
                await paired_saver.checkpoint(journal,adapter,optimizer,mixer,0,
                                              journal_tip=final_tip,force=True)
                await paired_saver.drain()
            else:
                await asyncio.to_thread(save_working,journal,adapter,optimizer,0,
                                        journal_tip=final_tip)
        if args.shard_state_manifest and args.shard_store:
            from .shards import export_state_sharded
            print(f'CHECKPOINT SAVE START: manifest={args.shard_state_manifest} '
                  f'objects={args.shard_store}',flush=True)
            saved=await asyncio.to_thread(
                export_state_sharded,model,args.shard_store,args.shard_state_manifest,
                model_id=adapter.model_id,revision=adapter.revision,peer_id=args.peer_id,
                seq=journal.count(),adapter=adapter,shard_bytes=args.shard_mb*1024**2,
                max_state_bytes=args.max_state_gb*1024**3)
            print(f'CHECKPOINT SAVE COMPLETE: {args.shard_state_manifest} '
                  f'bytes={saved["total_tensor_bytes"]}',flush=True)
            # A clean shutdown can produce a generation boundary WITHOUT
            # retaining every signed event body in the NEXT room. The
            # descriptor does not authorize deleting this room's log yet.
            if signer is not None and room_only:
                from .generation_rollover import sign_live_checkpoint
                descriptor=sign_live_checkpoint(journal,args.shard_state_manifest,
                                                signer=signer)
                print(f'KAGAMI GENERATION CHECKPOINT SIGNED: {descriptor}',flush=True)
        if args.full_state_output:
            from .state import export_state
            export_state(model,args.full_state_output,model_id=adapter.model_id,
                revision=adapter.revision,peer_id=args.peer_id,seq=journal.count(),
                adapter=adapter,max_bytes=args.max_state_gb*1024**3)
    finally:
        # A cancelled to_thread wrapper does not stop its OS worker. Drain the
        # exporter before releasing journals, signer and shard-store resources.
        if compact_export_task is not None:
            try:
                await asyncio.shield(compact_export_task)
                # Publish an already completed signed candidate while the
                # journal and compact ledger are still open on clean exit.
                await maybe_propose_compact_epoch()
            except Exception as exc:
                log.warning('outstanding background export failed on shutdown: %s',exc)
        if auto_epoch_prepare is not None:
            auto_epoch_prepare.cancel()
            await asyncio.gather(auto_epoch_prepare,return_exceptions=True)
        if auto_epoch_service is not None:
            await auto_epoch_service.close()
        try:
            await paired_saver.drain()
        except Exception as exc:
            log.error('paired Actor/Critic snapshot failed on shutdown: %s',exc)
        local_critic=getattr(args,'_v65_mixer',None)
        if local_critic is not None:
            try:
                await asyncio.to_thread(local_critic.close)
            except Exception as exc:
                log.error('PPO critic state could not be saved on shutdown: %s',exc)
        # Keep any in-flight filesystem commit alive even if the enclosing
        # coroutine is cancelled (thread cancellation cannot undo fsync).
        await working_saver.drain()
        telemetry.close()
        shutdown.set()
        if wal_maintenance_task is not None:
            await asyncio.gather(wal_maintenance_task,return_exceptions=True)
        if network_receiver_task is not None:
            network_receiver_task.cancel()
            await asyncio.gather(network_receiver_task,return_exceptions=True)
        await inbound_pipeline.stop()
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task,return_exceptions=True)
        if watchdog_task is not None:
            watchdog_task.cancel()
            await asyncio.gather(watchdog_task,return_exceptions=True)
        if room_only:
            room_stop.set()
            lobby_task.cancel()
            await asyncio.gather(lobby_task,return_exceptions=True)
        await asyncio.gather(*(stop_slot(slot) for slot in tuple(slots.values())),return_exceptions=True)
        # Allow bounded RTC generation cleanup on shutdown while other peers
        # stop concurrently, not serially 15s per channel.
        retired=[slot.retiring for slot in slots.values() if slot.retiring is not None]
        if retired:
            await asyncio.gather(*retired,return_exceptions=True)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try: loop.remove_signal_handler(sig)
            except (RuntimeError, NotImplementedError): pass
        for sig, handler in previous_signals.items():
            try: signal.signal(sig,handler)
            except ValueError: pass
        if epoch_paxos is not None:epoch_paxos.close()
        if generation_gc_task is not None:
            await asyncio.gather(generation_gc_task,return_exceptions=True)
        if generation_gc_runtime is not None:generation_gc_runtime.close()
        if generation_ledger is not None:generation_ledger.close()
        if generation_compact_control is not None:generation_compact_control.close()
        if generation_compact_ledger is not None:generation_compact_ledger.close()
        if carry_relay is not None:carry_relay.close()
        if trajectory_gossip is not None:trajectory_gossip.close()
        if critic_gossip is not None:critic_gossip.close()
        journal.close()
        if hasattr(optimizer,'close'):optimizer.close()
        # An abandoned shadow should not hold an SSD optimizer allocation.
        for prepared in auto_epoch_prepared.values():
            shadow_optimizer=prepared.generation.optimizer
            if shadow_optimizer is not optimizer and hasattr(shadow_optimizer,'close'):
                shadow_optimizer.close()
        if carry_archive is not None:
            carry_archive.close()
