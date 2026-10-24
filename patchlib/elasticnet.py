"""Signed elastic anti-entropy over Mummy signaling + aiortc DataChannels."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
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
FRONTIER_ORIGIN_LIMIT = 96
FRONTIER_PENDING_LIMIT = 4096


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


def can_supply_snapshot(*, room_only: bool, bootstrap) -> bool:
    """Verified state, never creator identity, confers snapshot-serving rights."""
    return not room_only or (bootstrap.snapshot_installed and not bootstrap.fresh_join)


def encode_control(msg: dict, signer=None) -> str:
    body = {'v':CTRL_VERSION, **msg}
    raw = _canonical_json(body)
    header = {'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
              'sig':signer(len(raw),hashlib.sha256(raw).hexdigest()) if signer else None}
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
                 snapshot_recv=None, snapshot_limit=32*1024**3, **kwargs):
        super().__init__(*args,**kwargs)
        self.origin,self.remote,self.genesis,self.unsafe=origin,remote,genesis,unsafe
        self.expert_bridge=None
        self.public_key=public_key
        self.on_authenticated=on_authenticated
        self.hello_sent=False
        self.hello_received=False
        self.outgoing = asyncio.Queue(maxsize=3)
        self.controls = asyncio.Queue(maxsize=128)
        self.counter = 0
        self.remote_counter = -1
        self.ready = False
        self.verified_at = 0.0
        self.pending: set[tuple[str,int]] = set()
        self.need_cursor = 0  # prevent an undeliverable low-ID prefix starving the rest
        self.backpressure_drops = 0  # only retryable traffic is discarded
        self.inventory_done = False
        self.inventory_cursor = 0
        self.last_inventory = 0.0
        self.next_inventory_at = 0.0
        self.next_need_retry_at = 0.0
        self.next_frontier_at = 0.0
        self.remote_frontiers: dict[str, int] = {}
        self.frontier_cursors: dict[str, int] = {}
        self.frontier_origin_cursor = 0
        self.next_frontier_scan_at = 0.0
        self.frontier_scanned = 0
        self.need_received = 0
        # Demand-driven anti-entropy responses have precedence over redundant
        # broadcast fanout. Keep only IDs here; payloads remain in SQLite.
        self.requested_events = OrderedDict()
        self.lookup_requested_event = None
        self.request_backlog_overflow = 0
        self.request_responses_sent = 0
        self.connected_since = 0.0
        # Signed availability advertisement: a cold joiner must not request
        # a full model from ANOTHER cold joiner, which has nothing to serve.
        self.can_serve_snapshot = False
        self.remote_can_serve_snapshot = False
        self.remote_supports_frontier = False
        self.snapshot_link = None
        self.snapshot_tip = None
        # At most one signed full-state offer is in flight per RTC link.
        # A critical manifest boundary must never be discarded merely because
        # the slow model-training reader filled the normal control queue.
        self.snapshot_tip_latch = None
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
        if kind != 'hello' and not self.hello_sent:
            if not self.control('hello',serves_snapshot=self.can_serve_snapshot, frontier_v1=True, **({'pub':self.public_key} if self.public_key else {})):
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
        """Eventually request every absent journal event despite SCTP backpressure.

        A failed control send never clears pending. The next call retries.
        """
        if not self.connected.is_set() or not self.pending or now < self.next_need_retry_at:
            return False
        ordered=sorted(self.pending)
        offset=self.need_cursor % len(ordered)
        # Wrap the page so that even a persistently blocked first event cannot
        # prevent later signed updates from being requested.
        page=(ordered if len(ordered)<=REQUEST_PAGE else
              ordered[offset:offset+REQUEST_PAGE] +
              ordered[:max(0, offset+REQUEST_PAGE-len(ordered))])
        success=self.control('need',events=[list(item) for item in page])
        if success:
            self.need_cursor=(offset+len(page)) % len(ordered)
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
        try:
            while True:
                data=None
                demand_selected=False
                if (self.requested_events and self.lookup_requested_event is not None
                        and (priority_streak < 8 or self.outgoing.empty())):
                    ident,_=self.requested_events.popitem(last=False)
                    data=self.lookup_requested_event(*ident)
                    if data is None:
                        continue  # upstream has not received it: requester retries
                    priority_streak += 1
                    demand_selected=True
                else:
                    try:
                        data=await asyncio.wait_for(self.outgoing.get(),timeout=.10)
                        priority_streak=0
                    except asyncio.TimeoutError:
                        continue
                self.sent.clear()
                # A permanently wedged SCTP send buffer must not keep this
                # authenticated-looking link unable to respond indefinitely.
                timeout=max(60.0, min(300.0, 60.0 + len(data)/(256*1024)))
                await asyncio.wait_for(
                    transmit(self._channel,data,max_bytes=self.max_bytes,signer=self.signer),
                    timeout=timeout)
                if demand_selected:
                    self.request_responses_sent += 1
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
        self.link_closed.clear()
        opened_once=False
        # aiortc may deliver an incoming signed hello before the local open
        # callback fires. Initialize per-channel auth HERE, never in on_open.
        self.hello_sent=False
        self.hello_received=False
        self.verified_at=0.0
        self.snapshot_tip_latch=None
        self.latest_snapshot_tip_counter=-1
        self.remote_can_serve_snapshot=False
        self.remote_supports_frontier=False
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
            if not self.control('hello',serves_snapshot=self.can_serve_snapshot, frontier_v1=True, **({'pub':self.public_key} if self.public_key else {})):
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
                    if decoded.get('genesis') != self.genesis:
                        raise HandshakeError('GENESIS MISMATCH: peer is training a different model/revision/rank/checkpoint; do not merge')
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
                        self.remote_supports_frontier = decoded.get('frontier_v1') is True
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
                            self.remote_frontiers[origin]=max(
                                seq,self.remote_frontiers.get(origin,0))
                        return
                    if decoded.get('type')=='snapshot-tip' and self.controls.full():
                        # Snapshot boundaries are non-droppable: when the
                        # trainer falls behind, coalesce the latest signed
                        # tip separately rather than tearing down good RTC.
                        # The ordinary non-full path remains FIFO-compatible.
                        self.snapshot_tip_latch=decoded
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
                        if decoded.get('type') in ('heartbeat','inventory','hello-ack','hello','snapshot-request','snapshot-status','epoch-v2-wire'):
                            self.backpressure_drops+=1
                            return
                        # Snapshot controls are NOT silently dropped. A fresh
                        # snapshot request / status must be retried instead.
                        raise PatchError('critical control queue saturated')
                    self.controls.put_nowait(decoded)
                else:
                    if not self.hello_received:
                        raise PatchError('data received before authenticated hello')
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


async def run_step_offloop(run_step, model, adapter, optimizer, batch_iter, args):
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
            batch=next(batch_iter)
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
    if not args.state_base: raise ValueError('elastic needs immutable --state-base as common genesis')
    if args.grad_accum<1 or args.local_steps<1 or args.rounds<0 or getattr(args,'durable_every',1)<0:
        raise ValueError('bad training step settings')
    if (args.max_journal_gb < 1 or args.peer_timeout <= args.heartbeat_interval or
        args.sync_timeout < 0 or args.inventory_interval <= 0 or
        args.inventory_cycle <= 0 or args.checkpoint_every < 0):
        raise ValueError('invalid elastic liveness, journal or checkpoint settings')
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
        args.local_steps=policy.local_steps
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
    key_bytes={}
    registry_path=Path(str(args.journal)+'.trusted.json')
    if registry_path.is_file():
        registry=json.loads(registry_path.read_text(encoding='utf-8'))
        if not isinstance(registry,dict) or len(registry)>257:
            raise PatchError('corrupt pinned identity registry')
        for peer,encoded in registry.items():
            if not PEER_ID.fullmatch(peer) or not isinstance(encoded,str):
                raise PatchError('bad archived public key')
            data=base64.b64decode(encoded,validate=True)
            # Obtain verifier from bytes without trusting a temp file.
            from cryptography.hazmat.primitives.serialization import load_pem_public_key
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            from cryptography.exceptions import InvalidSignature
            from .identity import _message
            pub=load_pem_public_key(data)
            if not isinstance(pub,Ed25519PublicKey):raise PatchError('non-Ed25519 key in archived registry')
            def make_verifier(k):
                def verify(msg):
                    try:
                        signature=base64.b64decode(msg['sig'],validate=True)
                        k.verify(signature,_message(msg['size'],msg['sha256']))
                    except (ValueError,TypeError,KeyError,InvalidSignature) as exc:
                        raise PatchError('invalid archived origin signature') from exc
                return verify
            trusted[peer]=make_verifier(pub)
            key_bytes[peer]=data

    def store_pin(peer, data, verifier):
        if peer in key_bytes and key_bytes[peer]!=data:
            raise PatchError(f'public key for {peer} changed; manual key migration required')
        if peer not in key_bytes:
            key_bytes[peer]=data
            trusted[peer]=verifier
            registry_path.parent.mkdir(parents=True,exist_ok=True)
            tmp=registry_path.with_name(registry_path.name+'.tmp')
            fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
            try:
                with os.fdopen(fd,'w',encoding='utf-8') as fh:
                    json.dump({p:base64.b64encode(b).decode('ascii') for p,b in key_bytes.items()},fh)
                    fh.flush();os.fsync(fh.fileno())
                os.replace(tmp,registry_path)
            finally:
                tmp.unlink(missing_ok=True)
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
        genesis_input['checkpoint_sha256']=semantic_state_sha256(Path(args.state_base))
        genesis_input['room_fingerprint']=hashlib.sha256(args.room_id.encode('ascii')).hexdigest()
    # Live elastic uses ONLY signed before/after effective delta events.
    # Legacy reset-v1 peers/journals must never join this new genesis.
    genesis_input['optimizer_protocol']='continual-elastic-fedsum-relora-v4'
    genesis_input['reinit_every']=getattr(args,'reinit_every',20)
    # Preserve existing AdamW genesis compatibility. LoRA-Muon journals must
    # not collide with AdamW journals because optimizer moments are different.
    if getattr(args,'optimizer','adamw')=='lora-muon':
        genesis_input['optimizer_name']='lora-muon'
        genesis_input['muon_momentum']=getattr(args,'muon_momentum',.9)
        genesis_input['muon_weight_decay']=getattr(args,'muon_weight_decay',0.)
    genesis=hashlib.sha256(_canonical_json(genesis_input)).hexdigest()
    from .snapshot_live import journal_tip, snapshot_inventory
    snapshot_root=Path(str(args.journal)+'.snapshots')
    snapshot_store=snapshot_root/'objects'
    snapshot_recv=snapshot_root/'incoming'
    journal=EventJournal(args.journal,genesis,max_bytes=args.max_journal_gb*1024**3)
    from .continual import ContinualElasticEngine
    # factor roundoff would leave nonzero residual between anchor and local.
    args.wire_dtype='float32'
    engine=ContinualElasticEngine(adapter,args.peer_id,genesis,journal,
                                  reinit_every=getattr(args,'reinit_every',20),signer=signer,trusted=trusted,
                                  unsafe=args.unsafe_unsigned,max_delta_rms=args.max_delta_rms,
                                  event_public=room_pub if room_only else None)
    print(f'elastic optimization=FedSum+ReLoRA-v4 reinit_every={engine.reinit_every} local events',flush=True)
    engine.replay()
    print(f'elastic peer={args.peer_id} genesis={genesis} recovered={journal.count()} events; log={args.journal}',flush=True)
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
        base=semantic_state_sha256(Path(args.state_base))
        epoch_id=hashlib.sha256(_canonical_json(['fedpara-epoch-probe-v1',genesis,base,sorted(committee)])).hexdigest()
        epoch_probe_value=make_value(epoch_id,base,tuple(sorted(committee)),[])
        epoch_paxos=EpochPaxos(Path(str(args.journal)+'.epoch-v2.sqlite3'),
                               peer=args.peer_id,committee=committee,epoch=epoch_id,base=base,
                               signer=signer,trusted=trusted)
        print(f'EPOCH V2 PROBE ARMED peer={args.peer_id} committee={len(committee)} '
              f'quorum={epoch_paxos.majority} epoch={epoch_id[:16]} '
              f'(no model mutation)',flush=True)
    slots:dict[str,PeerSlot]={}
    room_stop=asyncio.Event() if room_only else None
    lobby=RoomDiscovery(args.ws,args.room_id,args.peer_id,args.token,
                         manifest=room_manifest,backup_urls=getattr(args,'ws_backup',[]),
                         publish=getattr(args,'room_public',False)) if room_only else None
    lobby_task=asyncio.create_task(lobby.run(room_stop)) if room_only else None
    def new_optimizer():
        from .lora_muon import make_adapter_optimizer
        return make_adapter_optimizer(adapter,optimizer=getattr(args,'optimizer','adamw'),
            lr=args.lr,weight_decay=(getattr(args,'muon_weight_decay',0.)
              if getattr(args,'optimizer','adamw')=='lora-muon' else args.weight_decay),
            muon_momentum=getattr(args,'muon_momentum',.9))
    optimizer=new_optimizer()
    optimizer.zero_grad(set_to_none=True)
    from .localprogress import restore_working, save_working, snapshot_path
    restored_steps=restore_working(journal,adapter,optimizer)
    if restored_steps:
        print(f'elastic restored unfinished local block: {restored_steps} steps and AdamW state',flush=True)
    batch_iter=training_batches(args)
    loop=asyncio.get_running_loop()
    local_steps=restored_steps
    emitted=journal.next_seq(args.peer_id)-1  # do not display a false zero after restart
    recent_loss=float('nan')
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
    last_snapshot_report=0.0
    snapshot_pool=SnapshotProviderPool()
    presence_hysteresis=PresenceHysteresis()

    def update_roster():
        desired=dict(roster)
        if room_only:
            # Symmetric bounded small-world overlay. Degree <= 8 even for
            # 96 participants; additional power-of-two chords reduce gossip
            # hop count versus the former ±1/±2 ring.
            # Never drop already-active links when signaling goes down.
            from .topology import neighbors
            # A signaling timeout can briefly remove a peer while its signed
            # RTC DataChannel is still alive. Debounce the WHOLE roster (not
            # individual edges) to avoid reconnect storms and >8 neighbors.
            members=presence_hysteresis.observe(
                set(lobby.members),time.monotonic(),online=lobby.online) | {args.peer_id}
            close_peers=neighbors(args.peer_id,members,
                max_degree=getattr(args,'max_neighbors',8))
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
            if room_only and lobby.online and slot.disabled and slot.task is None:
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
            # In ring gossip topology only up to four neighbors connect
            # directly, so neighbor count is NOT the FedAvg-style divisor.
            if lobby.online:return 1+len(lobby.members)
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
        for slot in slots.values():
            if slot.peer == exclude_peer or slot.peer == origin_peer:
                # No need to echo an event to its authenticated sender or
                # back to its original creator. This cuts redundant traffic
                # in a densely connected 8-peer swarm.
                continue
            node=slot.node
            if node and node.connected.is_set() and not slot.disabled and not node.outgoing.full():
                await node.outgoing.put(wire)
        # A full queue doesn't lose data: journal inventory announces it later.

    async def seal_if_needed():
        nonlocal local_steps,emitted,optimizer
        if not local_steps: return
        # Only unfinished accumulation needs one final optimizer step.
        if pending[0]:
            def finish_optimizer_step():
                with adapter.lock:
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
            await asyncio.to_thread(finish_optimizer_step)
            pending[0]=0
        wire=await engine.seal_local_responsive(1,dtype=args.wire_dtype,
                                                max_wire_bytes=args.max_patch_mb*1024**2)
        emitted+=1;local_steps=0
        if engine.should_reinitialize(emitted):
            # _apply_checked committed delta and deterministically refreshed
            # local factors+anchors as part of this durable signed event.
            # AdamW moments refer to the OLD factor coordinates: drop them.
            optimizer=new_optimizer()
            optimizer.zero_grad(set_to_none=True)
            print(f'RELORA REFRESH event={emitted} factors=reset optimizer=reset '
                  f'base=preserved',flush=True)
        nonlocal_optimizer[0]=optimizer
        # Save after the signed event is durable, with the NEW moments on refresh.
        from .localprogress import _tip
        tip=_tip(journal)
        await asyncio.to_thread(save_working,journal,adapter,optimizer,0,journal_tip=tip)
        await fanout(wire)
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
        if slot.task:
            slot.task.cancel()
            await asyncio.gather(slot.task,return_exceptions=True)
        slot.task=None;slot.node=None

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
                    snapshot_recv=snapshot_recv if room_only else None,
                    snapshot_limit=args.max_state_gb*1024**3)
                node.lookup_requested_event=journal.get
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
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if not node.control('snapshot-tip',manifest_sha256=digest,events=count,
                            inventory_sha256=tip,cutoffs=cutoffs):
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
        for slot in list(slots.values()):
            node=slot.node
            if node is None or slot.disabled: continue
            for _ in range(32):
                if node.snapshot_tip_latch is not None:
                    msg=node.snapshot_tip_latch
                    node.snapshot_tip_latch=None
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
                    if not isinstance(entries,list) or len(entries)>INVENTORY_PAGE:
                        raise PatchError('malformed inventory')
                    missing=[]
                    for item in entries:
                        if not (isinstance(item,list) and len(item)==3 and
                                isinstance(item[0],str) and PEER_ID.fullmatch(item[0]) and
                                type(item[1]) is int and 1<=item[1]<2**63 and
                                isinstance(item[2],str) and len(item[2])==64):
                            raise PatchError('malformed event identity in inventory')
                        original=journal.has(item[0],item[1])
                        if original and original[0]!=item[2]:
                            raise PatchError('conflicting event in inventory')
                        if original is None and (item[0],item[1]) not in node.pending:
                            missing.append([item[0],item[1]])
                            node.pending.add((item[0],item[1]))
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
                    # Validate bounded vector clock before allocating/processing.
                    snapshot_inventory(journal,cutoffs)
                    snapshot_tips[slot.peer]=(digest,tip,count,cutoffs)
                    node.latest_snapshot_tip_counter=msg['counter']
                    print(f'SNAPSHOT TIP peer={slot.peer} events={count}',flush=True)
                elif typ in ('hello-ack','heartbeat'):
                    if controller is not None:
                        controller.observe(slot.peer,msg.get('cap'))
                else:
                    raise PatchError('unsupported elastic control')
            for _ in range(16):
                if node.incoming.empty():break
                wire=node.incoming.get_nowait()
                if not isinstance(wire,bytes):raise PatchError('nonbinary elastic event')
                # Preserve local work against current base before accepting an
                # external update. Replacing factors without sealing loses steps.
                from .elastic import unpack_event
                origin,meta,_,_=unpack_event(wire,trusted,unsafe=args.unsafe_unsigned)
                old=journal.has(origin,meta['seq'])
                if old:
                    await engine.ingest_responsive(wire)  # verify retransmission off-loop
                    node.pending.discard((origin,meta['seq']))
                    if node.inventory_done and not node.pending: node.ready=True
                    continue
                if origin!=args.peer_id and local_steps:
                    await seal_if_needed()
                try:
                    if await engine.ingest_responsive(wire):
                        # Anchor removes previously committed local factors,
                        # so inbound deltas must NOT erase optimizer moments.
                        from .localprogress import _tip
                        await asyncio.to_thread(save_working,journal,adapter,
                            nonlocal_optimizer[0],0,journal_tip=_tip(journal))
                        await fanout(wire, exclude_peer=slot.peer, origin_peer=origin)
                        # Relay immutable origin-signed event to OTHER neighbors
                finally:
                    # Any one source can fill a need requested from another relay.
                    for ps in slots.values():
                        if ps.node is not None:
                            ps.node.pending.discard((origin,meta['seq']))
                            if ps.node.inventory_done and not ps.node.pending:
                                ps.node.ready=True

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
            current_digest,current_count,_=snapshot_inventory(journal,cutoffs)
            if (current_digest,current_count)!=(events_digest,event_count):
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
            assembled=snapshot_root/f'joined-{expected_sha}.fstate'
            await asyncio.to_thread(assemble_state,manifest,snapshot_store,assembled,
                max_state_bytes=args.max_state_gb*1024**3)
            await asyncio.to_thread(apply_effective_state,model,adapter,assembled,
                model_id=adapter.model_id,revision=adapter.revision,
                max_bytes=args.max_state_gb*1024**3)
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
            optimizer=new_optimizer();optimizer.zero_grad(set_to_none=True)
            nonlocal_optimizer[0]=optimizer
            from .localprogress import _tip
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
                  f'events={event_count}, suffix_events={suffix}, model={assembled}',flush=True)
            print(f'JOIN READY: verified-full-model-snapshot; '
                  f'events={journal.count()} peer={args.peer_id}',flush=True)
            return

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

    # Use one mutable reference when updating optimizer in nested receive handler.
    nonlocal_optimizer=[optimizer]
    heartbeat_task=None
    watchdog_task=None
    last_swarm_report=0.0
    try:
        update_roster()
        heartbeat_task=asyncio.create_task(heartbeat_daemon(),name='fedpara-heartbeat')
        watchdog_task=asyncio.create_task(loop_stall_watchdog(shutdown),name='fedpara-loop-watchdog')
        while not shutdown.is_set() and (args.rounds == 0 or emitted < args.rounds):
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
                    if node.control('frontier',heads=frontier_cache):
                        node.next_frontier_at=now+(0.7 if paused else 2.5)
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
                                min(args.inventory_cycle,0.3) if (
                                    getattr(args,'swarm_pause_file',None) and
                                    Path(args.swarm_pause_file).exists()) else args.inventory_cycle)
            # Flush network events before training the next minibatch.
            await read_peer_events()
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
                    received_hashes[slot.peer]=hashlib.sha256(
                        received_snapshots[slot.peer].read_bytes()).hexdigest()
                    print(f'SNAPSHOT DOWNLOADED peer={slot.peer}',flush=True)
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
                args.local_steps=controller.tune(timeout=args.peer_timeout).local_steps
            await maybe_checkpoint()
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
            loss=await run_step_offloop(run_step,model,adapter,optimizer,batch_iter,args)
            local_steps+=1; pending[0]+=1; recent_loss=loss
            if pending[0]>=args.grad_accum:
                def advance_optimizer():
                    with adapter.lock:
                        optimizer.step();optimizer.zero_grad(set_to_none=True)
                await asyncio.to_thread(advance_optimizer)
                pending[0]=0
            # Durable local-only adapter+AdamW state: an abrupt process death
            # must not reset every unsealed optimizer step to genesis.
            if pending[0] == 0 and getattr(args,'durable_every',1) and local_steps % getattr(args,'durable_every',1) == 0:
                # Preserve sqlite3's same-thread contract: capture the journal
                # tip on-loop and serialize safetensors/AdamW buffers off-loop.
                from .localprogress import _tip
                tip=_tip(journal)
                await asyncio.to_thread(save_working,journal,adapter,optimizer,
                                        local_steps,journal_tip=tip)
            if local_steps>=args.local_steps:
                await seal_if_needed()
                nonlocal_optimizer[0]=optimizer
                await maybe_checkpoint()
            if local_steps==1 or local_steps%args.log_steps==0:
                channels=sum(bool(s.node and s.node.connected.is_set())
                             for s in slots.values() if not s.disabled)
                authed=sum(bool(s.node and s.node.verified_at)
                           for s in slots.values() if not s.disabled)
                print(f'elastic steps={local_steps} emitted={emitted} loss={recent_loss:.5g} '
                      f'active={active_count()} living={living_count()} '
                      f'channels={channels} authed={authed} synced={synced_count()} '
                      f'peer={args.peer_id}',flush=True)
            await asyncio.sleep(0)
        if local_steps:
            await seal_if_needed()
        if args.shard_state_manifest and args.shard_store:
            from .shards import export_state_sharded
            export_state_sharded(model,args.shard_store,args.shard_state_manifest,
                model_id=adapter.model_id,revision=adapter.revision,peer_id=args.peer_id,
                seq=journal.count(),adapter=adapter,shard_bytes=args.shard_mb*1024**2,
                max_state_bytes=args.max_state_gb*1024**3)
        if args.full_state_output:
            from .state import export_state
            export_state(model,args.full_state_output,model_id=adapter.model_id,
                revision=adapter.revision,peer_id=args.peer_id,seq=journal.count(),
                adapter=adapter,max_bytes=args.max_state_gb*1024**3)
    finally:
        shutdown.set()
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
        for slot in slots.values(): await stop_slot(slot)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try: loop.remove_signal_handler(sig)
            except (RuntimeError, NotImplementedError): pass
        for sig, handler in previous_signals.items():
            try: signal.signal(sig,handler)
            except ValueError: pass
        if epoch_paxos is not None:epoch_paxos.close()
        journal.close()
