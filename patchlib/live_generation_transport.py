"""Bounded authenticated WebRTC anti-entropy for certified Kagami generations.

The *outer* elastic control envelope is signed by the direct peer. The *inner*
certificate is independently quorum-verified, so relaying cannot impersonate
committee members. Control frames carry no tensor bytes. They are safe to
process while optimization proceeds and after reconnect.

This does not itself authorize discarding a trainer's old weight/journal state.
Only a completed runtime adoption may authorize local GC.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from .core import PatchError, _canonical_json
from .online_generation import AutomaticGenerationControl, GenerationLedger, _digest

MAX_GENERATION_FRAME = 10_000  # leaves room for signed elastic wrapper (16 KiB)
MAX_RECOVERY_BATCH = 4
# Idle advertisements are anti-entropy, not a clock for optimizer progress.
# Changes are still announced on the very next tick; old peers sync on reconnect.
IDLE_HEAD_SECONDS = 30.0
IDLE_HAVE_SECONDS = 120.0
MAX_RECOVERY_BACKOFF_SECONDS = 30.0
MAX_SNAPSHOT_BACKOFF_SECONDS = 60.0


def validate_control(kind, message):
    if not isinstance(message, dict) or len(_canonical_json(message)) > MAX_GENERATION_FRAME:
        raise PatchError('generation control exceeds bounded SCTP frame')
    if kind == 'generation-head':
        if set(message) != {'epoch', 'digest'} or type(message['epoch']) is not int or message['epoch'] < 0 or message['epoch'] >= 2**32:
            raise PatchError('invalid generation head')
        if not isinstance(message['digest'], str) or len(message['digest']) != 64 or any(c not in '0123456789abcdef' for c in message['digest']):
            raise PatchError('invalid generation digest')
    elif kind == 'generation-snapshot-request':
        if (set(message) != {'epoch','manifest'} or type(message['epoch']) is not int
            or not 1 <= message['epoch'] < 2**32 or not isinstance(message['manifest'],str)
            or len(message['manifest']) != 64 or any(c not in '0123456789abcdef' for c in message['manifest'])):
            raise PatchError('invalid certified checkpoint request')
    elif kind == 'generation-have':
        if (set(message) != {'epoch','manifest'} or type(message['epoch']) is not int
            or not 1 <= message['epoch'] < 2**32 or not isinstance(message['manifest'],str)
            or len(message['manifest']) != 64 or any(c not in '0123456789abcdef' for c in message['manifest'])):
            raise PatchError('invalid generation availability hint')
    elif kind == 'generation-need':
        if set(message) != {'after'} or type(message['after']) is not int or not 0 <= message['after'] < 2**32:
            raise PatchError('invalid generation catch-up cursor')
    elif kind == 'generation-compact-head':
        if (set(message) != {'epoch','digest'} or type(message['epoch']) is not int or
                not 0 <= message['epoch'] < 2**32 or
                not isinstance(message['digest'],str) or len(message['digest'])!=64 or
                any(x not in '0123456789abcdef' for x in message['digest'])):
            raise PatchError('invalid compact generation head')
    elif kind in ('generation-compact-have','generation-compact-snapshot-request'):
        if (set(message) != {'epoch','manifest'} or type(message['epoch']) is not int or
            not 1 <= message['epoch'] < 2**32 or not isinstance(message['manifest'],str) or
            len(message['manifest'])!=64 or any(c not in '0123456789abcdef' for c in message['manifest'])):
            raise PatchError('invalid compact checkpoint control')
    elif kind == 'generation-compact-need':
        if set(message) != {'after'} or type(message['after']) is not int or not 0 <= message['after'] < 2**32:
            raise PatchError('invalid compact generation recovery cursor')
    elif kind == 'generation-wire':
        if set(message) != {'payload'} or not isinstance(message['payload'], dict):
            raise PatchError('invalid generation wire payload')
        if set(message['payload']) != {'kind', 'data'}:
            raise PatchError('unexpected generation message')
    else:
        raise PatchError('unsupported generation control type')


@dataclass
class Neighbor:
    last_head: float = 0
    sent_head_digest: str | None = None
    last_wire: float = 0
    advertised_epoch: int = 0
    requested_after: int = -1
    forwarded: set[str] = field(default_factory=set)
    last_snapshot_request: float = 0
    last_have: float = 0
    sent_have_manifest: str | None = None
    last_snapshot_key: str | None = None
    snapshot_backoff: float = 3.0
    need_backoff: float = 0.0
    available_manifest: str | None = None
    available_epoch: int = 0
    advertised_compact_epoch: int = 0
    compact_available_epoch: int = 0
    compact_available_manifest: str | None = None
    sent_compact_have: str | None = None
    last_compact_have: float = 0.0
    compact_missing_key: str | None = None
    compact_snapshot_backoff: float = 3.0
    last_compact_snapshot_request: float = 0.0
    requested_compact_after: int = -1
    sent_compact_head: str | None = None
    last_compact_head: float = 0
    last_compact_need: float = 0
    last_compact_page_need: float = 0
    compact_reply: dict | None = None
    # Small rotating set: room degree, not total participants, determines memory.


class GenerationWebRTCBridge:
    """Requires an already-authenticated ElasticNode for send and receive.

    Serverless gossip is loss-tolerant: a failed DataChannel send never moves a
    cursor, while a reconnect clears the per-link state and re-requests a chain.
    """
    def __init__(self, controller: AutomaticGenerationControl, *, interval=.6,
                 snapshot_available=None, compact_ledger=None, compact_control=None):
        if not isinstance(controller, AutomaticGenerationControl):
            raise TypeError('verified generation control required')
        if not 0.05 <= interval <= 30:
            raise ValueError('invalid generation gossip interval')
        self.control = controller
        self.ledger: GenerationLedger = controller.ledger
        # Optional independently verified v2 quorum heads; v1 peers are still
        # supported but cannot forge or vote for a v2 certificate.
        self.compact_ledger = compact_ledger
        self.compact_control = compact_control
        if compact_control is not None and compact_control.ledger is not compact_ledger:
            raise ValueError("compact controller ledger mismatch")
        self.interval = interval
        self.snapshot_available = snapshot_available or (lambda _manifest: False)
        self.neighbors: OrderedDict[str, Neighbor] = OrderedDict()
        self._last_certificate = self.local_epoch()
        self._last_compact_certificate = 0
        # Drain the producer once per TICK, never once per neighbor.  A first
        # neighbor (or failed send) must not consume a vote intended for the
        # entire authenticated overlay.  Bound retained advertisements; lost
        # certificates are also recovered from the durable ledger by NEED.
        self._pending_broadcast: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self._max_broadcast = 128
        self._broadcast_lifetime = 180.0

    def local_epoch(self):
        latest = self.ledger.latest()
        return latest['proposal']['epoch'] if latest else 0

    def _digest(self):
        return self.ledger.head()

    def disconnect(self, peer):
        self.neighbors.pop(peer, None)

    def _neighbor(self, peer):
        state = self.neighbors.get(peer)
        if state is not None:
            self.neighbors.move_to_end(peer)
            return state
        # Ordinary peers need state for active neighbors, not all identities
        # that have ever contacted the network in a long-running room.
        if len(self.neighbors) >= 256:
            self.neighbors.popitem(last=False)
        state = Neighbor()
        self.neighbors[peer] = state
        return state

    def receive(self, sender: str, kind: str, fields: dict) -> bool:
        """Must be called ONLY after ElasticNode checked direct signature.

        Signatures of relayed votes/certificates are *also* checked by the
        GenerationLedger, including on idempotent replay. Never accept a
        remote 'head' as committed state.
        """
        validate_control(kind, fields)
        state = self._neighbor(sender)
        if kind == 'generation-compact-head':
            if self.compact_ledger is None:
                raise PatchError('compact generation is not configured')
            here=self.compact_ledger.latest()
            current=here['proposal']['epoch'] if here else 0
            if fields['epoch']==current and here and fields['digest']!=_digest(here['proposal']):
                raise PatchError('conflicting compact epoch head')
            state.advertised_compact_epoch=fields['epoch']
            if fields['epoch']>current:
                state.requested_compact_after=current
            return False
        if kind == 'generation-compact-have':
            if self.compact_ledger is None:
                raise PatchError('compact generation is not configured')
            state.compact_available_epoch=fields['epoch']
            state.compact_available_manifest=fields['manifest']
            return False
        if kind == 'generation-compact-snapshot-request':
            if self.compact_ledger is None:
                raise PatchError('compact generation is not configured')
            # Actual bytes are sent ONLY after the authenticated trainer
            # rechecks this exact cert and every requested object hash.
            return False
        if kind == 'generation-compact-need':
            if self.compact_ledger is None:
                raise PatchError('compact generation is not configured')
            state.requested_compact_after=fields['after']
            return False
        if (kind == 'generation-wire' and
                fields['payload'].get('kind','').startswith('compact-') and
                fields['payload'].get('kind')!='compact-certificate'):
            if self.compact_control is None:
                raise PatchError('compact candidate traffic requires trusted local coordinator')
            reply=self.compact_control.receive(sender,fields['payload'])
            if reply is not None:
                state.compact_reply=reply
            return False
        if kind == 'generation-wire' and fields['payload'].get('kind')=='compact-certificate':
            if self.compact_ledger is None:
                raise PatchError('compact generation certificate without pinned committee')
            return self.compact_ledger.accept(fields['payload']['data'])
        if kind == 'generation-head':
            state.advertised_epoch = fields['epoch']
            if fields['epoch'] == self.local_epoch() and fields['digest'] != self._digest():
                raise PatchError('generation fork detected: equal epoch, different digest')
            if fields['epoch'] > self.local_epoch():
                state.requested_after = self.local_epoch()
            return False
        if kind == 'generation-have':
            # Hints are not proofs: the receiver still checks *all* shard hashes.
            # Keep just the latest advertised manifest per direct neighbor.
            state.available_epoch = fields['epoch']
            state.available_manifest = fields['manifest']
            return False
        if kind == 'generation-snapshot-request':
            # This request is served by the training node from *certified*
            # manifests only, never from arbitrary peer-supplied paths.
            return False
        if kind == 'generation-need':
            # A stale/reconnecting peer can ask any verified holder, not just a
            # designated coordinator. Requests are rate-limited during tick().
            state.requested_after = fields['after']
            return False
        before = self.local_epoch()
        self.control.receive(fields['payload'])
        return self.local_epoch() > before

    def _certs_since(self, after: int):
        # Indexed, bounded disk scan; avoid loading a 1000-epoch history into RAM.
        query = self.ledger.db.execute(
            'SELECT wire FROM commits WHERE epoch>? ORDER BY epoch LIMIT ?',
            (after, MAX_RECOVERY_BATCH))
        return [json.loads(row[0]) for row in query]

    def tick(self, peers, *, now=None):
        """Send bounded control traffic to verified, generation-capable links.

        Each item in peers is (peer_id, elastic_node); returns number sent.
        Ordinary peers retain no per-network-membership state.
        """
        now = time.monotonic() if now is None else now
        count = 0
        # One SQLite query and one filesystem availability check per tick,
        # not per neighbor. At N=1000, this avoids O(degree) repeated IO.
        cert = self.ledger.latest()
        epoch = cert['proposal']['epoch'] if cert else 0
        head_digest = _digest(cert['proposal']) if cert else self.ledger.genesis
        manifest = cert['proposal']['manifest'] if cert else None
        available = bool(manifest and self.snapshot_available(manifest))
        compact = self.compact_ledger.latest() if self.compact_ledger else None
        compact_epoch = compact['proposal']['epoch'] if compact else 0
        if compact_epoch != self._last_compact_certificate:
            self._last_compact_certificate=compact_epoch
            for state in self.neighbors.values():state.forwarded.clear()
        compact_manifest=(compact['proposal']['manifest'] if compact else None)
        compact_available=bool(compact_manifest and self.snapshot_available(compact_manifest))
        compact_digest = (_digest(compact['proposal']) if compact else
                          self.compact_ledger.genesis if self.compact_ledger else None)
        compact_messages=(self.compact_control.announcements()
            if self.compact_control is not None else ())
        compact_missing=(iter(self.compact_control.missing_pages(
            limit=8)) if self.compact_control is not None else iter(()))
        if epoch != self._last_certificate:
            self._last_certificate = epoch
            for state in self.neighbors.values():
                state.forwarded.clear()
                state.sent_head_digest = None
                state.need_backoff = 0.0
        # Drop obsolete PROPOSALS and VOTES after a quorum commit. Replaying
        # a proposal against an already committed head is not idempotent in the
        # legacy controller. The durable certificate remains recoverable.
        def obsolete(message):
            if not isinstance(message, dict):
                return False
            kind = message.get('kind')
            if kind == 'proposal':
                return message.get('data', {}).get('epoch', 2**32) <= epoch
            if kind == 'vote':
                return message.get('data', {}).get('payload', {}).get('epoch', 2**32) <= epoch
            return False
        for digest, (msg, _) in list(self._pending_broadcast.items()):
            if obsolete(msg):
                self._pending_broadcast.pop(digest, None)
        # Materialize authenticated controls only once before iterating links.
        # No live neighbors? Keep bounded messages until a peer reconnects.
        for msg in self.control.drain(maximum=16):
            if obsolete(msg):
                continue
            digest = hashlib.sha256(_canonical_json(msg)).hexdigest()
            if len(_canonical_json({'payload':msg})) > MAX_GENERATION_FRAME:
                raise PatchError('generation proposal/certificate requires paging')
            self._pending_broadcast[digest] = (msg, now)
            self._pending_broadcast.move_to_end(digest)
        while self._pending_broadcast and (
                len(self._pending_broadcast) > self._max_broadcast or
                now - next(iter(self._pending_broadcast.values()))[1] > self._broadcast_lifetime):
            self._pending_broadcast.popitem(last=False)
        for peer, node in peers:
            if not getattr(node, 'verified_at', 0) or not getattr(node, 'remote_supports_generation', False):
                continue
            state = self._neighbor(peer)
            if self.compact_ledger is not None:
                if self.compact_control is not None:
                    # Only one outstanding bounded page response per neighbor.
                    if state.compact_reply is not None:
                        if node.control('generation-wire',payload=state.compact_reply):
                            state.compact_reply=None
                            count+=1
                    for msg in compact_messages:
                        digest=_digest(msg)
                        if digest in state.forwarded:continue
                        if len(_canonical_json({'payload':msg}))>MAX_GENERATION_FRAME:
                            raise PatchError('compact control message exceeds signed SCTP frame')
                        if not node.control('generation-wire',payload=msg):break
                        if len(state.forwarded)>=128:state.forwarded.clear()
                        state.forwarded.add(digest)
                        count+=1
                    if now-state.last_compact_page_need>=max(self.interval,0.5):
                        msg=next(compact_missing,None)
                        if msg is not None and node.control('generation-wire',payload=msg):
                            state.last_compact_page_need=now
                            count+=1
                if (state.sent_compact_head != compact_digest or
                        now-state.last_compact_head >= IDLE_HEAD_SECONDS):
                    if node.control('generation-compact-head',epoch=compact_epoch,digest=compact_digest):
                        state.last_compact_head=now
                        state.sent_compact_head=compact_digest
                        count+=1
                if compact is not None and compact_available and (
                        state.sent_compact_have!=compact_manifest or
                        now-state.last_compact_have>=IDLE_HAVE_SECONDS):
                    if node.control('generation-compact-have',epoch=compact_epoch,
                                    manifest=compact_manifest):
                        state.sent_compact_have=compact_manifest
                        state.last_compact_have=now
                        count+=1
                if compact is not None and not compact_available and (
                        peer in compact['proposal']['replicas'] or
                        (state.compact_available_epoch==compact_epoch and
                         state.compact_available_manifest==compact_manifest)):
                    if state.compact_missing_key!=compact_manifest:
                        state.compact_missing_key=compact_manifest
                        state.compact_snapshot_backoff=3.0
                        state.last_compact_snapshot_request=float('-inf')
                    if now-state.last_compact_snapshot_request>=state.compact_snapshot_backoff:
                        if node.control('generation-compact-snapshot-request',
                                        epoch=compact_epoch,manifest=compact_manifest):
                            state.last_compact_snapshot_request=now
                            state.compact_snapshot_backoff=min(MAX_SNAPSHOT_BACKOFF_SECONDS,
                                                               state.compact_snapshot_backoff*2)
                            count+=1
                if state.advertised_compact_epoch > compact_epoch:
                    if now-state.last_compact_need >= self.interval:
                        if node.control('generation-compact-need',after=compact_epoch):
                            state.last_compact_need=now;count+=1
                elif 0 <= state.requested_compact_after < compact_epoch:
                    cur=self.compact_ledger.db.execute(
                        'SELECT wire FROM commits WHERE epoch>? ORDER BY epoch LIMIT ?',
                        (state.requested_compact_after,MAX_RECOVERY_BATCH))
                    for row in cur:
                        cert_v2=json.loads(row[0])
                        payload={'kind':'compact-certificate','data':cert_v2}
                        if len(_canonical_json({'payload':payload}))>MAX_GENERATION_FRAME:
                            raise PatchError('compact epoch certificate exceeds bounded frame')
                        if not node.control('generation-wire',payload=payload):
                            break
                        state.requested_compact_after=cert_v2['proposal']['epoch']
                        count+=1
            # Advertise immediately on connection or a new certified epoch;
            # otherwise idle heartbeats are deliberately rare.
            if (state.sent_head_digest != head_digest or
                now - state.last_head >= IDLE_HEAD_SECONDS):
                if node.control('generation-head', epoch=epoch, digest=head_digest):
                    state.last_head = now
                    state.sent_head_digest = head_digest
                    count += 1
            if cert is not None and available and (
                state.sent_have_manifest != manifest or
                now - state.last_have >= IDLE_HAVE_SECONDS):
                if node.control('generation-have',epoch=epoch,manifest=manifest):
                    state.last_have=now
                    state.sent_have_manifest=manifest
                    count+=1
            if cert is not None and not available and (
                peer in cert['proposal']['replicas'] or
                (state.available_epoch == epoch and state.available_manifest == manifest)):
                # Missing shard requests back off instead of polling every 3s
                # forever when a peer is offline or cannot serve the model.
                if state.last_snapshot_key != manifest:
                    state.last_snapshot_key = manifest
                    state.snapshot_backoff = 3.0
                    state.last_snapshot_request = float('-inf')
                if now - state.last_snapshot_request >= state.snapshot_backoff:
                    if node.control('generation-snapshot-request',epoch=epoch,manifest=manifest):
                        state.last_snapshot_request=now
                        state.snapshot_backoff=min(MAX_SNAPSHOT_BACKOFF_SECONDS,state.snapshot_backoff * 2)
                        count+=1
            if state.advertised_epoch > epoch:
                # Signed request retries are exponential; once an epoch is
                # installed they reset. Reconnect clears Neighbor entirely.
                backoff = max(self.interval,state.need_backoff)
                if now - state.last_wire >= backoff:
                    if node.control('generation-need', after=epoch):
                        state.last_wire=now
                        state.need_backoff=min(MAX_RECOVERY_BACKOFF_SECONDS,max(self.interval,state.need_backoff*2))
                        count+=1
                continue
            state.need_backoff=0.0
            if state.requested_after >= 0 and state.requested_after < epoch:
                certs = self._certs_since(state.requested_after)
                for receipt in certs:
                    payload={'kind':'certificate','data':receipt}
                    if len(_canonical_json({'payload':payload})) > MAX_GENERATION_FRAME:
                        raise PatchError('generation certificate must be paged; cannot send')
                    if not node.control('generation-wire',payload=payload):
                        break
                    state.requested_after=receipt['proposal']['epoch']
                    count+=1
            if now - state.last_wire >= self.interval:
                for digest, (msg, _created) in self._pending_broadcast.items():
                    if digest in state.forwarded:
                        continue
                    if not node.control('generation-wire',payload=msg):
                        break  # Retry this exact frame on the next successful tick.
                    count+=1
                    if len(state.forwarded) >= self._max_broadcast:
                        state.forwarded.clear()
                    state.forwarded.add(digest)
                state.last_wire=now
        return count
