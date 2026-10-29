"""Per-connection SDP/ICE safety during simultaneous joins, pauses and churn.

Never feed an SDP answer to a stable PeerConnection or a second offer into an
already answered connection. Reject unbounded early ICE candidates. A new
PatchNode owns a new guard on each *actual* peer connection; stale callbacks
are fenced by the connection identity in p2p.py.
"""
from __future__ import annotations

class NegotiationGuard:
    def __init__(self, role: str, *, max_candidates: int = 96):
        if role not in ('initiator','answerer'):
            raise ValueError('invalid signaling role')
        self.role=role
        self.max_candidates=max_candidates
        self.offer_applied=False
        self.answer_applied=False
        self.pending=[]
        self.seen=set()
        self.ignored_offers=0
        self.ignored_answers=0
        self.ignored_candidates=0

    def accept_offer(self, signaling_state: str) -> bool:
        if self.role!='answerer' or self.offer_applied or signaling_state!='stable':
            self.ignored_offers+=1
            return False
        self.offer_applied=True
        return True

    def accept_answer(self, signaling_state: str) -> bool:
        if self.role!='initiator' or self.answer_applied or signaling_state!='have-local-offer':
            self.ignored_answers+=1
            return False
        self.answer_applied=True
        return True

    def push_candidate(self, key: str, candidate, remote_ready: bool):
        """Return whether ICE may be applied immediately; None means ignored.

        A candidate received during negotiation must be bound to this RTCPeerConnection;
        pending candidates are forgotten when that connection is replaced.
        """
        if not isinstance(key,str) or len(key)>1024 or key in self.seen:
            self.ignored_candidates+=1
            return None
        if len(self.seen)>=self.max_candidates * 4:
            self.ignored_candidates+=1
            return None
        self.seen.add(key)
        if remote_ready:
            return True
        if len(self.pending)>=self.max_candidates:
            self.ignored_candidates+=1
            return None
        self.pending.append(candidate)
        return False

    def release(self):
        pending,self.pending=self.pending,[]
        # An unauthenticated peer may have filled the early candidate cache.
        # Bound it without making all future, valid post-SDP ICE candidates
        # permanently undeliverable (an early-candidate DoS).
        if len(self.seen) >= self.max_candidates * 4:
            self.seen.clear()
        return pending
