"""Bounded-degree, leaderless four-ring overlay.

Four independent deterministic ring orderings give at most eight undirected
WebRTC neighbors per peer. The base ring guarantees connectivity; alternate
hash orders yield stable long-distance contacts without full mesh edges.
An insertion/removal changes at most eight existing ring edges (16 endpoints),
regardless of room size.  Both the signaling server and simulator MUST use
this precise FNV64 order (not Python's process-salted hash()).
"""
from __future__ import annotations
from functools import lru_cache

ROOM_PEER_LIMIT = 4096
MAX_DEGREE = 8
_MASK = (1 << 64) - 1
_FNV_PRIME = 0x100000001b3
_FNV_OFFSET = 0xcbf29ce484222325


def _ring_key(peer: str, ring: int) -> tuple[int, str]:
    h = _FNV_OFFSET
    for b in (f'kagami-v49-ring-{ring}:'.encode('ascii') + peer.encode('ascii')):
        h = ((h ^ b) * _FNV_PRIME) & _MASK
    h ^= h >> 33
    h = h * 0xff51afd7ed558ccd & _MASK
    h ^= h >> 33
    h = h * 0xc4ceb9fe1a85ec53 & _MASK
    h ^= h >> 33
    return (h, peer)


@lru_cache(maxsize=8)
def _build(roster: tuple[str, ...], degree: int):
    adj = {p: set() for p in roster}
    # Preserve small room behavior to minimize 16-peer interoperability churn.
    if len(roster) <= 32:
        for i, p in enumerate(roster):
            steps=(1, 2, 4, 8)[:degree // 2]
            if len(roster)>1:
                adj[p].update(roster[(i+s*sign)%len(roster)]
                              for s in steps for sign in (-1,1))
            adj[p].discard(p)
            if len(roster)==16 and degree==8:
                adj[p].add(roster[(i+(3 if i%2==0 else -3))%16])
    else:
        for ring in range(degree // 2):
            order = roster if ring == 0 else tuple(sorted(roster,
                               key=lambda p: _ring_key(p,ring)))
            for i,p in enumerate(order):
                left,right=order[(i-1)%len(order)],order[(i+1)%len(order)]
                if left != p: adj[p].add(left)
                if right != p: adj[p].add(right)
    return {p:frozenset(v) for p,v in adj.items()}


def _canonical_roster(member_ids):
    # Discovery already provides strictly sorted peer lists. Fast-path those
    # without building a 4,096-entry set and re-sorting it for EACH peer query.
    # We still copy lists to a tuple, so later in-place roster edits cannot
    # mutate the topology cache key. Unordered/duplicate input is canonicalized.
    if isinstance(member_ids,(list,tuple)) and all(
            member_ids[i]<member_ids[i+1] for i in range(len(member_ids)-1)):
        return tuple(member_ids)
    return tuple(sorted(set(member_ids)))


def neighbors(peer: str, member_ids, *, max_degree: int = MAX_DEGREE) -> set[str]:
    if isinstance(member_ids,str):
        raise ValueError('members must be an iterable of peer IDs')
    roster=_canonical_roster(member_ids)
    if peer not in roster:
        raise ValueError('local peer is missing from room roster')
    if not (2<=max_degree<=MAX_DEGREE and max_degree%2==0):
        raise ValueError('max_degree must be even in [2,8]')
    if len(roster)>ROOM_PEER_LIMIT:
        raise ValueError('room exceeds supported participant limit')
    return set(_build(roster,max_degree)[peer])


def topology_metrics(member_ids, *, max_degree=MAX_DEGREE):
    roster=_canonical_roster(member_ids)
    adj=_build(roster,max_degree)
    undirected=all(p in adj[q] for p in adj for q in adj[p])
    return {'peers':len(roster),'max_degree':max((len(v) for v in adj.values()),default=0),
            'undirected':undirected,'links':sum(map(len,adj.values()))//2}
