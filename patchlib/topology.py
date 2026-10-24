"""Bounded-degree deterministic small-world overlay for room discovery.

Every node computes the SAME undirected edges from the same sorted roster.
The overlay needs only O(N*log N) edge checks per roster update and limits
open WebRTC channels to a constant number. It does not promise convergence
under network partitions or failure of all known bootstrap sources.
"""
from __future__ import annotations

ROOM_PEER_LIMIT = 96
MAX_DEGREE = 8


def neighbors(peer: str, member_ids, *, max_degree: int = MAX_DEGREE) -> set[str]:
    if isinstance(member_ids, str):
        raise ValueError('members must be an iterable of peer IDs')
    roster = sorted(set(member_ids))
    if peer not in roster:
        raise ValueError('local peer is missing from room roster')
    if not (2 <= max_degree <= MAX_DEGREE and max_degree % 2 == 0):
        raise ValueError('max_degree must be even in [2,8]')
    if len(roster) > ROOM_PEER_LIMIT:
        raise ValueError('room exceeds supported participant limit')
    if len(roster) == 1:
        return set()
    pos = roster.index(peer)
    # Ring edges survive every roster ordering; power-of-two chords reduce
    # gossip diameter while keeping degree <= 8, even at 96 members.
    strides = (1, 2, 4, 8)[:max_degree // 2]
    result = {roster[(pos + sign * step) % len(roster)]
              for step in strides for sign in (-1, 1)} - {peer}
    # A 16-node ring has only seven distinct peers with strides (1,2,4,8):
    # +8 and -8 refer to the same opposite node. Add a symmetric matching
    # (+3 on even positions / -3 on odd) to reach exactly eight peers.
    if len(roster) == 16 and max_degree == 8:
        partner = (pos + (3 if pos % 2 == 0 else -3)) % 16
        result.add(roster[partner])
    return result


def topology_metrics(member_ids, *, max_degree: int = MAX_DEGREE):
    roster = sorted(set(member_ids))
    adj = {p: neighbors(p, roster, max_degree=max_degree) for p in roster}
    undirected = all(p in adj[q] for p in adj for q in adj[p])
    return {'peers': len(roster), 'max_degree': max((len(v) for v in adj.values()), default=0),
            'undirected': undirected, 'links': sum(map(len, adj.values())) // 2}
