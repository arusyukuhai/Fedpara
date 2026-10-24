"""Deterministic peer-specific sampling without server coordination.

Only identifiers, not training examples, are exchanged. Seeds remain stable
across restarts of the same persistent peer identity, and different peers use
independent shuffles of the same underlying dataset.
"""
from __future__ import annotations

import hashlib
import random


def peer_sampling_seed(peer_id: str | None, room_seed: int = 42) -> int:
    """Domain-separated 63-bit sampling seed; no dependence on Python hash()."""
    if type(room_seed) is not int or not 0 <= room_seed < 2**63:
        raise ValueError('room dataset seed out of range')
    identity = peer_id or 'standalone'
    if not isinstance(identity, str) or len(identity) > 256:
        raise ValueError('invalid sampling identity')
    material = (b'fedpara/data-peer-seed/v1\0' + room_seed.to_bytes(8,'big') +
                identity.encode('utf-8'))
    return int.from_bytes(hashlib.sha256(material).digest()[:8], 'big') & ((1 << 63) - 1)


def shuffled_cycle(items, seed: int):
    """Infinite shuffled epochs; uses an independent local RNG per iterator."""
    if not items:
        raise ValueError('empty training dataset')
    rng = random.Random(seed)
    size = len(items)
    while True:
        positions = list(range(size))
        rng.shuffle(positions)
        for i in positions:
            yield items[i]
