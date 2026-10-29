"""Lossless multi-origin, signed-event aggregation for bounded P2P transport.

An aggregate is *not* an unauditable relay-authored sum. Each constituent
carries its original author's signed FPEL1 envelope; the receiving engine
checks every author's signature, checksum, genesis, sequence and tensor
bounds before applying the exact sum of previously unseen contributions.

The compressed frame saves repeated transport framing and redundant bytes,
not the information-theoretic cost of shipping all independently verifiable
contributions. Partial overlap is resolved at per-(origin,seq) granularity.
"""
from __future__ import annotations

import hashlib
import struct
import zlib

from .core import PatchError

MAGIC = b'FPAG1\0'
HEADER = struct.Struct('!6sHII32s')
MAX_EVENTS = 8
MAX_EXPANSION = 8
MAX_RAW_BYTES = 64 * 1024 * 1024


def is_aggregate(wire: bytes) -> bool:
    return isinstance(wire, bytes) and wire.startswith(MAGIC)


def pack_aggregate(wires: list[bytes], *, max_bytes: int) -> bytes:
    """Emit one compressed transport frame with signed, independently auditable events."""
    if not 2 <= len(wires) <= MAX_EVENTS or not max_bytes > 0:
        raise ValueError('aggregate requires 2..8 events and a positive wire cap')
    if any(not isinstance(wire, bytes) or not wire.startswith(b'FPEL1\0')
           or not 0 < len(wire) <= max_bytes for wire in wires):
        raise PatchError('aggregate contains invalid/oversize event')
    # Do not allow duplicate frames even before expensive cryptographic checks.
    if len({hashlib.sha256(wire).digest() for wire in wires}) != len(wires):
        raise PatchError('duplicate aggregate frame')
    raw = b''.join(struct.pack('!I', len(wire)) + wire for wire in wires)
    if len(raw) > min(max_bytes * MAX_EVENTS, MAX_RAW_BYTES):
        raise PatchError('aggregate uncompressed limit exceeded')
    compressed = zlib.compress(raw, level=3)
    packed = HEADER.pack(MAGIC, len(wires), len(raw), len(compressed),
                         hashlib.sha256(raw).digest()) + compressed
    if len(packed) > max_bytes:
        raise PatchError('aggregate transport cap exceeded')
    return packed


def unpack_aggregate(wire: bytes, *, max_bytes: int) -> list[bytes]:
    """Bound compressed and decompressed size; never trust a relay's counts."""
    if not is_aggregate(wire):
        return [wire]
    if not isinstance(max_bytes, int) or max_bytes <= 0 or len(wire) > max_bytes or len(wire) <= HEADER.size:
        raise PatchError('invalid aggregate wire length')
    magic, n, raw_len, compressed_len, digest = HEADER.unpack_from(wire)
    if (magic != MAGIC or not 2 <= n <= MAX_EVENTS or
            compressed_len != len(wire) - HEADER.size or
            not 0 < raw_len <= min(max_bytes * MAX_EXPANSION, MAX_RAW_BYTES)):
        raise PatchError('invalid aggregate header')
    try:
        decomp = zlib.decompressobj()
        raw = decomp.decompress(wire[HEADER.size:], raw_len + 1)
        if decomp.unconsumed_tail or not decomp.eof or decomp.unused_data:
            raise PatchError('aggregate stream not terminated cleanly')
        raw += decomp.flush()
    except zlib.error as exc:
        raise PatchError('invalid aggregate compression') from exc
    if len(raw) != raw_len or hashlib.sha256(raw).digest() != digest:
        raise PatchError('aggregate length/checksum mismatch')
    out=[]
    offset=0
    for _ in range(n):
        if offset + 4 > len(raw):
            raise PatchError('truncated aggregate index')
        size=struct.unpack_from('!I', raw, offset)[0]
        offset += 4
        if not 0 < size <= max_bytes or offset+size > len(raw):
            raise PatchError('aggregate subevent exceeds bounds')
        part=raw[offset:offset+size]
        if not part.startswith(b'FPEL1\0'):
            raise PatchError('aggregate subevent lacks signed envelope')
        out.append(part)
        offset += size
    if offset != len(raw) or len({hashlib.sha256(part).digest() for part in out}) != len(out):
        raise PatchError('aggregate trailing bytes or duplicate events')
    return out


def adaptive_publish_steps(baseline: int, room_peers: int, *, max_steps: int = 64) -> int:
    """Double local work per 4x room size; 5 peers preserve original behavior.

    This controls ONLY local event publication, not heartbeat, NEED, or recovery.
    The hard cap prevents arbitrarily stale local updates in huge rooms.
    """
    if any(type(v) is not int for v in (baseline, room_peers, max_steps)) or min(baseline, room_peers, max_steps) < 1:
        raise ValueError('invalid adaptive publish policy')
    if baseline > max_steps:
        return baseline  # never silently shorten an explicitly larger local block
    factor = 1
    cohort = max(1, room_peers)
    threshold = 20
    while cohort >= threshold and factor * baseline < max_steps:
        factor *= 2
        threshold *= 4
    return min(max_steps, baseline * factor)
