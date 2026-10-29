"""Authenticated, bounded paging for large signed snapshot vector clocks.

Every page is transmitted as an individually signed elastic control. The digest
commits to the complete ordered vector and the snapshot metadata. No incomplete
or conflicting collection is eligible for installation.
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict

from .core import PEER_ID, PatchError, _canonical_json

PAGE_ORIGINS = 48
MAX_BOUNDARY_ORIGINS = 16384
MAX_BOUNDARY_PAGES = math.ceil(MAX_BOUNDARY_ORIGINS / PAGE_ORIGINS)


def _hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_cutoffs(cutoffs):
    if (not isinstance(cutoffs, dict) or len(cutoffs) > MAX_BOUNDARY_ORIGINS or
        any(not isinstance(peer, str) or not PEER_ID.fullmatch(peer) or
            type(seq) is not int or not 1 <= seq < 2**63
            for peer, seq in cutoffs.items())):
        raise PatchError('invalid snapshot inventory cutoffs')
    return cutoffs


def boundary_digest(manifest_sha256, inventory_sha256, events, cutoffs, request_id=''):
    validate_cutoffs(cutoffs)
    if (not _hex64(manifest_sha256) or not _hex64(inventory_sha256) or
        type(events) is not int or not 0 <= events < 2**63 or
        not isinstance(request_id, str) or len(request_id) > 32):
        raise PatchError('invalid snapshot boundary metadata')
    payload = [manifest_sha256, inventory_sha256, events, request_id,
               [[p, cutoffs[p]] for p in sorted(cutoffs)]]
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def boundary_pages(manifest_sha256, inventory_sha256, events, cutoffs, request_id=''):
    digest = boundary_digest(manifest_sha256, inventory_sha256, events, cutoffs, request_id)
    items = [[p, cutoffs[p]] for p in sorted(cutoffs)]
    parts = max(1, math.ceil(len(items) / PAGE_ORIGINS))
    for index in range(parts):
        yield dict(
            manifest_sha256=manifest_sha256, inventory_sha256=inventory_sha256,
            events=events, request_id=request_id, boundary_sha256=digest,
            part=index, parts=parts,
            cutoffs=items[index * PAGE_ORIGINS:(index + 1) * PAGE_ORIGINS],
        )


class SnapshotBoundaryAssembler:
    """Reconstruct exactly one signed boundary; no unbounded fragment backlog.

    Pages arriving after part zero may be reordered; the complete set is
    required. A new offer begins at part zero; old trailing parts are ignored.
    """
    def __init__(self):
        self.clear()

    def clear(self):
        self.identity = None
        self.pages = {}

    def accept(self, msg):
        if msg.get('type') != 'snapshot-tip-page':
            raise PatchError('unexpected boundary page type')
        part, parts, entries = msg.get('part'), msg.get('parts'), msg.get('cutoffs')
        if (type(part) is not int or type(parts) is not int or
            not 1 <= parts <= MAX_BOUNDARY_PAGES or not 0 <= part < parts or
            not isinstance(entries, list) or len(entries) > PAGE_ORIGINS or
            (part < parts - 1 and len(entries) != PAGE_ORIGINS)):
            raise PatchError('invalid snapshot boundary pagination')
        cutoffs = {}
        previous = None
        for entry in entries:
            if (not isinstance(entry, list) or len(entry) != 2 or
                not isinstance(entry[0], str) or not PEER_ID.fullmatch(entry[0]) or
                type(entry[1]) is not int or not 1 <= entry[1] < 2**63 or
                (previous is not None and entry[0] <= previous)):
                raise PatchError('invalid snapshot boundary entry')
            cutoffs[entry[0]] = entry[1]
            previous = entry[0]
        if (not _hex64(msg.get('boundary_sha256')) or
            not _hex64(msg.get('manifest_sha256')) or
            not _hex64(msg.get('inventory_sha256')) or
            type(msg.get('events')) is not int or not 0 <= msg['events'] < 2**63 or
            not isinstance(msg.get('request_id'), str) or len(msg['request_id']) > 32):
            raise PatchError('invalid boundary page header')
        identity = (msg['boundary_sha256'], msg['manifest_sha256'],
                    msg['inventory_sha256'], msg['events'], msg['request_id'], parts)
        if self.identity != identity:
            if part != 0:
                # The first page of a new offer may have been delayed or
                # superseded; never blend two different snapshot boundaries.
                return None
            self.clear()
            self.identity = identity
        old = self.pages.get(part)
        if old is not None and old != entries:
            raise PatchError('conflicting snapshot boundary page')
        self.pages[part] = entries
        if len(self.pages) != parts:
            return None
        merged = OrderedDict()
        prev = None
        for i in range(parts):
            for peer, seq in self.pages[i]:
                if prev is not None and peer <= prev:
                    raise PatchError('duplicate or disordered snapshot boundary origin')
                merged[peer] = seq
                prev = peer
        if len(merged) > MAX_BOUNDARY_ORIGINS or boundary_digest(
                identity[1], identity[2], identity[3], merged, identity[4]) != identity[0]:
            raise PatchError('snapshot boundary digest mismatch')
        self.clear()
        return dict(type='snapshot-tip', manifest_sha256=identity[1],
                    inventory_sha256=identity[2], events=identity[3],
                    request_id=identity[4], cutoffs=dict(merged), counter=msg['counter'])
