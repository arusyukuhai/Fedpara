"""Compact immutable frontier commitment for future Kagami epoch certificates.

A proposed epoch needs only the digest of ``header``. Pages of at most 32
peer/sequence pairs travel on demand. Committee validators MUST fetch and
verify *every* page before certifying checkpoint completeness; a membership
proof for one page alone is not a valid checkpoint frontier certification.

Not wired into the legacy cutoffs protocol: older trainers cannot safely GC
using this summary until their event coverage scanner is updated.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

from .core import PatchError, _canonical_json

PAGE_SIZE = 32
MAX_ENTRIES = 200_000
ALGORITHM = 'kagami-paged-frontier-v1'


def _h(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _leaf(index: int, entries: list) -> bytes:
    return _h(b'Kagami-frontier-leaf-v1\0' + _canonical_json([index, entries]))


def _parent(left: bytes, right: bytes) -> bytes:
    return _h(b'Kagami-frontier-node-v1\0' + left + right)


def _height(pages: int) -> int:
    h = 0
    while pages > 1:
        pages = (pages + 1) // 2
        h += 1
    return h


def _valid_entry(key, val) -> bool:
    return (isinstance(key, str) and 1 <= len(key) <= 128 and
            type(val) is int and 0 <= val < 2**63)


def _validate_header(header):
    if (not isinstance(header, dict) or set(header) !=
            {'protocol', 'page_size', 'total', 'pages', 'root'} or
            header['protocol'] != ALGORITHM or
            header['page_size'] != PAGE_SIZE or
            type(header['total']) is not int or
            not 0 <= header['total'] <= MAX_ENTRIES or
            type(header['pages']) is not int or
            header['pages'] != max(1, (header['total'] + PAGE_SIZE - 1)//PAGE_SIZE) or
            not isinstance(header['root'], str) or len(header['root']) != 64):
        raise PatchError('invalid paged-frontier commitment')
    try:
        bytes.fromhex(header['root'])
    except ValueError as ex:
        raise PatchError('invalid frontier root') from ex


def build_frontier_pages(cutoffs: Mapping[str, int]):
    """Return constant-sized signed-header candidate and complete page proofs.

    The function is run once per candidate checkpoint, not once per training
    iteration. Memory is O(participants), while each wire page is O(32).
    """
    if not isinstance(cutoffs, Mapping) or len(cutoffs) > MAX_ENTRIES:
        raise PatchError('invalid frontier size')
    if any(not _valid_entry(peer, seq) for peer, seq in cutoffs.items()):
        raise PatchError('invalid frontier entry')
    entries = [[peer, seq] for peer, seq in sorted(cutoffs.items())]
    chunks = [entries[i:i+PAGE_SIZE] for i in range(0, len(entries), PAGE_SIZE)] or [[]]
    levels = [[_leaf(i, part) for i, part in enumerate(chunks)]]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        levels.append([_parent(cur[i], cur[i+1] if i+1 < len(cur) else cur[i])
                       for i in range(0, len(cur), 2)])
    header = {'protocol': ALGORITHM, 'page_size': PAGE_SIZE,
              'total': len(entries), 'pages': len(chunks),
              'root': levels[-1][0].hex()}
    pages = []
    for i, part in enumerate(chunks):
        pos = i
        siblings = []
        for nodes in levels[:-1]:
            pair = pos ^ 1
            siblings.append(nodes[pair if pair < len(nodes) else pos].hex())
            pos //= 2
        pages.append({'index': i, 'entries': part, 'siblings': siblings})
    return header, pages


def verify_frontier_page(header: dict, page: dict) -> list:
    """Check membership in a committed frontier. Does NOT prove full coverage."""
    _validate_header(header)
    if not isinstance(page, dict) or set(page) != {'index', 'entries', 'siblings'}:
        raise PatchError('invalid frontier page')
    i = page['index']
    if type(i) is not int or not 0 <= i < header['pages']:
        raise PatchError('frontier page index outside commitment')
    entries = page['entries']
    expected = min(PAGE_SIZE, header['total']-i*PAGE_SIZE)
    if (not isinstance(entries, list) or len(entries) != expected or
        any(not isinstance(e, list) or len(e) != 2 or not _valid_entry(*e) for e in entries) or
        any(a[0] >= b[0] for a, b in zip(entries, entries[1:]))):
        raise PatchError('malformed/noncanonical frontier page')
    proof = page['siblings']
    if not isinstance(proof, list) or len(proof) != _height(header['pages']):
        raise PatchError('frontier page has wrong proof length')
    h = _leaf(i, entries)
    position = i
    width = header['pages']
    for sibling in proof:
        if not isinstance(sibling, str) or len(sibling) != 64:
            raise PatchError('invalid frontier proof')
        try: other = bytes.fromhex(sibling)
        except ValueError as ex: raise PatchError('invalid frontier proof') from ex
        if position % 2 == 0:
            if position+1 >= width and other != h:
                raise PatchError('invalid frontier odd-leaf padding')
            h = _parent(h, other)
        else:
            h = _parent(other, h)
        position //= 2
        width = (width+1)//2
    if h.hex() != header['root']:
        raise PatchError('frontier page proof does not match root')
    return entries


def verify_full_frontier(header: dict, pages) -> dict[str,int]:
    """Require *all* pages and global uniqueness/order, before a committee vote."""
    _validate_header(header)
    if len(pages) != header['pages']:
        raise PatchError('incomplete checkpoint frontier')
    result = {}
    last = None
    for i, page in enumerate(pages):
        if not isinstance(page, dict) or page.get('index') != i:
            raise PatchError('missing/unsorted frontier page')
        for peer, seq in verify_frontier_page(header, page):
            if last is not None and peer <= last:
                raise PatchError('duplicate/out-of-order frontier peer')
            result[peer] = seq
            last = peer
    if len(result) != header['total']:
        raise PatchError('incomplete frontier count')
    return result


def header_digest(header: dict) -> str:
    _validate_header(header)
    return _h(b'Kagami-frontier-header-v1\0' + _canonical_json(header)).hex()
