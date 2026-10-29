"""Signed locally fenced model/cut exports for compact crash-fault epochs.

Only call `export_at_step_boundary` from the owner of the trainer optimizer
loop, with local gradients SEALED and all journal append paths serialized by
`inbound_commit_lock`. This trust assumption is NOT enforced by a remote flag.
The exporter holds that lock while streaming model shards, so it is correct
but NOT a strict zero-pause/300B exporter. Receipt verifies the pinned honest
producer's assertion (CFT), not a Byzantine proof of model computation.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
from .core import PatchError, _canonical_json
from .frontier_commitment_v44 import build_frontier_pages
from .epoch_certificate_v46 import proposal_v2
from .online_generation import _digest, _sign, _verify
from .shards import (export_state_sharded,read_manifest,semantic_shard_sha256,
                     shard_map,_blob_path,verify_shards)
from .epoch_live_rebase_v52 import verify_certified_cut

RECEIPT_PROTOCOL='kagami-fenced-trainer-export-v57'
MAX_EXPORT_ORIGINS=4096


def _indexed_frontier(journal):
    """O(origins) memory; reject holes instead of certifying max(seq)."""
    frontier={}
    sha=hashlib.sha256()
    for origin,seq,digest in journal.db.execute('SELECT origin,seq,digest FROM events ORDER BY origin,seq'):
        if seq!=frontier.get(origin,0)+1:
            raise PatchError('cannot export noncontiguous signed journal prefix')
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise PatchError('invalid signed journal event digest')
        frontier[origin]=seq
        if len(frontier)>MAX_EXPORT_ORIGINS:
            raise PatchError('more than 4096 active origins in compact export')
        sha.update(_canonical_json([origin,seq,digest])+b'\n')
    return frontier,sha.hexdigest()


def verify_receipt(wire,proposal,*,trusted,committee):
    payload=_verify(wire,trusted)
    expected={'protocol':RECEIPT_PROTOCOL,'room':proposal['room'],
              'epoch':proposal['epoch'],'proposal':_digest(proposal),
              'manifest':proposal['manifest'],'inventory':proposal['inventory'],
              'frontier_digest':proposal['frontier_digest']}
    if payload!=expected or wire['peer'] not in committee or wire['peer'] not in proposal['replicas']:
        raise PatchError('untrusted or mismatching trainer cut receipt')
    return True


def export_at_step_boundary(*, journal, model, adapter, store, incoming, room, epoch,
                            parent, replicas, signer, peer, max_state_bytes,
                            shard_bytes=4*1024**2, frozen=None):
    """Synchronous under a caller-owned TRAINER writer fence/ingress lock.

    Must not be called from an arbitrary network callback; no concurrent
    optimizer step, model apply, or journal writer may run in this interval.
    The returned receipt is an independently signed assertion by the exporter.
    """
    if peer not in replicas or not callable(signer):
        raise PatchError('exporter must be pinned committee replica')
    if frozen is None:
        frontier, inventory=_indexed_frontier(journal)
        effective_adapter=adapter
        export_seq=journal.count()
    else:
        if frozen.frontier is None or frozen.inventory is None:
            raise PatchError('unfinalized frozen checkpoint frontier')
        # A frozen, immutable low-rank view can be serialized while the owner
        # continues training. Recheck its shared base before and after export.
        frozen.assert_immutable_base()
        frontier, inventory=frozen.frontier, frozen.inventory
        effective_adapter=frozen.adapter
        export_seq=frozen.event_count
    header,pages=build_frontier_pages(frontier)
    incoming=Path(incoming);store=Path(store)
    if incoming.is_symlink() or store.is_symlink():
        raise PatchError('unsafe export location')
    incoming.mkdir(parents=True,exist_ok=True)
    store.mkdir(parents=True,exist_ok=True)
    # Publish only a complete content-addressed manifest. No consumer may
    # treat the unpublished name as a verified checkpoint.
    fd,temporary=tempfile.mkstemp(prefix='.export-',suffix='.fmanifest',dir=incoming)
    os.close(fd)
    temp=Path(temporary)
    try:
        meta=export_state_sharded(model,store,temp,
            model_id=adapter.model_id,revision=adapter.revision,peer_id=peer,
            seq=export_seq,adapter=effective_adapter,shard_bytes=shard_bytes,
            max_state_bytes=max_state_bytes,
            frozen_base_sources=(frozen.frozen_base_sources if frozen is not None else None))
        # Harden the content-addressed blobs against a sudden power failure.
        # The regular streaming exporter omits per-shard fsync for throughput;
        # a *certified* checkpoint must not sign undurable shard bytes.
        verify_shards(meta,store)
        directories=set()
        for digest in shard_map(meta):
            shard=_blob_path(store,digest)
            if shard.is_symlink():raise PatchError('unsafe export shard symlink')
            with shard.open('rb') as fp:os.fsync(fp.fileno())
            directories.add(shard.parent)
        for directory in directories:
            fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        # A full indexed cut is checked again after export, preventing any
        # accidentally unfenced DB mutation from silently becoming certified.
        if frozen is None:
            if _indexed_frontier(journal)!=(frontier,inventory):
                raise PatchError('signed journal changed during snapshot export')
        else:
            frozen.assert_immutable_base()
            # Suffix events are allowed: verify exactly the captured *prefix*
            # against the durable journal rather than requiring a frozen DB.
            verify_certified_cut(journal,frontier,inventory)
        with temp.open('rb') as fp:
            manifest_hash=hashlib.file_digest(fp,'sha256').hexdigest()
        p=proposal_v2(room=room,epoch=epoch,parent=parent,
            manifest=manifest_hash,semantic=semantic_shard_sha256(meta),
            inventory=inventory,frontier_header=header,replicas=list(replicas))
        verify_certified_cut(journal,frontier,inventory)
        published=incoming/(manifest_hash+'.fmanifest')
        if published.is_symlink():raise PatchError('unsafe published manifest')
        if published.exists():
            if hashlib.sha256(published.read_bytes()).hexdigest()!=manifest_hash:
                raise PatchError('colliding published manifest')
        else:
            # fsync manifest and enclosing directory before signing receipt;
            # export_state_sharded uses atomic rename but does not fsync itself.
            with temp.open('rb') as fp:os.fsync(fp.fileno())
            os.replace(temp,published)
            dfd=os.open(incoming,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(dfd)
            finally:os.close(dfd)
        receipt=_sign({'protocol':RECEIPT_PROTOCOL,'room':room,'epoch':epoch,
             'proposal':_digest(p),'manifest':manifest_hash,
             'inventory':inventory,'frontier_digest':p['frontier_digest']},peer,signer)
        return p,pages,receipt
    finally:
        temp.unlink(missing_ok=True)
