"""Off-loop, quota-controlled checkpoint preparation for LiveEpochSwitchService.

Transport is supplied as authenticated ranged RPCs; no remote paths/code.
The caller must run a generation-aware data plane before enabling adoption.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil

from .epoch_objects_v50 import EpochObjectStore, ByteRateLimiter, ResumableEpochFetch, CacheError, fsync_dir, digest_ok
from .core import PatchError
from .shards import parse_manifest, shard_map, semantic_shard_sha256, verify_shards, _blob_path


@dataclass(frozen=True)
class StagedCheckpoint:
    manifest: Path
    store: Path
    owner: str


class EpochCacheWorker:
    """All fsync/hash/SQLite operations run on ONE dedicated disk thread."""
    def __init__(self, root, **limits):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='epoch-cache')
        self._opening = self.pool.submit(EpochObjectStore, root, **limits)
        self.closed = False

    async def call(self, method, *args, **kwargs):
        if self.closed:
            raise RuntimeError('closed epoch cache')
        def run():
            store = self._opening.result()
            return getattr(store, method)(*args, **kwargs)
        # Cancellation does not interrupt a started fsync/SQL commit. Subsequent
        # calls on this executor run AFTER it and observe the committed offset.
        return await asyncio.wrap_future(self.pool.submit(run))

    async def status(self,*args):return await self.call('status',*args)
    async def append(self,*args):return await self.call('append',*args)
    async def path(self,*args):return await self.call('path',*args)

    async def close(self):
        if self.closed:return
        try:
            await self.call('close')
        finally:
            self.closed = True
            # All previously submitted jobs are complete; do not block the
            # relay thread on worker teardown after a cancelled transfer.
            await asyncio.to_thread(self.pool.shutdown, wait=True)


class BoundedCheckpointStager:
    """Prepare a verified checkpoint while the old trainer remains admitted.

    fetch_manifest(peer, digest, max_bytes) and fetch_block(peer,digest,offset,n)
    are authenticated application callbacks. The transport must enforce the
    supplied length caps before allocating. All candidate shard bytes are
    reserved atomically before any shard transfer. Failed stages stay resumable.

    No journal GC authority is granted here. ``retire`` only removes this
    cache's views; lifecycle integration must prove the model no longer uses
    them before calling it. Never release the current base on elapsed time.
    """
    def __init__(self, root, *, peers, fetch_manifest, fetch_block,
                 quota_bytes=8*1024**3, bytes_per_second=4*1024**2,
                 max_manifest_bytes=16*1024**2, max_neighbors=8,
                 max_objects=100_000, max_state_bytes=2*1024**4):
        if not 1 <= max_manifest_bytes <= 96*1024**2:
            raise ValueError('invalid manifest size limit')
        self.root = Path(root)
        self.peers = peers
        self.fetch_manifest = fetch_manifest
        self.max_manifest_bytes = max_manifest_bytes
        self.max_neighbors = max_neighbors
        self.max_state_bytes = max_state_bytes
        self.cache = EpochCacheWorker(self.root/'cache',quota_bytes=quota_bytes,
            max_objects=max_objects,max_object_bytes=max(64*1024**2,max_manifest_bytes))
        self.limiter = ByteRateLimiter(bytes_per_second)
        self.fetcher = ResumableEpochFetch(self.cache,fetch_block,limiter=self.limiter,max_neighbors=max_neighbors)
        self.lock = asyncio.Lock()

    async def close(self):
        await self.cache.close()

    def _peers(self):
        selected=[]
        for peer in self.peers():
            if peer not in selected:selected.append(peer)
            if len(selected)==self.max_neighbors:break
        return selected

    async def prepare(self, proposal):
        async with self.lock:
            digest = proposal.get('manifest')
            semantic = proposal.get('semantic')
            if not digest_ok(digest) or not digest_ok(semantic):
                raise PatchError('invalid certified checkpoint digest')
            owner='checkpoint-'+digest
            peers=self._peers()
            raw=None
            try:
                path=await self.cache.path(digest)
                raw=await asyncio.to_thread(path.read_bytes)
            except CacheError:
                pass
            if raw is None:
                last_error=None
                for peer in peers:
                    try:
                        # Manifest admission is also rate-budgeted. Reserve its
                        # maximum upfront, so a lying source cannot bypass rate.
                        remaining=self.max_manifest_bytes
                        while remaining:
                            amount=min(remaining,self.limiter.burst)
                            await self.limiter.take(amount)
                            remaining-=amount
                        raw=await asyncio.wait_for(self.fetch_manifest(peer,digest,self.max_manifest_bytes),10.)
                        if (not isinstance(raw,bytes) or not 0<len(raw)<=self.max_manifest_bytes
                                or hashlib.sha256(raw).hexdigest()!=digest):
                            raise CacheError('manifest hash/length mismatch')
                        break
                    except (OSError,TimeoutError,CacheError) as exc:
                        last_error=exc
                        raw=None
                if raw is None:
                    raise ConnectionError('no verified checkpoint manifest source') from last_error
            meta=await asyncio.to_thread(parse_manifest,raw,max_state_bytes=self.max_state_bytes,
                                         max_manifest_bytes=self.max_manifest_bytes)
            if semantic_shard_sha256(meta)!=semantic:
                raise PatchError('checkpoint semantic digest differs from certificate')
            specs=shard_map(meta)
            if digest in specs and specs[digest]!=len(raw):
                raise PatchError('conflicting manifest/object size')
            specs[digest]=len(raw)
            await self.cache.call('pin',owner,specs)
            size,offset,complete=await self.cache.status(digest)
            if not complete:
                while offset<len(raw):
                    chunk=raw[offset:offset+256*1024]
                    await self.cache.append(digest,offset,chunk)
                    offset+=len(chunk)
            for shard in specs:
                if shard != digest:
                    await self.fetcher.download(shard,peers)
            # A view is a tree of hardlinks into pinned objects, never copied
            # payloads. Shared shards across generations consume disk once.
            target=self.root/'views'/digest
            paths={key:await self.cache.path(key) for key in specs}
            def build_view():
                target.mkdir(parents=True,exist_ok=True)
                if target.is_symlink() or target.parent.is_symlink():
                    raise CacheError('unsafe checkpoint view')
                def link(source,dest):
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.is_symlink():raise CacheError('unsafe checkpoint link')
                    if dest.exists():
                        if not os.path.samefile(source,dest):
                            raise CacheError('unexpected checkpoint view object')
                    else:
                        os.link(source,dest)
                    fsync_dir(dest.parent)
                link(paths[digest],target/'checkpoint.fmanifest')
                for key in specs:
                    if key!=digest:link(paths[key],_blob_path(target,key))
                verify_shards(meta,target)
                fsync_dir(target)
                fsync_dir(target.parent)
            await _finish_disk_mutation(build_view)
            return StagedCheckpoint(target/'checkpoint.fmanifest',target,owner)

    async def retire(self, checkpoint:StagedCheckpoint):
        """Call only after local durable retirement proof; no remote GC command."""
        async with self.lock:
            digest=checkpoint.owner.removeprefix('checkpoint-')
            target=self.root/'views'/digest
            if (not digest_ok(digest) or checkpoint.store!=target or
                    checkpoint.manifest!=target/'checkpoint.fmanifest'):
                raise CacheError('foreign checkpoint retirement')
            def remove():
                if target.is_symlink():raise CacheError('unsafe checkpoint view')
                if target.exists():shutil.rmtree(target)
                if target.parent.exists():fsync_dir(target.parent)
            await _finish_disk_mutation(remove)
            await self.cache.call('release',checkpoint.owner)
            await self.cache.call('collect',all_unpinned=True)


async def _finish_disk_mutation(fn):
    # Do not release a lifecycle lock while an uncancellable filesystem job
    # can still create hardlinks to objects another coroutine may unpin.
    task=asyncio.create_task(asyncio.to_thread(fn))
    cancelled=False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled=True
    result=task.result()
    if cancelled:raise asyncio.CancelledError
    return result
