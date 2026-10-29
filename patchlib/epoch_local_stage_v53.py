"""Atomic, zero-copy local view of a previously downloaded certified checkpoint.

Every model shard is validated against its content-addressed manifest before
linking.  Publication uses a sibling directory rename after directory fsync;
concurrent stage operations are serialized with a process-safe flock.  A crash
cannot make a partially hardlinked view look complete.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from .core import PatchError
from .shards import read_manifest, shard_map, semantic_shard_sha256, verify_shards, _blob_path
from .live_epoch_switch_v48 import verify_candidate_files


def _sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|getattr(os,'O_NOFOLLOW',0))
    try:os.fsync(fd)
    finally:os.close(fd)


def _safe_dir(path):
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise PatchError('unsafe checkpoint view path')


def stage_verified_local_view(proposal, *, root, incoming, objects,
                              max_state_bytes=2*1024**4):
    """Stage a certified view with hardlinks and an atomic publish boundary.

    No transfer/GC/voting is authorized by this function.  Its flock ensures
    that only *abandoned* temporary views from a crashed process are reclaimed.
    An already published but corrupt view is never quietly replaced.
    """
    from .epoch_objects_v50 import digest_ok
    digest=proposal.get('manifest')
    if not digest_ok(digest) or not digest_ok(proposal.get('semantic')):
        raise PatchError('invalid certified candidate hashes')
    root=Path(root);incoming=Path(incoming);objects=Path(objects)
    views=root/'checkpoints'/'views'
    target=views/digest
    for directory in (root,root/'checkpoints',views,incoming,objects):
        _safe_dir(directory)
    # Missing input must not create an empty checkpoint cache on each retry.
    source=incoming/(digest+'.fmanifest')
    if not target.exists() and (source.is_symlink() or not source.is_file()):
        raise FileNotFoundError('certified checkpoint not locally downloaded')
    views.mkdir(parents=True,exist_ok=True)
    lock_path=views/'.stage.lock'
    if lock_path.is_symlink():raise PatchError('unsafe checkpoint lock')
    lock_fd=os.open(lock_path,os.O_CREAT|os.O_RDWR|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        fcntl.flock(lock_fd,fcntl.LOCK_EX)
        # No other local invocation can own an incomplete work directory once
        # we hold the lock (flock is released automatically after a crash).
        for abandoned in views.iterdir():
            if abandoned.name.startswith('.building-'):
                if abandoned.is_symlink() or not abandoned.is_dir():
                    raise PatchError('unexpected checkpoint staging object')
                shutil.rmtree(abandoned)
        _safe_dir(target)
        current=target/'checkpoint.fmanifest'
        if target.exists():
            if not current.is_file() or current.is_symlink():
                raise PatchError('unverified legacy partial checkpoint view; isolate manually')
            candidate=SimpleNamespace(manifest=current,store=target)
            verify_candidate_files(candidate,proposal,max_state_bytes=max_state_bytes)
            return candidate
        source=incoming/(digest+'.fmanifest')
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError('certified checkpoint not locally downloaded')
        h=hashlib.sha256()
        with source.open('rb') as fp:
            for part in iter(lambda:fp.read(1024*1024),b''):h.update(part)
        if h.hexdigest()!=digest:
            raise PatchError('downloaded certified manifest hash mismatch')
        meta=read_manifest(source,max_state_bytes=max_state_bytes)
        if semantic_shard_sha256(meta)!=proposal['semantic']:
            raise PatchError('downloaded checkpoint semantic hash mismatch')
        verify_shards(meta,objects)
        building=Path(tempfile.mkdtemp(prefix='.building-',dir=views))
        try:
            parent_dirs=set()
            def link(src,dst):
                if src.is_symlink() or not src.is_file():
                    raise PatchError('unsafe snapshot source shard')
                dst.parent.mkdir(parents=True,exist_ok=True)
                if dst.is_symlink() or dst.exists():
                    raise PatchError('collision in unpublished checkpoint')
                os.link(src,dst)
                parent_dirs.add(dst.parent)
            link(source,building/'checkpoint.fmanifest')
            for key in shard_map(meta):
                link(_blob_path(objects,key),_blob_path(building,key))
            candidate=SimpleNamespace(manifest=building/'checkpoint.fmanifest',store=building)
            verify_candidate_files(candidate,proposal,max_state_bytes=max_state_bytes)
            # Link entries are durable after fsync of each containing directory.
            # O(dirs), not one fsync syscall PER shard.
            for directory in sorted(parent_dirs,key=lambda p:len(p.parts),reverse=True):
                _sync_dir(directory)
            _sync_dir(building)
            _sync_dir(views)
            os.rename(building,target)
            _sync_dir(views)
            return SimpleNamespace(manifest=current,store=target)
        finally:
            if building.exists():shutil.rmtree(building)
    finally:
        fcntl.flock(lock_fd,fcntl.LOCK_UN)
        os.close(lock_fd)
