"""Separate mutable disk Linear scratch per live model/epoch.

A shadow model must NEVER truncate/overwrite the current worker's mmap base.
Crash leftovers are reclaimed under non-blocking file locks. Only locations
created by this allocator (scratch-*) are eligible for automatic deletion.
"""
from __future__ import annotations
import fcntl
import os
import shutil
import uuid
import weakref
from pathlib import Path


def _release(lease, scratch):
    try:
        lease.close()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def allocate(root):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    for scratch in root.glob('scratch-*'):
        if not scratch.is_dir() or scratch.is_symlink() or not (scratch/'lease').is_file():
            continue
        try:
            with (scratch/'lease').open('rb') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                shutil.rmtree(scratch)
        except (OSError,BlockingIOError):
            pass
    scratch=root/('scratch-'+uuid.uuid4().hex)
    scratch.mkdir(mode=0o700)
    lease=(scratch/'lease').open('wb')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return scratch,lease


def attach(model,scratch,lease):
    weakref.finalize(model,_release,lease,scratch)
    return model
