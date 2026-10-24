"""One durable per-peer journal per GUI instance, locked across processes.

The room's immutable genesis and dataset manifest stay shared. A window gets an
exclusive, persistent peer profile. If all windows exit, the same profiles can
be reclaimed and their cryptographic identities/journals are preserved.
"""
from __future__ import annotations

import errno
import os
import secrets
from pathlib import Path

try:
    import fcntl
except ImportError:  # POSIX/macOS only; fail closed instead of identity collisions.
    fcntl = None


class PeerProfile:
    def __init__(self, name: str, journal: Path, lock_fd: int):
        self.name = name
        self.journal = journal
        self._lock_fd = lock_fd

    def close(self):
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None


def _acquire(lock_path: Path):
    if fcntl is None:
        raise RuntimeError('GUI multi-peer profiles require POSIX flock; unsupported on this system')
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except OSError as e:
        os.close(fd)
        if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
            return None
        raise


def allocate_profile(room_dir: Path) -> PeerProfile:
    """Reclaim a dormant profile or create an independent one for this window.

    Existing legacy events.sqlite3 is retained at its original path to avoid
    migrating SQLite WAL files or losing an earlier single-window training run.
    """
    room_dir = Path(room_dir)
    room_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    profiles = room_dir / 'peers'
    profiles.mkdir(parents=True, exist_ok=True, mode=0o700)
    candidates = []
    legacy = room_dir / 'events.sqlite3'
    if legacy.exists() or Path(str(legacy) + '.identity.pem').exists():
        candidates.append(('legacy', legacy, profiles / 'legacy.lock'))
    for directory in sorted(profiles.iterdir()):
        if directory.is_dir() and not directory.is_symlink() and directory.name.startswith('peer-'):
            candidates.append((directory.name, directory / 'events.sqlite3', directory / 'active.lock'))
    for name, journal, lock in candidates:
        fd = _acquire(lock)
        if fd is not None:
            return PeerProfile(name, journal, fd)
    for _ in range(10):
        name = 'peer-' + secrets.token_hex(8)
        directory = profiles / name
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            continue
        fd = _acquire(directory / 'active.lock')
        if fd is None:
            raise RuntimeError('new peer profile unexpectedly locked')
        return PeerProfile(name, directory / 'events.sqlite3', fd)
    raise RuntimeError('unable to allocate an independent peer profile')
