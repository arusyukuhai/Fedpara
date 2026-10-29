"""Opt-in disk-paged AdamW for the live elastic trainer.

Working snapshots use *independent* copy-on-write reflinks where supported;
fall back to streamed copies (slower and longer writer fence). Never hardlink
mutable mmap moment files. An optimizer scratch directory is expendable:
recovery trusts ONLY the journal-tip-bound, hash-checked working snapshot.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import shutil
import uuid
from pathlib import Path

from .core import PatchError
from .disk_adamw import DiskAdamW, adapter_named_parameters

FICLONE = 0x40049409
CHUNK = 4 * 1024 * 1024


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try: os.fsync(fd)
    finally: os.close(fd)


def _copy_immutable(src: Path, dst: Path):
    """Reflink, NOT hardlink; destination keeps original bytes after next step."""
    with src.open('rb') as inp, dst.open('xb') as out:
        try:
            fcntl.ioctl(out.fileno(), FICLONE, inp.fileno())
        except OSError as exc:
            if exc.errno not in (errno.EOPNOTSUPP, errno.ENOTTY, errno.EXDEV,
                                 errno.EINVAL, errno.ENOSYS):
                raise
            inp.seek(0)
            while True:
                b = inp.read(CHUNK)
                if not b: break
                out.write(b)
        out.flush(); os.fsync(out.fileno())


def _hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(CHUNK): h.update(block)
    return h.hexdigest()


class LiveDiskAdamW(DiskAdamW):
    """One-parameter-at-a-time optimizer, crash-safe via separate snapshots.

    The active mmap files are intentionally NOT trusted after an unclean exit.
    Each run gets its own scratch path. Only matching, hashed working snapshots
    can restore moments and factors together.
    """
    live_disk_optimizer_v59 = True

    def __init__(self, adapter, root, *, lr, weight_decay, max_bytes=32*1024**3):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # Recover space left by crashed processes, but never touch a live run.
        for old in self.root.glob('scratch-*'):
            if old.is_symlink() or not old.is_dir() or not (old/'lease').is_file():
                continue
            try:
                with (old/'lease').open('rb') as lease:
                    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    shutil.rmtree(old)
            except (OSError, BlockingIOError):
                pass
        self.max_bytes = int(max_bytes)
        params = adapter_named_parameters(adapter)
        required = sum(p.numel()*8 for _, p in params)
        if required > self.max_bytes or required > shutil.disk_usage(self.root).free:
            raise PatchError('disk AdamW moment storage exceeds configured limit or free disk')
        # A new run MUST NOT resume partially modified mmap files.
        scratch = self.root / ('scratch-' + uuid.uuid4().hex)
        scratch.mkdir(mode=0o700)
        self._lease = (scratch / 'lease').open('wb')
        fcntl.flock(self._lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._closed = False
        self._poisoned = False
        try:
            super().__init__(params, scratch, lr=lr, weight_decay=weight_decay)
        except BaseException:
            self._lease.close()
            shutil.rmtree(scratch)
            raise

    def step(self, closure=None):
        if self._closed or self._poisoned:
            raise PatchError('disk optimizer is closed or a prior step failed')
        try:
            return super().step(closure)
        except BaseException:
            # The original in-place optimizer may have updated a strict prefix
            # of parameters; the trainer must terminate and restart from the
            # last journal-bound working snapshot, not attempt a second step.
            self._poisoned=True
            raise

    def snapshot_moments(self, target):
        """Must be invoked between optimizer steps under adapter.lock.

        Reflink/copy happens here, before the writer resumes. The expensive
        checksum of immutable cloned files happens off-lock in finalize.
        """
        if self._closed or self._poisoned:
            raise PatchError('cannot checkpoint closed or partially failed disk optimizer')
        target = Path(target)
        destination = target.parent / (target.name + '.moments-' + uuid.uuid4().hex)
        destination.mkdir(mode=0o700)
        try:
            self.flush()
            desc = []
            for _, p, key in self.entries:
                for suffix in ('m', 'v'):
                    name = f'{key}.{suffix}.f32'
                    _copy_immutable(self.directory / name, destination / name)
                    desc.append({'file':name, 'bytes':p.numel()*4})
            _sync_dir(destination)
            _sync_dir(destination.parent)
            return {'format':'live-disk-adamw-v59', 'folder':destination.name,
                    'layout':self._description, 'steps':{
                        key:int(self.state[p].get('step', 0)) for _,p,key in self.entries},
                    'files':desc}
        except BaseException:
            shutil.rmtree(destination, ignore_errors=True)
            raise

    def finalize_moments(self, meta, parent):
        dest = Path(parent) / meta['folder']
        for item in meta['files']:
            file = dest / item['file']
            if file.stat().st_size != item['bytes']:
                raise PatchError('disk optimizer snapshot truncated')
            item['sha256'] = _hash(file)

    def restore_moments(self, meta, parent):
        if (meta.get('format') != 'live-disk-adamw-v59' or
                meta.get('layout') != self._description or
                set(meta.get('steps', {})) != {key for _,_,key in self.entries}):
            raise PatchError('disk optimizer working snapshot layout mismatch')
        folder = meta.get('folder')
        # Directory must be directly adjacent to its owning working snapshot.
        if (not isinstance(folder, str) or
                '.working.safetensors.moments-' not in folder or
                Path(folder).name != folder):
            raise PatchError('unsafe optimizer snapshot path')
        source = Path(parent) / folder
        if source.is_symlink() or not source.is_dir():
            raise PatchError('missing or unsafe optimizer snapshot directory')
        expected = {f'{key}.{kind}.f32' for _,_,key in self.entries for kind in ('m','v')}
        entries = meta.get('files')
        if not isinstance(entries, list) or {x.get('file') for x in entries if isinstance(x,dict)} != expected or len(entries) != len(expected):
            raise PatchError('incomplete optimizer snapshot')
        # Validate every shard before changing ANY live mmap moment file.
        for key, step in meta['steps'].items():
            if type(step) is not int or step < 0 or step > 2**63-1:
                raise PatchError('invalid optimizer step counter')
        for item in entries:
            f = source / item['file']
            if f.is_symlink() or not f.is_file() or f.stat().st_size != item['bytes'] or _hash(f) != item.get('sha256'):
                raise PatchError('damaged optimizer snapshot; refusing partial restore')
        for item in entries:
            dst = self.directory / item['file']
            tmp = self.directory / (item['file'] + '.' + uuid.uuid4().hex + '.tmp')
            try:
                _copy_immutable(source / item['file'],tmp)
                os.replace(tmp,dst)
            finally:tmp.unlink(missing_ok=True)
        for _,p,key in self.entries:
            step = meta['steps'][key]
            if type(step) is not int or step < 0:
                raise PatchError('invalid optimizer step counter')
            self.state[p]['step'] = step
        self.iteration = max(meta['steps'].values(),default=0)
        self.flush()

    def close(self):
        if not self._closed:
            self._closed = True
            self._lease.close()
            shutil.rmtree(self.directory, ignore_errors=True)


def create_live_disk_optimizer(adapter, root, *, lr, weight_decay, max_gb=32):
    return LiveDiskAdamW(adapter,root,lr=lr,weight_decay=weight_decay,
                         max_bytes=max_gb*1024**3)
