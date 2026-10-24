"""Safe, reproducible local genesis creation for packaged Hugging Face models.

The control plane never receives model bytes. Creation happens in the training
subprocess, not on the GUI/pywebview thread. Only trusted packaged models can be
automatically imported. A peer-assisted bootstrap transport is a separate concern.
"""
from __future__ import annotations

import copy
import os
import secrets
import time
from pathlib import Path

from .state import export_state, inspect_state, verify_state, semantic_state_sha256
from .core import PatchError


def ensure_state(args, *, load_model):
    """Resolve --state-base, creating a checkpoint if missing when --auto-state.

    Side effects: writes only to the requested user-local cache. Existing files
    are verified, never overwritten. Lock guards concurrent GUI launches.
    """
    if not getattr(args, 'auto_state', False):
        return
    if (getattr(args,'factory_file',None) or getattr(args,'trust_remote_code',False)
            or getattr(args,'allow_legacy_bin',False)):
        raise PermissionError('auto-bootstrap cannot execute custom model code or load pickle checkpoints')
    if getattr(args,'room_manifest',None):
        from .room_manifest import load_manifest
        room = load_manifest(args.room_manifest)
        if room['model']['id'] != args.model or room['model']['revision'] != args.revision:
            raise PatchError('room model and local bootstrap model disagree')
        if room['code']['sha256']:
            raise PermissionError('room includes executable model code; explicit offline sandbox bootstrap required')
    if not args.state_base:
        if not getattr(args,'room_id',None):
            raise ValueError('--auto-state without --state-base requires --room-id')
        args.state_base = str(Path('.fedpara-gui')/'rooms'/args.room_id/'genesis.fstate')
    path = Path(args.state_base).expanduser().resolve()
    if path.suffix != '.fstate' or path.is_dir():
        raise ValueError('automatic genesis path must be a .fstate file')
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.chmod(path.parent,0o700)
    if path.exists():
        meta = verify_state(path,max_bytes=args.max_state_gb*1024**3)
        if meta['model_id'] != (args.model_id or args.model):
            raise PatchError('cached genesis belongs to another model; refusing to overwrite')
        print('bootstrap: verified cached genesis, tensor content:',semantic_state_sha256(path),flush=True)
        args.state_base=str(path)
        return
    # Serialise creation across *different GUI processes*.  The old O_EXCL
    # ".creating" marker made the second window exit immediately, and a crash
    # left a permanent marker. flock is reclaimed by the kernel on process exit.
    # The fixed lock filename must never be deleted (another waiter may hold it).
    import fcntl
    lock = path.with_name(path.name + '.creating.lock')
    fd = os.open(lock, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        os.fchmod(fd, 0o600)
        started = time.monotonic()
        next_report = started
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                now = time.monotonic()
                if now-started > 600:
                    raise TimeoutError('Another GUI has not completed the shared genesis '
                                       'checkpoint after 600s; inspect its log and retry')
                if now >= next_report:
                    print('bootstrap: waiting for the other GUI to finish shared genesis creation',
                          flush=True)
                    next_report = now + 10
                time.sleep(.25)
        # Recheck after acquiring: the first process may have just completed.
        if path.exists():
            meta=verify_state(path,max_bytes=args.max_state_gb*1024**3)
            if meta['model_id'] != (args.model_id or args.model):
                raise PatchError('cached genesis belongs to another model; refusing to overwrite')
            print('bootstrap: reused peer-created genesis',semantic_state_sha256(path),flush=True)
            args.state_base=str(path)
            return
        local = copy.copy(args)
        local.state_base = None
        local.device = 'cpu'  # avoid GPU memory spikes just to serialize genesis
        local.moe_plan = None  # genesis must describe the unsliced model
        print('bootstrap: creating shared genesis from model config / loading pretrained',args.model,
              'revision',args.revision,flush=True)
        model, revision = load_model(local,use_adapter=False)
        # export_state uses an atomic temp-file + rename internally.
        export_state(model,path,model_id=args.model_id or args.model,
                     revision=revision,peer_id='genesis',seq=0,
                     max_bytes=args.max_state_gb*1024**3)
        digest=semantic_state_sha256(path,verify=True)
        print('bootstrap: saved',path,'semantic_sha256='+digest,flush=True)
        args.state_base=str(path)
    finally:
        os.close(fd)
