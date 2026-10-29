"""Stable Kagami data-root selection across ZIP/app upgrades.

Never copy, rename, delete, or silently merge training state. An explicit
KAGAMI_DATA_DIR wins; otherwise a pinned pointer is used across restarts.
Legacy relative .fedpara-gui stores are adopted only when unambiguous.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _config_file(home: Path) -> Path:
    return home / '.config' / 'kagami' / 'data-root.txt'


def _has_models(path: Path) -> bool:
    try:
        return (path / '.fedpara-gui' / 'rooms').is_dir()
    except OSError:
        return False


def choose_data_root(*, cwd: Path | None = None, home: Path | None = None,
                     environment: dict | None = None) -> Path:
    """Resolve a persistent root without modifying existing room data."""
    env = os.environ if environment is None else environment
    current = (Path.cwd() if cwd is None else Path(cwd)).expanduser().resolve()
    user_home = (Path.home() if home is None else Path(home)).expanduser().resolve()
    explicit = env.get('KAGAMI_DATA_DIR', '').strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    pin = _config_file(user_home)
    if pin.is_file() and not pin.is_symlink():
        stored = pin.read_text(encoding='utf8').strip()
        if stored:
            remembered = Path(stored).expanduser().resolve()
            if _has_models(remembered):
                return remembered
    if _has_models(current):
        return current
    # A newly extracted distribution often lives directly beside the old ZIP
    # extraction folder. Only adopt a SINGLE matching candidate, so two old
    # training histories are never conflated by a heuristic.
    try:
        candidates = [p.resolve() for p in current.parent.iterdir()
                      if p.is_dir() and not p.is_symlink() and
                      p.name.lower().startswith(('kagami-', 'kagami_', 'fedpara-', 'fedpara_'))
                      and _has_models(p)]
    except OSError:
        candidates = []
    if len(candidates) == 1:
        return candidates[0]
    if sys.platform == 'darwin':
        return user_home / 'Library' / 'Application Support' / 'Kagami'
    if sys.platform == 'win32':
        return Path(env.get('LOCALAPPDATA') or user_home / 'AppData' / 'Local') / 'Kagami'
    return Path(env.get('XDG_DATA_HOME') or user_home / '.local' / 'share') / 'kagami'


def pin_data_root(root: Path, *, home: Path | None = None) -> None:
    """Persist only a directory pointer; never migrate source files."""
    user_home = (Path.home() if home is None else Path(home)).expanduser().resolve()
    destination = _config_file(user_home)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f'.{os.getpid()}.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as handle:
            handle.write(str(root.resolve()) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
