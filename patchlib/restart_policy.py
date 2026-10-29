"""Pin genesis-affecting launch options per peer for safe GUI restarts.

A resume MUST reconstruct the same immutable genesis and adapter layout.
Optimizer family / ReLoRA interval / bootstrap checkpoint are not UI defaults.
"""
from __future__ import annotations
import json
import os
import secrets
from pathlib import Path

from .core import PatchError

FIELDS = ('manifest_digest', 'initialization', 'streaming_base', 'base',
          'auto_state', 'optimizer', 'reinit_every')


def policy_path(journal: Path) -> Path:
    return Path(str(journal) + '.launch.json')


def read_policy(journal: Path, digest: str):
    path = policy_path(journal)
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding='utf8'))
    except (OSError, ValueError) as exc:
        raise PatchError('saved model launch policy is corrupt; do not reset the journal') from exc
    if (type(obj) is not dict or obj.get('version') != 1 or
            any(key not in obj for key in FIELDS) or
            obj.get('manifest_digest') != digest or
            obj.get('initialization') not in ('pretrained','config','random') or
            type(obj.get('streaming_base')) is not bool or
            type(obj.get('auto_state')) is not bool or
            obj.get('optimizer') not in ('adamw','lora-muon') or
            type(obj.get('reinit_every')) is not int or
            not 0 <= obj['reinit_every'] <= 100000 or
            (obj['base'] is not None and not isinstance(obj['base'], str)) or
            (obj.get('sharded_base_manifest') is not None and
             not isinstance(obj['sharded_base_manifest'],str)) or
            (obj.get('sharded_base_store') is not None and
             not isinstance(obj['sharded_base_store'],str)) or
            bool(obj.get('sharded_base_manifest')) != bool(obj.get('sharded_base_store'))):
        raise PatchError('saved model launch policy conflicts with this immutable model; do not reset the journal')
    return obj


def write_policy(journal: Path, *, manifest_digest: str, initialization: str,
                 streaming_base: bool, base: Path | None, auto_state: bool,
                 optimizer: str, reinit_every: int,
                 sharded_base_manifest: Path | None = None,
                 sharded_base_store: Path | None = None) -> None:
    path = policy_path(journal)
    obj = {'version':1, 'manifest_digest':manifest_digest, 'initialization':initialization,
           'streaming_base':streaming_base,
           'base':str(base) if base is not None else None,
           'auto_state':auto_state,'optimizer':optimizer,'reinit_every':reinit_every}
    if sharded_base_manifest is not None:
        obj['sharded_base_manifest']=str(sharded_base_manifest)
        obj['sharded_base_store']=str(sharded_base_store)
    if path.is_file():
        saved = read_policy(journal,manifest_digest)
        if saved != obj:
            raise PatchError('refusing to replace pinned resume policy of an existing journal')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+secrets.token_hex(8)+'.tmp')
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        with os.fdopen(fd,'w',encoding='utf8') as out:
            json.dump(obj,out,sort_keys=True)
            out.write('\n');out.flush();os.fsync(out.fileno())
        # A running journal holds the profile lock. Multiple windows must
        # not acquire the same profile simultaneously.
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)
