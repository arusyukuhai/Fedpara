"""Deterministic and immutable expert placement epochs for MoE P2P.

One placement must be pinned for an entire execution/optimizer epoch. Joining
and departing peers are NOT automatically allowed to change owners without
copying a version-consistent expert checkpoint to the replacement owner.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from .expert_parallel import expert_owner
from .core import _canonical_json
from .roomid import validate_room


def plan_experts(room_id:str, layers:list[dict], capacity:dict[str,int],
                 *, replicas:int=2, generation:str='genesis'):
    validate_room(room_id)
    if not isinstance(layers,list) or len(layers)>1024 or not isinstance(capacity,dict):
        raise ValueError('invalid expert plan inputs')
    if len(capacity)>32 or not 1<=len(capacity):raise ValueError('1..32 peers required')
    if not isinstance(generation,str) or not 1<=len(generation)<=128:raise ValueError('invalid generation')
    placed={}
    for info in layers:
        path=info.get('experts'); count=info.get('count')
        if not isinstance(path,str) or len(path)>240 or type(count) is not int or not 1<=count<=100_000:
            raise ValueError('invalid expert layer')
        if path in placed:raise ValueError('duplicate layer')
        placed[path]={str(i):list(expert_owner(room_id,f'{path}.{i}',capacity,replicas)) for i in range(count)}
    plan={'version':1,'room_sha256':hashlib.sha256(room_id.encode()).hexdigest(),
          'generation':generation,'capacity':capacity,'placements':placed}
    if len(_canonical_json(plan))>8*1024*1024:raise ValueError('plan exceeds 8MiB')
    return plan


def plan_digest(plan):return hashlib.sha256(_canonical_json(plan)).hexdigest()


def selected_local_experts(plan,peer):
    result={}
    for path,mapping in plan['placements'].items():
        result[path]={int(i) for i,owners in mapping.items() if peer in owners}
    return result


def remote_experts(plan,peer):
    return {path:{int(i):owners[0] for i,owners in mapping.items() if peer not in owners}
            for path,mapping in plan['placements'].items()}


def save_plan(path:Path,plan):
    path=Path(path);path.write_text(json.dumps(plan,sort_keys=True,indent=2))
    return plan_digest(plan)
