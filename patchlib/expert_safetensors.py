"""Checkpoint-selective safetensors loading for unpacked ModuleList experts.

Supports local single-file safetensors and HF shard index files. All tensors
except non-owned individual experts are loaded. Safe lazy/meta instantiation is
required to avoid initially allocating all expert weights.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import torch
from torch import nn
from safetensors import safe_open

EXPERT=re.compile(r'(^|\.)(experts)\.(\d+)\.')


def checkpoint_files(path:Path):
    path=Path(path)
    if path.is_file() and path.suffix=='.safetensors':return [path]
    if not path.is_dir():raise FileNotFoundError(path)
    index=path/'model.safetensors.index.json'
    if index.is_file():
        obj=json.loads(index.read_text())
        weight_map=obj.get('weight_map')
        if not isinstance(weight_map,dict) or len(weight_map)>1_000_000:raise ValueError('bad checkpoint index')
        names=set(weight_map.values())
        if any(not isinstance(n,str) or '/' in n or '\\' in n or not n.endswith('.safetensors') for n in names):
            raise ValueError('unsafe checkpoint paths')
        return [path/name for name in sorted(names)]
    single=path/'model.safetensors'
    if single.is_file():return [single]
    raise FileNotFoundError('only safetensors checkpoints with valid shard index supported')


def expert_tensor_key(name):
    m=EXPERT.search(name)
    if m is None:return None
    return name[:m.start(3)-1],int(m.group(3))


def iter_selected_tensors(path,selected:dict[str,set[int]]):
    """Yield only target tensors; unselected experts never enter RAM."""
    if not all(isinstance(v,(set,list,tuple)) for v in selected.values()):
        raise ValueError('bad selected expert sets')
    for file in checkpoint_files(Path(path)):
        with safe_open(str(file),framework='pt',device='cpu') as f:
            for name in f.keys():
                ex=expert_tensor_key(name)
                if ex is not None and ex[1] not in selected.get(ex[0],set()):continue
                yield name,f.get_tensor(name)


def load_into_meta_model(model:nn.Module,path,selected:dict[str,set[int]],*,device='cpu',max_elements=10_000_000_000):
    """Materialize checkpoint into meta-initialized nn.Module, assigning tensors one at a time.

    Caller must replace all unselected experts with remote proxies or stubs.
    Nonexpert checkpoint tensors always load. Requires exact parameter/buffer
    registration and rejects leftover meta tensors not belonging to removed experts.
    """
    loaded=0
    for name,tensor in iter_selected_tensors(path,selected):
        if tensor.numel()>max_elements:raise ValueError('oversized tensor')
        components=name.split('.')
        try:mod=model.get_submodule('.'.join(components[:-1])) if len(components)>1 else model
        except AttributeError:continue  # Remote expert submodule absent, expected
        leaf=components[-1]
        if leaf in mod._parameters:
            old=mod._parameters[leaf]
            mod.register_parameter(leaf,nn.Parameter(tensor.to(device),requires_grad=old.requires_grad))
        elif leaf in mod._buffers:
            mod.register_buffer(leaf,tensor.to(device),persistent=(leaf not in mod._non_persistent_buffers_set))
        else:continue
        loaded+=1
    missing=[name for name,p in model.named_parameters() if p.device.type=='meta']
    missing += [name for name,b in model.named_buffers() if b.device.type=='meta']
    if missing:raise ValueError('checkpoint incomplete / unmapped meta tensors: '+', '.join(missing[:8]))
    return loaded
