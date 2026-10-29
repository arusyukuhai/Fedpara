"""Bounded-memory, integrity-checked Kagami shard -> HF safetensors conversion.

Exports trained effective weights as standard HF sharded safetensors *without*
instantiating their dense CPU tensors. A whole checkpoint can greatly exceed RAM.
Only well-formed SHA-256-pinned data shards are read; never torch.load or pickle.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

from .core import PatchError
from .shards import _blob_path, read_manifest

_DTYPE = {
    'torch.float16': 'F16', 'torch.bfloat16': 'BF16', 'torch.float32': 'F32',
    'torch.float64': 'F64', 'torch.int8': 'I8', 'torch.uint8': 'U8',
    'torch.int16': 'I16', 'torch.int32': 'I32', 'torch.int64': 'I64',
    'torch.bool': 'BOOL',
}


def _pack(groups, size_limit):
    pending=[]; total=0
    for tensor in groups:
        if pending and total+tensor['bytes']>size_limit:
            yield pending
            pending=[];total=0
        pending.append(tensor)
        total+=tensor['bytes']
    if pending:yield pending


def export_hf_safetensors(checkpoint: Path, store: Path, dest: Path, *,
                          max_state_bytes: int=2048*1024**3,
                          max_shard_bytes: int=512*1024**2):
    """Write standard HF index and safetensors files using <=1MiB IO buffers.

    Files are committed only after every source shard + full tensor hash matches.
    The caller must create an isolated temporary destination directory so any
    interrupted export can be discarded as a unit.
    """
    if not 16*1024**2 <= max_shard_bytes <= 4*1024**3:
        raise ValueError('invalid safetensors packing budget')
    meta=read_manifest(checkpoint,max_state_bytes=max_state_bytes)
    dest=Path(dest)
    dest.mkdir(parents=True,exist_ok=True,mode=0o700)
    if any(dest.iterdir()):
        raise FileExistsError('safetensors destination must be empty')
    # Tensors not supported by safetensors must fail *before* writing partial files.
    unsupported=[t['name'] for t in meta['tensors'] if t['dtype'] not in _DTYPE]
    if unsupported:
        raise PatchError('unsupported safetensors dtype: '+unsupported[0])
    weight_map={}
    groups=list(_pack(meta['tensors'],max_shard_bytes))
    try:
        for index, specs in enumerate(groups,1):
            fname=f'model-{index:05d}-of-{len(groups):05d}.safetensors'
            final=dest/fname
            part=dest/(fname+'.part')
            offsets={};pos=0
            for spec in specs:
                offsets[spec['name']]={'dtype':_DTYPE[spec['dtype']],
                                      'shape':spec['shape'],
                                      'data_offsets':[pos,pos+spec['bytes']]}
                pos+=spec['bytes']
                weight_map[spec['name']]=fname
            header=json.dumps({'__metadata__':{'format':'pt'},**offsets},
                separators=(',',':'),ensure_ascii=False).encode('utf8')
            # safetensors header length includes right-padding to an 8-byte boundary.
            header+=b' '*((-len(header))%8)
            if len(header)>100*1024**2:
                raise PatchError('safetensors header exceeds format limit')
            fd=os.open(part,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'wb',buffering=1024*1024) as output:
                output.write(struct.pack('<Q',len(header)))
                output.write(header)
                for spec in specs:
                    whole=hashlib.sha256()
                    written=0
                    for chunk in spec['shards']:
                        path=_blob_path(store,chunk['sha256'])
                        # Avoid following symlinks from writable stores.
                        flags=os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)
                        fd_in=os.open(path,flags)
                        with os.fdopen(fd_in,'rb') as source:
                            blockhash=hashlib.sha256();remaining=chunk['size']
                            while remaining:
                                block=source.read(min(1024*1024,remaining))
                                if not block:raise PatchError('truncated checkpoint shard')
                                blockhash.update(block)
                                whole.update(block)
                                output.write(block)
                                remaining-=len(block)
                                written+=len(block)
                            if source.read(1):raise PatchError('trailing bytes in checkpoint shard')
                        if blockhash.hexdigest()!=chunk['sha256']:
                            raise PatchError('checkpoint shard digest mismatch')
                    if whole.hexdigest()!=spec['sha256'] or written!=spec['bytes']:
                        raise PatchError('checkpoint tensor digest mismatch: '+spec['name'])
                output.flush();os.fsync(output.fileno())
            os.replace(part,final)
        # This index follows HF save_pretrained output exactly.
        index_data={'metadata':{'total_size':meta['total_tensor_bytes']},
                    'weight_map':weight_map}
        index_path=dest/'model.safetensors.index.json'
        temp=dest/'.index.part'
        with temp.open('w',encoding='utf8') as fp:
            json.dump(index_data,fp,ensure_ascii=False,sort_keys=True)
            fp.flush();os.fsync(fp.fileno())
        os.replace(temp,index_path)
    except BaseException:
        # Only the isolated caller-owned temporary directory is touched.
        for item in dest.glob('*.part'):item.unlink(missing_ok=True)
        raise
    return {'tensors':len(weight_map),'bytes':meta['total_tensor_bytes'],
            'files':len(groups),'model_id':meta['model_id'],
            'revision':meta['revision']}
