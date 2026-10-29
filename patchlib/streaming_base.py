"""Lazy virtual-memory loading of trusted, local Hugging Face safetensors.

Build the architecture on the meta device, then bind each tensor directly to the
safetensors mmap (immutable base) or a writable, temporary disk-backed mapping
(for continual base updates). No full-state dict, full model copy, pickle, or
single-model-size CPU allocation is made by this loader.

The calling model must support construction on meta, and every material tensor
must be present in the safetensors checkpoint. Arbitrary custom forward kernels
are NOT automatically made out-of-core by this loader.
"""
from __future__ import annotations

import json
import itertools
import math
import os
import tempfile
import hashlib
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open


class StreamingModelError(RuntimeError):
    pass


def _files(root: Path):
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise StreamingModelError('streaming base requires a local snapshot directory')
    index = root / 'model.safetensors.index.json'
    if index.is_file():
        obj = json.loads(index.read_text(encoding='utf8'))
        weights = obj.get('weight_map')
        if not isinstance(weights, dict) or not weights:
            raise StreamingModelError('invalid safetensors weight map')
        grouped = {}
        for name, file in weights.items():
            if not isinstance(name, str) or not isinstance(file, str) or not file.endswith('.safetensors'):
                raise StreamingModelError('invalid safetensors index entry')
            path = (root / file).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise StreamingModelError('safetensors shard path escapes snapshot')
            grouped.setdefault(path, set()).add(name)
    else:
        path = root / 'model.safetensors'
        if not path.is_file():
            raise StreamingModelError('need model.safetensors or model.safetensors.index.json; legacy bin/pickle disallowed')
        grouped = {path: None}
    return grouped


def _named_tensors(model):
    params={}; buffers={}; nonpersistent={}
    for prefix, mod in model.named_modules():
        for name,p in mod._parameters.items():
            if p is not None:
                params[f'{prefix}.{name}'.lstrip('.')] = (mod,name,p)
        for name,b in mod._buffers.items():
            if b is not None:
                full=f'{prefix}.{name}'.lstrip('.')
                if name not in mod._non_persistent_buffers_set:
                    buffers[full]=(mod,name,b)
                elif b.device.type == 'meta' and b.numel():
                    nonpersistent[full]=(mod,name,b)
    return params,buffers,nonpersistent




def _rebuild_nonpersistent(module, name, reference):
    """Regenerate local, deterministic rotary buffers never saved by HF.

    Never guess for unknown modules or scaled RoPE implementations; an
    incorrect RoPE spectrum could silently corrupt every training step.
    """
    if name != 'inv_freq' or 'rotary' not in type(module).__name__.lower():
        raise StreamingModelError(f'unsupported nonpersistent meta buffer: {type(module).__name__}.{name}')
    fn = getattr(module, 'rope_init_fn', None)
    cfg = getattr(module, 'config', None)
    if callable(fn) and cfg is not None:
        try:
            value, _scale = fn(cfg, device=torch.device('cpu'))
        except (TypeError, ValueError, AttributeError, RuntimeError) as exc:
            raise StreamingModelError('failed to reinitialize built-in rotary frequency buffer') from exc
    else:
        if cfg is None:
            raise StreamingModelError('rotary buffer lacks its pretrained config')
        rope_scaling=getattr(cfg,'rope_scaling',None)
        if rope_scaling and rope_scaling.get('rope_type',rope_scaling.get('type','default')) != 'default':
            raise StreamingModelError('scaled RoPE requires a supported local rope_init_fn')
        dim = getattr(cfg,'head_dim',None) or (getattr(cfg,'hidden_size',0)//max(1,getattr(cfg,'num_attention_heads',1)))
        dim = int(dim*float(getattr(cfg,'partial_rotary_factor',1.0)))
        if not (2 <= dim <= 65536 and dim%2==0):
            raise StreamingModelError('invalid default rotary dimension')
        theta=float(getattr(cfg,'rope_theta',10000.))
        if not math.isfinite(theta) or theta <= 0:
            raise StreamingModelError('invalid rotary theta')
        value = 1.0 / (theta ** (torch.arange(0,dim,2,dtype=torch.float32)/dim))
    if not isinstance(value,torch.Tensor) or value.device.type=='meta' or tuple(value.shape)!=tuple(reference.shape) or not torch.isfinite(value.float()).all().item():
        raise StreamingModelError('recreated RoPE inv_freq has unexpected shape or contents')
    return value.to(dtype=reference.dtype)

def _write_tensor_slice(src, path: Path, *, data_type, chunk_bytes: int):
    """Copy through one sliced tensor at a time to a preallocated backing file."""
    shape=tuple(src.get_shape())
    total=math.prod(shape)
    bytes_total=total*data_type.itemsize
    # mmap file is created in exclusive, ephemeral namespace, never a checkpoint.
    with path.open('xb') as out:
        out.truncate(bytes_total)
    target=torch.from_file(str(path),shared=True,size=total,dtype=data_type).reshape(shape)
    if total:
        if not shape:
            target.copy_(src[:])
        else:
            # Safetensors get_slice supports multidimensional slicing. Split
            # wide rows as well as the leading axis: no row-sized allocation.
            axis=next((d for d in range(len(shape))
                       if math.prod(shape[d+1:])*data_type.itemsize<=chunk_bytes),None)
            if axis is None:raise StreamingModelError('stream chunk smaller than a dtype element')
            suffix=math.prod(shape[axis+1:])
            step=max(1,chunk_bytes//max(1,suffix*data_type.itemsize))
            prefixes=itertools.product(*(range(v) for v in shape[:axis]))
            for prefix in prefixes:
                for start in range(0,shape[axis],step):
                    index=(*prefix,slice(start,min(shape[axis],start+step)),
                           *(slice(None) for _ in shape[axis+1:]))
                    target[index].copy_(src[index])
    # data is durable enough for the process lifetime; journal recovery does
    # not depend on this temp file. Pages will flush lazily during execution.
    return target


def materialize_meta_from_safetensors(model: nn.Module, snapshot_dir,
                                       *, writable=False, scratch_dir=None,
                                       chunk_mib=8, strict=True, hash_genesis=False,
                                       preserve_checkpoint_dtype=False):
    """Bind meta params/buffers from local verified-by-OS safetensors files.

    This is shape validation, NOT cryptographic verification: pin the HF revision
    or externally verify the snapshot before using. For writable=True, writes
    go to a private temporary directory, NEVER the immutable source files.
    Caller must keep model alive while using these mapped weights.
    """
    if not isinstance(model,nn.Module):raise TypeError('model must be nn.Module')
    if not 1<=chunk_mib<=128:raise ValueError('stream chunk size must be 1..128 MiB')
    params,buffers,nonpersistent=_named_tensors(model)
    for key,(_,_,tensor) in {**params,**buffers}.items():
        if tensor.device.type!='meta':
            raise StreamingModelError(f'model needs meta initialization; already resident: {key}')
    # Prepare all deterministic non-persistent tensors *before* changing the
    # model.  Modern Llama/Qwen RoPE inv_freq is typically omitted by HF.
    rebuilt_nonpersistent={}
    for key,(mod,leaf,ref) in nonpersistent.items():
        rebuilt_nonpersistent[key]=_rebuild_nonpersistent(mod,leaf,ref)
    grouped=_files(Path(snapshot_dir))
    genesis_digest=None
    if hash_genesis:
        # Hash the complete trusted starting weights without ever materializing
        # a complete checkpoint in RAM. This binds room genesis to exact data,
        # not a caller-supplied repository revision alone.
        digest=hashlib.sha256(b'fedpara-streaming-safetensors-v1\n')
        root=Path(snapshot_dir).resolve()
        for file in sorted(grouped):
            digest.update(str(file.relative_to(root)).encode('utf8')+b'\0')
            with file.open('rb') as src:
                h=hashlib.sha256()
                for part in iter(lambda:src.read(4*1024*1024),b''):
                    h.update(part)
            digest.update(h.digest())
        genesis_digest=digest.hexdigest()
    scratch=None
    if writable:
        if scratch_dir is not None:Path(scratch_dir).mkdir(parents=True,exist_ok=True)
        scratch=tempfile.TemporaryDirectory(prefix='fedpara-weights-',dir=scratch_dir)
    by_name={}
    count=0;total_bytes=0
    try:
        for file, expected in grouped.items():
            with safe_open(str(file),framework='pt',device='cpu') as reader:
                keys=set(reader.keys())
                if expected is not None and not expected.issubset(keys):
                    raise StreamingModelError(f'index references absent tensors in {file.name}: {sorted(expected-keys)[:3]}')
                if expected is None:expected=keys
                for key in sorted(expected):
                    if key in by_name:raise StreamingModelError(f'duplicate weight name: {key}')
                    if key not in params and key not in buffers:
                        if strict:raise StreamingModelError(f'unexpected checkpoint key {key}')
                        continue
                    ref=(params.get(key) or buffers.get(key))[2]
                    mapped_value=reader.get_tensor(key)
                    if mapped_value.is_floating_point() and mapped_value.dtype not in (
                            torch.float16, torch.bfloat16, torch.float32, torch.float64):
                        raise StreamingModelError(
                            f'{key}: floating/FP8 checkpoint dtype {mapped_value.dtype} has no validated Kagami training kernel; '
                            'use BF16/FP16/FP32 safetensors instead')
                    if mapped_value.dtype!=ref.dtype:
                        if not (preserve_checkpoint_dtype and ref.is_floating_point() and mapped_value.is_floating_point()):
                            raise StreamingModelError(f'{key} checkpoint dtype mismatch: {mapped_value.dtype} != {ref.dtype}')
                    tensor_slice=reader.get_slice(key) if mapped_value.ndim else None
                    from_shape=tuple(tensor_slice.get_shape()) if tensor_slice is not None else tuple(mapped_value.shape)
                    if from_shape!=tuple(ref.shape):
                        raise StreamingModelError(f'{key} checkpoint shape mismatch: {from_shape} != {tuple(ref.shape)}')
                    if writable:
                        filename=hashlib.sha256(key.encode('utf8')).hexdigest()+'.raw'
                        if tensor_slice is None:
                            filename_path=Path(scratch.name)/filename
                            with filename_path.open('xb') as output:output.write(mapped_value.reshape(1).view(torch.uint8).numpy().tobytes())
                            value=torch.from_file(str(filename_path),shared=True,size=1,dtype=mapped_value.dtype).view(())
                        else:
                            value=_write_tensor_slice(tensor_slice,Path(scratch.name)/filename,
                                                  data_type=mapped_value.dtype,chunk_bytes=chunk_mib*1024**2)
                        if value.dtype!=mapped_value.dtype:
                            raise StreamingModelError(f'{key} dtype mismatch: {value.dtype} != {mapped_value.dtype}')
                        file_path=str(Path(scratch.name)/filename)
                    else:
                        # safetensors presents tensor views backed by memory maps.
                        # Do not call clone, float, or to(device) here.
                        value=mapped_value
                        if value.dtype!=mapped_value.dtype:
                            raise StreamingModelError(f'{key} dtype mismatch: {value.dtype} != {mapped_value.dtype}')
                        file_path=None
                    by_name[key]=(value,file_path)
                    count+=1;total_bytes+=value.numel()*value.element_size()
        # Resolve tied aliases using the meta tensor object's identity. HF
        # checkpoints usually omit duplicate names of tied input/output heads.
        aliases={}
        for name,(_,_,param) in params.items():
            aliases.setdefault(id(param),[]).append(name)
        for names in aliases.values():
            present=[n for n in names if n in by_name]
            if not present:raise StreamingModelError(f'missing parameter group {names[:3]}')
            chosen=by_name[present[0]]
            for alias in present[1:]:
                other=by_name[alias]
                if chosen[0].data_ptr()!=other[0].data_ptr() and not torch.equal(chosen[0],other[0]):
                    raise StreamingModelError(f'conflicting tied parameter tensors {present[:2]}')
            for alias in names:by_name[alias]=chosen
        missing=[name for name in buffers if name not in by_name]
        if missing:raise StreamingModelError(f'missing persistent model buffers: {missing[:4]}')
        # Prevalidate everything before mutating the model.
        new_params={}
        for ident,names in aliases.items():
            sample=params[names[0]][2]
            tensor,path=by_name[names[0]]
            if tuple(tensor.shape)!=tuple(sample.shape) or (tensor.dtype!=sample.dtype and not (preserve_checkpoint_dtype and tensor.is_floating_point() and sample.is_floating_point())):
                raise StreamingModelError(f'tied alias incompatible: {names[0]}')
            p=nn.Parameter(tensor,requires_grad=sample.requires_grad)
            if path:p._fedpara_raw_path=path
            new_params[ident]=p
        for name,(module,leaf,orig) in params.items():
            module._parameters[leaf]=new_params[id(orig)]
        for name,(module,leaf,_) in buffers.items():
            module._buffers[leaf]=by_name[name][0]
        for name,(module,leaf,_) in nonpersistent.items():
            module._buffers[leaf]=rebuilt_nonpersistent[name]
        if scratch:
            object.__setattr__(model,'_fedpara_ephemeral_weights',scratch)
        object.__setattr__(model,'_fedpara_streaming_base_info',{'tensors':count,'bytes':total_bytes,
                           'mode':'writable-ephemeral' if writable else 'read-only-mmap',
                           'source':str(Path(snapshot_dir).resolve()),
                           'genesis_digest':genesis_digest})
        return model
    except Exception:
        if scratch:scratch.cleanup()
        raise
