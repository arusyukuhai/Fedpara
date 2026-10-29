"""Data-only full PyTorch state snapshots, streamed as uncompressed ZIP entries.

No pickle, Python source, import hooks or executable files are accepted on input.
Snapshots contain registered floating/int/bool tensors (parameters AND persistent
buffers), but NOT tokenizer/config or any code required to instantiate a model.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import tempfile
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from .core import PatchError, PEER_ID, _canonical_json

if TYPE_CHECKING:
    from .core import UniversalAdapter

VERSION = 1
META_NAME = 'manifest.json'
MAX_MANIFEST = 16 * 1024 * 1024
MAX_TENSORS = 200_000
CHUNK_BYTES = 4 * 1024 * 1024
NAME = re.compile(r'^[A-Za-z0-9_][A-Za-z0-9_.-]{0,511}$')
DTYPES = {
    str(dtype): dtype for dtype in (
        torch.float16, torch.float32, torch.float64, torch.bfloat16,
        torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
        torch.bool, torch.complex64, torch.complex128,
    )
}


def _state_items(model: torch.nn.Module, adapter: 'UniversalAdapter | None'):
    if adapter is None:
        return dict(model.state_dict(keep_vars=True))
    # state_dict() under torch parametrizations contains internal factor names,
    # not the publicly visible *effective* weights. Replace them explicitly.
    with adapter.lock, torch.no_grad():
        items = {k: v for k, v in model.state_dict(keep_vars=True).items()
                 if '.parametrizations.' not in ('.' + k + '.') and not k.startswith('parametrizations.')}
        for g in adapter.groups:
            for alias in g.aliases:
                mod, _, pname = alias.rpartition('.')
                items[alias] = getattr(model.get_submodule(mod), pname).detach()
        return items


def _valid_name(name: str) -> bool:
    return bool(NAME.fullmatch(name)) and '..' not in name and '/' not in name and '\\' not in name


def _nbytes(shape, dtype):
    if not isinstance(shape, list) or len(shape) > 64 or not all(type(x) is int and 0 <= x <= (1<<48) for x in shape):
        raise PatchError('invalid tensor shape')
    if dtype not in DTYPES:
        raise PatchError('unsupported state tensor dtype')
    return math.prod(shape) * DTYPES[dtype].itemsize


def _tensor_chunks(tensor: torch.Tensor, chunk_bytes: int = CHUNK_BYTES):
    if sys.byteorder != 'little':
        raise PatchError('state format currently requires a little-endian machine')
    if tensor.is_sparse or tensor.device.type == 'meta' or tensor.layout != torch.strided:
        raise PatchError('sparse/meta/non-strided tensors are not supported')
    if str(tensor.dtype) not in DTYPES:
        raise PatchError(f'unsupported dtype {tensor.dtype}')
    with torch.no_grad():
        # For a contiguous tensor reshape is a view; this also supports bf16.
        flat = tensor.detach().reshape(-1)
        count = max(1, chunk_bytes // tensor.element_size())
        for pos in range(0, flat.numel(), count):
            raw = flat[pos:pos+count].to(device='cpu').contiguous().view(torch.uint8).numpy().tobytes()
            yield raw


def export_state(model: torch.nn.Module, path: str | Path, *, model_id: str,
                 revision: str, peer_id: str, seq: int = 0,
                 adapter: 'UniversalAdapter | None' = None,
                 max_bytes: int = 32 * 1024**3) -> dict:
    """Export all named weights/buffers, including *effective* adapted weights.

    Writes directly to disk. A single CUDA tensor is copied in bounded chunks.
    Does not export model Python implementation or optimizer state.
    """
    if not PEER_ID.fullmatch(peer_id) or type(seq) is not int or seq < 0:
        raise PatchError('invalid peer_id or seq')
    if not isinstance(model_id, str) or not model_id or not isinstance(revision, str) or not revision:
        raise PatchError('model_id and revision are required')
    items = _state_items(model, adapter)
    if not 0 < len(items) <= MAX_TENSORS:
        raise PatchError('invalid state tensor count')
    specs, total = [], 0
    for idx, (key, t) in enumerate(sorted(items.items())):
        if not _valid_name(key) or not isinstance(t, torch.Tensor):
            raise PatchError(f'unsupported state entry {key!r}')
        if str(t.dtype) not in DTYPES:
            raise PatchError(f'unsupported dtype for {key}: {t.dtype}')
        size = _nbytes(list(t.shape), str(t.dtype))
        total += size
        if total > max_bytes:
            raise PatchError(f'model tensors exceed export quota of {max_bytes} bytes')
        specs.append({'name':key, 'entry':f'tensors/{idx:06d}.bin',
                      'dtype':str(t.dtype), 'shape':list(t.shape), 'bytes':size})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name+'.',suffix='.part',dir=path.parent)
    os.close(fd)
    temp = Path(tmp_path)
    try:
        with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for spec in specs:
                digest, written = hashlib.sha256(), 0
                with z.open(spec['entry'], 'w', force_zip64=True) as dst:
                    for piece in _tensor_chunks(items[spec['name']]):
                        dst.write(piece)
                        digest.update(piece)
                        written += len(piece)
                if written != spec['bytes']:
                    raise PatchError(f'serialized byte count mismatch: {spec["name"]}')
                spec['sha256'] = digest.hexdigest()
            manifest = {'format':'fedpara-state', 'version':VERSION, 'model_id':model_id,
                        'revision':revision, 'peer_id':peer_id, 'seq':seq,
                        'total_tensor_bytes':total, 'tensors':specs}
            metadata = _canonical_json(manifest)
            if len(metadata) > MAX_MANIFEST:
                raise PatchError('snapshot manifest too large')
            z.writestr(META_NAME, metadata, compress_type=zipfile.ZIP_STORED)
        if temp.stat().st_size > max_bytes + 64 * 1024 * 1024:
            raise PatchError('snapshot archive exceeds quota')
        os.replace(temp, path)
        return manifest
    finally:
        temp.unlink(missing_ok=True)


def inspect_state(path: str | Path, *, max_bytes: int = 32 * 1024**3) -> dict:
    """Strictly validate a data-only archive structure/manifest, without loading weights.

    Tensor SHA-256 hashes are checked separately in verify_state or apply_state.
    """
    path = Path(path)
    if path.stat().st_size > max_bytes + 64 * 1024 * 1024:
        raise PatchError('oversized state archive')
    try:
        with zipfile.ZipFile(path, 'r') as z:
            entries = z.infolist()
            by_name = {info.filename: info for info in entries}
            if len(entries) != len(by_name) or META_NAME not in by_name:
                raise PatchError('duplicate/missing archive entry')
            for info in entries:
                if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1 or info.is_dir():
                    raise PatchError('compressed/encrypted/directory entries are forbidden')
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise PatchError('symlinks forbidden')
            if by_name[META_NAME].file_size > MAX_MANIFEST:
                raise PatchError('oversized manifest')
            meta = json.loads(z.read(META_NAME))
            if not isinstance(meta, dict) or meta.get('format') != 'fedpara-state' or meta.get('version') != VERSION:
                raise PatchError('invalid snapshot format')
            if not isinstance(meta.get('model_id'), str) or not meta['model_id'] or not isinstance(meta.get('revision'), str) or not meta['revision']:
                raise PatchError('missing model identity')
            if not isinstance(meta.get('peer_id'), str) or not PEER_ID.fullmatch(meta['peer_id']) or type(meta.get('seq')) is not int or meta['seq'] < 0:
                raise PatchError('invalid sender identity/sequence')
            specs = meta.get('tensors')
            if not isinstance(specs, list) or not 0 < len(specs) <= MAX_TENSORS:
                raise PatchError('bad tensor list')
            expected = {META_NAME}
            names, total = set(), 0
            for idx, spec in enumerate(specs):
                if not isinstance(spec, dict) or not _valid_name(spec.get('name', '')) or spec['name'] in names:
                    raise PatchError('invalid/duplicate tensor name')
                entry = f'tensors/{idx:06d}.bin'
                if spec.get('entry') != entry or entry not in by_name:
                    raise PatchError('invalid tensor entry')
                digest = spec.get('sha256')
                if not isinstance(digest,str) or not re.fullmatch('[a-f0-9]{64}',digest):
                    raise PatchError('invalid tensor digest')
                n = _nbytes(spec.get('shape'), spec.get('dtype'))
                if type(spec.get('bytes')) is not int or spec['bytes'] != n or by_name[entry].file_size != n:
                    raise PatchError('tensor length mismatch')
                total += n
                if total > max_bytes:
                    raise PatchError('state tensor bytes exceed quota')
                expected.add(entry); names.add(spec['name'])
            if expected != set(by_name) or meta.get('total_tensor_bytes') != total:
                raise PatchError('unexpected entries or size')
            return meta
    except (zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
        raise PatchError(f'bad snapshot archive: {exc}') from exc


def _verify_members(path: Path, specs: list[dict]):
    with zipfile.ZipFile(path) as z:
        for spec in specs:
            digest, length = hashlib.sha256(), 0
            with z.open(spec['entry']) as src:
                for part in iter(lambda: src.read(CHUNK_BYTES), b''):
                    digest.update(part); length += len(part)
                    dtype = DTYPES[spec['dtype']]
                    if dtype.is_floating_point or dtype.is_complex:
                        values = torch.frombuffer(bytearray(part),dtype=torch.uint8).view(dtype)
                        if not torch.isfinite(values).all().item():
                            raise PatchError(f'non-finite tensor: {spec["name"]}')
            if length != spec['bytes'] or digest.hexdigest() != spec['sha256']:
                raise PatchError(f'corrupt tensor: {spec["name"]}')


def verify_state(path: str | Path, *, max_bytes: int = 32 * 1024**3) -> dict:
    meta = inspect_state(path, max_bytes=max_bytes)
    _verify_members(Path(path), meta['tensors'])
    return meta


def semantic_state_sha256(path: str | Path, *, verify: bool = False) -> str:
    """Stable identity for tensor contents irrespective of ZIP timestamps/peer_id.

    verify=True checks actual shard bytes against tensor SHA-256 values; callers
    importing untrusted data must verify before trusting the identity.
    """
    meta = verify_state(path) if verify else inspect_state(path)
    tensors = [{'name':t['name'],'dtype':t['dtype'],'shape':t['shape'],
                'sha256':t['sha256']} for t in meta['tensors']]
    return hashlib.sha256(_canonical_json({'model_id':meta['model_id'],
                   'revision':meta['revision'],'tensors':tensors})).hexdigest()


def apply_state(model: torch.nn.Module, path: str | Path, *, model_id: str,
                revision: str, max_bytes: int = 32 * 1024**3) -> dict:
    """Verify complete incoming archive *before* mutating model parameters.

    Uses streaming per-tensor copying after preflight. A later OS/device failure
    may still cause a partially updated model; import to a fresh model/process.
    """
    meta = inspect_state(path, max_bytes=max_bytes)
    if (meta['model_id'], meta['revision']) != (model_id, revision):
        raise PatchError('snapshot model ID/revision mismatch')
    own = model.state_dict(keep_vars=True)
    keys = set(own)
    if keys != {spec['name'] for spec in meta['tensors']}:
        raise PatchError('snapshot tensor names differ from target model')
    for spec in meta['tensors']:
        t = own[spec['name']]
        if (not isinstance(t, torch.Tensor) or tuple(t.shape) != tuple(spec['shape'])
                or str(t.dtype) != spec['dtype'] or t.device.type == 'meta'):
            raise PatchError('snapshot dtype/shape/device mismatch: '+spec['name'])
    _verify_members(Path(path), meta['tensors'])
    # Load in bounded chunks. This avoids allocating a second entire model.
    with zipfile.ZipFile(path) as z, torch.no_grad():
        for spec in meta['tensors']:
            t = own[spec['name']]
            if not t.is_contiguous():
                # Fallback for rare non-contiguous registered parameters.
                raw = z.read(spec['entry'])
                src = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(DTYPES[spec['dtype']]).reshape(spec['shape'])
                t.copy_(src.to(t.device))
                continue
            dest = t.reshape(-1)
            num_elements = max(1, CHUNK_BYTES // t.element_size())
            with z.open(spec['entry']) as inp:
                for offset in range(0, dest.numel(), num_elements):
                    count = min(num_elements, dest.numel()-offset)
                    raw = inp.read(count * t.element_size())
                    if len(raw) != count * t.element_size():
                        raise PatchError('unexpected EOF applying snapshot')
                    src = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(t.dtype)
                    dest[offset:offset+count].copy_(src.to(t.device))
    return meta


def apply_effective_state(model: torch.nn.Module, adapter: 'UniversalAdapter', path: str | Path, *,
                          model_id: str, revision: str, max_bytes: int = 32*1024**3) -> dict:
    """Apply a *fully verified* peer snapshot to parametrized model base tensors.

    This is only safe at a quiescent training boundary and after checking the
    signed snapshot event tip matches the local durable event journal. The
    snapshot includes effective LoRA/LoHA weights, not internal factor tensors.
    """
    meta=verify_state(path,max_bytes=max_bytes)
    if (meta['model_id'],meta['revision'])!=(model_id,revision):
        raise PatchError('incoming snapshot has wrong model or revision')
    own=model.state_dict(keep_vars=True)
    mapped={k:v for k,v in own.items() if '.parametrizations.' not in ('.'+k+'.')
            and not k.startswith('parametrizations.')}
    with adapter.lock:
        for group in adapter.groups:
            for alias in group.aliases:
                module,_,attr=alias.rpartition('.')
                mapped[alias]=model.get_submodule(module).parametrizations[attr].original
        if set(mapped)!={x['name'] for x in meta['tensors']}:
            raise PatchError('snapshot state names differ from local model')
        for spec in meta['tensors']:
            dest=mapped[spec['name']]
            if tuple(dest.shape)!=tuple(spec['shape']) or str(dest.dtype)!=spec['dtype']:
                raise PatchError('snapshot dtype/shape mismatch: '+spec['name'])
        with zipfile.ZipFile(path) as z,torch.no_grad():
            for spec in meta['tensors']:
                dest=mapped[spec['name']]
                # Bounded copies for common contiguous tensors.
                if dest.is_contiguous():
                    flat=dest.reshape(-1)
                    block=max(1,CHUNK_BYTES//dest.element_size())
                    with z.open(spec['entry']) as inp:
                        for start in range(0,flat.numel(),block):
                            length=min(block,flat.numel()-start)
                            data=inp.read(length*dest.element_size())
                            if len(data)!=length*dest.element_size():
                                raise PatchError('truncated snapshot tensor')
                            src=torch.frombuffer(bytearray(data),dtype=torch.uint8).view(dest.dtype)
                            flat[start:start+length].copy_(src.to(device=dest.device))
                else:
                    data=z.read(spec['entry'])
                    src=torch.frombuffer(bytearray(data),dtype=torch.uint8).view(dest.dtype).reshape(dest.shape)
                    dest.copy_(src.to(device=dest.device))
        # Snapshot already includes all committed deltas. Newly initialized
        # factors must not re-add an old local update.
        from .relora import _refresh_spec
        with torch.no_grad():
            for group in adapter.groups:
                factor = group.overlay.local
                _refresh_spec(factor)
                # A received snapshot already includes the learned effective
                # weight. Reset both local and its anchor: otherwise stale
                # anchors re-add an old pre-snapshot local update.
                if hasattr(group.overlay, 'anchor'):
                    for index, current in enumerate(factor.parts):
                        getattr(group.overlay.anchor, f'part{index}').copy_(current.detach())
                # Remove overlay-only remotes, if any: snapshot supersedes them.
                group.overlay.remotes.clear()
                group.overlay._peer_ids.clear()
    return meta
