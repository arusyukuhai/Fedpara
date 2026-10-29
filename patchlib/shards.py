"""Content-addressed, tensor-aware shards for Fedpara .fstate snapshots.

Each tensor is split independently, preserving deduplication when other tensors
change size. Manifests and shards are DATA ONLY: no pickle, Python code, paths
supplied by a peer or executable payloads are evaluated on receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from pathlib import Path

from .core import PatchError, PEER_ID, _canonical_json
from .state import (inspect_state, verify_state, MAX_TENSORS, MAX_MANIFEST,
                    META_NAME, _valid_name, _nbytes, _state_items, _tensor_chunks, DTYPES)

FORMAT = 'fedpara-sharded-state'
VERSION = 1
DEFAULT_SHARD_BYTES = 4 * 1024 * 1024
MAX_SHARD_BYTES = 64 * 1024 * 1024
MIN_SHARD_BYTES = 64 * 1024
MAX_SHARDS = 750_000
MAX_SHARD_MANIFEST = 96 * 1024 * 1024
HEX = re.compile(r'^[0-9a-f]{64}$')


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check_hash(value):
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise PatchError('invalid shard digest')
    return value


def _blob_path(store: Path, digest: str) -> Path:
    _check_hash(digest)
    return Path(store) / 'objects' / digest[:2] / (digest + '.shard')


def _check_object(path: Path, digest: str, size: int) -> bool:
    """Untrusted cache entries never become trusted merely by existing."""
    try:
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode) or path.stat().st_size != size:
            return False
        h = hashlib.sha256()
        with path.open('rb') as f:
            for piece in iter(lambda: f.read(1024*1024), b''):
                h.update(piece)
        return h.hexdigest() == digest
    except FileNotFoundError:
        return False


def has_object(store: Path, digest: str, size: int) -> bool:
    _check_hash(digest)
    if type(size) is not int or size <= 0 or size > MAX_SHARD_BYTES:
        raise PatchError('bad shard length')
    return _check_object(_blob_path(Path(store), digest), digest, size)


def _atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.incoming-', suffix='.part', dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(contents)
            stream.flush()
        os.replace(name,path)
    finally:
        Path(name).unlink(missing_ok=True)


def put_object(store: Path, digest: str, contents: bytes) -> Path:
    _check_hash(digest)
    if not 0 < len(contents) <= MAX_SHARD_BYTES or _sha(contents) != digest:
        raise PatchError('shard content hash/size mismatch')
    target = _blob_path(Path(store), digest)
    if not has_object(store, digest, len(contents)):
        _atomic_write(target, contents)
    return target


def manifest_bytes(meta: dict, *, max_bytes: int = MAX_SHARD_MANIFEST) -> bytes:
    payload = _canonical_json(meta)
    if len(payload) > max_bytes:
        raise PatchError('shard manifest exceeds limit')
    return payload


def parse_manifest(data: bytes, *, max_state_bytes: int = 32*1024**3,
                   max_manifest_bytes: int = MAX_SHARD_MANIFEST) -> dict:
    if not isinstance(data, bytes) or not 0 < len(data) <= max_manifest_bytes:
        raise PatchError('invalid shard manifest length')
    try:
        meta = json.loads(data)
    except (ValueError, UnicodeDecodeError) as e:
        raise PatchError('bad shard manifest JSON') from e
    if not isinstance(meta,dict) or meta.get('format') != FORMAT or type(meta.get('version')) is not int or meta['version'] != VERSION:
        raise PatchError('unsupported shard manifest format')
    if set(meta) != {'format','version','model_id','revision','peer_id','seq','shard_bytes',
                     'total_tensor_bytes','tensors'}:
        raise PatchError('invalid shard manifest fields')
    if not isinstance(meta.get('model_id'),str) or not 0 < len(meta['model_id']) <= 1024:
        raise PatchError('bad model_id')
    if not isinstance(meta.get('revision'),str) or not 0 < len(meta['revision']) <= 1024:
        raise PatchError('bad revision')
    if not isinstance(meta.get('peer_id'),str) or not PEER_ID.fullmatch(meta['peer_id']):
        raise PatchError('bad peer_id')
    if type(meta.get('seq')) is not int or not 0 <= meta['seq'] < 2**63:
        raise PatchError('bad sequence')
    sb = meta.get('shard_bytes')
    if type(sb) is not int or not MIN_SHARD_BYTES <= sb <= MAX_SHARD_BYTES:
        raise PatchError('invalid shard size')
    specs = meta.get('tensors')
    if not isinstance(specs,list) or not 0 < len(specs) <= MAX_TENSORS:
        raise PatchError('bad tensor count')
    names = set()
    total, nshards = 0, 0
    for idx, spec in enumerate(specs):
        if not isinstance(spec,dict) or set(spec) != {'name','entry','dtype','shape','bytes','sha256','shards'}:
            raise PatchError('invalid tensor spec')
        if not _valid_name(spec['name']) or spec['name'] in names:
            raise PatchError('bad tensor name')
        names.add(spec['name'])
        if spec['entry'] != f'tensors/{idx:06d}.bin':
            raise PatchError('invalid tensor entry')
        size = _nbytes(spec['shape'],spec['dtype'])
        if type(spec['bytes']) is not int or spec['bytes'] != size:
            raise PatchError('tensor byte length differs from shape/dtype')
        _check_hash(spec['sha256'])
        total += size
        if total > max_state_bytes:
            raise PatchError('snapshot exceeds configured byte budget')
        chunks = spec['shards']
        if not isinstance(chunks,list) or (size > 0 and not chunks) or (size == 0 and chunks):
            raise PatchError('invalid tensor shard count')
        expected_count = (size + sb - 1)//sb
        if len(chunks) != expected_count:
            raise PatchError('invalid tensor shard count/length')
        piece_total = 0
        for j, chunk in enumerate(chunks):
            if not isinstance(chunk, dict) or set(chunk) != {'sha256','size'}:
                raise PatchError('invalid shard specification')
            _check_hash(chunk['sha256'])
            expected_size = min(sb, size-j*sb)
            if type(chunk['size']) is not int or chunk['size'] != expected_size:
                raise PatchError('invalid shard byte count')
            piece_total += chunk['size']
        if piece_total != size:
            raise PatchError('tensor shards do not cover full tensor')
        nshards += len(chunks)
        if nshards > MAX_SHARDS:
            raise PatchError('too many shards')
    if type(meta.get('total_tensor_bytes')) is not int or meta['total_tensor_bytes'] != total:
        raise PatchError('mismatched snapshot total size')
    return meta


def read_manifest(path: Path, **kwargs) -> dict:
    path = Path(path)
    if path.stat().st_size > MAX_SHARD_MANIFEST:
        raise PatchError('shard manifest too large')
    return parse_manifest(path.read_bytes(), **kwargs)


def shard_map(meta: dict) -> dict[str,int]:
    """Unique shards in first-use order; same content need transfer only once."""
    result = {}
    for spec in meta['tensors']:
        for chunk in spec['shards']:
            sha, size = chunk['sha256'],chunk['size']
            if sha in result and result[sha] != size:
                raise PatchError('same hash with incompatible size')
            result[sha] = size
    return result


def export_state_sharded(model, store: Path, output: Path, *, model_id: str,
                         revision: str, peer_id: str, seq: int = 0,
                         adapter=None, shard_bytes: int = DEFAULT_SHARD_BYTES,
                         max_state_bytes: int = 32*1024**3,
                         frozen_base_sources=None) -> dict:
    """Export live PyTorch state directly into shards, with no intermediate .fstate.

    Includes effective LoRA/LoHA-adapted weights, embeddings, and buffers.
    CUDA tensors are moved to CPU in individually bounded blocks.
    """
    import torch
    # A sparse page COW engine deliberately leaves the raw nn.Parameter at
    # immutable genesis values. Exporting it without materializing would
    # silently certify INCORRECT weights, so require an explicit full export.
    manager=getattr(model,'_fedpara_lazy_mutable_weights',None)
    if manager is not None and getattr(manager,'stores',None):
        raise PatchError('paged COW needs explicit manager.materialize_all(adapter) before full-state export')
    # The factor ledger supports direct bounded sharded export; the immutable
    # original Parameter alone is NOT the effective model.  Thread-local scope
    # suppresses its state_dict hook only for this controlled serialization.
    if type(shard_bytes) is not int or not MIN_SHARD_BYTES <= shard_bytes <= MAX_SHARD_BYTES:
        raise PatchError('invalid shard_bytes')
    if not isinstance(model_id,str) or not model_id or not isinstance(revision,str) or not revision:
        raise PatchError('model_id and revision required')
    if not isinstance(peer_id,str) or not PEER_ID.fullmatch(peer_id) or type(seq) is not int or seq < 0:
        raise PatchError('invalid peer_id/seq')
    from .shard_runtime import effective_chunks, effective_state_sources
    import torch
    # Access the *original* parametrized weight, never the effective .weight
    # property (which eagerly allocates an entire dense LoHa/LoRA weight).
    guard = adapter.lock if adapter is not None else None
    from contextlib import nullcontext
    ledger=(manager if getattr(manager,'_fedpara_is_factor_ledger',False) else None)
    with guard if guard is not None else nullcontext(), torch.no_grad(), (ledger.export_context() if ledger is not None else nullcontext()):
        if ledger is not None:ledger._healthy()
        sources_by_alias=(frozen_base_sources if frozen_base_sources is not None
                          else ({alias:ledger.sources[g.canonical]
                          for g in adapter.groups for alias in g.aliases
                          if g.canonical in ledger.sources} if ledger is not None else {}))
        if frozen_base_sources is not None and ledger is None:
            raise PatchError('frozen factor ledger sources require a factor-ledger manager')
        tensors = effective_state_sources(model, adapter)
        if not 0 < len(tensors) <= MAX_TENSORS:
            raise PatchError('invalid number of registered tensors')
        meta = {'format':FORMAT,'version':VERSION,'model_id':model_id,'revision':revision,
                'peer_id':peer_id,'seq':seq,'shard_bytes':shard_bytes,
                'total_tensor_bytes':0,'tensors':[]}
        for idx, (name, (tensor, overlay)) in enumerate(sorted(tensors.items())):
            if not _valid_name(name) or not isinstance(tensor,torch.Tensor):
                raise PatchError('unsupported state entry')
            if str(tensor.dtype) not in DTYPES:
                raise PatchError('unsupported tensor dtype: '+str(tensor.dtype))
            size = _nbytes(list(tensor.shape),str(tensor.dtype))
            meta['total_tensor_bytes'] += size
            if meta['total_tensor_bytes'] > max_state_bytes:
                raise PatchError('state exceeds max_state_bytes')
            spec = {'name':name,'entry':f'tensors/{idx:06d}.bin',
                    'shape':list(tensor.shape),'dtype':str(tensor.dtype),
                    'bytes':size,'shards':[]}
            tensor_hash=hashlib.sha256()
            buffer=bytearray()
            written=0
            for piece in effective_chunks(tensor,overlay,chunk_bytes=shard_bytes,
                                          base_source=sources_by_alias.get(name)):
                tensor_hash.update(piece)
                written += len(piece)
                buffer.extend(piece)
                while len(buffer) >= shard_bytes:
                    data=bytes(buffer[:shard_bytes])
                    del buffer[:shard_bytes]
                    digest=_sha(data)
                    put_object(store,digest,data)
                    spec['shards'].append({'sha256':digest,'size':len(data)})
            if buffer:
                data=bytes(buffer)
                digest=_sha(data)
                put_object(store,digest,data)
                spec['shards'].append({'sha256':digest,'size':len(data)})
            if written != size:
                raise PatchError('unexpected tensor byte count')
            spec['sha256'] = tensor_hash.hexdigest()
            meta['tensors'].append(spec)
    payload = manifest_bytes(meta)
    parse_manifest(payload,max_state_bytes=max_state_bytes)
    _atomic_write(Path(output),payload)
    return meta


def shard_state(source: Path, store: Path, output: Path, *, shard_bytes: int = DEFAULT_SHARD_BYTES,
                max_state_bytes: int = 32*1024**3) -> dict:
    """Convert existing .fstate ZIP into tensor-aligned content-addressed shards.

    Never allocates an entire tensor. For repeated exports, an unchanged shard
    is reused and does not consume another copy in the object store.
    """
    if type(shard_bytes) is not int or not MIN_SHARD_BYTES <= shard_bytes <= MAX_SHARD_BYTES:
        raise PatchError('shard_bytes must be between 64KiB and 64MiB')
    info = inspect_state(source,max_bytes=max_state_bytes)
    meta = {k:info[k] for k in ('model_id','revision','peer_id','seq','total_tensor_bytes')}
    meta.update({'format':FORMAT,'version':VERSION,'shard_bytes':shard_bytes,'tensors':[]})
    with zipfile.ZipFile(source) as archive:
        for spec in info['tensors']:
            local = {k:spec[k] for k in ('name','entry','dtype','shape','bytes','sha256')}
            local['shards'] = []
            tensor_hash = hashlib.sha256()
            with archive.open(spec['entry']) as inp:
                remaining = spec['bytes']
                while remaining:
                    data = inp.read(min(shard_bytes,remaining))
                    if not data:
                        raise PatchError('truncated state tensor')
                    tensor_hash.update(data)
                    digest = _sha(data)
                    put_object(store,digest,data)
                    local['shards'].append({'sha256':digest,'size':len(data)})
                    remaining -= len(data)
                if inp.read(1):
                    raise PatchError('trailing tensor payload')
            if tensor_hash.hexdigest() != spec['sha256']:
                raise PatchError('source tensor checksum invalid: '+spec['name'])
            meta['tensors'].append(local)
    payload = manifest_bytes(meta)
    parse_manifest(payload,max_state_bytes=max_state_bytes)
    _atomic_write(Path(output),payload)
    return meta


def verify_shards(meta: dict, store: Path) -> tuple[int,int]:
    objects = shard_map(meta)
    missing = [(h,n) for h,n in objects.items() if not has_object(store,h,n)]
    if missing:
        raise PatchError(f'missing or corrupt shards: {len(missing)} of {len(objects)}')
    return len(objects),sum(objects.values())


def assemble_state(manifest: Path, store: Path, output: Path, *,
                   max_state_bytes: int = 32*1024**3) -> dict:
    """Rebuild a legacy .fstate file and verify its per-tensor hashes before commit."""
    meta = read_manifest(manifest,max_state_bytes=max_state_bytes)
    verify_shards(meta,store)
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix='.state-assembling-',suffix='.part',dir=output.parent)
    os.close(fd)
    temp = Path(filename)
    try:
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            for spec in meta['tensors']:
                h = hashlib.sha256()
                with archive.open(spec['entry'],'w',force_zip64=True) as stream:
                    for chunk in spec['shards']:
                        p = _blob_path(store,chunk['sha256'])
                        with p.open('rb') as inp:
                            for data in iter(lambda:inp.read(1024*1024),b''):
                                h.update(data)
                                stream.write(data)
                if h.hexdigest() != spec['sha256']:
                    raise PatchError('tensor hash mismatch while assembling: '+spec['name'])
            specs = [{k:s[k] for k in ('name','entry','dtype','shape','bytes','sha256')}
                     for s in meta['tensors']]
            legacy = {k:meta[k] for k in ('model_id','revision','peer_id','seq','total_tensor_bytes')}
            legacy.update({'format':'fedpara-state','version':1,'tensors':specs})
            legacy_json = _canonical_json(legacy)
            if len(legacy_json) > MAX_MANIFEST:
                raise PatchError('assembled legacy state manifest exceeds 16MiB limit')
            archive.writestr(META_NAME,legacy_json,compress_type=zipfile.ZIP_STORED)
        verify_state(temp,max_bytes=max_state_bytes)
        os.replace(temp,output)
        return legacy
    finally:
        temp.unlink(missing_ok=True)


def _sharded_destinations(model, adapter=None):
    """Raw destination tensors, without accessing dense parametrized weights."""
    from .shard_runtime import effective_state_sources
    return {name: pair[0] for name,pair in effective_state_sources(model,adapter).items()}


def _install_verified_shards(meta, store, destinations, *, chunk_bytes=4*1024*1024):
    """Copy a PRE-VERIFIED snapshot into preallocated parameters without assembly.

    This mutates weights as it progresses. Call only at a quiescent boundary or
    on a newly constructed model; a device failure during install may leave a
    partially installed model and must be treated as fatal.
    """
    import torch
    if type(chunk_bytes) is not int or not 64*1024 <= chunk_bytes <= MAX_SHARD_BYTES:
        raise ValueError('invalid bounded read size')
    if set(destinations) != {spec['name'] for spec in meta['tensors']}:
        raise PatchError('sharded snapshot tensor names differ from local model')
    # Every tied alias must represent the SAME bytes. Fail before mutation.
    tied = {}
    for spec in meta['tensors']:
        t=destinations[spec['name']]
        if (not isinstance(t,torch.Tensor) or t.device.type=='meta'
            or tuple(t.shape)!=tuple(spec['shape']) or str(t.dtype)!=spec['dtype']):
            raise PatchError('sharded snapshot destination shape/dtype mismatch: '+spec['name'])
        key=(t.untyped_storage().data_ptr(),t.storage_offset(),tuple(t.shape),tuple(t.stride()))
        if key in tied and tied[key] != spec['sha256']:
            raise PatchError('shared/tied snapshot weights disagree')
        tied[key]=spec['sha256']
    verify_shards(meta,store)  # All referenced data is checked before first write.
    with torch.no_grad():
        already=set()
        for spec in meta['tensors']:
            target=destinations[spec['name']]
            key=(target.untyped_storage().data_ptr(),target.storage_offset(),tuple(target.shape),tuple(target.stride()))
            if key in already:
                continue
            already.add(key)
            flat=target.view(-1) if target.is_contiguous() else None
            def store_range(start,values):
                if flat is not None:
                    flat[start:start+values.numel()].copy_(values)
                    return
                coords=[]
                indices=torch.arange(start,start+values.numel(),device=target.device,dtype=torch.long)
                for dim in reversed(tuple(target.shape)):
                    coords.append(indices.remainder(dim))
                    indices=torch.div(indices,dim,rounding_mode='floor')
                target[tuple(reversed(coords))]=values
            offset=0
            full_hash=hashlib.sha256()
            for chunk in spec['shards']:
                path=_blob_path(store,chunk['sha256'])
                with path.open('rb') as src:
                    remaining=chunk['size']
                    while remaining:
                        amount=min(chunk_bytes,remaining)
                        # Respect tensor element boundaries even across file reads.
                        amount-=amount % target.element_size()
                        if not amount:
                            raise PatchError('unaligned tensor shard')
                        data=src.read(amount)
                        if len(data)!=amount:
                            raise PatchError('unexpected EOF in verified shard')
                        full_hash.update(data)
                        raw=torch.frombuffer(bytearray(data),dtype=torch.uint8).view(target.dtype)
                        count=amount//target.element_size()
                        store_range(offset,raw.to(target.device))
                        offset+=count
                        remaining-=amount
                    if src.read(1):
                        raise PatchError('trailing shard data')
            if offset!=target.numel() or full_hash.hexdigest()!=spec['sha256']:
                raise PatchError('tensor digest changed while applying: '+spec['name'])


def apply_sharded_state(model, manifest: Path, store: Path, *, model_id: str,
                        revision: str, max_state_bytes: int = 32*1024**3,
                        chunk_bytes: int = 4*1024*1024):
    """Load data-only sharded state directly into an EXISTING model, no ZIP.

    Does not construct a model or offload its base weights; the model still
    needs adequate CPU/device memory (or an external offload-capable backend).
    """
    manager=getattr(model,'_fedpara_lazy_mutable_weights',None)
    if manager is not None and (getattr(manager,'stores',None) or getattr(manager,'sources',None)):
        raise PatchError('cannot import into a virtual base; materialize all first')
    meta=read_manifest(manifest,max_state_bytes=max_state_bytes)
    if (meta['model_id'],meta['revision'])!=(model_id,revision):
        raise PatchError('sharded snapshot model identity/revision mismatch')
    _install_verified_shards(meta,store,_sharded_destinations(model),chunk_bytes=chunk_bytes)
    return meta


def apply_sharded_effective_state(model, adapter, manifest: Path, store: Path, *,
                                  model_id: str, revision: str,
                                  max_state_bytes: int = 32*1024**3,
                                  chunk_bytes: int = 4*1024*1024):
    """Direct cold-join import, preserving tied weights and continual anchors."""
    import torch
    meta=read_manifest(manifest,max_state_bytes=max_state_bytes)
    if (meta['model_id'],meta['revision'])!=(model_id,revision):
        raise PatchError('sharded snapshot model identity/revision mismatch')
    with adapter.lock:
        # v60: cold-join / shadow installation must never write through a
        # read-only safetensors genesis mapping. Prepare private scratch first.
        from .lazy_mutable_weights_v60 import materialize_all
        materialize_all(adapter)
        _install_verified_shards(meta,store,_sharded_destinations(model,adapter),
                                 chunk_bytes=chunk_bytes)
        from .relora import _refresh_spec
        with torch.no_grad():
            for group in adapter.groups:
                factor=group.overlay.local
                _refresh_spec(factor)
                if hasattr(group.overlay,'anchor'):
                    for i,part in enumerate(factor.parts):
                        getattr(group.overlay.anchor,f'part{i}').copy_(part.detach())
                group.overlay.remotes.clear()
                group.overlay._peer_ids.clear()
    return meta


def semantic_shard_sha256(meta: dict) -> str:
    """Identity digest matching semantic_state_sha256 on assembled .fstate."""
    tensors=[{k:spec[k] for k in ('name','dtype','shape','sha256')} for spec in meta['tensors']]
    return hashlib.sha256(_canonical_json({'model_id':meta['model_id'],
                      'revision':meta['revision'],'tensors':tensors})).hexdigest()
