"""Bounded-memory effective-weight generation for sharded Fedpara snapshots.

Only selected row/flat ranges are materialized.  No call to a parametrized
module's ``weight`` property is made for a complete adapted parameter.
"""
from __future__ import annotations

import math
import sys
from collections.abc import Iterator

import torch

from .core import PatchError, _cp
from .state import DTYPES


def effective_state_sources(model, adapter=None):
    """State keys mapped to (original Tensor, optional Overlay), no dense delta.

    Aliased/tied weights intentionally point at the same original and overlay.
    """
    entries = {key: (tensor, None) for key, tensor in model.state_dict(keep_vars=True).items()
               if '.parametrizations.' not in ('.' + key + '.')
               and not key.startswith('parametrizations.')}
    if adapter is not None:
        for group in adapter.groups:
            for alias in group.aliases:
                path, _, attr = alias.rpartition('.')
                module = model.get_submodule(path)
                entries[alias] = (module.parametrizations[attr].original, group.overlay)
    return entries


def _parts(spec):
    if hasattr(spec, 'parts'):
        return list(spec.parts)
    return [getattr(spec, f'part{i}') for i in range(spec.nparts)]


def _flat_factor(spec, first: int, last: int, shape: tuple[int, ...]) -> torch.Tensor:
    """Calculate arbitrary flattened range with O((end-start) * rank) space.

    A row-spanning fallback for extremely wide/oddly shaped tensors, avoiding
    any complete row or dense update allocation.
    """
    parts = _parts(spec)
    if spec.dense:
        return parts[0].reshape(-1)[first:last].float()
    indices = torch.arange(first, last, device=parts[0].device, dtype=torch.long)
    coordinates = []
    for dim in reversed(shape):
        coordinates.append(indices.remainder(dim))
        indices = torch.div(indices, dim, rounding_mode='floor')
    coordinates.reverse()
    nd = len(shape)

    def branch(offset):
        values = parts[offset][coordinates[0]]
        for axis in range(1, nd):
            values = values * parts[offset + axis][coordinates[axis]]
        return values.sum(dim=-1)

    result = branch(0)
    if spec.kind == 'loha':
        result = result * branch(nd)
    return result * (spec.alpha / spec.rank)


def _row_factor(spec, first: int, last: int) -> torch.Tensor:
    parts = _parts(spec)
    if spec.dense:
        return parts[0][first:last].float()
    ndim = len(spec.shape)
    result = _cp([parts[0][first:last], *parts[1:ndim]])
    if spec.kind == 'loha':
        result = result * _cp([parts[ndim][first:last], *parts[ndim+1:]])
    return result * (spec.alpha / spec.rank)


def effective_chunks(original: torch.Tensor, overlay=None, *, chunk_bytes: int = 4 * 1024 * 1024,
                     work_elements: int = 131072, base_source=None) -> Iterator[bytes]:
    """Serialize effective tensor in bounded working blocks (including huge rows).

    For ordinary 2-D matrices, use native low-rank matmul on full row stripes;
    for a row exceeding the budget use flattened factor gathers.  Multiple
    remote overlays and the continual anchor preserve Overlay.forward semantics.
    """
    if sys.byteorder != 'little':
        raise PatchError('shard runtime requires little-endian tensors')
    if original.layout != torch.strided or original.device.type == 'meta' or original.is_sparse:
        raise PatchError('cannot stream sparse/meta model tensors')
    if str(original.dtype) not in DTYPES:
        raise PatchError('unsupported tensor dtype')
    if not isinstance(chunk_bytes, int) or chunk_bytes <= 0 or not isinstance(work_elements, int) or work_elements <= 0:
        raise ValueError('invalid chunk budget')
    block = max(1, min(chunk_bytes // original.element_size(), work_elements))
    shape = tuple(original.shape)
    with torch.no_grad():
        flat = original.detach().view(-1) if original.is_contiguous() and base_source is None else None
        total = original.numel()
        def base_range(lo, end):
            if base_source is not None:
                if len(shape)!=2:raise PatchError('factor ledger requires 2-D base')
                width=shape[1]
                pieces=[]
                offset=lo
                while offset<end:
                    row,col=divmod(offset,width)
                    stop=min(end,offset+width-col)
                    pieces.append(base_source[row:row+1,col:col+stop-offset].reshape(-1))
                    offset=stop
                return torch.cat(pieces) if pieces else torch.empty(0,dtype=original.dtype)
            if flat is not None:
                return flat[lo:end]
            indices = torch.arange(lo, end, dtype=torch.long, device=original.device)
            coords = []
            for dim in reversed(shape):
                coords.append(indices.remainder(dim))
                indices = torch.div(indices, dim, rounding_mode='floor')
            return original[tuple(reversed(coords))]
        if overlay is None:
            for lo in range(0, total, block):
                src = base_range(lo, min(total, lo+block)).to('cpu').contiguous()
                yield src.view(torch.uint8).numpy().tobytes()
            return
        if not shape:
            # A scalar dense fallback has no row/coordinate axes.
            value = overlay.forward(original.to(overlay.local.parts[0].device))
            yield value.detach().to('cpu').contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            return
        specs = [(overlay.local, 1.0)]
        if hasattr(overlay, 'anchor'):
            specs.append((overlay.anchor, -1.0))
        specs.extend((remote, 1.0) for remote in overlay.remotes.values())
        denominator = 1 + len(overlay.remotes)
        row_elems = math.prod(shape[1:]) if len(shape) > 1 else 1
        # Whole-row computation matches PyTorch's existing factor-matrix
        # matmul, keeping the usual case fast and numerically compatible.
        rows = block // row_elems if shape and row_elems <= block else 0
        if rows:
            for row in range(0, shape[0], rows):
                end = min(shape[0], row+rows)
                original_part = (base_source.rows(row,end) if base_source is not None
                                 else original[row:end])
                delta = None
                for spec, sign in specs:
                    value = _row_factor(spec, row, end)
                    delta = sign*value if delta is None else delta+sign*value
                result = original_part.to(delta.device) + (delta/denominator).to(original.dtype)
                yield result.detach().to('cpu').contiguous().view(torch.uint8).numpy().tobytes()
        else:
            for lo in range(0, total, block):
                end = min(total, lo+block)
                delta = None
                for spec, sign in specs:
                    value = _flat_factor(spec, lo, end, shape)
                    delta = sign*value if delta is None else delta+sign*value
                result = base_range(lo,end).to(delta.device) + (delta/denominator).to(original.dtype)
                yield result.detach().to('cpu').contiguous().view(torch.uint8).numpy().tobytes()
