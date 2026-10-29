"""Universal tensor LoRA / LoHA with exact additive multi-peer overlays.

This adapter operates on ordinary PyTorch Parameters, rather than selected
Linear/Embedding/Conv module types. For tensors of order >=2 where every
axis >= rank, we use a rank-r CP expansion (matrix LoRA for order 2). For
LoHA, two CP expansions are multiplied elementwise. Other tensors use a
full dense trainable delta by design.

No unpickling, dynamic code execution, or arbitrary model hooks are used.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import threading
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn.utils import parametrize
from safetensors.torch import load as safe_load, save as safe_save

MAGIC = b"FPAT1"
MAX_META = 8 * 1024 * 1024
MAX_PATCH = 512 * 1024 * 1024
PEER_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

class PatchError(ValueError):
    pass


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _cp(factors: list[torch.Tensor]) -> torch.Tensor:
    # For matrix weights the dense update is the usual A @ B.
    if len(factors) == 2:
        return factors[0] @ factors[1].T
    # CP expansion, supporting higher-order weights without an einsum-index cap.
    x = factors[0]
    for axis, f in enumerate(factors[1:], start=1):
        x = x.unsqueeze(-2) * f.reshape(*([1] * axis), *f.shape)
    return x.sum(dim=-1)


class FactorDelta(nn.Module):
    """Trainable factors. Base tensors are never modified by this class."""
    def __init__(self, shape: tuple[int, ...], kind: str, rank: int, alpha: float,
                 init_style: str = 'balanced'):
        super().__init__()
        if kind not in ("lora", "loha") or rank <= 0 or alpha <= 0:
            raise ValueError("kind must be lora/loha; rank and alpha must be positive")
        self.shape = tuple(int(x) for x in shape)
        if init_style not in ('legacy','balanced'):
            raise ValueError('invalid factor initialization')
        self.init_style = init_style
        self.kind, self.rank, self.alpha = kind, rank, float(alpha)
        self.dense = len(shape) < 2 or min(shape) < rank
        self.parts = nn.ParameterList()
        if self.dense:
            # Explicitly retain full-shape parameters when any dimension < rank.
            self.parts.append(nn.Parameter(torch.zeros(shape, dtype=torch.float32)))
        else:
            groups = 1 if kind == "lora" else 2
            for group in range(groups):
                for axis, dimension in enumerate(shape):
                    x = torch.empty(dimension, rank, dtype=torch.float32)
                    # Initialize one factor in the only trainable LoRA branch at 0.
                    # LoHA: its *second* CP branch has one zero factor; the first
                    # CP branch is nonzero, preventing the dead-gradient pitfall.
                    if (kind == "lora" and axis == len(shape)-1) or (kind == "loha" and group == 1 and axis == len(shape)-1):
                        nn.init.zeros_(x)
                    elif init_style == 'balanced' and len(shape)==2 and kind=='lora':
                        nn.init.normal_(x,mean=0,std=(1. / math.sqrt(shape[1]) if axis==0 else 1.))
                    elif init_style == 'balanced' and len(shape) == 2 and kind == 'loha':
                        # The first CP product has RMS~1; the *other* product
                        # is exactly zero at step zero but has a nonzero
                        # derivative wrt its last factor. Legacy std=.02 in
                        # BOTH branches made the LoHa startup gradient tiny.
                        std = rank ** (-.25) if group == 0 else rank ** (-.5)
                        nn.init.normal_(x, mean=0, std=std)
                    else:
                        nn.init.normal_(x, mean=0, std=0.2 if len(shape) > 2 else 0.02)
                    self.parts.append(nn.Parameter(x))

    def expected_shapes(self) -> list[tuple[int, ...]]:
        if self.dense:
            return [self.shape]
        return [(d, self.rank) for _ in range(1 if self.kind == 'lora' else 2) for d in self.shape]

    def delta(self) -> torch.Tensor:
        if self.dense:
            return self.parts[0]
        n = len(self.shape)
        a = _cp(list(self.parts[:n]))
        if self.kind == 'loha':
            a = a * _cp(list(self.parts[n:]))
        return a * (self.alpha / self.rank)


class FrozenDelta(nn.Module):
    """An untrusted remote patch converted to immutable, validated buffers."""
    def __init__(self, spec: FactorDelta, tensors: list[torch.Tensor]):
        super().__init__()
        self.shape, self.kind = spec.shape, spec.kind
        self.rank, self.alpha, self.dense = spec.rank, spec.alpha, spec.dense
        for i, tensor in enumerate(tensors):
            self.register_buffer(f"part{i}", tensor.detach().to(dtype=torch.float32).clone(), persistent=False)
        self.nparts = len(tensors)

    def delta(self):
        parts = [getattr(self, f"part{i}") for i in range(self.nparts)]
        if self.dense:
            return parts[0]
        n = len(self.shape)
        a = _cp(parts[:n])
        if self.kind == "loha":
            a = a * _cp(parts[n:])
        return a * (self.alpha / self.rank)


class Overlay(nn.Module):
    def __init__(self, local: FactorDelta):
        super().__init__()
        self.local = local
        self.remotes = nn.ModuleDict()
        self._peer_ids: dict[str, str] = {}

    def install(self, peer_id: str, parts: list[torch.Tensor]) -> None:
        key = hashlib.sha256(peer_id.encode()).hexdigest()
        self.remotes[key] = FrozenDelta(self.local, parts).to(self.local.parts[0].device)
        self._peer_ids[peer_id] = key

    def remove(self, peer_id: str) -> None:
        key = self._peer_ids.pop(peer_id, None)
        if key is not None and key in self.remotes:
            del self.remotes[key]

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        # Crucially average the *updates*, NOT the LoRA/LoHA factor matrices.
        delta = self.local.delta()
        # Continual elastic mode keeps optimizer/factors across commits. A frozen
        # low-rank anchor is subtracted so previously committed local factors
        # cannot be counted twice. Legacy overlays have no anchor.
        if hasattr(self, 'anchor'):
            delta = delta - self.anchor.delta()
        for remote in self.remotes.values():
            delta = delta + remote.delta()
        delta = delta / (1 + len(self.remotes))
        return base + delta.to(dtype=base.dtype)


@dataclass
class Group:
    canonical: str
    aliases: tuple[str, ...]
    shape: tuple[int, ...]
    overlay: Overlay


class UniversalAdapter:
    def __init__(self, model: nn.Module, model_id: str, revision: str, *,
                 kind: str = 'loha', rank: int = 16, alpha: float | None = None,
                 max_peers: int = 8, target_scope: str = 'all',
                 init_style: str = 'balanced', wide_lora_rank: int = 4):
        if not isinstance(wide_lora_rank,int) or not 1 <= wide_lora_rank <= 16:
            raise ValueError('wide_lora_rank must be between 1 and 16')
        self.wide_lora_rank=wide_lora_rank
        if rank <= 0 or max_peers < 1:
            raise ValueError('rank and max_peers must be positive')
        if kind not in ('lora', 'loha'):
            raise ValueError('kind must be lora or loha')
        self.model, self.model_id, self.revision = model, model_id, revision
        self.kind, self.rank = kind, rank
        self.alpha = float(rank if alpha is None else alpha)
        if target_scope not in ('all','gpt_neo_linear','gpt_neo_full','auto','linear','universal'):
            raise ValueError('unsupported target_scope')
        self.target_scope = target_scope
        self.init_style = init_style
        self.max_peers = max_peers
        self.lock = threading.RLock()
        self.groups: list[Group] = []
        self.remote_versions: dict[str, int] = {}
        self._attach()
        if self.target_scope in ('gpt_neo_linear','gpt_neo_full','auto','linear','universal'):
            # Filtering adapter targets is NOT enough: excluded pretrained
            # parameters must be frozen too. Otherwise autograd still computes
            # embedding/LM head/LayerNorm gradients (and optimizer scope is
            # misleading), wasting memory and potentially leaking grads.
            trainable_ids={id(param) for g in self.groups for param in g.overlay.local.parts}
            for param in self.model.parameters():
                if id(param) not in trainable_ids:
                    param.requires_grad_(False)
        if not self.groups:
            raise ValueError(
                'No compatible trainable LoRA/LoHA targets for this architecture '
                f'(scope={self.target_scope!r}, rank={self.rank}, wide_lora_rank={self.wide_lora_rank}). '
                'For narrow matrices use a smaller SHARED --rank/--wide-lora-rank '
                'in a NEW P2P room. For custom convolution/N-D tensors, '
                'explicitly select --target-scope universal after a successful '
                'training-preflight; never change a running room adapter layout.')
        self.layout = self._layout_digest()

    def _attach(self):
        """Bind each original Parameter exactly once, including all tied aliases.

        `auto` is architecture-neutral: embeddings and output heads use a small
        LoRA, as do strongly rectangular 2-D projections; other linear/2-D
        weights use the requested LoHa. Names and get_output_embeddings() are
        used as hints, but rank selection is always recorded in the layout.
        """
        owners: dict[int, list[tuple[str, nn.Module, str, nn.Parameter]]] = {}
        # Universal mode preserves names of shared module aliases. Legacy
        # modes retain their original signed layout digests.
        bound_fields = set()
        iterator = (self.model.named_modules(remove_duplicate=False) if
                    self.target_scope == 'universal' else self.model.named_modules())
        for module_path, module in iterator:
            for pname, param in list(module.named_parameters(recurse=False)):
                if not isinstance(param, nn.Parameter) or not param.is_floating_point() or param.is_sparse:
                    continue
                fullname = (module_path + '.' + pname).lstrip('.')
                owners.setdefault(id(param), []).append((fullname, module, pname, param))

        output_id = None
        get_output = getattr(self.model, 'get_output_embeddings', None)
        if callable(get_output):
            try:
                output = get_output()
                if isinstance(output, nn.Module):
                    weight = getattr(output, 'weight', None)
                    if isinstance(weight, nn.Parameter):
                        output_id = id(weight)
            except (AttributeError, NotImplementedError):
                pass

        for entries in owners.values():
            name, _, _, param = entries[0]
            aliases = tuple(item[0] for item in entries)
            shape = tuple(param.shape)
            if self.target_scope == 'universal':
                # Any ordinary differentiable floating Parameter may be
                # targeted, including custom MoE/router, 1-D or N-D kernels.
                if not shape or any(dim <= 0 for dim in shape):
                    continue
                if param.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                    raise PatchError(f'{name}: unsupported floating dtype {param.dtype}; '
                                     'dequantize with a trusted backend before attaching')
                if (len(shape)<2 or min(shape)<self.rank) and math.prod(shape)>1_048_576:
                    raise PatchError(f'{name}: dense adapter fallback exceeds 1,048,576 elements; '
                                     'reduce rank or use a custom factor backend')
                wide = False
            elif self.target_scope in ('gpt_neo_linear', 'gpt_neo_full'):
                linear = any(
                    pname == 'weight' and isinstance(module, nn.Linear)
                    and re.fullmatch(r'transformer\.h\.\d+\.(?:attn\.attention\.(?:q_proj|k_proj|v_proj|out_proj)|mlp\.(?:c_fc|c_proj))', path.rpartition('.')[0])
                    for path, module, pname, _ in entries
                )
                embedding = any(
                    pname == 'weight' and isinstance(module, nn.Embedding)
                    and path in ('transformer.wte.weight', 'transformer.wpe.weight')
                    for path, module, pname, _ in entries
                )
                head = any(path == 'lm_head.weight' and isinstance(module, nn.Linear)
                           for path, module, pname, _ in entries)
                if not (linear or (self.target_scope == 'gpt_neo_full' and (embedding or head))):
                    continue
                if len(shape) != 2 or min(shape) < self.rank:
                    continue
                wide = (self.target_scope == 'gpt_neo_full' and self.kind == 'loha'
                        and (embedding or head))
            elif self.target_scope == 'linear':
                # Architecture-neutral LoRA-Muon / projection-only policy.
                # Do not accidentally train LayerNorms, embeddings, heads or
                # unused router tensors; never create a dense fallback.
                if (len(shape)!=2 or min(shape)<self.rank or
                    any(pname=='weight' and isinstance(module,(nn.Embedding,nn.EmbeddingBag)) for _,module,pname,_ in entries) or
                    output_id == id(param) or
                    not any(pname=='weight' and
                            (isinstance(module,nn.Linear) or type(module).__name__=='Conv1D')
                            for _,module,pname,_ in entries)):
                    continue
                wide=False
            elif self.target_scope == 'auto':
                # Skip biases, norms and arbitrary dense 1-D parameters. Pick
                # any 2-D weight (not only GPT-Neo paths), including GPT-2's
                # Conv1D and custom HF modules. Small matrices cannot support
                # the requested factor rank and are skipped, not made dense.
                if len(shape) != 2 or not any(pname == 'weight' for _, _, pname, _ in entries):
                    continue
                is_embedding = any(
                    pname == 'weight' and isinstance(module, (nn.Embedding, nn.EmbeddingBag))
                    for _, module, pname, _ in entries
                )
                is_output = output_id == id(param)
                is_head = any(
                    pname == 'weight' and path.rsplit('.', 1)[0].rsplit('.', 1)[-1].lower()
                    in ('lm_head', 'embed_out', 'output_projection',
                        'output_layer', 'language_model_head', 'classifier', 'score')
                    for path, _, pname, _ in entries
                )
                # Linear projections whose major dimension is >=8x the minor
                # dimension have much larger LoHa wire factors than LoRA.
                # Even unusual dimensions (min < LoHa rank) can be LoRA if
                # min >= low-rank LoRA size.
                rectangular = min(shape) > 0 and max(shape) >= 8 * min(shape)
                wide = (self.kind == 'loha' and
                        (is_embedding or is_output or is_head or rectangular))
                required_rank = self.wide_lora_rank if wide else self.rank
                if min(shape) < required_rank:
                    continue
            else:
                wide = False  # Explicit `all` is backward-compatible.
            if param.device.type == 'meta':
                raise ValueError(f'Meta-device param {name} must be materialized first')
            group_kind = 'lora' if wide else self.kind
            group_rank = self.wide_lora_rank if wide else self.rank
            group_alpha = float(group_rank) if wide else self.alpha
            overlay = Overlay(FactorDelta(shape, group_kind, group_rank, group_alpha,
                                          init_style=self.init_style)).to(device=param.device)
            param.requires_grad_(False)
            # Some weight-tied models expose the same Parameter through several
            # modules. Register the exact same Overlay on ALL aliases.
            for _, module, pname, _ in entries:
                if self.target_scope == 'universal' and (id(module),pname) in bound_fields:
                    continue
                bound_fields.add((id(module),pname))
                parametrize.register_parametrization(module, pname, overlay,
                    unsafe=bool(getattr(self.model,'_fedpara_streaming_base_info',None)))
            self.groups.append(Group(name, aliases, shape, overlay))
        self.groups.sort(key=lambda x: x.canonical)

    def _layout_digest(self):
        schema = [[x.canonical, list(x.aliases), list(x.shape),
                   x.overlay.local.kind, x.overlay.local.rank, x.overlay.local.alpha]
                  for x in self.groups]
        return hashlib.sha256(_canonical_json(schema)).hexdigest()

    def trainable_parameters(self):
        seen = set()
        for g in self.groups:
            for param in g.overlay.local.parameters():
                if id(param) not in seen:
                    seen.add(id(param))
                    yield param

    def counts(self) -> dict[str, int]:
        params = sum(math.prod(g.shape) for g in self.groups)
        patch = sum(sum(math.prod(shape) for shape in g.overlay.local.expected_shapes()) for g in self.groups)
        return {"original": params, "patch": patch, "groups": len(self.groups),
                "dense_fallback": sum(g.overlay.local.dense for g in self.groups),
                "wide_lora_groups": sum(g.overlay.local.kind=='lora' for g in self.groups),
                "wide_lora_rank": self.wide_lora_rank}

    def _metadata(self, peer_id: str, seq: int) -> dict[str, Any]:
        return {"format": 1, "model_id": self.model_id, "revision": self.revision,
                "layout": self.layout, "kind": self.kind, "rank": self.rank,
                "alpha": self.alpha, "peer_id": peer_id, "seq": seq,
                "targets": [{"name": g.canonical, "aliases": list(g.aliases),
                             "shape": list(g.shape), "dense": g.overlay.local.dense,
                             "kind":g.overlay.local.kind, "rank":g.overlay.local.rank,
                             "alpha":g.overlay.local.alpha}
                            for g in self.groups]}

    def export(self, peer_id: str, seq: int = 0, dtype: str = 'float32', *,
               extra_meta: dict[str, Any] | None = None) -> bytes:
        if not PEER_ID.fullmatch(peer_id) or seq < 0:
            raise PatchError('invalid peer ID / sequence number')
        if dtype not in ('float32', 'float16'):
            raise PatchError('dtype must be float32 or float16')
        with self.lock:
            weights = {}
            for idx, group in enumerate(self.groups):
                for j, param in enumerate(group.overlay.local.parts):
                    x = param.detach().cpu().contiguous()
                    weights[f'p{idx}.f{j}'] = x.to(getattr(torch, dtype))
            meta = self._metadata(peer_id, seq)
            if extra_meta:
                if set(extra_meta) & set(meta):
                    raise PatchError('extra metadata must not override patch identity')
                meta.update(extra_meta)
            return encode_patch(meta, weights)

    def _validate(self, meta: dict[str, Any], weights: dict[str, torch.Tensor]) -> list[list[torch.Tensor]]:
        reference = self._metadata(meta.get('peer_id', ''), meta.get('seq', -1))
        for key in ('format', 'model_id', 'revision', 'layout', 'kind', 'rank', 'alpha', 'targets'):
            if meta.get(key) != reference[key]:
                raise PatchError(f'patch incompatible: {key}')
        if not isinstance(meta.get('seq'), int) or meta['seq'] < 0:
            raise PatchError('invalid patch sequence')
        if not isinstance(meta.get('peer_id'), str) or not PEER_ID.fullmatch(meta['peer_id']):
            raise PatchError('invalid patch peer ID')
        all_parts = []
        expected = set()
        for i, g in enumerate(self.groups):
            spec = g.overlay.local
            parts = []
            for j, shape in enumerate(spec.expected_shapes()):
                key = f'p{i}.f{j}'
                expected.add(key)
                if key not in weights:
                    raise PatchError(f'missing tensor {key}')
                t = weights[key]
                if tuple(t.shape) != shape or t.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                    raise PatchError(f'invalid shape/dtype: {key}')
                if not torch.isfinite(t).all().item():
                    raise PatchError(f'nonfinite values: {key}')
                parts.append(t)
            all_parts.append(parts)
        if set(weights) != expected:
            raise PatchError('unexpected/missing tensors')
        return all_parts

    def load_local(self, data: bytes, expected_peer: str | None = None) -> str:
        """Resume editable local factors (do not count the checkpoint as a remote)."""
        meta, weights = decode_patch(data)
        if expected_peer is not None and meta.get('peer_id') != expected_peer:
            raise PatchError('local checkpoint sender mismatch')
        with self.lock, torch.no_grad():
            parts = self._validate(meta, weights)
            for g, group_parts in zip(self.groups, parts):
                for param, tensor in zip(g.overlay.local.parts, group_parts):
                    param.copy_(tensor.to(param.device, dtype=param.dtype))
            return meta['peer_id']

    def import_remote(self, data: bytes, expected_peer: str | None = None) -> str:
        meta, weights = decode_patch(data)
        if expected_peer is not None and meta.get('peer_id') != expected_peer:
            raise PatchError('sender identity mismatch')
        with self.lock:
            parts = self._validate(meta, weights)
            peer_id, seq = meta['peer_id'], meta['seq']
            if seq <= self.remote_versions.get(peer_id, -1):
                raise PatchError('replayed or stale patch')
            if peer_id not in self.remote_versions and len(self.remote_versions) >= self.max_peers:
                raise PatchError('too many remote peers')
            # Validation and allocations all complete before swapping live state.
            prepared = [FrozenDelta(g.overlay.local, ps).to(g.overlay.local.parts[0].device)
                        for g, ps in zip(self.groups, parts)]
            key = hashlib.sha256(peer_id.encode()).hexdigest()
            for g, remote in zip(self.groups, prepared):
                g.overlay.remotes[key] = remote
                g.overlay._peer_ids[peer_id] = key
            self.remote_versions[peer_id] = seq
            return peer_id

    def remove_remote(self, peer_id: str):
        with self.lock:
            for g in self.groups:
                g.overlay.remove(peer_id)
            self.remote_versions.pop(peer_id, None)

    def merge_into_base(self) -> None:
        """Merge the current ensemble into the original weights and detach adapters.

        This is irreversible in-place, may need full tensor memory, and is
        intended for exporting a standalone ordinary model after training.
        """
        with self.lock, torch.no_grad():
            for g in self.groups:
                for alias in g.aliases:
                    mod_path, _, attr = alias.rpartition('.')
                    module = self.model.get_submodule(mod_path)
                    # Beware: tied weights require merging exactly once.
                    if alias == g.canonical:
                        p = module.parametrizations[attr].original
                        delta = g.overlay.local.delta()
                        for peer_delta in g.overlay.remotes.values():
                            delta = delta + peer_delta.delta()
                        p.add_((delta / (1 + len(g.overlay.remotes))).to(p.dtype))
                    parametrize.remove_parametrizations(module, attr, leave_parametrized=False)


def encode_patch(meta: dict[str, Any], tensors: dict[str, torch.Tensor]) -> bytes:
    header = _canonical_json(meta)
    if len(header) > MAX_META:
        raise PatchError('oversized patch metadata')
    payload = safe_save({k: v.detach().cpu().contiguous() for k, v in tensors.items()})
    data = MAGIC + struct.pack('!I', len(header)) + header + payload
    if len(data) > MAX_PATCH:
        raise PatchError('patch exceeds maximum size')
    return data


def decode_patch(data: bytes, max_bytes: int = MAX_PATCH) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if len(data) > max_bytes or len(data) < len(MAGIC) + 4 or not data.startswith(MAGIC):
        raise PatchError('invalid or oversized patch')
    hlen = struct.unpack('!I', data[len(MAGIC):len(MAGIC)+4])[0]
    if hlen > MAX_META or len(data) <= len(MAGIC)+4+hlen:
        raise PatchError('invalid metadata length')
    try:
        meta = json.loads(data[len(MAGIC)+4:len(MAGIC)+4+hlen])
        if not isinstance(meta, dict):
            raise PatchError('metadata must be object')
        tensors = safe_load(data[len(MAGIC)+4+hlen:])
    except (ValueError, RuntimeError, TypeError) as e:
        raise PatchError(f'patch decoding failed: {e}') from e
    return meta, tensors
