"""Conservative, deterministic training defaults. No untrusted code execution.

These are memory/network *estimates*, not a claim of OOM-freedom. Network
membership remains an observed live quantity, never a user-set peer target.
"""
from __future__ import annotations

import math

_POSITION_KEYS = ('max_position_embeddings', 'n_positions', 'n_ctx',
                  'max_seq_len', 'max_sequence_length', 'seq_length',
                  'model_max_length')


def estimate_training_policy(*, config: dict | None, available_ram_bytes: int = 0,
                             available_vram_bytes: int = 0, rank: int = 16,
                             kind: str = 'loha') -> dict:
    if type(rank) is not int or not 1 <= rank <= 256 or kind not in ('loha', 'lora'):
        raise ValueError('invalid adapter configuration')
    cfg = config if isinstance(config, dict) else {}
    bounds = [int(cfg[key]) for key in _POSITION_KEYS
              if type(cfg.get(key)) is int and 16 <= cfg[key] <= 2**20]
    hard_max = min(bounds) if bounds else 1024  # unknown => application safety ceiling
    h = int(cfg.get('hidden_size') or cfg.get('n_embd') or cfg.get('d_model') or 768)
    depth = int(cfg.get('num_hidden_layers') or cfg.get('n_layer') or
                cfg.get('num_layers') or cfg.get('num_hidden_layers') or 12)
    h = max(32, min(131072, h))
    depth = max(1, min(512, depth))
    vram = max(0, int(available_vram_bytes or 0))
    ram = max(0, int(available_ram_bytes or 0))
    # User asks to use available memory, not arbitrarily cap to a tiny context.
    # Reserve base model/optimizer/allocator headroom; paging is not free.
    usable = int((vram * .24) if vram >= 768*1024**2 else (ram * .10))
    if usable <= 0:
        usable = 384*1024**2
    # Approximate activation checkpointing budget for batch size 1-2.
    bytes_per_token = max(16384, h * depth * 44)
    fitting = min(hard_max, max(32, usable // bytes_per_token))
    # Avoid odd sizes; no exception for non-power-of-two architecture limits.
    length = max(32, 1 << (int(fitting).bit_length()-1))
    length = min(length, hard_max, 8192)
    # LoRA/LoHa delta proxy: attention + feedforward projections and adapter factors.
    # Per-event wire is usually larger than a single rank factor; plan cautiously.
    estimate_bytes = h * depth * rank * 7 * 2 * 2 * (2 if kind == 'loha' else 1)
    mib = estimate_bytes / (1024*1024)
    if mib < 8: baseline = 4
    elif mib < 16: baseline = 8
    elif mib < 32: baseline = 16
    elif mib < 64: baseline = 32
    else: baseline = 64
    return {'max_length':int(length), 'local_steps':int(baseline),
            'model_context_limit':int(hard_max),
            'estimated_adapter_wire_mib':round(mib,2),
            'model_metadata_available':bool(cfg),
            'online_peers':None,
            'reason':('Context is a device-memory/architecture estimate; '
                      'publish interval also adapts to observed peer count. '
                      'Live topology is measured after connecting.')}


# The wire layout is SHARED and immutable for a given model. Auto-rank runs once
# at model creation; peers must use the pinned result, not their own preference.
RANK_CHOICES = (4, 8, 16, 32, 64)


def estimate_adapter_rank(*, config: dict | None, available_ram_bytes: int = 0,
                          available_vram_bytes: int = 0, kind: str = 'loha') -> dict:
    """Advisory capacity + network-conscious genesis rank, using metadata only.

    Deliberately uses the same worst-case proxy as context/publication planning.
    Never treats disk space as fast working memory. Unknown architecture receives
    the legacy conservative rank rather than an unjustified high-rank estimate.
    """
    if kind not in ('loha', 'lora'):
        raise ValueError('unknown adapter kind')
    cfg = config if isinstance(config, dict) else {}
    h = int(cfg.get('hidden_size') or cfg.get('n_embd') or cfg.get('d_model') or 768)
    depth = int(cfg.get('num_hidden_layers') or cfg.get('n_layer') or cfg.get('num_layers') or 12)
    h = max(32, min(131072, h))
    depth = max(1, min(512, depth))
    ram = max(0, int(available_ram_bytes or 0))
    vram = max(0, int(available_vram_bytes or 0))
    # Use accelerator capacity when one is present. CPU/MPS uses system RAM;
    # this is an upper bound, not proof that an OOM will not happen.
    working = vram if vram else ram
    def estimate(rank):
        # 7 projections, 2 matrices each, LoHa has two factor pairs.
        factors = h * depth * rank * 7 * 2 * 4 * (2 if kind == 'loha' else 1)
        return factors * 4  # factors + grads + Adam moments
    mem_limit = int(working * .15) if working else 0
    feasible = [rank for rank in RANK_CHOICES if not working or estimate(rank) <= mem_limit]
    max_safe = (max(feasible) if feasible else 0) if working else None
    if not cfg:
        chosen = 16 if 16 in feasible else (max(feasible) if feasible else 4)
    else:
        # Max ~32 MiB/commit where reasonable; the continual-gossip wire
        # traffic grows with rank even if a large GPU could fit rank=64.
        wire_ceiling = 32 * 1024**2
        recommended = [rank for rank in feasible if estimate(rank) // 4 <= wire_ceiling]
        chosen = max(recommended) if recommended else (min(feasible) if feasible else 4)
    return {'rank':chosen,'rank_capacity':max_safe,
            'memory_estimate_bytes':estimate(chosen),
            'wire_estimate_bytes':estimate(chosen) // 4,
            'metadata_available':bool(cfg),
            'warning':'Metadata-only estimation. Real peaks and bandwidth may differ; room rank is immutable.'}
