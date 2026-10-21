"""Conservative local resource adaptation; peer telemetry is untrusted advisory data.

Never adjust wire protocol identity (genesis, model, rank, architecture) live.
The current adapter format requires a common rank and layout. Only safe
per-device settings, such as batch size before loader creation, local steps,
checkpointing and wire encoding, are adapted.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import os
import time


def _mem_available_gib() -> float:
    try:
        import psutil
        return max(0., psutil.virtual_memory().available / 2**30)
    except ImportError:
        if hasattr(os, 'sysconf'):
            try:
                return max(0., os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 2**30)
            except (ValueError, OSError):
                pass
        return 2.0  # safest fallback


@dataclass(frozen=True)
class DeviceProfile:
    cpu_threads: int
    ram_free_gib: float
    accelerator: str
    vram_free_gib: float
    available_storage_gib: float

    def public_summary(self) -> dict:
        # Don't expose exact hardware characteristics to anonymous peers.
        return {'tier': self.tier, 'accelerator': self.accelerator != 'cpu'}

    @property
    def tier(self) -> int:
        memory = self.vram_free_gib if self.accelerator != 'cpu' else self.ram_free_gib
        if memory < 3 or self.cpu_threads <= 2: return 0
        if memory < 8 or self.cpu_threads <= 4: return 1
        if memory < 20: return 2
        return 3


def detect_device() -> DeviceProfile:
    import shutil
    import torch
    acc, vram = 'cpu', 0.
    if torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info()
            acc, vram = 'cuda', free / 2**30
        except RuntimeError:
            pass
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        acc, vram = 'mps', _mem_available_gib() * .60  # shared unified memory
    return DeviceProfile(max(1, os.cpu_count() or 1), _mem_available_gib(),
                         acc, max(0., vram), shutil.disk_usage('.').free / 2**30)


@dataclass(frozen=True)
class Policy:
    device: str
    batch_size: int
    grad_accum: int
    gradient_checkpointing: bool
    wire_dtype: str
    local_steps: int
    preferred_neighbors: int
    memory_fraction: float

    def json(self) -> dict: return asdict(self)


def initial_policy(profile: DeviceProfile, *, remote_tiers: list[int] | None = None,
                   requested_local_steps: int = 32, max_batch: int = 8) -> Policy:
    tier = profile.tier
    low_others = sum(int(t <= 1) for t in (remote_tiers or []))
    batch = min(max_batch, (1, 1, 2, 4)[tier])
    accum = max(1, 4 // batch)
    # Rarely advertise more than 4 connections; neighbors counted per device.
    neighbors = min(4, (2, 2, 3, 4)[tier])
    # In the presence of struggling peers, publish less frequently to lower
    # transfer volume. Keep max delay bounded to ensure eventual sync.
    steps = min(256, max(4, requested_local_steps * (2 if low_others else 1)))
    return Policy(profile.accelerator, batch, accum, tier < 2,
                  'float16' if tier <= 2 else 'float32', steps,
                  neighbors, (0.35, 0.45, 0.55, 0.65)[tier])


class AdaptiveController:
    """Use peer signed DataChannel heartbeat telemetry, not central server state.

    Remote tiers are capped and can only influence *local* pacing, not remote
    privileges or signed event divisor. A hysteresis prevents oscillations.
    """
    def __init__(self, profile: DeviceProfile, policy: Policy):
        self.profile, self.policy = profile, policy
        self.peers: dict[str, tuple[float, int]] = {}
        self.last_change = 0.

    def observe(self, peer: str, cap: dict, now: float | None = None):
        if not isinstance(cap, dict): return
        tier = cap.get('tier')
        if type(tier) is int and 0 <= tier <= 3:
            self.peers[peer] = (time.monotonic() if now is None else now, tier)

    def tune(self, *, now: float | None = None, timeout: float = 30.) -> Policy:
        now = time.monotonic() if now is None else now
        self.peers = {k:v for k,v in self.peers.items() if now-v[0] <= timeout}
        if now-self.last_change < 10.: return self.policy
        next_policy = initial_policy(self.profile,
                                     remote_tiers=[v[1] for v in self.peers.values()],
                                     requested_local_steps=max(4,self.policy.local_steps // (2 if self.policy.local_steps>64 else 1)),
                                     max_batch=self.policy.batch_size)
        # Only adjust the runtime pacing. The minibatch iterator and device are fixed.
        if next_policy.local_steps != self.policy.local_steps:
            self.policy = Policy(**{**asdict(self.policy), 'local_steps':next_policy.local_steps})
            self.last_change=now
        return self.policy
