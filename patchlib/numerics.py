"""Numerically stable validation helpers that work on Apple MPS.

MPS has no float64, so avoid ``delta.double()`` during distributed-update
validation. Scaling by the maximum magnitude also avoids FP32 sum-of-squares
overflow before the Python float accumulator receives the result.
"""
from __future__ import annotations

import math
import torch


def squared_l2(delta: torch.Tensor) -> float:
    """Return sum(delta[i]**2) as a Python float without allocating fp64 tensors.

    Called on small reconstructed update blocks after finite-value validation.
    Only two scalar synchronizations are required; no full tensor is copied
    back to CPU, even with an MPS model.
    """
    if not delta.is_floating_point():
        raise TypeError('squared_l2 expects a floating-point tensor')
    if delta.numel() == 0:
        return 0.0
    # All validation arithmetic stays on the original accelerator in FP32.
    d = delta.detach().float()
    max_abs = float(d.abs().amax().item())
    if not math.isfinite(max_abs):
        return math.inf
    if max_abs == 0.0:
        return 0.0
    # 0 <= abs(d/max_abs) <= 1, so the reduction cannot overflow float32
    # for any realistically representable tensor. Python float handles the
    # exponent range when undoing the scale.
    unit_energy = float((d / max_abs).square().sum().item())
    return (max_abs * max_abs) * unit_energy
