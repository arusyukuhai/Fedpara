"""Regression tests for unsupported float64 cast on MPS during RMS checks."""
import math
import pytest
import torch
from patchlib.numerics import squared_l2

@pytest.mark.parametrize('shape', [(), (21,), (15,17), (2,3,4)])
def test_squared_l2_matches_cpu_float64(shape):
    torch.manual_seed(123)
    x = torch.randn(shape, dtype=torch.float32)
    expected = x.double().square().sum().item()
    assert squared_l2(x) == pytest.approx(expected, rel=2e-6, abs=1e-12)

def test_scaled_squared_l2_handles_float32_overflow_and_underflow():
    for x in [torch.full((4096,),1e25),torch.full((4096,),1e-25),torch.zeros(300),torch.tensor(3e37)]:
        actual = squared_l2(x)
        expected = x.double().square().sum().item()
        assert math.isfinite(actual)
        assert actual == pytest.approx(expected, rel=1e-5, abs=1e-65)

def test_nonfinite_never_passes_norm_gate():
    assert math.isinf(squared_l2(torch.tensor([float('inf')])))
    assert math.isinf(squared_l2(torch.tensor([float('nan')])))

@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Apple MPS required')
def test_actual_mps_has_no_fp64_cast():
    x = torch.randn(1024, device='mps',dtype=torch.float32)
    expected = x.cpu().double().square().sum().item()
    assert squared_l2(x) == pytest.approx(expected,rel=2e-6)
