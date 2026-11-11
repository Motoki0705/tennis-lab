"""Official forward reference and all original channel-width numerical gradchecks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch
from protocol import ROOT, write

from src.utils.models.components.ops.msda.vendor.functions.ms_deform_attn_func import (
    MSDeformAttnFunction,
    ms_deform_attn_core_pytorch,
)

torch.manual_seed(3)
shapes = torch.tensor([[6, 4], [3, 2]], device="cuda", dtype=torch.long)
starts = torch.tensor([0, 24], device="cuda", dtype=torch.long)
results = []
for dtype in (torch.float32, torch.float64):
    value = torch.rand(1, 30, 2, 64, device="cuda", dtype=dtype) * 0.01
    locations = torch.rand(1, 2, 2, 2, 2, 2, device="cuda", dtype=dtype)
    weights = torch.rand(1, 2, 2, 2, 2, device="cuda", dtype=dtype) + 1e-5
    weights = weights / weights.sum((-1, -2), keepdim=True)
    actual = MSDeformAttnFunction.apply(value, shapes, starts, locations, weights, 2)
    reference = ms_deform_attn_core_pytorch(value, shapes, locations, weights)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-7)
    results.append(
        dict(dtype=str(dtype), max_abs_error=float((actual - reference).abs().max()))
    )
for channels in (30, 32, 64, 71, 1025, 2048, 3096):
    value = (
        torch.rand(1, 30, 2, channels, device="cuda", dtype=torch.float64) * 0.01
    ).requires_grad_()
    locations = torch.rand(
        1, 2, 2, 2, 2, 2, device="cuda", dtype=torch.float64, requires_grad=True
    )
    weights = torch.rand(1, 2, 2, 2, 2, device="cuda", dtype=torch.float64)
    weights = (weights / weights.sum((-1, -2), keepdim=True)).requires_grad_()
    assert torch.autograd.gradcheck(
        MSDeformAttnFunction.apply,
        (value, shapes, starts, locations, weights, 2),
        fast_mode=True,
        nondet_tol=1e-10,
    )
    results.append(
        dict(channels=channels, gradcheck=True, fast_mode=True, nondet_tol=1e-10)
    )
write(ROOT / "msda-numerical-check.json", results)
