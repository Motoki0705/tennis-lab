"""Performance smoke tests for deformable attention."""

from __future__ import annotations

import time

import pytest
import torch

from src.utils.models.components.ops.deformable.api import multi_scale_deformable_attention
from src.utils.models.components.ops.deformable.kernels.msda_dispatch import get_msda_extension
from src.utils.models.components.ops.deformable.utils import build_level_start_index


def _make_case(device: torch.device, dtype: torch.dtype):
    torch.manual_seed(17)
    bsz, n_query, n_heads, head_dim = 2, 64, 8, 16
    spatial_shapes = torch.tensor([[24, 32], [12, 16]], device=device, dtype=torch.long)
    n_levels, n_points = int(spatial_shapes.shape[0]), 4
    total_tokens = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())
    level_start_index = build_level_start_index(spatial_shapes)

    value = torch.randn(bsz, total_tokens, n_heads, head_dim, device=device, dtype=dtype)
    sampling_locations = torch.rand(bsz, n_query, n_heads, n_levels, n_points, 2, device=device, dtype=dtype)
    attention_weights = torch.rand(bsz, n_query, n_heads, n_levels, n_points, device=device, dtype=dtype)
    attention_weights = attention_weights / attention_weights.sum(dim=(-1, -2), keepdim=True)
    return value, spatial_shapes, level_start_index, sampling_locations, attention_weights


@pytest.mark.cuda
def test_msda_cuda_perf_smoke() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    ext = get_msda_extension()
    if ext is None:
        pytest.skip("MSDA CUDA extension is not available.")

    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = _make_case(
        torch.device("cuda"), torch.float32
    )

    for _ in range(3):
        _ = multi_scale_deformable_attention(
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
            prefer_cuda=True,
        )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    out = multi_scale_deformable_attention(
        value=value,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        sampling_locations=sampling_locations,
        attention_weights=attention_weights,
        prefer_cuda=True,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    assert out.is_cuda
    assert torch.isfinite(out).all()
    assert elapsed < 2.0
