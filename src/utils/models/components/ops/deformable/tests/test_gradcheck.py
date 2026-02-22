"""Gradcheck tests for multi-scale deformable attention."""

from __future__ import annotations

import pytest
import torch

from src.common.models.components.ops.deformable.api import multi_scale_deformable_attention
from src.common.models.components.ops.deformable.kernels.msda_dispatch import get_msda_extension
from src.common.models.components.ops.deformable.utils import build_level_start_index


def _gradcheck_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(11)
    bsz, n_query, n_heads, head_dim = 1, 2, 2, 2
    spatial_shapes = torch.tensor([[2, 3], [1, 2]], device=device, dtype=torch.long)
    n_levels, n_points = int(spatial_shapes.shape[0]), 2
    total_tokens = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())

    value = torch.randn(
        bsz,
        total_tokens,
        n_heads,
        head_dim,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    )
    sampling_locations = torch.rand(
        bsz,
        n_query,
        n_heads,
        n_levels,
        n_points,
        2,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    ) * 0.8 + 0.1
    attention_weights = torch.randn(
        bsz,
        n_query,
        n_heads,
        n_levels,
        n_points,
        dtype=torch.double,
        device=device,
        requires_grad=True,
    ).softmax(dim=-1)
    level_start_index = build_level_start_index(spatial_shapes)
    return value, spatial_shapes, level_start_index, sampling_locations, attention_weights


def test_msda_gradcheck_fallback_cpu() -> None:
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = _gradcheck_inputs(
        torch.device("cpu")
    )

    def fn(v: torch.Tensor, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return multi_scale_deformable_attention(
            value=v,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            sampling_locations=s,
            attention_weights=a,
            prefer_cuda=False,
        )

    assert torch.autograd.gradcheck(fn, (value, sampling_locations, attention_weights), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.cuda
def test_msda_gradcheck_cuda_extension() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    ext = get_msda_extension()
    if ext is None:
        pytest.skip("MSDA CUDA extension is not available.")

    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = _gradcheck_inputs(
        torch.device("cuda")
    )

    def fn(v: torch.Tensor, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return multi_scale_deformable_attention(
            value=v,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            sampling_locations=s,
            attention_weights=a,
            prefer_cuda=True,
        )

    assert torch.autograd.gradcheck(fn, (value, sampling_locations, attention_weights), eps=1e-6, atol=3e-3, rtol=3e-3)
