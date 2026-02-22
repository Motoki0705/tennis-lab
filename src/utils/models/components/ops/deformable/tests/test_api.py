"""API smoke tests for deformable operator layer."""

from __future__ import annotations

import torch

from src.common.models.components.ops.deformable import (
    MSDeformAttnConfig,
    MultiScaleDeformableAttention,
    multi_scale_deformable_attention,
)
from src.common.models.components.ops.deformable.utils import build_level_start_index


def test_api_import_and_module_forward_smoke() -> None:
    cfg = MSDeformAttnConfig(dim=32, num_heads=4, num_levels=2, num_points=2, use_cuda_kernel=False)
    mod = MultiScaleDeformableAttention(cfg)

    bsz, n_query, n_heads, head_dim = 2, 5, 4, 8
    spatial_shapes = torch.tensor([[3, 4], [2, 2]], dtype=torch.long)
    total_tokens = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())
    n_levels, n_points = int(spatial_shapes.shape[0]), 2

    query = torch.randn(bsz, n_query, cfg.dim)
    value = torch.randn(bsz, total_tokens, cfg.dim)
    sampling_locations = torch.rand(bsz, n_query, n_heads, n_levels, n_points, 2)
    attention_weights = torch.rand(bsz, n_query, n_heads, n_levels, n_points)
    attention_weights = attention_weights / attention_weights.sum(dim=(-1, -2), keepdim=True)

    out = mod(
        query=query,
        value=value,
        spatial_shapes=spatial_shapes,
        sampling_locations=sampling_locations,
        attention_weights=attention_weights,
        level_start_index=build_level_start_index(spatial_shapes),
    )

    assert out.shape == (bsz, n_query, cfg.dim)
    assert torch.isfinite(out).all()


def test_function_api_smoke() -> None:
    bsz, n_query, n_heads, head_dim = 1, 3, 2, 4
    spatial_shapes = torch.tensor([[2, 3]], dtype=torch.long)
    total_tokens = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())

    value = torch.randn(bsz, total_tokens, n_heads, head_dim)
    sampling_locations = torch.rand(bsz, n_query, n_heads, 1, 2, 2)
    attention_weights = torch.rand(bsz, n_query, n_heads, 1, 2)
    attention_weights = attention_weights / attention_weights.sum(dim=(-1, -2), keepdim=True)

    out = multi_scale_deformable_attention(
        value=value,
        spatial_shapes=spatial_shapes,
        level_start_index=build_level_start_index(spatial_shapes),
        sampling_locations=sampling_locations,
        attention_weights=attention_weights,
        prefer_cuda=False,
    )
    assert out.shape == (bsz, n_query, n_heads, head_dim)
