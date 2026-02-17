"""Fallback correctness tests for deformable attention."""

from __future__ import annotations

import os

import torch

from src.common.models.components.ops.deformable.api import multi_scale_deformable_attention
from src.common.models.components.ops.deformable.kernels import msda_dispatch
from src.common.models.components.ops.deformable.reference import ms_deform_attn_reference
from src.common.models.components.ops.deformable.utils import build_level_start_index


def test_msda_fallback_matches_reference() -> None:
    torch.manual_seed(3)
    bsz, n_query, n_heads, head_dim = 2, 4, 3, 5
    spatial_shapes = torch.tensor([[3, 4], [2, 2]], dtype=torch.long)
    n_levels, n_points = int(spatial_shapes.shape[0]), 3
    total_tokens = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())

    value = torch.randn(bsz, total_tokens, n_heads, head_dim)
    sampling_locations = torch.rand(bsz, n_query, n_heads, n_levels, n_points, 2)
    attention_weights = torch.rand(bsz, n_query, n_heads, n_levels, n_points)
    attention_weights = attention_weights / attention_weights.sum(dim=(-1, -2), keepdim=True)
    level_start_index = build_level_start_index(spatial_shapes)

    out_api = multi_scale_deformable_attention(
        value=value,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        sampling_locations=sampling_locations,
        attention_weights=attention_weights,
        prefer_cuda=False,
    )
    out_ref = ms_deform_attn_reference(
        value=value,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        sampling_locations=sampling_locations,
        attention_weights=attention_weights,
    )

    assert out_api.shape == (bsz, n_query, n_heads, head_dim)
    assert torch.allclose(out_api, out_ref, atol=1e-6, rtol=1e-6)


def test_msda_force_fallback_env(monkeypatch) -> None:
    monkeypatch.setenv("MSDA_FORCE_FALLBACK", "1")
    msda_dispatch._MSDA_EXT = None
    msda_dispatch._MSDA_EXT_LOAD_ATTEMPTED = False
    # Ensure env setup is visible to loader before op invocation.
    assert os.environ.get("MSDA_FORCE_FALLBACK") == "1"

    torch.manual_seed(31)
    bsz, n_query, n_heads, head_dim = 1, 3, 2, 4
    spatial_shapes = torch.tensor([[2, 3], [1, 2]], dtype=torch.long)
    n_levels, n_points = int(spatial_shapes.shape[0]), 2
    total_tokens = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())

    value = torch.randn(bsz, total_tokens, n_heads, head_dim, requires_grad=True)
    sampling_locations = torch.rand(bsz, n_query, n_heads, n_levels, n_points, 2, requires_grad=True)
    attention_weights = torch.rand(bsz, n_query, n_heads, n_levels, n_points, requires_grad=True)
    attention_weights = attention_weights / attention_weights.sum(dim=(-1, -2), keepdim=True)

    out = multi_scale_deformable_attention(
        value=value,
        spatial_shapes=spatial_shapes,
        level_start_index=build_level_start_index(spatial_shapes),
        sampling_locations=sampling_locations,
        attention_weights=attention_weights,
        prefer_cuda=True,
    )
    loss = out.square().mean()
    loss.backward()

    assert torch.isfinite(out).all().item()
    assert value.grad is not None and torch.isfinite(value.grad).all().item()
    assert sampling_locations.grad is not None and torch.isfinite(sampling_locations.grad).all().item()
