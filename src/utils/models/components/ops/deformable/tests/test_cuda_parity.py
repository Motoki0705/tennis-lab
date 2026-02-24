"""CUDA parity tests for multi-scale deformable attention."""

from __future__ import annotations

import pytest
import torch

from src.utils.models.components.ops.deformable.api import multi_scale_deformable_attention
from src.utils.models.components.ops.deformable.kernels.msda_dispatch import get_msda_extension
from src.utils.models.components.ops.deformable.utils import build_level_start_index


def _sample_inputs(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(7)
    bsz, n_query, n_heads, head_dim = 2, 6, 4, 8
    spatial_shapes = torch.tensor([[5, 6], [3, 4]], device=device, dtype=torch.long)
    n_levels, n_points = int(spatial_shapes.shape[0]), 4
    total_tokens = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())

    value = torch.randn(bsz, total_tokens, n_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    sampling_locations = torch.rand(
        bsz,
        n_query,
        n_heads,
        n_levels,
        n_points,
        2,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    attention_weights = torch.randn(
        bsz,
        n_query,
        n_heads,
        n_levels,
        n_points,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    attention_weights = attention_weights.softmax(dim=-1)
    attention_weights.retain_grad()

    level_start_index = build_level_start_index(spatial_shapes)
    return value, spatial_shapes, level_start_index, sampling_locations, attention_weights


@pytest.mark.cuda
def test_msda_cuda_forward_parity() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    ext = get_msda_extension()
    if ext is None:
        pytest.skip("MSDA CUDA extension is not available.")

    device = torch.device("cuda")
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = _sample_inputs(
        device, torch.float32
    )

    out_cuda = multi_scale_deformable_attention(
        value=value,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        sampling_locations=sampling_locations,
        attention_weights=attention_weights,
        prefer_cuda=True,
    )
    out_ref = multi_scale_deformable_attention(
        value=value.detach().clone(),
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        sampling_locations=sampling_locations.detach().clone(),
        attention_weights=attention_weights.detach().clone(),
        prefer_cuda=False,
    )

    assert out_cuda.shape == out_ref.shape
    assert torch.allclose(out_cuda, out_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.cuda
def test_msda_cuda_backward_parity() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    ext = get_msda_extension()
    if ext is None:
        pytest.skip("MSDA CUDA extension is not available.")

    device = torch.device("cuda")
    value, spatial_shapes, level_start_index, sampling_locations, attention_weights = _sample_inputs(
        device, torch.float32
    )

    out_cuda = multi_scale_deformable_attention(
        value=value,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        sampling_locations=sampling_locations,
        attention_weights=attention_weights,
        prefer_cuda=True,
    )
    loss_cuda = out_cuda.square().mean()
    loss_cuda.backward()

    grad_value_cuda = value.grad.detach().clone()
    grad_sampling_cuda = sampling_locations.grad.detach().clone()
    grad_attn_cuda = attention_weights.grad.detach().clone()

    value_ref = value.detach().clone().requires_grad_(True)
    sampling_ref = sampling_locations.detach().clone().requires_grad_(True)
    attn_ref = attention_weights.detach().clone().requires_grad_(True)

    out_ref = multi_scale_deformable_attention(
        value=value_ref,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        sampling_locations=sampling_ref,
        attention_weights=attn_ref,
        prefer_cuda=False,
    )
    loss_ref = out_ref.square().mean()
    loss_ref.backward()

    assert torch.allclose(grad_value_cuda, value_ref.grad, atol=3e-3, rtol=3e-3)
    assert torch.allclose(grad_sampling_cuda, sampling_ref.grad, atol=4e-3, rtol=4e-3)
    assert torch.allclose(grad_attn_cuda, attn_ref.grad, atol=3e-3, rtol=3e-3)
