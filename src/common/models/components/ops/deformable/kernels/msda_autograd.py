"""Autograd-facing entrypoint for deformable attention kernels."""

from __future__ import annotations

import torch
from torch import Tensor

from src.common.models.components.ops.deformable.kernels.msda_dispatch import get_msda_extension, ms_deform_attn_dispatch


class _MSDeformAttnCudaFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        ext,
    ) -> Tensor:
        out = ext.ms_deform_attn_forward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )
        ctx.ext = ext
        ctx.save_for_backward(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_value, grad_sampling_locations, grad_attention_weights = ctx.ext.ms_deform_attn_backward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
        )
        # None for spatial_shapes, level_start_index and ext handle
        return grad_value, None, None, grad_sampling_locations, grad_attention_weights, None


def ms_deform_attn(
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    *,
    prefer_cuda: bool = True,
) -> Tensor:
    """Differentiable deformable attention op with CUDA extension fallback."""
    if prefer_cuda and value.is_cuda:
        ext = get_msda_extension()
        if ext is not None and hasattr(ext, "ms_deform_attn_forward") and hasattr(ext, "ms_deform_attn_backward"):
            return _MSDeformAttnCudaFn.apply(
                value.contiguous(),
                spatial_shapes.contiguous(),
                level_start_index.contiguous(),
                sampling_locations.contiguous(),
                attention_weights.contiguous(),
                ext,
            )

    return ms_deform_attn_dispatch(
        value=value,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        sampling_locations=sampling_locations,
        attention_weights=attention_weights,
        prefer_cuda=False,
    )
