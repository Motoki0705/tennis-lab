"""Autograd-facing entrypoint for deformable attention kernels."""

from __future__ import annotations

import torch
from torch import Tensor

from src.common.models.components.ops.deformable.kernels.msda_cuda_bridge import (
    run_cuda_backward,
    run_cuda_forward,
)
from src.common.models.components.ops.deformable.kernels.msda_dtype import (
    cast_grad_output_for_compute,
    promote_inputs_for_compute,
    restore_backward_grad_dtypes,
    restore_forward_output_dtype,
)
from src.common.models.components.ops.deformable.kernels.msda_dispatch import get_msda_extension, ms_deform_attn_dispatch
from src.common.models.components.ops.deformable.kernels.msda_state import (
    load_ctx_ext,
    load_ctx_meta,
    save_ctx_state,
)


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
        value_in, sampling_in, attention_in, meta = promote_inputs_for_compute(
            value=value,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
        )
        out = run_cuda_forward(
            ext=ext,
            value=value_in,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            sampling_locations=sampling_in,
            attention_weights=attention_in,
        )

        save_ctx_state(ctx, meta=meta, ext=ext)
        ctx.save_for_backward(value_in, spatial_shapes, level_start_index, sampling_in, attention_in)
        return restore_forward_output_dtype(out, meta=meta)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        meta = load_ctx_meta(ctx)
        ext = load_ctx_ext(ctx)

        grad_output_in = cast_grad_output_for_compute(grad_output, promote_compute=meta.promote_compute)
        grad_value, grad_sampling_locations, grad_attention_weights = run_cuda_backward(
            ext=ext,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
            grad_output=grad_output_in,
        )

        grad_value, grad_sampling_locations, grad_attention_weights = restore_backward_grad_dtypes(
            grad_value,
            grad_sampling_locations,
            grad_attention_weights,
            meta=meta,
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
