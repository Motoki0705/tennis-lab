"""CUDA extension bridge wrappers for deformable attention kernels."""

from __future__ import annotations

from torch import Tensor


def run_cuda_forward(
    ext: object,
    *,
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Invoke extension forward with a normalized contiguous tensor contract."""
    return ext.ms_deform_attn_forward(  # type: ignore[attr-defined]
        value.contiguous(),
        spatial_shapes.contiguous(),
        level_start_index.contiguous(),
        sampling_locations.contiguous(),
        attention_weights.contiguous(),
    )


def run_cuda_backward(
    ext: object,
    *,
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    grad_output: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Invoke extension backward with a normalized contiguous tensor contract."""
    return ext.ms_deform_attn_backward(  # type: ignore[attr-defined]
        value.contiguous(),
        spatial_shapes.contiguous(),
        level_start_index.contiguous(),
        sampling_locations.contiguous(),
        attention_weights.contiguous(),
        grad_output.contiguous(),
    )

