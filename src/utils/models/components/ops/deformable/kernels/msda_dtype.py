"""Dtype policy helpers for mixed-precision deformable attention kernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class MsdaDtypeMeta:
    """Metadata required to restore gradient dtypes after promoted compute."""

    promote_compute: bool
    value_dtype: torch.dtype
    sampling_dtype: torch.dtype
    attention_dtype: torch.dtype


def should_promote_compute(value: Tensor) -> bool:
    """Return True when kernel compute should be promoted to fp32."""
    return value.dtype in (torch.float16, torch.bfloat16)


def promote_inputs_for_compute(
    value: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> tuple[Tensor, Tensor, Tensor, MsdaDtypeMeta]:
    """Cast input tensors to compute dtype and return restoration metadata."""
    promote_compute = should_promote_compute(value)
    value_in = value.float() if promote_compute else value
    sampling_in = sampling_locations.float() if promote_compute else sampling_locations
    attention_in = attention_weights.float() if promote_compute else attention_weights
    meta = MsdaDtypeMeta(
        promote_compute=promote_compute,
        value_dtype=value.dtype,
        sampling_dtype=sampling_locations.dtype,
        attention_dtype=attention_weights.dtype,
    )
    return value_in, sampling_in, attention_in, meta


def cast_grad_output_for_compute(grad_output: Tensor, *, promote_compute: bool) -> Tensor:
    """Cast grad_output to compute dtype when forward promoted precision."""
    if promote_compute:
        return grad_output.float().contiguous()
    return grad_output.contiguous()


def restore_forward_output_dtype(output: Tensor, *, meta: MsdaDtypeMeta) -> Tensor:
    """Restore forward output dtype to the original value dtype when promoted."""
    return output.to(meta.value_dtype) if meta.promote_compute else output


def restore_backward_grad_dtypes(
    grad_value: Tensor,
    grad_sampling_locations: Tensor,
    grad_attention_weights: Tensor,
    *,
    meta: MsdaDtypeMeta,
) -> tuple[Tensor, Tensor, Tensor]:
    """Restore gradient tensors to original input dtypes when promoted."""
    if not meta.promote_compute:
        return grad_value, grad_sampling_locations, grad_attention_weights
    return (
        grad_value.to(meta.value_dtype),
        grad_sampling_locations.to(meta.sampling_dtype),
        grad_attention_weights.to(meta.attention_dtype),
    )
