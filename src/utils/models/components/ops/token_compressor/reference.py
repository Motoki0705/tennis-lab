"""Gather-based reference pooling for the token compressor."""

from __future__ import annotations

import torch
from torch import Tensor

from src.utils.models.components.ops.token_compressor.layout import (
    build_token_compressor_layout,
)

_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _validate_reference_inputs(
    raw_kv: Tensor,
    raw_gate: Tensor,
    state_valid: Tensor,
    *,
    compression_ratio: int,
) -> tuple[int, int, int]:
    if raw_kv.ndim != 4:
        raise ValueError(
            f"raw_kv must have shape [N, T, 2, Dh], got {tuple(raw_kv.shape)}"
        )
    if raw_gate.shape != raw_kv.shape:
        raise ValueError(
            "raw_gate shape must equal raw_kv shape, got "
            f"{tuple(raw_gate.shape)} and {tuple(raw_kv.shape)}"
        )
    n, sequence_length, branches, head_dim = raw_kv.shape
    if n <= 0 or sequence_length <= 0 or head_dim <= 0:
        raise ValueError("raw_kv batch, time, and head dimensions must be positive")
    if branches != 2:
        raise ValueError(f"raw_kv branch dimension must be 2, got {branches}")
    for name, tensor in (("raw_kv", raw_kv), ("raw_gate", raw_gate)):
        if tensor.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must be floating point, got {tensor.dtype}")
        if tensor.device != raw_kv.device:
            raise ValueError(
                f"{name} must be on device {raw_kv.device}, got {tensor.device}"
            )
    if state_valid.shape != (n, sequence_length):
        raise ValueError(
            f"state_valid must have shape {(n, sequence_length)}, "
            f"got {tuple(state_valid.shape)}"
        )
    if state_valid.dtype != torch.bool:
        raise TypeError(f"state_valid must have dtype bool, got {state_valid.dtype}")
    if state_valid.device != raw_kv.device:
        raise ValueError(
            f"state_valid must be on device {raw_kv.device}, got {state_valid.device}"
        )
    if type(compression_ratio) is not int or compression_ratio < 2:
        raise ValueError(
            f"compression_ratio must be an int of at least 2, got {compression_ratio!r}"
        )
    return n, sequence_length, head_dim


def reference_token_compressor_pool(
    raw_kv: Tensor,
    raw_gate: Tensor,
    state_valid: Tensor,
    *,
    compression_ratio: int,
) -> tuple[Tensor, Tensor]:
    """Pool previous/current branches with a channel-wise masked softmax.

    The result has shape ``[N, ceil(T / ratio), Dh]``. Float64 inputs retain
    float64 for numerical checking; all other floating inputs accumulate and
    return in float32. Invalid sources, including non-finite sentinel values,
    are removed before weighted arithmetic. All-invalid rows are exact zero.
    """
    _, sequence_length, _ = _validate_reference_inputs(
        raw_kv,
        raw_gate,
        state_valid,
        compression_ratio=compression_ratio,
    )
    accumulation_dtype = (
        torch.float64
        if raw_kv.dtype == torch.float64 or raw_gate.dtype == torch.float64
        else torch.float32
    )
    layout = build_token_compressor_layout(
        sequence_length, compression_ratio, raw_kv.device
    )
    gathered_kv = raw_kv[:, layout.source_indices, layout.source_branches, :].to(
        dtype=accumulation_dtype
    )
    gathered_gate = raw_gate[:, layout.source_indices, layout.source_branches, :].to(
        dtype=accumulation_dtype
    )
    source_valid = (
        layout.boundary_valid.unsqueeze(0) & state_valid[:, layout.source_indices]
    )
    channel_valid = source_valid.unsqueeze(-1)
    safe_kv = torch.where(channel_valid, gathered_kv, torch.zeros_like(gathered_kv))
    minimum = torch.finfo(accumulation_dtype).min
    masked_gate = gathered_gate.masked_fill(~channel_valid, minimum)
    maximum = masked_gate.amax(dim=2, keepdim=True)
    numerator = torch.exp(masked_gate - maximum)
    numerator = torch.where(channel_valid, numerator, torch.zeros_like(numerator))
    denominator = numerator.sum(dim=2, keepdim=True)
    weights = torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(torch.finfo(accumulation_dtype).tiny),
        torch.zeros_like(numerator),
    )
    pooled = (weights * safe_kv).sum(dim=2)
    pooled_valid = source_valid.any(dim=2)
    pooled = torch.where(pooled_valid.unsqueeze(-1), pooled, torch.zeros_like(pooled))
    return pooled, pooled_valid


__all__ = ["reference_token_compressor_pool"]
