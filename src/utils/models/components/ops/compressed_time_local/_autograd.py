"""Autograd boundary for fused compressed time-local CUDA attention."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import Tensor

from src.utils.models.components.ops.loader import (
    require_compressed_time_local_cuda_extension,
)

_EXTENSION = require_compressed_time_local_cuda_extension()
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_MAX_WINDOW_RADIUS = 64


def _validate_inputs(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    query_valid: Tensor,
    key_valid: Tensor,
    compression_ratio: int,
    window_radius: int,
    dropout_p: float,
    training: bool,
) -> None:
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must have shape [N, H, T, Dh], got {tuple(tensor.shape)}"
            )
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(
                f"compressed time-local CUDA supports float16, bfloat16, and "
                f"float32; {name} has {tensor.dtype}"
            )
    n, heads, query_len, head_dim = query.shape
    key_n, key_heads, key_len, key_head_dim = key.shape
    if value.shape != key.shape:
        raise ValueError("value shape must equal key shape")
    if (key_n, key_heads, key_head_dim) != (n, heads, head_dim):
        raise ValueError(
            "query and key must have the same batch, head, and head dimensions"
        )
    if n <= 0 or heads <= 0 or query_len <= 0 or key_len <= 0 or head_dim <= 0:
        raise ValueError("query and key dimensions must all be positive")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise TypeError("query, key, and value must have the same dtype")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same CUDA device")
    if query_valid.shape != (n, query_len):
        raise ValueError(
            f"query_valid must have shape {(n, query_len)}, "
            f"got {tuple(query_valid.shape)}"
        )
    if key_valid.shape != (n, key_len):
        raise ValueError(
            f"key_valid must have shape {(n, key_len)}, got {tuple(key_valid.shape)}"
        )
    for name, mask in (("query_valid", query_valid), ("key_valid", key_valid)):
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must have dtype bool, got {mask.dtype}")
        if not mask.is_cuda or mask.device != query.device:
            raise ValueError(f"{name} must be on device {query.device}")
    if type(compression_ratio) is not int or compression_ratio < 2:
        raise ValueError(
            f"compression_ratio must be an int of at least 2, got {compression_ratio!r}"
        )
    expected_key_len = (query_len + compression_ratio - 1) // compression_ratio
    if key_len != expected_key_len:
        raise ValueError(
            "key length must equal ceil(query length / compression_ratio): "
            f"expected {expected_key_len}, got {key_len}"
        )
    if type(window_radius) is not int or not 0 <= window_radius <= _MAX_WINDOW_RADIUS:
        raise ValueError(
            f"window_radius must be an int in [0, {_MAX_WINDOW_RADIUS}], "
            f"got {window_radius!r}"
        )
    if isinstance(dropout_p, bool) or not isinstance(dropout_p, (int, float)):
        raise TypeError(f"dropout_p must be a real number, got {dropout_p!r}")
    if not math.isfinite(float(dropout_p)) or not 0.0 <= float(dropout_p) < 1.0:
        raise ValueError(f"dropout_p must be finite and in [0, 1), got {dropout_p!r}")
    if type(training) is not bool:
        raise TypeError(f"training must be a bool, got {training!r}")
    if training and float(dropout_p) != 0.0:
        raise RuntimeError(
            "compressed time-local CUDA does not support attention dropout; "
            "set attn_dropout=0 or select backend='reference'"
        )


class _CompressedTimeLocalAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_valid: Tensor,
        key_valid: Tensor,
        compression_ratio: int,
        window_radius: int,
    ) -> tuple[Tensor, Tensor]:
        contiguous_query = query.contiguous()
        contiguous_key = key.contiguous()
        contiguous_value = value.contiguous()
        contiguous_query_valid = query_valid.contiguous()
        contiguous_key_valid = key_valid.contiguous()
        output, logsumexp, invalid_row = _EXTENSION.forward(
            contiguous_query,
            contiguous_key,
            contiguous_value,
            contiguous_query_valid,
            contiguous_key_valid,
            compression_ratio,
            window_radius,
        )
        ctx.save_for_backward(
            contiguous_query,
            contiguous_key,
            contiguous_value,
            contiguous_query_valid,
            contiguous_key_valid,
            output,
            logsumexp,
        )
        ctx.mark_non_differentiable(invalid_row)
        ctx.compression_ratio = compression_ratio
        ctx.window_radius = window_radius
        return cast(Tensor, output), cast(Tensor, invalid_row)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
        _grad_invalid_row: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, None, None, None, None]:
        (
            query,
            key,
            value,
            query_valid,
            key_valid,
            output,
            logsumexp,
        ) = ctx.saved_tensors
        grad_query, grad_key, grad_value = _EXTENSION.backward(
            grad_output.contiguous(),
            query,
            key,
            value,
            query_valid,
            key_valid,
            output,
            logsumexp,
            ctx.compression_ratio,
            ctx.window_radius,
        )
        return (
            cast(Tensor, grad_query),
            cast(Tensor, grad_key),
            cast(Tensor, grad_value),
            None,
            None,
            None,
            None,
        )


def cuda_compressed_time_local_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    query_valid: Tensor,
    key_valid: Tensor,
    compression_ratio: int,
    window_radius: int,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tensor:
    """Run fused online-softmax CUDA attention without gathered K/V windows."""
    _validate_inputs(
        query,
        key,
        value,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=compression_ratio,
        window_radius=window_radius,
        dropout_p=dropout_p,
        training=training,
    )
    raw_result = _CompressedTimeLocalAttention.apply(
        query,
        key,
        value,
        query_valid,
        key_valid,
        compression_ratio,
        window_radius,
    )
    result, invalid_row = cast(tuple[Tensor, Tensor], raw_result)
    if int(invalid_row.item()) != 0:
        raise RuntimeError(
            "valid query has no valid compressed key in its local window"
        )
    return result


__all__ = ["cuda_compressed_time_local_attention"]
