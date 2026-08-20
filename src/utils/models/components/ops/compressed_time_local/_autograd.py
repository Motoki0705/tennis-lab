"""Autograd boundary for fused compressed time-local CUDA attention."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor

from src.utils.models.components.ops.compressed_time_local._dispatcher import (
    compressed_time_local_forward,
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_MAX_WINDOW_RADIUS = 64
_INVALID_ROW_MESSAGE = "valid query has no valid compressed key in its local window"


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
    if (key_n, key_head_dim) != (n, head_dim) or key_heads not in (1, heads):
        raise ValueError(
            "query and key must have the same batch and head dimensions, and "
            "key heads must be 1 or equal query heads"
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


def _normalize_rope_phasors(
    freqs_cis: Tensor,
    *,
    name: str,
    batch_size: int,
    heads: int,
    sequence_length: int,
    head_dim: int,
    device: torch.device,
) -> Tensor:
    """Return a rank-five float32 real view without expanding broadcast axes."""
    if freqs_cis.requires_grad:
        raise ValueError(f"{name} must not require gradients for fused RoPE attention")
    if freqs_cis.dtype != torch.complex64:
        raise TypeError(f"{name} must have dtype complex64, got {freqs_cis.dtype}")
    if freqs_cis.device != device:
        raise ValueError(f"{name} must be on device {device}, got {freqs_cis.device}")
    if freqs_cis.ndim not in (3, 4):
        raise ValueError(
            f"{name} must have rank 3 or 4, got shape {tuple(freqs_cis.shape)}"
        )

    normalized = freqs_cis.unsqueeze(0) if freqs_cis.ndim == 3 else freqs_cis
    expected_pairs = head_dim // 2
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for fused RoPE, got {head_dim}")
    if normalized.shape[0] not in (1, batch_size):
        raise ValueError(
            f"{name} batch dimension must be 1 or {batch_size}, "
            f"got {normalized.shape[0]}"
        )
    if normalized.shape[1] != sequence_length:
        raise ValueError(
            f"{name} sequence dimension must be {sequence_length}, "
            f"got {normalized.shape[1]}"
        )
    if normalized.shape[2] not in (1, heads):
        raise ValueError(
            f"{name} head dimension must be 1 or {heads}, got {normalized.shape[2]}"
        )
    if normalized.shape[3] != expected_pairs:
        raise ValueError(
            f"{name} pair dimension must be {expected_pairs}, got {normalized.shape[3]}"
        )
    return torch.view_as_real(normalized)


def _normalize_rope_pair(
    query: Tensor,
    key: Tensor,
    query_freqs_cis: Tensor | None,
    key_freqs_cis: Tensor | None,
) -> tuple[Tensor | None, Tensor | None]:
    if (query_freqs_cis is None) != (key_freqs_cis is None):
        raise ValueError(
            "query_freqs_cis and key_freqs_cis must either both be provided or "
            "both be omitted"
        )
    if query_freqs_cis is None or key_freqs_cis is None:
        return None, None
    query_phasors_real = _normalize_rope_phasors(
        query_freqs_cis,
        name="query_freqs_cis",
        batch_size=query.shape[0],
        heads=query.shape[1],
        sequence_length=query.shape[2],
        head_dim=query.shape[3],
        device=query.device,
    )
    key_phasors_real = _normalize_rope_phasors(
        key_freqs_cis,
        name="key_freqs_cis",
        batch_size=key.shape[0],
        heads=key.shape[1],
        sequence_length=key.shape[2],
        head_dim=key.shape[3],
        device=key.device,
    )
    return query_phasors_real, key_phasors_real


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
    query_freqs_cis: Tensor | None = None,
    key_freqs_cis: Tensor | None = None,
) -> Tensor:
    """Run fused CUDA attention, optionally rotating full-head Q/K in-kernel."""
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
    query_phasors_real, key_phasors_real = _normalize_rope_pair(
        query,
        key,
        query_freqs_cis,
        key_freqs_cis,
    )
    # The CUDA boundary reads the supported NHTD query through explicit strides.
    # K/V and masks retain the simpler contiguous kernel contract.
    contiguous_key = key.contiguous()
    contiguous_value = value.contiguous()
    contiguous_query_valid = query_valid.contiguous()
    contiguous_key_valid = key_valid.contiguous()
    raw_result = compressed_time_local_forward(
        query,
        contiguous_key,
        contiguous_value,
        contiguous_query_valid,
        contiguous_key_valid,
        query_phasors_real,
        key_phasors_real,
        compression_ratio,
        window_radius,
    )
    result, _logsumexp, invalid_row = cast(tuple[Tensor, Tensor, Tensor], raw_result)
    torch._assert_async(invalid_row == 0, _INVALID_ROW_MESSAGE)
    return result


__all__ = ["cuda_compressed_time_local_attention"]
