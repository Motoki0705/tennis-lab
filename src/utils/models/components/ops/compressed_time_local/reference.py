"""Gather-based PyTorch reference for compressed time-local attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from src.utils.models.components.ops.compressed_time_local.layout import (
    build_compressed_sliding_window_layout,
)

_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _validate_reference_inputs(
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
) -> tuple[int, int, int, int, int]:
    tensors = {"query": query, "key": key, "value": value}
    for name, tensor in tensors.items():
        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must have shape [N, H, T, Dh], got {tuple(tensor.shape)}"
            )
        if tensor.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must be floating point, got {tensor.dtype}")

    n, heads, query_len, head_dim = query.shape
    key_n, key_heads, key_len, key_head_dim = key.shape
    if value.shape != key.shape:
        raise ValueError(
            "value shape must equal key shape, got "
            f"{tuple(value.shape)} and {tuple(key.shape)}"
        )
    if key_n != n or key_head_dim != head_dim or key_heads not in (1, heads):
        raise ValueError(
            "query and key must have the same batch and head dimensions, and "
            "key heads must be 1 or equal query heads"
        )
    if query_len <= 0 or key_len <= 0 or n <= 0 or heads <= 0 or head_dim <= 0:
        raise ValueError("query and key dimensions must all be positive")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise TypeError("query, key, and value must have the same dtype")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device")

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
        if mask.device != query.device:
            raise ValueError(
                f"{name} must be on device {query.device}, got {mask.device}"
            )

    if type(compression_ratio) is not int or compression_ratio < 2:
        raise ValueError(
            f"compression_ratio must be an int of at least 2, got {compression_ratio!r}"
        )
    if type(window_radius) is not int or window_radius < 0:
        raise ValueError(
            f"window_radius must be a non-negative int, got {window_radius!r}"
        )
    expected_key_len = (query_len + compression_ratio - 1) // compression_ratio
    if key_len != expected_key_len:
        raise ValueError(
            "key length must equal ceil(query length / compression_ratio): "
            f"expected {expected_key_len}, got {key_len}"
        )
    if isinstance(dropout_p, bool) or not isinstance(dropout_p, (int, float)):
        raise TypeError(f"dropout_p must be a real number, got {dropout_p!r}")
    if not math.isfinite(float(dropout_p)) or not 0.0 <= float(dropout_p) < 1.0:
        raise ValueError(f"dropout_p must be finite and in [0, 1), got {dropout_p!r}")
    if type(training) is not bool:
        raise TypeError(f"training must be a bool, got {training!r}")
    return n, heads, query_len, key_len, head_dim


def reference_compressed_time_local_attention(
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
    """Attend each query only to a gathered window of compressed K/V.

    Query has shape ``[N,H,T,Dh]`` while key and value have shape
    ``[N,Hkv,Tc,Dh]`` where ``Hkv`` is either ``1`` (multi-query attention) or
    ``H``, and ``Tc=ceil(T/compression_ratio)``.  The function materializes
    only gathered KV windows and relies on SDPA broadcasting when ``Hkv=1``;
    it never copies KV across heads or creates a dense ``[T,Tc]`` score/mask.

    Invalid queries return exactly zero.  A valid query with no valid key in
    its compressed window violates the caller/compressor contract and raises.
    """
    n, _, query_len, key_len, _ = _validate_reference_inputs(
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
    indices, index_valid = build_compressed_sliding_window_layout(
        query_len=query_len,
        key_len=key_len,
        compression_ratio=compression_ratio,
        window_radius=window_radius,
        device=query.device,
    )

    # Zero invalid values before score/value computation.  This keeps padding
    # values (including non-finite sentinels) from leaking through masked math.
    safe_query = torch.where(
        query_valid[:, None, :, None], query, torch.zeros_like(query)
    )
    safe_key = torch.where(key_valid[:, None, :, None], key, torch.zeros_like(key))
    safe_value = torch.where(
        key_valid[:, None, :, None], value, torch.zeros_like(value)
    )
    gathered_key = safe_key[:, :, indices, :]
    gathered_value = safe_value[:, :, indices, :]

    window_keep = (
        index_valid.unsqueeze(0) & key_valid[:, indices] & query_valid[:, :, None]
    )
    valid_without_key = query_valid & ~window_keep.any(dim=-1)
    if bool(valid_without_key.any()):
        bad_rows = valid_without_key.nonzero(as_tuple=False).tolist()
        raise RuntimeError(
            "valid query has no valid compressed key in its local window; "
            f"rows={bad_rows}"
        )

    # SDPA must not receive an empty row.  Invalid queries use one safe,
    # already-zeroed gathered state and are zeroed again after attention.
    safe_window_keep = window_keep.clone()
    empty_rows = ~safe_window_keep.any(dim=-1)
    safe_window_keep[..., 0] |= empty_rows

    output = F.scaled_dot_product_attention(
        safe_query.unsqueeze(-2),
        gathered_key,
        gathered_value,
        attn_mask=safe_window_keep[:, None, :, None, :],
        dropout_p=float(dropout_p) if training else 0.0,
        is_causal=False,
    ).squeeze(-2)
    return torch.where(query_valid[:, None, :, None], output, torch.zeros_like(output))
