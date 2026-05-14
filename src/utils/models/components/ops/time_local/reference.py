from __future__ import annotations

import torch
import torch.nn.functional as F

from src.utils.models.components.ops.time_local.layout import normalize_valid_mask


def build_local_attention_keep_mask(
    valid_mask: torch.Tensor,
    window_radius: int,
) -> torch.Tensor:
    valid_fixed = _normalize_valid_mask(valid_mask)
    seq_len = valid_fixed.shape[1]
    positions = torch.arange(seq_len, device=valid_fixed.device)

    if window_radius < 0:
        raise ValueError(f"window_radius must be non-negative, got {window_radius}")

    local_keep = (positions[:, None] - positions[None, :]).abs() <= window_radius
    keep_mask = valid_fixed[:, None, :] & local_keep.unsqueeze(0)
    fallback_keep = valid_fixed[:, None, :].expand_as(keep_mask)
    return _ensure_rows_have_keys(keep_mask, fallback_keep)


def reference_time_local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    window_radius: int,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, value must have shape [B, H, T, D]")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, value must have the same shape")

    keep_mask = build_local_attention_keep_mask(valid_mask, window_radius)
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=keep_mask[:, None, :, :],
        dropout_p=dropout_p if training else 0.0,
        is_causal=False,
    )


def _normalize_valid_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    return normalize_valid_mask(valid_mask)


def _ensure_rows_have_keys(attn_mask: torch.Tensor, fallback_keep: torch.Tensor) -> torch.Tensor:
    empty_rows = ~attn_mask.any(dim=-1)
    if empty_rows.any():
        attn_mask = attn_mask.clone()
        attn_mask[empty_rows] = fallback_keep[empty_rows]
    return attn_mask