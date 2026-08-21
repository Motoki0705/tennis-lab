"""Padding-derived masks for fixed-query multi-view models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.utils.models.transformer_utils import build_self_attn_mask


@dataclass(frozen=True, slots=True)
class FixedQueryPaddingMasks:
    """Validity tensors and attention keep-masks derived from ``(B,V,T)`` padding."""

    context_valid: Tensor
    frame_valid: Tensor
    camera_state_valid: Tensor
    spatial_attention_keep_mask: Tensor
    object_temporal_state_valid: Tensor
    object_temporal_attention_keep_mask: Tensor
    query_temporal_state_valid: Tensor
    query_temporal_attention_keep_mask: Tensor


def build_fixed_query_padding_masks(
    padding_mask: Tensor,
    *,
    num_queries: int,
) -> FixedQueryPaddingMasks:
    """Build fixed-query model masks from a multi-view padding mask.

    Args:
        padding_mask: Boolean ``(B,V,T)`` mask where ``True`` marks padding.
            The mask may be nonrectangular: each view may pad different frames.
        num_queries: Fixed query/state width ``Q``.

    Returns:
        Raw validity tensors (``True=valid``) and dense attention keep-masks
        (``True=keep``). Dense masks use :func:`build_self_attn_mask`, including
        its token-0 repair for rows whose raw state is entirely invalid.
    """
    if not isinstance(padding_mask, Tensor):
        raise TypeError("padding_mask must be a torch.Tensor.")
    if padding_mask.dtype != torch.bool:
        raise TypeError(
            f"padding_mask must have dtype torch.bool, got {padding_mask.dtype}."
        )
    if padding_mask.ndim != 3:
        raise ValueError(
            "padding_mask must have shape (B,V,T), "
            f"got rank {padding_mask.ndim} and shape {tuple(padding_mask.shape)}."
        )
    if any(axis_size == 0 for axis_size in padding_mask.shape):
        raise ValueError(
            "padding_mask axes B, V, and T must all be nonempty, "
            f"got shape {tuple(padding_mask.shape)}."
        )
    if type(num_queries) is not int:
        raise TypeError("num_queries must be exactly int.")
    if num_queries <= 0:
        raise ValueError(f"num_queries must be positive, got {num_queries}.")

    batch_size, num_views, num_frames = padding_mask.shape
    context_valid = ~padding_mask
    frame_valid = context_valid.any(dim=1)
    camera_state_valid = context_valid.unsqueeze(-1).expand(
        batch_size,
        num_views,
        num_frames,
        num_queries,
    )

    query_spatial_valid = frame_valid.unsqueeze(-1).expand(
        batch_size,
        num_frames,
        num_queries,
    )
    camera_spatial_valid = camera_state_valid.permute(0, 2, 1, 3).reshape(
        batch_size,
        num_frames,
        num_views * num_queries,
    )
    spatial_valid = torch.cat(
        (query_spatial_valid, camera_spatial_valid),
        dim=-1,
    ).flatten(0, 1)
    spatial_attention_keep_mask, _ = build_self_attn_mask(spatial_valid)

    object_temporal_state_valid = context_valid.reshape(
        batch_size * num_views,
        num_frames,
    )
    object_temporal_attention_keep_mask, _ = build_self_attn_mask(
        object_temporal_state_valid
    )

    query_temporal_state_valid = (
        frame_valid[:, None, :]
        .expand(batch_size, num_queries, num_frames)
        .reshape(batch_size * num_queries, num_frames)
    )
    query_temporal_attention_keep_mask, _ = build_self_attn_mask(
        query_temporal_state_valid
    )

    return FixedQueryPaddingMasks(
        context_valid=context_valid,
        frame_valid=frame_valid,
        camera_state_valid=camera_state_valid,
        spatial_attention_keep_mask=spatial_attention_keep_mask,
        object_temporal_state_valid=object_temporal_state_valid,
        object_temporal_attention_keep_mask=object_temporal_attention_keep_mask,
        query_temporal_state_valid=query_temporal_state_valid,
        query_temporal_attention_keep_mask=query_temporal_attention_keep_mask,
    )


__all__ = [
    "FixedQueryPaddingMasks",
    "build_fixed_query_padding_masks",
]
