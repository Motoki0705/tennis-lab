"""Derive BLCS-internal validity and attention masks from public padding masks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from src.utils.models import build_self_attn_mask


@dataclass(frozen=True, slots=True)
class SingleViewPaddingMasks:
    """Single-view masks derived from ``padding_mask[B,T]``."""

    frame_valid: Tensor
    attention_keep_mask: Tensor


@dataclass(frozen=True, slots=True)
class MultiViewPaddingMasks:
    """Iterative multiview masks derived from ``padding_mask[B,V,T]``."""

    context_valid: Tensor
    frame_valid: Tensor
    query_attention_keep_mask: Tensor
    cross_attention_keep_mask: Tensor
    frame_token_valid: Tensor


@dataclass(frozen=True, slots=True)
class AxialPaddingMasks:
    """Axial multiview masks derived from ``padding_mask[B,V,T]``."""

    context_valid: Tensor
    frame_valid: Tensor
    camera_attention_keep_mask: Tensor
    time_attention_keep_mask: Tensor
    sliding_attention_keep_mask: Tensor


def _validate_padding_mask(padding_mask: Tensor, *, expected_rank: int) -> None:
    if not isinstance(padding_mask, Tensor):
        raise TypeError("padding_mask must be a torch.Tensor.")
    if padding_mask.dtype != torch.bool:
        raise TypeError(
            f"padding_mask must have dtype torch.bool, got {padding_mask.dtype}."
        )
    if padding_mask.ndim != expected_rank:
        raise ValueError(
            f"padding_mask must have rank {expected_rank}, "
            f"got shape {tuple(padding_mask.shape)}."
        )
    if any(axis_size == 0 for axis_size in padding_mask.shape):
        raise ValueError(
            "padding_mask axes must all be nonempty, "
            f"got shape {tuple(padding_mask.shape)}."
        )


def _validate_num_court_tokens(num_court_tokens: int) -> None:
    if type(num_court_tokens) is not int:
        raise TypeError("num_court_tokens must be exactly int.")
    if num_court_tokens <= 0:
        raise ValueError("num_court_tokens must be positive.")


def _build_local_attention_keep_mask(
    valid_mask: Tensor,
    *,
    window_radius: int,
) -> Tensor:
    """Restrict valid temporal keys to a symmetric local window."""
    seq_len = valid_mask.shape[1]
    positions = torch.arange(seq_len, device=valid_mask.device)
    local_keep = (positions[:, None] - positions[None, :]).abs() <= window_radius
    keep_mask = valid_mask[:, None, :] & local_keep.unsqueeze(0)
    fallback_keep = valid_mask[:, None, :].expand_as(keep_mask)
    has_key = keep_mask.any(dim=-1, keepdim=True)
    return torch.where(has_key, keep_mask, fallback_keep)


def build_single_view_padding_masks(
    padding_mask: Tensor,
    *,
    num_court_tokens: int,
) -> SingleViewPaddingMasks:
    """Build single-view attention keep masks from ``True=padding`` input."""
    _validate_padding_mask(padding_mask, expected_rank=2)
    _validate_num_court_tokens(num_court_tokens)
    frame_valid = ~padding_mask
    sample_valid = frame_valid.any(dim=1, keepdim=True)
    court_valid = sample_valid.expand(-1, num_court_tokens)
    token_valid = torch.cat((court_valid, frame_valid), dim=1)
    attention_keep_mask, _ = build_self_attn_mask(token_valid)
    return SingleViewPaddingMasks(
        frame_valid=frame_valid,
        attention_keep_mask=attention_keep_mask,
    )


def build_multiview_padding_masks(
    padding_mask: Tensor,
    *,
    num_court_tokens: int,
) -> MultiViewPaddingMasks:
    """Build iterative multiview masks from ``True=padding`` input."""
    _validate_padding_mask(padding_mask, expected_rank=3)
    _validate_num_court_tokens(num_court_tokens)
    batch_size, num_views, num_frames = padding_mask.shape
    context_valid = ~padding_mask
    frame_valid = context_valid.any(dim=1)
    query_attention_keep_mask, _ = build_self_attn_mask(frame_valid)

    court_valid = context_valid.unsqueeze(-1).expand(
        batch_size,
        num_views,
        num_frames,
        num_court_tokens,
    )
    per_context_valid = torch.cat(
        (court_valid, context_valid.unsqueeze(-1)),
        dim=-1,
    )
    frame_token_valid = per_context_valid.permute(0, 2, 1, 3).reshape(
        batch_size,
        num_frames,
        num_views * (num_court_tokens + 1),
    )
    _, repaired_frame_token_valid = build_self_attn_mask(
        frame_token_valid.flatten(0, 1)
    )
    cross_attention_keep_mask = repaired_frame_token_valid[:, None, :]
    return MultiViewPaddingMasks(
        context_valid=context_valid,
        frame_valid=frame_valid,
        query_attention_keep_mask=query_attention_keep_mask,
        cross_attention_keep_mask=cross_attention_keep_mask,
        frame_token_valid=frame_token_valid,
    )


def build_axial_padding_masks(
    padding_mask: Tensor,
    *,
    time_window_radius: int,
) -> AxialPaddingMasks:
    """Build axial camera/time attention masks from ``True=padding`` input."""
    _validate_padding_mask(padding_mask, expected_rank=3)
    if type(time_window_radius) is not int:
        raise TypeError("time_window_radius must be exactly int.")
    if time_window_radius < 0:
        raise ValueError("time_window_radius must be non-negative.")
    batch_size, num_views, num_frames = padding_mask.shape
    context_valid = ~padding_mask
    frame_valid = context_valid.any(dim=1)
    camera_valid = context_valid.permute(0, 2, 1).reshape(
        batch_size * num_frames,
        num_views,
    )
    time_valid = context_valid.reshape(batch_size * num_views, num_frames)
    camera_attention_keep_mask, _ = build_self_attn_mask(camera_valid)
    time_attention_keep_mask, repaired_time_valid = build_self_attn_mask(time_valid)
    sliding_attention_keep_mask = _build_local_attention_keep_mask(
        repaired_time_valid,
        window_radius=time_window_radius,
    )
    return AxialPaddingMasks(
        context_valid=context_valid,
        frame_valid=frame_valid,
        camera_attention_keep_mask=camera_attention_keep_mask,
        time_attention_keep_mask=time_attention_keep_mask,
        sliding_attention_keep_mask=sliding_attention_keep_mask,
    )


def mask_trajectory_outputs(
    outputs: Mapping[str, Tensor],
    frame_valid: Tensor,
) -> dict[str, Tensor]:
    """Zero padded ``(B,T,...)`` trajectory outputs after repaired attention."""
    masked: dict[str, Tensor] = {}
    for name, value in outputs.items():
        if value.shape[:2] != frame_valid.shape:
            raise ValueError(
                f"Trajectory output {name!r} must start with "
                f"{tuple(frame_valid.shape)}, got {tuple(value.shape)}."
            )
        trailing_axes = (1,) * (value.ndim - 2)
        masked[name] = value * frame_valid.reshape(*frame_valid.shape, *trailing_axes)
    return masked


__all__ = [
    "AxialPaddingMasks",
    "build_axial_padding_masks",
    "build_multiview_padding_masks",
    "build_single_view_padding_masks",
    "mask_trajectory_outputs",
    "MultiViewPaddingMasks",
    "SingleViewPaddingMasks",
]
