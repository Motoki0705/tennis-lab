"""Attention-mask preparation at the BLCS model-I/O boundary."""

from __future__ import annotations

import torch
from torch import Tensor

from src.utils.models import build_self_attn_mask
from src.utils.models.components.ops.time_local import (
    build_local_attention_keep_mask,
)


def prepare_single_attention_mask(
    padding_mask: Tensor,
    *,
    num_court_tokens: int,
) -> Tensor:
    """Prepare the single-view court/ball self-attention mask."""
    court_valid = torch.ones(
        padding_mask.shape[0],
        num_court_tokens,
        device=padding_mask.device,
        dtype=torch.bool,
    )
    token_valid = torch.cat((court_valid, ~padding_mask), dim=1)
    attention_mask, _ = build_self_attn_mask(token_valid)
    return attention_mask


def prepare_multiview_attention_masks(
    padding_mask: Tensor,
    *,
    num_court_tokens: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Prepare query self-attention state and per-frame cross-attention masks."""
    batch_size, num_cameras, num_frames = padding_mask.shape
    ball_valid = ~padding_mask
    query_valid = ball_valid.any(dim=1)
    query_mask, query_state_valid = build_self_attn_mask(query_valid)

    court_valid = ball_valid.any(dim=2)
    court_valid_expanded = court_valid[:, :, None, None].expand(
        batch_size,
        num_cameras,
        num_frames,
        num_court_tokens,
    )
    per_camera_valid = torch.cat(
        (court_valid_expanded, ball_valid.unsqueeze(-1)),
        dim=3,
    )
    frame_valid = per_camera_valid.permute(0, 2, 1, 3).reshape(
        batch_size * num_frames,
        num_cameras * (num_court_tokens + 1),
    )
    _, frame_state_valid = build_self_attn_mask(frame_valid)
    cross_attention_mask = frame_state_valid[:, None, :]
    frame_token_valid = frame_valid.reshape(
        batch_size,
        num_frames,
        num_cameras * (num_court_tokens + 1),
    )
    return (
        query_mask,
        query_state_valid,
        cross_attention_mask,
        frame_token_valid,
    )


def prepare_axial_attention_masks(
    padding_mask: Tensor,
    *,
    time_window_radius: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Prepare camera, global-time, and local-time axial masks."""
    batch_size, num_cameras, num_frames = padding_mask.shape
    token_valid = (~padding_mask).permute(0, 2, 1)
    camera_valid = token_valid.reshape(batch_size * num_frames, num_cameras)
    time_valid = token_valid.permute(0, 2, 1).reshape(
        batch_size * num_cameras,
        num_frames,
    )
    camera_mask, _ = build_self_attn_mask(camera_valid)
    time_mask, time_state_valid = build_self_attn_mask(time_valid)
    sliding_mask = build_local_attention_keep_mask(
        time_state_valid,
        time_window_radius,
    )
    return camera_mask, time_mask, sliding_mask


__all__ = [
    "prepare_axial_attention_masks",
    "prepare_multiview_attention_masks",
    "prepare_single_attention_mask",
]
