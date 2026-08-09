"""Attention-mask preparation at the BLCS model-I/O boundary."""

from __future__ import annotations

import torch
from torch import Tensor

from src.utils.models import build_self_attn_mask
from src.utils.models.components.ops.time_local import (
    build_local_attention_keep_mask,
)


def prepare_single_attention_mask(
    ball_mask: Tensor,
    *,
    num_court_tokens: int,
) -> Tensor:
    """Prepare the single-view court/ball self-attention mask."""
    court_valid = torch.ones(
        ball_mask.shape[0],
        num_court_tokens,
        device=ball_mask.device,
        dtype=torch.bool,
    )
    token_valid = torch.cat((court_valid, ball_mask.bool()), dim=1)
    attention_mask, _ = build_self_attn_mask(token_valid)
    return attention_mask


def prepare_multiview_attention_masks(
    ball_mask: Tensor,
    *,
    num_court_tokens: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Prepare query self-attention state and per-frame cross-attention masks."""
    batch_size, num_cameras, num_frames = ball_mask.shape
    ball_valid = ball_mask.bool()
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
    ball_mask: Tensor,
    *,
    time_window_radius: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Prepare camera, global-time, and local-time axial masks."""
    batch_size, num_cameras, num_frames = ball_mask.shape
    token_valid = ball_mask.bool().permute(0, 2, 1)
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


def prepare_tracking_attention_masks(
    *,
    ball_visible: Tensor,
    court_visible: Tensor,
    frame_mask: Tensor,
    view_mask: Tensor,
    num_queries: int,
    mask_invisible_observations: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Prepare track-query state, spatial/time, and point-fusion masks."""
    batch_size, num_views, num_frames, num_detections = ball_visible.shape
    context_valid = (
        view_mask[:, :, None, None] & frame_mask[:, None, :, None]
    ).expand(-1, -1, -1, num_detections)
    observation_valid = (
        context_valid & ball_visible
        if mask_invisible_observations
        else context_valid
    )
    observation_state_valid = observation_valid.permute(0, 2, 1, 3)

    slot_valid = frame_mask[:, :, None].expand(-1, -1, num_queries)
    spatial_valid = torch.cat(
        (
            slot_valid,
            observation_state_valid.reshape(batch_size, num_frames, -1),
        ),
        dim=2,
    ).flatten(0, 1)
    spatial_mask, _ = build_self_attn_mask(spatial_valid)
    temporal_valid = (
        frame_mask[:, None, :]
        .expand(-1, num_queries, -1)
        .reshape(batch_size * num_queries, num_frames)
    )
    temporal_mask, _ = build_self_attn_mask(temporal_valid)

    _, point_mask = prepare_point_attention_mask(
        ball_visible=ball_visible,
        court_visible=court_visible,
        context_valid=context_valid[..., 0],
        mask_invisible_observations=mask_invisible_observations,
    )
    return observation_state_valid, spatial_mask, temporal_mask, point_mask


def prepare_point_attention_mask(
    *,
    ball_visible: Tensor,
    court_visible: Tensor,
    context_valid: Tensor,
    mask_invisible_observations: bool,
) -> tuple[Tensor, Tensor]:
    """Prepare point-fusion state validity and its complete attention mask."""
    ball_context_valid = context_valid.unsqueeze(-1).expand_as(ball_visible)
    ball_state_valid = (
        ball_context_valid & ball_visible
        if mask_invisible_observations
        else ball_context_valid
    )
    court_key_valid = court_visible & context_valid.unsqueeze(-1)
    point_valid = torch.cat((court_key_valid, ball_state_valid), dim=-1)
    point_mask, _ = build_self_attn_mask(
        point_valid.reshape(-1, point_valid.shape[-1])
    )
    return ball_state_valid, point_mask


__all__ = [
    "prepare_axial_attention_masks",
    "prepare_multiview_attention_masks",
    "prepare_point_attention_mask",
    "prepare_single_attention_mask",
    "prepare_tracking_attention_masks",
]
