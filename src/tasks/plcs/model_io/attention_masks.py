"""Attention-mask preparation at the PLCS model-I/O boundary."""

from __future__ import annotations

import torch
from torch import Tensor

from src.utils.models import build_self_attn_mask


def prepare_axial_attention_masks(human_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Prepare camera/time masks for every axial PLCS architecture."""
    batch_size, num_cameras, num_frames = human_mask.shape
    token_valid = human_mask.bool().permute(0, 2, 1)
    camera_valid = token_valid.reshape(batch_size * num_frames, num_cameras)
    time_valid = token_valid.permute(0, 2, 1).reshape(
        batch_size * num_cameras,
        num_frames,
    )
    camera_mask, _ = build_self_attn_mask(camera_valid)
    time_mask, _ = build_self_attn_mask(time_valid)
    return camera_mask, time_mask


def prepare_tracking_attention_masks(
    *,
    detection_mask: Tensor,
    frame_mask: Tensor,
    view_mask: Tensor,
    num_queries: int,
    mask_invisible_observations: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Prepare tracking observation state and spatial/time attention masks."""
    batch_size, num_views, num_frames, num_detections = detection_mask.shape
    context_valid = (
        view_mask[:, :, None, None] & frame_mask[:, None, :, None]
    ).expand(-1, -1, -1, num_detections)
    camera_valid = (
        context_valid & detection_mask
        if mask_invisible_observations
        else context_valid
    ).permute(0, 2, 1, 3)
    slot_valid = frame_mask[:, :, None].expand(-1, -1, num_queries)
    spatial_valid = torch.cat(
        (
            slot_valid,
            camera_valid.reshape(batch_size, num_frames, num_views * num_detections),
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
    return camera_valid, spatial_mask, temporal_mask


__all__ = [
    "prepare_axial_attention_masks",
    "prepare_tracking_attention_masks",
]
