"""Attention-mask preparation at the PLCS model-I/O boundary."""

from __future__ import annotations

from torch import Tensor

from src.utils.models import build_self_attn_mask


def prepare_axial_attention_masks(padding_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Prepare camera/time masks for every axial PLCS architecture."""
    batch_size, num_cameras, num_frames = padding_mask.shape
    token_valid = (~padding_mask).permute(0, 2, 1)
    camera_valid = token_valid.reshape(batch_size * num_frames, num_cameras)
    time_valid = token_valid.permute(0, 2, 1).reshape(
        batch_size * num_cameras,
        num_frames,
    )
    camera_mask, _ = build_self_attn_mask(camera_valid)
    time_mask, _ = build_self_attn_mask(time_valid)
    return camera_mask, time_mask


__all__ = [
    "prepare_axial_attention_masks",
]
