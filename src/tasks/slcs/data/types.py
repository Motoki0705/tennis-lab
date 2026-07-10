"""Type definitions for SLCS data structures.

Sample / batch schemas for the monocular temporal multitask model. Shapes use
the following symbols:

- ``P``: player slots (fixed by config, canonical order: near side first)
- ``T``: window size in frames (fixed by config; padded slots masked)
- ``K``: court keypoints per frame
- ``T_d``: DINOv3 token samples inside the window (variable; padded per batch)
- ``S``: DINOv3 patch tokens per sampled frame (``grid_h * grid_w``)
- ``C``: DINOv3 embedding width

2D coordinates are normalized image UV in ``[0, 1]``; 3D positions are
normalized court coordinates (meters divided by
``src.utils.schema.court.COURT_COORD_SCALE_XYZ``); rotations are ``(cos, sin)``
of court-frame yaw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch


class SLCSSample(TypedDict):
    """One dataset item: a single-camera temporal window of one clip."""

    # Observations
    player_kp: torch.Tensor  # (P, T, J, 2) float32
    player_kp_vis: torch.Tensor  # (P, T, J) float32
    player_valid: torch.Tensor  # (P, T) bool — per-frame observation validity
    ball_uv: torch.Tensor  # (T, 2) float32
    ball_vis: torch.Tensor  # (T,) bool
    court_kp: torch.Tensor  # (T, K, 2) float32
    court_vis: torch.Tensor  # (T, K) float32
    dino_tokens: torch.Tensor  # (T_d, S, C) float32
    dino_frame_idx: torch.Tensor  # (T_d,) int64 — window-relative frame index
    dino_valid: torch.Tensor  # (T_d,) bool
    # Timeline
    frame_idx: torch.Tensor  # (T,) int64 — absolute clip frame index
    timestamp: torch.Tensor  # (T,) float32 seconds
    frame_mask: torch.Tensor  # (T,) bool — window padding mask (True=real frame)
    # Targets (pseudo-labels)
    target_player_position: torch.Tensor  # (P, T, 3) float32, normalized
    target_player_rotation: torch.Tensor  # (P, T, 2) float32, (cos, sin)
    target_player_valid: torch.Tensor  # (P, T) bool
    target_player_weight: torch.Tensor  # (P, T) float32
    target_ball_position: torch.Tensor  # (T, 3) float32, normalized
    target_ball_valid: torch.Tensor  # (T,) bool
    target_ball_weight: torch.Tensor  # (T,) float32


class SLCSBatch(TypedDict):
    """Collated batch: sample tensors with a leading batch axis.

    ``dino_tokens`` / ``dino_frame_idx`` / ``dino_valid`` are right-padded to
    the largest ``T_d`` in the batch (padding marked invalid).
    """

    player_kp: torch.Tensor  # (B, P, T, J, 2)
    player_kp_vis: torch.Tensor  # (B, P, T, J)
    player_valid: torch.Tensor  # (B, P, T)
    ball_uv: torch.Tensor  # (B, T, 2)
    ball_vis: torch.Tensor  # (B, T)
    court_kp: torch.Tensor  # (B, T, K, 2)
    court_vis: torch.Tensor  # (B, T, K)
    dino_tokens: torch.Tensor  # (B, T_d, S, C)
    dino_frame_idx: torch.Tensor  # (B, T_d)
    dino_valid: torch.Tensor  # (B, T_d)
    frame_idx: torch.Tensor  # (B, T)
    timestamp: torch.Tensor  # (B, T)
    frame_mask: torch.Tensor  # (B, T)
    target_player_position: torch.Tensor  # (B, P, T, 3)
    target_player_rotation: torch.Tensor  # (B, P, T, 2)
    target_player_valid: torch.Tensor  # (B, P, T)
    target_player_weight: torch.Tensor  # (B, P, T)
    target_ball_position: torch.Tensor  # (B, T, 3)
    target_ball_valid: torch.Tensor  # (B, T)
    target_ball_weight: torch.Tensor  # (B, T)


@dataclass(frozen=True)
class SLCSWindowMeta:
    """Provenance of one dataset item (kept out of the tensor batch)."""

    clip_id: str
    recording_id: str
    camera_id: str
    window_start: int
    window_length: int


__all__ = ["SLCSBatch", "SLCSSample", "SLCSWindowMeta"]
