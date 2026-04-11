"""Type definitions for trajectory completion data."""

from __future__ import annotations

from typing import TypedDict

import torch


class TrajectoryCompletionSample(TypedDict):
    """Schema for a single trajectory completion sample."""

    ball_uv: torch.Tensor  # (T, 2) corrupted inputs
    ball_vis: torch.Tensor  # (T,) observed mask for model input (1=observed)
    ball_uv_gt: torch.Tensor  # (T, 2) ground truth UV
    ball_gt_vis: torch.Tensor  # (T,) GT visibility for supervision (1=visible)
    ball_in_frame_gt: torch.Tensor  # (T,) in-frame GT label (1=in frame, 0=out of frame)
    court_kp: torch.Tensor  # (K, 2) court keypoints
    court_vis: torch.Tensor  # (K,) court keypoint visibility
    seq_len: torch.Tensor  # scalar sequence length
