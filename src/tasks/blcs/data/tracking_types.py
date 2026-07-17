"""Tensor contracts for multi-ball BLCS tracking."""

from __future__ import annotations

from typing import TypedDict

from torch import Tensor


class BLCSTrackingBatch(TypedDict):
    """Padded multi-ball input, GT, and debug tensors.

    The authoritative shapes and mask semantics are documented in
    ``src/tasks/blcs/README.md``. ``candidate_gt_index`` is debug-only
    and must never be passed to a model.
    """

    scene_format_version: Tensor
    ball_uv: Tensor
    ball_score: Tensor
    ball_candidate_mask: Tensor
    ball_visible: Tensor
    court_kp: Tensor
    court_vis: Tensor
    frame_mask: Tensor
    view_mask: Tensor
    position_3d: Tensor
    ball_present: Tensor
    target_ball_mask: Tensor
    ball_uv_gt: Tensor
    ball_visible_gt: Tensor
    candidate_gt_index: Tensor


class BLCSTrackingPrediction(TypedDict):
    """Fixed-query multi-ball model outputs."""

    position: Tensor
    presence_logits: Tensor
