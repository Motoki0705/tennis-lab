"""Tensor contracts for multi-ball BLCS tracking."""

from __future__ import annotations

from typing import TypedDict

from torch import Tensor


class BLCSTrackingBatch(TypedDict):
    """Padded multi-ball input, GT, and debug tensors.

    The authoritative shapes and mask semantics are documented in
    ``src/tasks/blcs/README.md``. ``ball_visible``, ``court_vis``, and
    ``candidate_gt_index`` are augmentation/debug-only and must never be
    passed to the tracking model.
    """

    scene_format_version: Tensor
    ball_uv: Tensor
    ball_visible: Tensor
    court_kp: Tensor
    court_vis: Tensor
    frame_mask: Tensor
    view_mask: Tensor
    target_position: Tensor
    target_velocity: Tensor
    target_presence: Tensor
    target_instance_id: Tensor
    target_slot_mask: Tensor
    clean_ball_uv: Tensor
    clean_ball_visible: Tensor
    candidate_gt_index: Tensor


class BLCSTrackingPrediction(TypedDict):
    """Fixed-query multi-ball model outputs."""

    position: Tensor
    presence_logits: Tensor
