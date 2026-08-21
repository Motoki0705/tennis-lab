"""Tensor contracts for multi-ball BLCS tracking."""

from __future__ import annotations

from typing import TypedDict

from torch import Tensor


class BLCSTrackingBatch(TypedDict):
    """Padded multi-ball input, GT, and debug tensors.

    The authoritative shapes and mask semantics are documented in
    ``src/tasks/blcs/README.md``. ``candidate_gt_index`` validates the
    lifecycle-packed observation axis and must never be passed to the tracking
    model. Visibility selects observed/invisible embeddings only; padding alone
    controls attention participation.
    """

    scene_format_version: Tensor
    ball_uv: Tensor
    ball_vis: Tensor
    court_kp: Tensor
    court_vis: Tensor
    padding_mask: Tensor
    target_position: Tensor
    target_velocity: Tensor
    target_presence: Tensor
    target_instance_id: Tensor
    target_slot_mask: Tensor
    clean_ball_uv: Tensor
    clean_ball_vis: Tensor
    candidate_gt_index: Tensor


class BLCSTrackingPrediction(TypedDict):
    """Fixed-query multi-ball model outputs."""

    position: Tensor
    presence_logits: Tensor
