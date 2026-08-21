"""Tensor contracts for multi-person PLCS tracking."""

from __future__ import annotations

from typing import TypedDict

from torch import Tensor


class PLCSTrackingBatch(TypedDict):
    """Padded multi-person input, GT, and debug tensors.

    ``human_vis`` selects the visible/invisible observation embedding but never
    controls attention. ``detection_gt_index`` is retained for augmentation and
    data validation only. ``padding_mask`` is the sole attention-padding input.
    """

    scene_format_version: Tensor
    human_kp: Tensor
    human_vis: Tensor
    court_kp: Tensor
    court_vis: Tensor
    padding_mask: Tensor
    target_position: Tensor
    target_rotation: Tensor
    target_canonical_pose_3d: Tensor
    target_human_kp_3d: Tensor
    target_presence: Tensor
    target_instance_id: Tensor
    target_slot_mask: Tensor
    clean_human_kp: Tensor
    clean_human_vis: Tensor
    detection_gt_index: Tensor


class PLCSTrackingPrediction(TypedDict):
    """Fixed-query multi-person model outputs."""

    position: Tensor
    rotation: Tensor
    presence_logits: Tensor
