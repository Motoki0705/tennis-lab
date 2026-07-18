"""Tensor contracts for multi-person PLCS tracking."""

from __future__ import annotations

from typing import TypedDict

from torch import Tensor


class PLCSTrackingBatch(TypedDict):
    """Padded multi-person input, GT, and debug tensors.

    ``human_vis`` and ``detection_gt_index`` are retained for augmentation/data
    validation only. The model consumes ``detection_mask`` for whole-person
    invisibility and ``court_vis`` to zero unavailable court coordinates before
    per-camera pooling. See ``src/tasks/plcs/README.md`` for shapes.
    """

    scene_format_version: Tensor
    human_kp: Tensor
    human_vis: Tensor
    detection_mask: Tensor
    court_kp: Tensor
    court_vis: Tensor
    frame_mask: Tensor
    view_mask: Tensor
    target_position: Tensor
    target_rotation: Tensor
    target_canonical_pose_3d: Tensor
    target_human_kp_3d: Tensor
    target_presence: Tensor
    target_instance_id: Tensor
    target_slot_mask: Tensor
    clean_human_kp: Tensor
    clean_human_visible: Tensor
    detection_gt_index: Tensor


class PLCSTrackingPrediction(TypedDict):
    """Fixed-query multi-person model outputs."""

    position: Tensor
    rotation: Tensor
    presence_logits: Tensor
