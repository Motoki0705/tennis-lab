"""Tensor contracts for multi-person PLCS tracking."""

from __future__ import annotations

from typing import TypedDict

from torch import Tensor


class PLCSTrackingBatch(TypedDict):
    """Padded multi-person input, GT, and debug tensors.

    ``human_vis`` and ``detection_gt_index`` are retained for augmentation/data
    validation only. ``detection_gt_index`` validates the object-ID-ordered
    observation axis and is never passed to the model. The model consumes
    ``detection_mask`` for whole-person invisibility and ``court_vis`` to zero
    unavailable court coordinates. See ``src/tasks/plcs/README.md`` for shapes.
    """

    scene_format_version: Tensor
    human_kp: Tensor
    human_vis: Tensor
    joint_visibility: Tensor
    detection_score: Tensor
    detection_mask: Tensor
    court_kp: Tensor
    court_vis: Tensor
    court_peak_uv: Tensor
    court_peak_score: Tensor
    court_peak_covariance: Tensor
    court_peak_valid: Tensor
    frame_mask: Tensor
    view_mask: Tensor
    reference_view_index: Tensor
    orientation_sign: Tensor
    camera_center: Tensor
    target_position: Tensor
    source_target_position: Tensor
    target_rotation: Tensor
    source_target_rotation: Tensor
    target_canonical_pose_3d: Tensor
    target_human_kp_3d: Tensor
    source_target_human_kp_3d: Tensor
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
