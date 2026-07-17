"""Tensor contracts for multi-person PLCS tracking."""

from __future__ import annotations

from typing import TypedDict

from torch import Tensor


class PLCSTrackingBatch(TypedDict):
    """Padded multi-person input, GT, and debug tensors.

    ``detection_gt_index`` is retained only for data validation and must never
    become a model input. See ``src/tasks/plcs/README.md`` for shapes.
    """

    scene_format_version: Tensor
    human_kp: Tensor
    human_vis: Tensor
    detection_mask: Tensor
    detection_score: Tensor
    bbox: Tensor
    court_kp: Tensor
    court_vis: Tensor
    frame_mask: Tensor
    view_mask: Tensor
    position: Tensor
    rotation: Tensor
    canonical_pose_3d: Tensor
    human_kp_3d: Tensor
    person_present: Tensor
    target_person_mask: Tensor
    clean_human_kp: Tensor
    clean_human_visible: Tensor
    detection_gt_index: Tensor


class PLCSTrackingPrediction(TypedDict):
    """Fixed-query multi-person model outputs."""

    position: Tensor
    rotation: Tensor
    presence_logits: Tensor
