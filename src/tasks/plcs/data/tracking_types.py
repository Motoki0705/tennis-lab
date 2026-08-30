"""Tensor contracts for multi-person PLCS tracking."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NotRequired, TypedDict

from torch import Tensor

from src.tasks.base.data import ReferenceViewSelection, StableCameraIdTable
from src.tasks.base.generate_dataset import CourtReferenceFrameProvenance


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
    human_kp_target: Tensor
    human_vis_target: Tensor
    detection_gt_index: Tensor
    camera_C: Tensor
    camera_R: Tensor
    camera_f: Tensor
    camera_cx: Tensor
    camera_cy: Tensor
    camera_w: Tensor
    camera_h: Tensor
    court_keypoint_metadata: NotRequired[tuple[Mapping[str, object], ...]]
    court_reference_provenance: NotRequired[
        tuple[CourtReferenceFrameProvenance, ...]
    ]
    selected_camera_ids: NotRequired[tuple[tuple[str, ...], ...]]
    reference_view_selection: NotRequired[tuple[ReferenceViewSelection, ...]]
    stable_camera_id_table: NotRequired[tuple[StableCameraIdTable, ...]]
    reference_camera_id_string: NotRequired[tuple[str, ...]]
    reference_view_index: NotRequired[Tensor]
    view_camera_ids: NotRequired[Tensor]
    reference_camera_id: NotRequired[Tensor]
    reference_from_physical: NotRequired[Tensor]
    physical_from_reference: NotRequired[Tensor]
    track_query_reference: NotRequired[Mapping[str, object]]


class PLCSTrackingPrediction(TypedDict):
    """Fixed-query multi-person model outputs."""

    position: Tensor
    rotation: Tensor
    presence_logits: Tensor
    canonical_pose: NotRequired[Tensor]
