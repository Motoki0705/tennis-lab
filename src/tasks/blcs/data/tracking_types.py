"""Tensor contracts for multi-ball BLCS tracking."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from torch import Tensor

from src.tasks.base.data import ReferenceViewSelection, StableCameraIdTable
from src.tasks.base.generate_dataset import CourtReferenceFrameProvenance


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
    court_reference_provenance: tuple[CourtReferenceFrameProvenance, ...]
    selected_camera_ids: tuple[tuple[str, ...], ...]
    reference_view_selection: NotRequired[tuple[ReferenceViewSelection, ...]]
    stable_camera_id_table: NotRequired[tuple[StableCameraIdTable, ...]]
    reference_camera_id_string: NotRequired[tuple[str, ...]]
    reference_view_index: NotRequired[Tensor]
    view_camera_ids: NotRequired[Tensor]
    reference_camera_id: NotRequired[Tensor]
    reference_from_physical: NotRequired[Tensor]
    physical_from_reference: NotRequired[Tensor]
    track_query_reference: NotRequired[dict[str, object]]


class BLCSTrackingPrediction(TypedDict):
    """Fixed-query multi-ball model outputs."""

    position: Tensor
    presence_logits: Tensor
