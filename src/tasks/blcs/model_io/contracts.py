"""Typed BLCS model calls, training targets, and decoded predictions."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from src.tasks.base.model_io import ModelCall


@dataclass(frozen=True, slots=True)
class BLCSTrajectoryPrediction:
    """Decoded single-ball trajectory model output."""

    position: Tensor
    velocity: Tensor | None


@dataclass(frozen=True, slots=True)
class BLCSTrackQueryPrediction:
    """Decoded lifecycle-query output including configured presence semantics."""

    position: Tensor
    presence_logits: Tensor
    presence_probability: Tensor
    presence: Tensor


@dataclass(frozen=True, slots=True)
class BLCSTrajectoryTrainingBatch:
    """Validated standard-model call and all tensors consumed by training."""

    call: ModelCall
    position: Tensor
    velocity: Tensor
    loss_mask: Tensor
    target_uv: Tensor
    target_vis: Tensor
    camera_R: Tensor
    camera_C: Tensor
    camera_f: Tensor
    camera_cx: Tensor
    camera_cy: Tensor
    camera_w: Tensor
    camera_h: Tensor


@dataclass(frozen=True, slots=True)
class BLCSTrackQueryTrainingBatch:
    """Validated tracking-model call and lifecycle supervision tensors."""

    call: ModelCall
    target_position: Tensor
    source_target_position: Tensor
    target_velocity: Tensor
    source_target_velocity: Tensor
    target_presence: Tensor
    target_instance_id: Tensor
    target_slot_mask: Tensor
    frame_mask: Tensor
    reference_view_index: Tensor
    orientation_sign: Tensor


__all__ = [
    "BLCSTrackQueryPrediction",
    "BLCSTrackQueryTrainingBatch",
    "BLCSTrajectoryPrediction",
    "BLCSTrajectoryTrainingBatch",
]
