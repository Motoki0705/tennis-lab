"""Typed BLCS model calls, training targets, and decoded predictions."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from src.tasks.base.generate_dataset import (
    CourtReferenceFrameProvenance,
    build_physical_court_provenance,
    court_points_target_to_physical,
    court_vectors_target_to_physical,
)
from src.tasks.base.model_io import ModelCall


@dataclass(frozen=True, slots=True)
class BLCSTrajectoryPrediction:
    """Decoded single-ball trajectory model output."""

    position: Tensor
    velocity: Tensor | None
    court_reference_provenance: tuple[CourtReferenceFrameProvenance, ...] = field(
        default_factory=tuple
    )
    coordinates_in_metres: bool = False


@dataclass(frozen=True, slots=True)
class BLCSTrackQueryPrediction:
    """Decoded lifecycle-query output including configured presence semantics."""

    position: Tensor
    presence_logits: Tensor
    presence_probability: Tensor
    presence: Tensor
    court_reference_provenance: tuple[CourtReferenceFrameProvenance, ...] = field(
        default_factory=tuple
    )
    coordinates_in_metres: bool = False


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
    court_reference_provenance: tuple[CourtReferenceFrameProvenance, ...] = field(
        default_factory=tuple
    )


@dataclass(frozen=True, slots=True)
class BLCSTrackQueryTrainingBatch:
    """Validated tracking-model call and lifecycle supervision tensors."""

    call: ModelCall
    target_position: Tensor
    target_velocity: Tensor
    target_presence: Tensor
    target_instance_id: Tensor
    target_slot_mask: Tensor
    frame_valid: Tensor
    court_reference_provenance: tuple[CourtReferenceFrameProvenance, ...] = field(
        default_factory=tuple
    )


def _physical_batch(
    value: Tensor,
    provenance: tuple[CourtReferenceFrameProvenance, ...],
    *,
    vector: bool,
) -> Tensor:
    if value.ndim < 2 or len(provenance) != value.shape[0]:
        raise ValueError(
            "BLCS prediction provenance must contain exactly one record per batch item."
        )
    rows: list[Tensor] = []
    for batch_index, record in enumerate(provenance):
        transformed = (
            court_vectors_target_to_physical(value[batch_index], record)
            if vector
            else court_points_target_to_physical(value[batch_index], record)
        )
        if not isinstance(transformed, Tensor):
            raise TypeError("BLCS prediction frame conversion returned a non-tensor.")
        rows.append(transformed)
    return torch.stack(rows)


def blcs_trajectory_prediction_to_physical(
    prediction: BLCSTrajectoryPrediction,
) -> BLCSTrajectoryPrediction:
    """Restore a metre-valued standard prediction to physical court space."""
    if not prediction.coordinates_in_metres:
        raise ValueError(
            "BLCS predictions must be denormalized to metres before frame restoration."
        )
    position = _physical_batch(
        prediction.position,
        prediction.court_reference_provenance,
        vector=False,
    )
    velocity = (
        None
        if prediction.velocity is None
        else _physical_batch(
            prediction.velocity,
            prediction.court_reference_provenance,
            vector=True,
        )
    )
    identity = tuple(
        build_physical_court_provenance()
        for _ in prediction.court_reference_provenance
    )
    return BLCSTrajectoryPrediction(
        position=position,
        velocity=velocity,
        court_reference_provenance=identity,
        coordinates_in_metres=True,
    )


def blcs_track_query_prediction_to_physical(
    prediction: BLCSTrackQueryPrediction,
) -> BLCSTrackQueryPrediction:
    """Restore a metre-valued tracking prediction to physical court space."""
    if not prediction.coordinates_in_metres:
        raise ValueError(
            "BLCS predictions must be denormalized to metres before frame restoration."
        )
    position = _physical_batch(
        prediction.position,
        prediction.court_reference_provenance,
        vector=False,
    )
    identity = tuple(
        build_physical_court_provenance()
        for _ in prediction.court_reference_provenance
    )
    return BLCSTrackQueryPrediction(
        position=position,
        presence_logits=prediction.presence_logits,
        presence_probability=prediction.presence_probability,
        presence=prediction.presence,
        court_reference_provenance=identity,
        coordinates_in_metres=True,
    )


__all__ = [
    "BLCSTrackQueryPrediction",
    "BLCSTrackQueryTrainingBatch",
    "BLCSTrajectoryPrediction",
    "BLCSTrajectoryTrainingBatch",
    "blcs_track_query_prediction_to_physical",
    "blcs_trajectory_prediction_to_physical",
]
