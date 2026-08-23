"""Lifecycle-aware localization and identity diagnostics for ball tracks."""

from __future__ import annotations

import torch

from src.tasks.base.training.tracking_metrics import (
    TrackingMetricConfig,
    common_lifecycle_tracking_metrics,
)
from src.tasks.blcs.model_io import (
    BLCSTrackQueryPrediction,
    BLCSTrackQueryTrainingBatch,
)
from src.tasks.blcs.training.tracking_losses import Assignment
from src.utils.schema.court_normalization import (
    CourtCoordinateNormalization,
    resolve_court_coordinate_normalization,
)


def _position_mae_meters(
    prediction: BLCSTrackQueryPrediction,
    batch: BLCSTrackQueryTrainingBatch,
    assignments: list[Assignment],
    normalization: CourtCoordinateNormalization,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return matched aggregate and per-axis errors in physical metres."""
    pred_position = prediction.position
    axis_terms: list[torch.Tensor] = []
    aggregate_terms: list[torch.Tensor] = []
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        for query_index, target_index in zip(
            query_indices.tolist(), target_indices.tolist(), strict=True
        ):
            active = (
                batch.target_presence[batch_index, :, target_index]
                & batch.frame_valid[batch_index]
            )
            if active.any():
                error_norm = (
                    pred_position[batch_index, active, query_index]
                    - batch.target_position[batch_index, active, target_index]
                )
                error_m = normalization.denormalize_position(error_norm)
                if not isinstance(error_m, torch.Tensor):
                    raise TypeError(
                        "BLCS tracking metric denormalization returned a non-tensor."
                    )
                axis_terms.append(error_m.abs().mean(0))
                aggregate_terms.append(torch.linalg.vector_norm(error_m, dim=-1).mean())
    if axis_terms:
        return torch.stack(aggregate_terms).mean(), torch.stack(axis_terms).mean(0)
    return pred_position.new_zeros(()), pred_position.new_zeros(3)


def blcs_tracking_metrics(
    prediction: BLCSTrackQueryPrediction,
    batch: BLCSTrackQueryTrainingBatch,
    assignments: list[Assignment],
    *,
    config: TrackingMetricConfig,
    normalization: CourtCoordinateNormalization | str = "v1",
) -> dict[str, torch.Tensor]:
    """Compute shared lifecycle metrics for BLCS predictions."""
    contract = (
        normalization
        if isinstance(normalization, CourtCoordinateNormalization)
        else resolve_court_coordinate_normalization(normalization)
    )
    metrics: dict[str, torch.Tensor] = common_lifecycle_tracking_metrics(
        {
            "position": prediction.position,
            "presence_logits": prediction.presence_logits,
        },
        {
            "target_position": batch.target_position,
            "target_presence": batch.target_presence,
            "target_instance_id": batch.target_instance_id,
            "frame_mask": batch.frame_valid,
        },
        assignments,
        config=config,
    )
    position_error_m, position_mae_m = _position_mae_meters(
        prediction,
        batch,
        assignments,
        contract,
    )
    metrics.update(
        {
            "position_error_m": position_error_m,
            "position_mae_x_m": position_mae_m[0],
            "position_mae_y_m": position_mae_m[1],
            "position_mae_z_m": position_mae_m[2],
        }
    )
    return metrics
