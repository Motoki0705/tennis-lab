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
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def _position_mae_meters(
    prediction: BLCSTrackQueryPrediction,
    batch: BLCSTrackQueryTrainingBatch,
    assignments: list[Assignment],
) -> torch.Tensor:
    """Return matched per-axis MAE in physical metres."""
    pred_position = prediction.position
    scale = pred_position.new_tensor(COURT_COORD_SCALE_XYZ)
    terms: list[torch.Tensor] = []
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        for query_index, target_index in zip(
            query_indices.tolist(), target_indices.tolist(), strict=True
        ):
            active = (
                batch.target_presence[batch_index, :, target_index]
                & batch.frame_mask[batch_index]
            )
            if active.any():
                terms.append(
                    (
                        pred_position[batch_index, active, query_index]
                        - batch.target_position[batch_index, active, target_index]
                    )
                    .abs()
                    .mean(0)
                    * scale
                )
    if terms:
        return torch.stack(terms).mean(0)
    return pred_position.new_zeros(3)


def _reference_orientation_metrics(
    prediction: BLCSTrackQueryPrediction,
    batch: BLCSTrackQueryTrainingBatch,
    assignments: list[Assignment],
) -> dict[str, torch.Tensor]:
    """Measure oriented and source-frame target errors under one reference."""
    scale_y = prediction.position.new_tensor(COURT_COORD_SCALE_XYZ[1])
    y_errors: list[torch.Tensor] = []
    y_sign_hits: list[torch.Tensor] = []
    source_y_errors: list[torch.Tensor] = []
    side_errors: dict[float, list[torch.Tensor]] = {1.0: [], -1.0: []}
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        orientation_sign = batch.orientation_sign[batch_index]
        for query_index, target_index in zip(
            query_indices.tolist(), target_indices.tolist(), strict=True
        ):
            active = (
                batch.target_presence[batch_index, :, target_index]
                & batch.frame_mask[batch_index]
            )
            if not active.any():
                continue
            predicted_y = prediction.position[batch_index, active, query_index, 1]
            target_y = batch.target_position[batch_index, active, target_index, 1]
            error = (predicted_y - target_y).abs() * scale_y
            y_errors.append(error)
            y_sign_hits.append(torch.sign(predicted_y).eq(torch.sign(target_y)).float())
            source_predicted_y = predicted_y * orientation_sign
            source_target_y = batch.source_target_position[
                batch_index, active, target_index, 1
            ]
            source_y_errors.append(
                (source_predicted_y - source_target_y).abs() * scale_y
            )
            side_errors[float(orientation_sign.item())].append(error)

    zero = prediction.position.new_zeros(())

    def mean(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(values).mean() if values else zero

    return {
        "y_sign_accuracy": mean(y_sign_hits),
        "source_frame_position_mae_y_m": mean(source_y_errors),
        "reference_positive_position_mae_y_m": mean(side_errors[1.0]),
        "reference_negative_position_mae_y_m": mean(side_errors[-1.0]),
        "reference_oriented_position_mae_y_m": mean(y_errors),
    }


def _paired_reference_consistency_y(
    prediction: BLCSTrackQueryPrediction,
    batch: BLCSTrackQueryTrainingBatch,
    assignments: list[Assignment],
    counterfactual_prediction: BLCSTrackQueryPrediction,
    counterfactual_assignments: list[Assignment],
    counterfactual_orientation_sign: torch.Tensor,
) -> torch.Tensor:
    """Compare paired same-scene predictions after restoring one source frame."""
    if counterfactual_orientation_sign.shape != batch.orientation_sign.shape:
        raise ValueError("counterfactual_orientation_sign must have shape (B,).")
    scale_y = prediction.position.new_tensor(COURT_COORD_SCALE_XYZ[1])
    terms: list[torch.Tensor] = []
    for batch_index, ((queries, targets), (cf_queries, cf_targets)) in enumerate(
        zip(assignments, counterfactual_assignments, strict=True)
    ):
        query_by_target = dict(zip(targets.tolist(), queries.tolist(), strict=True))
        cf_query_by_target = dict(
            zip(cf_targets.tolist(), cf_queries.tolist(), strict=True)
        )
        shared_targets = sorted(set(query_by_target) & set(cf_query_by_target))
        for target_index in shared_targets:
            active = (
                batch.target_presence[batch_index, :, target_index]
                & batch.frame_mask[batch_index]
            )
            if not active.any():
                continue
            primary_y = (
                prediction.position[
                    batch_index, active, query_by_target[target_index], 1
                ]
                * batch.orientation_sign[batch_index]
            )
            counterfactual_y = (
                counterfactual_prediction.position[
                    batch_index, active, cf_query_by_target[target_index], 1
                ]
                * counterfactual_orientation_sign[batch_index]
            )
            terms.append((primary_y - counterfactual_y).abs() * scale_y)
    if not terms:
        raise ValueError(
            "paired reference consistency has no shared active target predictions."
        )
    return torch.cat(terms).mean()


def blcs_tracking_metrics(
    prediction: BLCSTrackQueryPrediction,
    batch: BLCSTrackQueryTrainingBatch,
    assignments: list[Assignment],
    *,
    counterfactual_prediction: BLCSTrackQueryPrediction,
    counterfactual_assignments: list[Assignment],
    counterfactual_orientation_sign: torch.Tensor,
    config: TrackingMetricConfig,
) -> dict[str, torch.Tensor]:
    """Compute shared lifecycle metrics for BLCS predictions."""
    metrics: dict[str, torch.Tensor] = common_lifecycle_tracking_metrics(
        {
            "position": prediction.position,
            "presence_logits": prediction.presence_logits,
        },
        {
            "target_position": batch.target_position,
            "target_presence": batch.target_presence,
            "target_instance_id": batch.target_instance_id,
            "frame_mask": batch.frame_mask,
        },
        assignments,
        config=config,
    )
    position_mae_m = _position_mae_meters(prediction, batch, assignments)
    metrics.update(
        {
            "position_mae_x_m": position_mae_m[0],
            "position_mae_y_m": position_mae_m[1],
            "position_mae_z_m": position_mae_m[2],
        }
    )
    metrics.update(_reference_orientation_metrics(prediction, batch, assignments))
    metrics["reference_consistency_y_m"] = _paired_reference_consistency_y(
        prediction,
        batch,
        assignments,
        counterfactual_prediction,
        counterfactual_assignments,
        counterfactual_orientation_sign,
    )
    return metrics
