"""Lifecycle-aware localization and identity diagnostics for ball tracks."""

from __future__ import annotations

import torch

from src.tasks.base.training.tracking_metrics import (
    TrackingMetricConfig,
    common_lifecycle_tracking_metrics,
)
from src.tasks.blcs.training.tracking_losses import Assignment
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def _position_mae_meters(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    assignments: list[Assignment],
) -> torch.Tensor:
    """Return matched per-axis MAE in physical metres."""
    pred_position = prediction["position"]
    scale = pred_position.new_tensor(COURT_COORD_SCALE_XYZ)
    terms: list[torch.Tensor] = []
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        for query_index, target_index in zip(
            query_indices.tolist(), target_indices.tolist(), strict=True
        ):
            active = (
                batch["target_presence"][batch_index, :, target_index]
                & batch["frame_mask"][batch_index]
            )
            if active.any():
                terms.append(
                    (
                        pred_position[batch_index, active, query_index]
                        - batch["target_position"][batch_index, active, target_index]
                    )
                    .abs()
                    .mean(0)
                    * scale
                )
    if terms:
        return torch.stack(terms).mean(0)
    return pred_position.new_zeros(3)


def blcs_tracking_metrics(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    assignments: list[Assignment],
    *,
    config: TrackingMetricConfig,
) -> dict[str, torch.Tensor]:
    """Compute shared lifecycle metrics for BLCS predictions."""
    metrics: dict[str, torch.Tensor] = common_lifecycle_tracking_metrics(
        prediction,
        batch,
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
    return metrics
