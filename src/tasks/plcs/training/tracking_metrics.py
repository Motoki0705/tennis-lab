"""Lifecycle-aware localization and identity diagnostics for player tracks."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.tasks.base.training.tracking_metrics import (
    TrackingMetricConfig,
    common_lifecycle_tracking_metrics,
)
from src.tasks.plcs.training.tracking_losses import Assignment
from src.utils.schema.court_normalization import (
    CourtCoordinateNormalization,
    resolve_court_coordinate_normalization,
)


def plcs_tracking_metrics(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    assignments: list[Assignment],
    *,
    config: TrackingMetricConfig,
    normalization: CourtCoordinateNormalization | str = "v1",
) -> dict[str, torch.Tensor]:
    """Compute shared lifecycle metrics plus matched angular error."""
    frame_valid = (~batch["padding_mask"]).any(dim=1)
    metrics: dict[str, torch.Tensor] = common_lifecycle_tracking_metrics(
        prediction,
        {
            "target_position": batch["target_position"],
            "target_presence": batch["target_presence"],
            "target_instance_id": batch["target_instance_id"],
            "frame_mask": frame_valid,
        },
        assignments,
        config=config,
    )
    contract = (
        normalization
        if isinstance(normalization, CourtCoordinateNormalization)
        else resolve_court_coordinate_normalization(normalization)
    )
    position_errors_m: list[torch.Tensor] = []
    axis_errors_m: list[torch.Tensor] = []
    angular_errors: list[torch.Tensor] = []
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        for query_index, target_index in zip(
            query_indices.tolist(), target_indices.tolist(), strict=True
        ):
            active = (
                batch["target_presence"][batch_index, :, target_index]
                & frame_valid[batch_index]
            )
            if not active.any():
                continue
            pred_position_m = contract.denormalize_position(
                prediction["position"][batch_index, active, query_index]
            )
            target_position_m = contract.denormalize_position(
                batch["target_position"][batch_index, active, target_index]
            )
            if not isinstance(pred_position_m, torch.Tensor) or not isinstance(
                target_position_m, torch.Tensor
            ):
                raise TypeError("PLCS tracking metric denormalization must preserve tensors.")
            difference_m = pred_position_m - target_position_m
            position_errors_m.append(torch.linalg.vector_norm(difference_m, dim=-1))
            axis_errors_m.append(difference_m.abs())
            cosine = (
                (
                    F.normalize(
                        prediction["rotation"][batch_index, active, query_index],
                        dim=-1,
                    )
                    * F.normalize(
                        batch["target_rotation"][batch_index, active, target_index],
                        dim=-1,
                    )
                )
                .sum(-1)
                .clamp(-1.0, 1.0)
            )
            angular_errors.append(torch.acos(cosine).mean() * (180.0 / math.pi))
    zero = prediction["position"].new_zeros(())
    metrics["angular_error_deg"] = (
        torch.stack(angular_errors).mean() if angular_errors else zero
    )
    if position_errors_m:
        all_position_errors_m = torch.cat(position_errors_m)
        all_axis_errors_m = torch.cat(axis_errors_m)
        metrics["position_error_m"] = all_position_errors_m.mean()
        metrics["x_error_m"] = all_axis_errors_m[:, 0].mean()
        metrics["y_error_m"] = all_axis_errors_m[:, 1].mean()
        metrics["z_error_m"] = all_axis_errors_m[:, 2].mean()
    else:
        metrics["position_error_m"] = zero
        metrics["x_error_m"] = zero
        metrics["y_error_m"] = zero
        metrics["z_error_m"] = zero
    return metrics
