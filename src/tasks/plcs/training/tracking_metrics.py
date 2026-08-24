"""Lifecycle-aware localization and identity diagnostics for player tracks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from src.tasks.base.generate_dataset import CourtReferenceFrameProvenance
from src.tasks.base.training.tracking_metrics import (
    TrackingMetricConfig,
    common_lifecycle_tracking_metrics,
)
from src.tasks.plcs.court_keypoint_contract import (
    headings_target_to_physical,
    normalized_points_target_to_physical,
)
from src.tasks.plcs.training.tracking_losses import Assignment
from src.utils.schema.court_normalization import denormalize_court_position


def plcs_tracking_metrics(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    assignments: list[Assignment],
    *,
    config: TrackingMetricConfig,
    court_reference_provenance: Sequence[
        CourtReferenceFrameProvenance
    ]
    | None = None,
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
    position_errors_m: list[torch.Tensor] = []
    axis_errors_m: list[torch.Tensor] = []
    angular_errors: list[torch.Tensor] = []
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        provenance = None
        if court_reference_provenance is not None:
            if not court_reference_provenance:
                raise ValueError("PLCS tracking metric provenance must not be empty.")
            if len(court_reference_provenance) == 1:
                provenance = court_reference_provenance[0]
            elif len(court_reference_provenance) == len(assignments):
                provenance = court_reference_provenance[batch_index]
            else:
                raise ValueError(
                    "PLCS tracking metric batch and Court provenance cardinality "
                    "do not match."
                )
        for query_index, target_index in zip(
            query_indices.tolist(), target_indices.tolist(), strict=True
        ):
            active = (
                batch["target_presence"][batch_index, :, target_index]
                & frame_valid[batch_index]
            )
            if not active.any():
                continue
            pred_position = prediction["position"][
                batch_index, active, query_index
            ]
            target_position = batch["target_position"][
                batch_index, active, target_index
            ]
            pred_rotation = prediction["rotation"][
                batch_index, active, query_index
            ]
            target_rotation = batch["target_rotation"][
                batch_index, active, target_index
            ]
            if provenance is None:
                pred_position_m = denormalize_court_position(pred_position)
                target_position_m = denormalize_court_position(target_position)
            else:
                pred_position_m = normalized_points_target_to_physical(
                    pred_position,
                    provenance,
                )
                target_position_m = normalized_points_target_to_physical(
                    target_position,
                    provenance,
                )
                pred_rotation = headings_target_to_physical(
                    pred_rotation,
                    provenance,
                )
                target_rotation = headings_target_to_physical(
                    target_rotation,
                    provenance,
                )
            difference_m = pred_position_m - target_position_m
            position_errors_m.append(torch.linalg.vector_norm(difference_m, dim=-1))
            axis_errors_m.append(difference_m.abs())
            cosine = (
                (
                    F.normalize(
                        pred_rotation,
                        dim=-1,
                    )
                    * F.normalize(
                        target_rotation,
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
