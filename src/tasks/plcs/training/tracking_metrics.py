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
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def plcs_tracking_metrics(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    assignments: list[Assignment],
    *,
    counterfactual_prediction: dict[str, torch.Tensor],
    counterfactual_assignments: list[Assignment],
    counterfactual_orientation_sign: torch.Tensor,
    config: TrackingMetricConfig,
) -> dict[str, torch.Tensor]:
    """Compute shared lifecycle metrics plus matched angular error."""
    metrics: dict[str, torch.Tensor] = common_lifecycle_tracking_metrics(
        prediction,
        batch,
        assignments,
        config=config,
    )
    angular_errors: list[torch.Tensor] = []
    position_errors: list[torch.Tensor] = []
    sign_hits: list[torch.Tensor] = []
    source_y_errors: list[torch.Tensor] = []
    source_heading_errors: list[torch.Tensor] = []
    side_y_errors: dict[float, list[torch.Tensor]] = {1.0: [], -1.0: []}
    scale = prediction["position"].new_tensor(COURT_COORD_SCALE_XYZ)
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        for query_index, target_index in zip(
            query_indices.tolist(), target_indices.tolist(), strict=True
        ):
            active = (
                batch["target_presence"][batch_index, :, target_index]
                & batch["frame_mask"][batch_index]
            )
            if not active.any():
                continue
            predicted_position = prediction["position"][
                batch_index, active, query_index
            ]
            target_position = batch["target_position"][
                batch_index, active, target_index
            ]
            position_error = (predicted_position - target_position).abs() * scale
            position_errors.append(position_error)
            sign_hits.append(
                torch.sign(predicted_position[..., 1])
                .eq(torch.sign(target_position[..., 1]))
                .float()
            )
            orientation_sign = batch["orientation_sign"][batch_index]
            source_predicted_y = predicted_position[..., 1] * orientation_sign
            source_target_y = batch["source_target_position"][
                batch_index, active, target_index, 1
            ]
            source_y_errors.append(
                (source_predicted_y - source_target_y).abs() * scale[1]
            )
            side_y_errors[float(orientation_sign.item())].append(
                position_error[..., 1]
            )
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
            source_prediction_rotation = prediction["rotation"][
                batch_index, active, query_index
            ].clone()
            source_prediction_rotation[..., 1] *= orientation_sign
            source_cosine = (
                F.normalize(source_prediction_rotation, dim=-1)
                * F.normalize(
                    batch["source_target_rotation"][
                        batch_index, active, target_index
                    ],
                    dim=-1,
                )
            ).sum(-1).clamp(-1.0, 1.0)
            source_heading_errors.append(
                torch.acos(source_cosine).mean() * (180.0 / math.pi)
            )
    zero = prediction["position"].new_zeros(())
    position_mae = (
        torch.cat(position_errors).mean(0)
        if position_errors
        else prediction["position"].new_zeros(3)
    )

    def mean(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([value.reshape(-1) for value in values]).mean() if values else zero

    metrics.update(
        {
            "position_mae_x_m": position_mae[0],
            "position_mae_y_m": position_mae[1],
            "position_mae_z_m": position_mae[2],
            "y_sign_accuracy": mean(sign_hits),
            "source_frame_position_mae_y_m": mean(source_y_errors),
            "source_frame_heading_error_deg": mean(source_heading_errors),
            "reference_positive_position_mae_y_m": mean(side_y_errors[1.0]),
            "reference_negative_position_mae_y_m": mean(side_y_errors[-1.0]),
        }
    )
    metrics["angular_error_deg"] = (
        torch.stack(angular_errors).mean() if angular_errors else zero
    )
    paired_y: list[torch.Tensor] = []
    paired_heading: list[torch.Tensor] = []
    if counterfactual_orientation_sign.shape != batch["orientation_sign"].shape:
        raise ValueError("counterfactual_orientation_sign must have shape (B,).")
    for batch_index, ((queries, targets), (cf_queries, cf_targets)) in enumerate(
        zip(assignments, counterfactual_assignments, strict=True)
    ):
        query_by_target = dict(zip(targets.tolist(), queries.tolist(), strict=True))
        cf_query_by_target = dict(
            zip(cf_targets.tolist(), cf_queries.tolist(), strict=True)
        )
        for target_index in sorted(set(query_by_target) & set(cf_query_by_target)):
            active = (
                batch["target_presence"][batch_index, :, target_index]
                & batch["frame_mask"][batch_index]
            )
            if not active.any():
                continue
            primary_query = query_by_target[target_index]
            counterfactual_query = cf_query_by_target[target_index]
            primary_y = (
                prediction["position"][batch_index, active, primary_query, 1]
                * batch["orientation_sign"][batch_index]
            )
            counterfactual_y = (
                counterfactual_prediction["position"][
                    batch_index, active, counterfactual_query, 1
                ]
                * counterfactual_orientation_sign[batch_index]
            )
            paired_y.append((primary_y - counterfactual_y).abs() * scale[1])
            primary_heading = prediction["rotation"][
                batch_index, active, primary_query
            ].clone()
            primary_heading[..., 1] *= batch["orientation_sign"][batch_index]
            counterfactual_heading = counterfactual_prediction["rotation"][
                batch_index, active, counterfactual_query
            ].clone()
            counterfactual_heading[..., 1] *= counterfactual_orientation_sign[
                batch_index
            ]
            cosine = (
                F.normalize(primary_heading, dim=-1)
                * F.normalize(counterfactual_heading, dim=-1)
            ).sum(-1).clamp(-1.0, 1.0)
            paired_heading.append(torch.acos(cosine) * (180.0 / math.pi))
    if not paired_y or not paired_heading:
        raise ValueError(
            "paired reference consistency has no shared active target predictions."
        )
    metrics["reference_consistency_y_m"] = mean(paired_y)
    metrics["reference_consistency_heading_deg"] = mean(paired_heading)
    return metrics
