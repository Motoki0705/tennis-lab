"""Lifecycle-aware localization and identity diagnostics for player tracks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from src.tasks.base.evaluation import (
    compute_axis_wise_position_error,
    compute_heading_error_radians,
    compute_y_sign_accuracy,
    stratify_metric_by_reference_view_index,
)
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
    reference_view_index: torch.Tensor | None = None,
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
    reference_predictions: list[torch.Tensor] = []
    reference_targets: list[torch.Tensor] = []
    reference_pred_headings: list[torch.Tensor] = []
    reference_target_headings: list[torch.Tensor] = []
    per_sample_reference_errors: dict[int, list[torch.Tensor]] = {}
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
            target_frame_pred_m = denormalize_court_position(pred_position)
            target_frame_target_m = denormalize_court_position(target_position)
            if not isinstance(target_frame_pred_m, torch.Tensor) or not isinstance(
                target_frame_target_m, torch.Tensor
            ):
                raise TypeError(
                    "PLCS tracking metric denormalization must preserve tensors."
                )
            reference_predictions.append(target_frame_pred_m)
            reference_targets.append(target_frame_target_m)
            reference_pred_headings.append(pred_rotation)
            reference_target_headings.append(target_rotation)
            per_sample_reference_errors.setdefault(batch_index, []).append(
                torch.linalg.vector_norm(
                    target_frame_pred_m - target_frame_target_m,
                    dim=-1,
                )
            )
            if provenance is None:
                pred_position_m = target_frame_pred_m
                target_position_m = target_frame_target_m
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
    metrics["heading_error_deg"] = metrics["angular_error_deg"]
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
    if reference_view_index is not None:
        if reference_view_index.shape != (prediction["position"].shape[0],):
            raise ValueError(
                "reference_view_index must match the PLCS tracking batch axis."
            )
        if reference_view_index.dtype != torch.int64:
            raise ValueError("reference_view_index must have dtype torch.int64.")
        if reference_view_index.device != prediction["position"].device:
            raise ValueError(
                "reference_view_index must share the PLCS tracking tensor device."
            )
        if reference_predictions:
            all_reference_predictions = torch.cat(reference_predictions)
            all_reference_targets = torch.cat(reference_targets)
            axis_error = compute_axis_wise_position_error(
                all_reference_predictions,
                all_reference_targets,
            )
            metrics["x_error_m"] = zero.new_tensor(axis_error.x)
            metrics["y_error_m"] = zero.new_tensor(axis_error.y)
            metrics["z_error_m"] = zero.new_tensor(axis_error.z)
            target_has_y_sign = bool(
                all_reference_targets[:, 1].ne(0).any().item()
            )
            metrics["y_sign_accuracy"] = (
                zero.new_tensor(
                    compute_y_sign_accuracy(
                        all_reference_predictions,
                        all_reference_targets,
                    )
                )
                if target_has_y_sign
                else zero
            )
            metrics["heading_error_deg"] = zero.new_tensor(
                compute_heading_error_radians(
                    torch.cat(reference_pred_headings),
                    torch.cat(reference_target_headings),
                )
                * (180.0 / math.pi)
            )
            sample_indices = sorted(per_sample_reference_errors)
            sample_errors = torch.stack(
                [
                    torch.cat(per_sample_reference_errors[index]).mean()
                    for index in sample_indices
                ]
            )
            strata = stratify_metric_by_reference_view_index(
                sample_errors,
                reference_view_index[
                    torch.tensor(
                        sample_indices,
                        dtype=torch.int64,
                        device=reference_view_index.device,
                    )
                ],
            )
            metrics.update(
                {
                    f"reference_index_{index}_position_error_m": zero.new_tensor(
                        value
                    )
                    for index, value in strata.items()
                }
            )
        else:
            metrics["y_sign_accuracy"] = zero
    return metrics
