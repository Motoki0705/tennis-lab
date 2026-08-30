"""Lifecycle-aware localization and identity diagnostics for player tracks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from src.tasks.base.generate_dataset import CourtReferenceFrameProvenance
from src.tasks.base.training.metric_logging import (
    ScalarMetricStatistic,
    compute_scalar_metric_statistics,
)
from src.tasks.base.training.tracking_metrics import (
    TrackingMetricConfig,
    common_lifecycle_tracking_statistics,
)
from src.tasks.plcs.court_keypoint_contract import (
    headings_target_to_physical,
    normalized_points_target_to_physical,
)
from src.tasks.plcs.training.tracking_losses import (
    Assignment,
    validate_tracking_projection_shapes,
)
from src.utils.geometry.court_pose import (
    canonical_pose_to_world_pose,
    world_pose_to_canonical_pose,
)
from src.utils.projection.differentiable_projection import (
    DifferentiablePinholeProjection,
)
from src.utils.schema.court_normalization import denormalize_court_position

_REPROJECTION_BATCH_FIELDS = frozenset(
    {
        "human_kp_target",
        "human_vis_target",
        "camera_R",
        "camera_C",
        "camera_f",
        "camera_cx",
        "camera_cy",
        "camera_w",
        "camera_h",
    }
)


def _scalar_statistic(
    total: torch.Tensor,
    count: int | torch.Tensor,
) -> ScalarMetricStatistic:
    """Build one scalar sum/count pair on the metric tensor's device."""
    if isinstance(count, torch.Tensor):
        denominator = count.to(dtype=total.dtype, device=total.device)
    else:
        denominator = total.new_tensor(float(count))
    return ScalarMetricStatistic(numerator=total, denominator=denominator)


def _zero_statistic(value: torch.Tensor) -> ScalarMetricStatistic:
    return _scalar_statistic(value.new_zeros(()), 0)


def plcs_tracking_statistics(
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
) -> dict[str, ScalarMetricStatistic]:
    """Compute PLCS tracking sufficient statistics.

    Frame-level localization and heading errors are returned as matched-frame
    sums and counts.  Reference-frame diagnostics use the target-frame values
    when reference metadata is available; local-index diagnostics deliberately
    retain one observation per sample so uneven clip lengths do not bias a
    stratum toward longer clips.
    """
    frame_valid = (~batch["padding_mask"]).any(dim=1)
    statistics: dict[str, ScalarMetricStatistic] = (
        common_lifecycle_tracking_statistics(
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
    )

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
        if (reference_view_index < 0).any():
            raise ValueError("reference_view_index cannot contain negative values.")

    zero = prediction["position"].new_zeros(())
    position_errors_m: list[torch.Tensor] = []
    axis_errors_m: list[torch.Tensor] = []
    angular_errors_deg: list[torch.Tensor] = []
    reference_predictions: list[torch.Tensor] = []
    reference_targets: list[torch.Tensor] = []
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
            angular_errors_deg.append(torch.acos(cosine) * (180.0 / math.pi))

    if position_errors_m:
        all_position_errors_m = torch.cat(position_errors_m)
        all_axis_errors_m = torch.cat(axis_errors_m)
        statistics.update(
            {
                "position_error_m": _scalar_statistic(
                    all_position_errors_m.sum(), all_position_errors_m.numel()
                ),
                "x_error_m": _scalar_statistic(
                    all_axis_errors_m[:, 0].sum(), all_axis_errors_m.shape[0]
                ),
                "y_error_m": _scalar_statistic(
                    all_axis_errors_m[:, 1].sum(), all_axis_errors_m.shape[0]
                ),
                "z_error_m": _scalar_statistic(
                    all_axis_errors_m[:, 2].sum(), all_axis_errors_m.shape[0]
                ),
            }
        )
    else:
        statistics.update(
            {
                "position_error_m": _zero_statistic(zero),
                "x_error_m": _zero_statistic(zero),
                "y_error_m": _zero_statistic(zero),
                "z_error_m": _zero_statistic(zero),
            }
        )

    if angular_errors_deg:
        all_angular_errors_deg = torch.cat(angular_errors_deg)
        statistics["angular_error_deg"] = _scalar_statistic(
            all_angular_errors_deg.sum(), all_angular_errors_deg.numel()
        )
    else:
        statistics["angular_error_deg"] = _zero_statistic(zero)

    if reference_view_index is not None:
        if reference_predictions:
            all_reference_predictions = torch.cat(reference_predictions)
            all_reference_targets = torch.cat(reference_targets)
            reference_axis_errors = (
                all_reference_predictions - all_reference_targets
            ).abs()
            matched_count = all_reference_targets.shape[0]
            statistics.update(
                {
                    "x_error_m": _scalar_statistic(
                        reference_axis_errors[:, 0].sum(), matched_count
                    ),
                    "y_error_m": _scalar_statistic(
                        reference_axis_errors[:, 1].sum(), matched_count
                    ),
                    "z_error_m": _scalar_statistic(
                        reference_axis_errors[:, 2].sum(), matched_count
                    ),
                }
            )
            eligible = all_reference_targets[:, 1].ne(0)
            correct = torch.sign(all_reference_predictions[:, 1]).eq(
                torch.sign(all_reference_targets[:, 1])
            )
            statistics["y_sign_accuracy"] = _scalar_statistic(
                correct[eligible].to(dtype=zero.dtype).sum(), eligible.sum()
            )
        else:
            statistics.update(
                {
                    "y_sign_accuracy": _zero_statistic(zero),
                }
            )

    if reference_view_index is not None and reference_predictions:
        sample_indices = sorted(per_sample_reference_errors)
        for index in sample_indices:
            sample_error = torch.cat(per_sample_reference_errors[index]).mean()
            reference_index = int(reference_view_index[index].item())
            key = f"reference_index_{reference_index}_position_error_m"
            if key in statistics:
                previous = statistics[key]
                statistics[key] = ScalarMetricStatistic(
                    numerator=previous.numerator + sample_error,
                    denominator=previous.denominator + sample_error.new_tensor(1.0),
                )
            else:
                statistics[key] = _scalar_statistic(sample_error, 1)

    statistics.update(
        _pose_metric_statistics(
            prediction,
            batch,
            assignments,
            frame_valid=frame_valid,
            zero=zero,
        )
    )
    return statistics


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
    """Compute PLCS tracking metrics from their sufficient statistics."""
    return compute_scalar_metric_statistics(
        plcs_tracking_statistics(
            prediction,
            batch,
            assignments,
            config=config,
            court_reference_provenance=court_reference_provenance,
            reference_view_index=reference_view_index,
        ),
        zero_denominator_value=0.0,
    )


def _pose_metric_statistics(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    assignments: list[Assignment],
    *,
    frame_valid: torch.Tensor,
    zero: torch.Tensor,
) -> dict[str, ScalarMetricStatistic]:
    """Return additive matched pose and clean reprojection statistics."""
    canonical_pose = prediction.get("canonical_pose")
    target_world_pose = batch.get("target_human_kp_3d")
    if canonical_pose is None or target_world_pose is None:
        return {}
    if canonical_pose.ndim != 5 or canonical_pose.shape[-2:] != (17, 3):
        raise ValueError(
            "prediction['canonical_pose'] must have shape (B,T,Q,17,3) for "
            f"tracking pose metrics, got {tuple(canonical_pose.shape)}."
        )
    if target_world_pose.ndim != 5 or target_world_pose.shape[-2:] != (17, 3):
        raise ValueError(
            "batch['target_human_kp_3d'] must have shape (B,T,S,17,3) for "
            f"tracking pose metrics, got {tuple(target_world_pose.shape)}."
        )

    pred_world_pose = canonical_pose_to_world_pose(
        canonical_pose,
        prediction["position"],
        prediction["rotation"],
    )
    canonical_errors: list[torch.Tensor] = []
    world_errors: list[torch.Tensor] = []
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
            target_world = target_world_pose[batch_index, active, target_index]
            target_canonical = world_pose_to_canonical_pose(
                target_world,
                batch["target_position"][batch_index, active, target_index],
                batch["target_rotation"][batch_index, active, target_index],
            )
            canonical_errors.append(
                torch.linalg.vector_norm(
                    canonical_pose[batch_index, active, query_index]
                    - target_canonical,
                    dim=-1,
                ).reshape(-1)
            )
            world_errors.append(
                torch.linalg.vector_norm(
                    pred_world_pose[batch_index, active, query_index]
                    - target_world,
                    dim=-1,
                ).reshape(-1)
            )
    if canonical_errors:
        all_canonical_errors = torch.cat(canonical_errors)
        all_world_errors = torch.cat(world_errors)
        statistics = {
            "canonical_mpjpe_m": _scalar_statistic(
                all_canonical_errors.sum(), all_canonical_errors.numel()
            ),
            "world_mpjpe_m": _scalar_statistic(
                all_world_errors.sum(), all_world_errors.numel()
            ),
        }
    else:
        statistics = {
            "canonical_mpjpe_m": _zero_statistic(zero),
            "world_mpjpe_m": _zero_statistic(zero),
        }

    if not _REPROJECTION_BATCH_FIELDS.issubset(batch):
        return statistics
    pred_uv, in_front = DifferentiablePinholeProjection()(
        world_points=pred_world_pose,
        camera_R=batch["camera_R"],
        camera_C=batch["camera_C"],
        camera_f=batch["camera_f"],
        camera_cx=batch["camera_cx"],
        camera_cy=batch["camera_cy"],
        camera_w=batch["camera_w"],
        camera_h=batch["camera_h"],
    )
    validate_tracking_projection_shapes(pred_uv, in_front, prediction, batch)
    reprojection_errors_px: list[torch.Tensor] = []
    behind_camera_indicators: list[torch.Tensor] = []
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        image_size = torch.stack(
            (batch["camera_w"][batch_index], batch["camera_h"][batch_index]),
            dim=-1,
        )
        for query_index, target_index in zip(
            query_indices.tolist(), target_indices.tolist(), strict=True
        ):
            target_uv = batch["human_kp_target"][
                batch_index, :, :, target_index
            ]
            target_vis = batch["human_vis_target"][
                batch_index, :, :, target_index
            ]
            predicted_uv = pred_uv[batch_index, :, :, query_index]
            if predicted_uv.shape != target_uv.shape:
                raise ValueError(
                    "Matched pose-metric reprojection tensors must share "
                    f"shape, got {tuple(predicted_uv.shape)} and "
                    f"{tuple(target_uv.shape)}."
                )
            active = batch["target_presence"][
                batch_index, :, target_index
            ]
            valid = (
                (target_vis > 0)
                & (~batch["padding_mask"][batch_index]).unsqueeze(-1)
                & active.unsqueeze(0).unsqueeze(-1)
            )
            error_px = torch.linalg.vector_norm(
                (predicted_uv - target_uv)
                * image_size[:, None, None, :].to(
                    device=predicted_uv.device,
                    dtype=predicted_uv.dtype,
                ),
                dim=-1,
            )
            if valid.any():
                reprojection_errors_px.append(error_px[valid])
                behind_camera_indicators.append(
                    (~in_front[batch_index, :, :, query_index])[valid].to(
                        dtype=zero.dtype
                    )
                )
    if reprojection_errors_px:
        all_reprojection_errors_px = torch.cat(reprojection_errors_px)
        all_behind_camera_indicators = torch.cat(behind_camera_indicators)
        statistics.update(
            {
                "reprojection_error_px": _scalar_statistic(
                    all_reprojection_errors_px.sum(),
                    all_reprojection_errors_px.numel(),
                ),
                "behind_camera_fraction": _scalar_statistic(
                    all_behind_camera_indicators.sum(),
                    all_behind_camera_indicators.numel(),
                ),
            }
        )
    else:
        statistics.update(
            {
                "reprojection_error_px": _zero_statistic(zero),
                "behind_camera_fraction": _zero_statistic(zero),
            }
        )
    return statistics
