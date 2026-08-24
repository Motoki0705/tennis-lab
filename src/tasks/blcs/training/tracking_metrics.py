"""Lifecycle-aware localization and identity diagnostics for ball tracks."""

from __future__ import annotations

import torch

from src.tasks.base.evaluation import (
    compute_axis_wise_position_error,
    compute_y_sign_accuracy,
    stratify_metric_by_reference_view_index,
)
from src.tasks.base.generate_dataset import (
    PHYSICAL_V1_SELECTOR,
    CourtKeypointContract,
    CourtKeypointContractMismatchError,
    CourtReferenceFrameProvenance,
    MissingCourtKeypointMetadataError,
    court_vectors_target_to_physical,
    resolve_court_keypoint_contract,
)
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

_PHYSICAL_COURT_KEYPOINT_CONTRACT = resolve_court_keypoint_contract(
    PHYSICAL_V1_SELECTOR
)


def _validate_court_provenance(
    provenance: tuple[CourtReferenceFrameProvenance, ...],
    *,
    batch_size: int,
    court_keypoint_contract: CourtKeypointContract,
) -> None:
    if not provenance:
        if court_keypoint_contract.selector != PHYSICAL_V1_SELECTOR:
            raise MissingCourtKeypointMetadataError(
                "BLCS camera_view_v2 tracking metrics require explicit Court "
                "reference provenance."
            )
        return
    if len(provenance) != batch_size:
        raise ValueError(
            "BLCS tracking metric provenance must contain one record per batch item."
        )
    for index, record in enumerate(provenance):
        if not isinstance(record, CourtReferenceFrameProvenance):
            raise TypeError(
                "BLCS tracking metric provenance entries must be validated records."
            )
        if record.contract != court_keypoint_contract:
            raise CourtKeypointContractMismatchError(
                f"BLCS tracking metric provenance[{index}] contract "
                f"{record.contract_id!r} does not match runtime "
                f"{court_keypoint_contract.contract_id!r}."
            )


def _position_metrics_meters(
    prediction: BLCSTrackQueryPrediction,
    batch: BLCSTrackQueryTrainingBatch,
    assignments: list[Assignment],
    reference_view_index: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Return matched physical distance and reference-target-frame diagnostics."""
    pred_position = prediction.position
    scale = pred_position.new_tensor(COURT_COORD_SCALE_XYZ)
    physical_error_terms: list[torch.Tensor] = []
    physical_axis_terms: list[torch.Tensor] = []
    reference_predictions: list[torch.Tensor] = []
    reference_targets: list[torch.Tensor] = []
    per_sample_reference_errors: dict[int, list[torch.Tensor]] = {}
    if reference_view_index is not None:
        if reference_view_index.shape != (pred_position.shape[0],):
            raise ValueError(
                "reference_view_index must match the BLCS tracking batch axis."
            )
        if reference_view_index.dtype != torch.int64:
            raise ValueError("reference_view_index must have dtype torch.int64.")
        if reference_view_index.device != pred_position.device:
            raise ValueError(
                "reference_view_index must share the BLCS tracking tensor device."
            )
    for batch_index, (query_indices, target_indices) in enumerate(assignments):
        for query_index, target_index in zip(
            query_indices.tolist(), target_indices.tolist(), strict=True
        ):
            active = (
                batch.target_presence[batch_index, :, target_index]
                & batch.frame_valid[batch_index]
            )
            if active.any():
                target_frame_pred_m = (
                    pred_position[batch_index, active, query_index] * scale
                )
                target_frame_target_m = (
                    batch.target_position[batch_index, active, target_index] * scale
                )
                reference_predictions.append(target_frame_pred_m)
                reference_targets.append(target_frame_target_m)
                reference_error = target_frame_pred_m - target_frame_target_m
                per_sample_reference_errors.setdefault(batch_index, []).append(
                    torch.linalg.vector_norm(reference_error, dim=-1)
                )
                error_m = reference_error
                if batch.court_reference_provenance:
                    error_m = court_vectors_target_to_physical(
                        error_m,
                        batch.court_reference_provenance[batch_index],
                    )
                    if not isinstance(error_m, torch.Tensor):
                        raise TypeError(
                            "BLCS tracking metric frame conversion returned a non-tensor."
                        )
                physical_axis_terms.append(error_m.abs())
                physical_error_terms.append(torch.linalg.vector_norm(error_m, dim=-1))
    zero = pred_position.new_zeros(())
    if physical_error_terms:
        physical_errors = torch.cat(physical_error_terms)
        physical_axis_errors = torch.cat(physical_axis_terms)
        aggregate = physical_errors.mean()
        axes = physical_axis_errors.mean(dim=0)
    else:
        aggregate = zero
        axes = pred_position.new_zeros(3)
    metrics = {
        "position_error_m": aggregate,
        "position_mae_x_m": axes[0],
        "position_mae_y_m": axes[1],
        "position_mae_z_m": axes[2],
        "x_error_m": axes[0],
        "y_error_m": axes[1],
        "z_error_m": axes[2],
    }
    if reference_view_index is not None and reference_predictions:
        all_reference_predictions = torch.cat(reference_predictions)
        all_reference_targets = torch.cat(reference_targets)
        axis_error = compute_axis_wise_position_error(
            all_reference_predictions,
            all_reference_targets,
        )
        metrics.update(
            {
                "position_mae_x_m": zero.new_tensor(axis_error.x),
                "position_mae_y_m": zero.new_tensor(axis_error.y),
                "position_mae_z_m": zero.new_tensor(axis_error.z),
                "x_error_m": zero.new_tensor(axis_error.x),
                "y_error_m": zero.new_tensor(axis_error.y),
                "z_error_m": zero.new_tensor(axis_error.z),
            }
        )
        if all_reference_targets[:, 1].ne(0).any().item():
            metrics["y_sign_accuracy"] = zero.new_tensor(
                compute_y_sign_accuracy(
                    all_reference_predictions,
                    all_reference_targets,
                )
            )
        sample_indices = sorted(per_sample_reference_errors)
        sample_errors = torch.stack(
            [
                torch.cat(per_sample_reference_errors[index]).mean()
                for index in sample_indices
            ]
        )
        sample_index_tensor = torch.tensor(
            sample_indices,
            dtype=torch.int64,
            device=reference_view_index.device,
        )
        strata = stratify_metric_by_reference_view_index(
            sample_errors,
            reference_view_index[sample_index_tensor],
        )
        metrics.update(
            {
                f"reference_index_{index}_position_error_m": zero.new_tensor(value)
                for index, value in strata.items()
            }
        )
    return metrics


def blcs_tracking_metrics(
    prediction: BLCSTrackQueryPrediction,
    batch: BLCSTrackQueryTrainingBatch,
    assignments: list[Assignment],
    *,
    config: TrackingMetricConfig,
    court_keypoint_contract: CourtKeypointContract = (
        _PHYSICAL_COURT_KEYPOINT_CONTRACT
    ),
    reference_view_index: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute shared lifecycle metrics for BLCS predictions."""
    canonical_court_contract = resolve_court_keypoint_contract(
        court_keypoint_contract.selector
    )
    if court_keypoint_contract != canonical_court_contract:
        raise CourtKeypointContractMismatchError(
            "BLCS tracking metric CourtKP20 contract must be canonical."
        )
    _validate_court_provenance(
        batch.court_reference_provenance,
        batch_size=int(prediction.position.shape[0]),
        court_keypoint_contract=canonical_court_contract,
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
    metrics.update(
        _position_metrics_meters(
            prediction,
            batch,
            assignments,
            reference_view_index,
        )
    )
    return metrics
