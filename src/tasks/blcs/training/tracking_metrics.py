"""Lifecycle-aware localization and identity diagnostics for ball tracks."""

from __future__ import annotations

import torch

from src.tasks.base.generate_dataset import (
    PHYSICAL_V1_SELECTOR,
    CourtKeypointContract,
    CourtKeypointContractMismatchError,
    CourtReferenceFrameProvenance,
    MissingCourtKeypointMetadataError,
    court_vectors_target_to_physical,
    resolve_court_keypoint_contract,
)
from src.tasks.base.training.metric_logging import (
    ScalarMetricStatistic,
    compute_scalar_metric_statistics,
)
from src.tasks.base.training.tracking_metrics import (
    TrackingMetricConfig,
    common_lifecycle_tracking_statistics,
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


def _position_metric_statistics(
    prediction: BLCSTrackQueryPrediction,
    batch: BLCSTrackQueryTrainingBatch,
    assignments: list[Assignment],
    reference_view_index: torch.Tensor | None,
) -> dict[str, ScalarMetricStatistic]:
    """Return additive statistics for matched physical tracking diagnostics."""
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
        if (reference_view_index < 0).any().item():
            raise ValueError(
                "reference_view_index cannot contain padding or negative values."
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
        matched_frame_count = zero.new_tensor(float(physical_errors.numel()))
        position_error_sum = physical_errors.sum()
        axis_error_sum = physical_axis_errors.sum(dim=0)
    else:
        matched_frame_count = zero
        position_error_sum = zero
        axis_error_sum = pred_position.new_zeros(3)
    metrics: dict[str, ScalarMetricStatistic] = {
        "position_error_m": ScalarMetricStatistic(
            position_error_sum,
            matched_frame_count,
        ),
        "x_error_m": ScalarMetricStatistic(
            axis_error_sum[0],
            matched_frame_count,
        ),
        "y_error_m": ScalarMetricStatistic(
            axis_error_sum[1],
            matched_frame_count,
        ),
        "z_error_m": ScalarMetricStatistic(
            axis_error_sum[2],
            matched_frame_count,
        ),
    }
    if reference_view_index is not None:
        reference_numerator = zero
        reference_denominator = zero
        if reference_predictions:
            all_reference_predictions = torch.cat(reference_predictions)
            all_reference_targets = torch.cat(reference_targets)
            eligible = all_reference_targets[:, 1].ne(0)
            if eligible.any().item():
                reference_denominator = eligible.sum().to(dtype=zero.dtype)
                reference_numerator = torch.sign(
                    all_reference_predictions[eligible, 1]
                ).eq(torch.sign(all_reference_targets[eligible, 1])).sum().to(
                    dtype=zero.dtype
                )
        metrics["y_sign_accuracy"] = ScalarMetricStatistic(
            reference_numerator,
            reference_denominator,
        )

        valid_sample_errors: dict[int, torch.Tensor] = {}
        for sample_index, sample_terms in per_sample_reference_errors.items():
            valid_sample_errors[sample_index] = torch.cat(sample_terms).mean()
        for reference_index in sorted(
            {
                int(reference_view_index[sample_index].item())
                for sample_index in valid_sample_errors
            }
        ):
            sample_errors = torch.stack(
                [
                    sample_error
                    for sample_index, sample_error in valid_sample_errors.items()
                    if int(reference_view_index[sample_index].item())
                    == reference_index
                ]
            )
            metrics[
                f"reference_index_{reference_index}_position_error_m"
            ] = ScalarMetricStatistic(
                sample_errors.sum(),
                zero.new_tensor(float(sample_errors.numel())),
            )
    return metrics


def blcs_tracking_statistics(
    prediction: BLCSTrackQueryPrediction,
    batch: BLCSTrackQueryTrainingBatch,
    assignments: list[Assignment],
    *,
    config: TrackingMetricConfig,
    court_keypoint_contract: CourtKeypointContract = (
        _PHYSICAL_COURT_KEYPOINT_CONTRACT
    ),
    reference_view_index: torch.Tensor | None = None,
) -> dict[str, ScalarMetricStatistic]:
    """Compute additive lifecycle and physical metrics for BLCS predictions."""
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
    metrics = common_lifecycle_tracking_statistics(
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
        _position_metric_statistics(
            prediction,
            batch,
            assignments,
            reference_view_index,
        )
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
    """Compute batch-local BLCS metrics with explicit zero-ratio fallback."""
    return compute_scalar_metric_statistics(
        blcs_tracking_statistics(
            prediction,
            batch,
            assignments,
            config=config,
            court_keypoint_contract=court_keypoint_contract,
            reference_view_index=reference_view_index,
        ),
        zero_denominator_value=0.0,
    )


__all__ = ["blcs_tracking_metrics", "blcs_tracking_statistics"]
