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
                & batch.frame_valid[batch_index]
            )
            if active.any():
                error_m = (
                    pred_position[batch_index, active, query_index]
                    - batch.target_position[batch_index, active, target_index]
                ) * scale
                if batch.court_reference_provenance:
                    error_m = court_vectors_target_to_physical(
                        error_m,
                        batch.court_reference_provenance[batch_index],
                    )
                    if not isinstance(error_m, torch.Tensor):
                        raise TypeError(
                            "BLCS tracking metric frame conversion returned a non-tensor."
                        )
                terms.append(error_m.abs().mean(0))
    if terms:
        return torch.stack(terms).mean(0)
    return pred_position.new_zeros(3)


def blcs_tracking_metrics(
    prediction: BLCSTrackQueryPrediction,
    batch: BLCSTrackQueryTrainingBatch,
    assignments: list[Assignment],
    *,
    config: TrackingMetricConfig,
    court_keypoint_contract: CourtKeypointContract = (
        _PHYSICAL_COURT_KEYPOINT_CONTRACT
    ),
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
    position_mae_m = _position_mae_meters(prediction, batch, assignments)
    metrics.update(
        {
            "position_mae_x_m": position_mae_m[0],
            "position_mae_y_m": position_mae_m[1],
            "position_mae_z_m": position_mae_m[2],
        }
    )
    return metrics
