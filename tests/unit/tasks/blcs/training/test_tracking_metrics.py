from __future__ import annotations

import pytest
import torch

from src.tasks.base.generate_dataset import (
    CourtKeypointContractMismatchError,
    CourtReferenceFrameProvenance,
    MissingCourtKeypointMetadataError,
    build_court_view_record,
    build_physical_court_provenance,
    build_reference_frame_provenance,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import ModelCall
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.blcs.model_io import (
    BLCSTrackQueryPrediction,
    BLCSTrackQueryTrainingBatch,
)
from src.tasks.blcs.training.tracking_metrics import blcs_tracking_metrics
from src.utils.schema.court import COURT_COORD_SCALE_XYZ
from src.utils.schema.court_normalization import normalize_court_position


def _positive_side_provenance() -> CourtReferenceFrameProvenance:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    view = build_court_view_record(
        camera_id="camera_positive",
        camera_center_court_m=(2.0, 12.0, 5.0),
        contract=contract,
    )
    return build_reference_frame_provenance(
        (view,),
        reference_camera_id=view.camera_id,
    )


def test_tracking_metrics_report_per_axis_physical_mae() -> None:
    logits = torch.tensor([[[20.0]]])
    prediction = BLCSTrackQueryPrediction(
        position=torch.tensor([[[[1.0, 2.0, 3.0]]]]),
        presence_logits=logits,
        presence_probability=logits.sigmoid(),
        presence=torch.ones(1, 1, 1, dtype=torch.bool),
    )
    batch = BLCSTrackQueryTrainingBatch(
        call=ModelCall(kwargs={}),
        target_position=torch.zeros(1, 1, 1, 3),
        target_velocity=torch.zeros(1, 1, 1, 3),
        target_presence=torch.ones(1, 1, 1, dtype=torch.bool),
        target_instance_id=torch.zeros(1, 1, 1, dtype=torch.long),
        target_slot_mask=torch.ones(1, 1, dtype=torch.bool),
        frame_valid=torch.ones(1, 1, dtype=torch.bool),
    )

    metrics = blcs_tracking_metrics(
        prediction,
        batch,
        [(torch.tensor([0]), torch.tensor([0]))],
        config=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
        ),
    )

    scale = torch.tensor(COURT_COORD_SCALE_XYZ)
    torch.testing.assert_close(metrics["position_mae_x_m"], scale[0])
    torch.testing.assert_close(metrics["position_mae_y_m"], 2.0 * scale[1])
    torch.testing.assert_close(metrics["position_mae_z_m"], 3.0 * scale[2])


def test_v2_tracking_metrics_reject_missing_and_mismatched_provenance() -> None:
    logits = torch.tensor([[[20.0]]])
    prediction = BLCSTrackQueryPrediction(
        position=torch.zeros(1, 1, 1, 3),
        presence_logits=logits,
        presence_probability=logits.sigmoid(),
        presence=torch.ones(1, 1, 1, dtype=torch.bool),
    )
    batch = BLCSTrackQueryTrainingBatch(
        call=ModelCall(kwargs={}),
        target_position=torch.zeros(1, 1, 1, 3),
        target_velocity=torch.zeros(1, 1, 1, 3),
        target_presence=torch.ones(1, 1, 1, dtype=torch.bool),
        target_instance_id=torch.zeros(1, 1, 1, dtype=torch.long),
        target_slot_mask=torch.ones(1, 1, dtype=torch.bool),
        frame_valid=torch.ones(1, 1, dtype=torch.bool),
    )
    assignments = [(torch.tensor([0]), torch.tensor([0]))]
    config = TrackingMetricConfig(
        presence_threshold=0.5,
        duplicate_distance=0.05,
    )
    contract = resolve_court_keypoint_contract("camera_view_v2")

    with pytest.raises(MissingCourtKeypointMetadataError):
        blcs_tracking_metrics(
            prediction,
            batch,
            assignments,
            config=config,
            court_keypoint_contract=contract,
        )

    mismatched_batch = BLCSTrackQueryTrainingBatch(
        call=batch.call,
        target_position=batch.target_position,
        target_velocity=batch.target_velocity,
        target_presence=batch.target_presence,
        target_instance_id=batch.target_instance_id,
        target_slot_mask=batch.target_slot_mask,
        frame_valid=batch.frame_valid,
        court_reference_provenance=(build_physical_court_provenance(),),
    )
    with pytest.raises(CourtKeypointContractMismatchError, match="does not match"):
        blcs_tracking_metrics(
            prediction,
            mismatched_batch,
            assignments,
            config=config,
            court_keypoint_contract=contract,
        )


def test_tracking_reference_metrics_report_target_axes_y_sign_and_local_strata() -> None:
    target_m = torch.tensor(
        [
            [[[1.0, 2.0, 0.5]], [[1.0, -2.0, 0.5]]],
            [[[2.0, 4.0, 1.0]], [[2.0, -4.0, 1.0]]],
        ]
    )
    prediction_m = target_m + torch.tensor(
        [
            [[[1.0, 1.0, 3.0]], [[1.0, 3.0, 3.0]]],
            [[[2.0, -1.0, 0.0]], [[2.0, -1.0, 0.0]]],
        ]
    )
    normalized_target = normalize_court_position(target_m)
    normalized_prediction = normalize_court_position(prediction_m)
    assert isinstance(normalized_target, torch.Tensor)
    assert isinstance(normalized_prediction, torch.Tensor)
    logits = torch.full((2, 2, 1), 20.0)
    prediction = BLCSTrackQueryPrediction(
        position=normalized_prediction,
        presence_logits=logits,
        presence_probability=logits.sigmoid(),
        presence=torch.ones(2, 2, 1, dtype=torch.bool),
    )
    provenance = _positive_side_provenance()
    batch = BLCSTrackQueryTrainingBatch(
        call=ModelCall(kwargs={}),
        target_position=normalized_target,
        target_velocity=torch.zeros_like(normalized_target),
        target_presence=torch.ones(2, 2, 1, dtype=torch.bool),
        target_instance_id=torch.zeros(2, 2, 1, dtype=torch.long),
        target_slot_mask=torch.ones(2, 1, dtype=torch.bool),
        frame_valid=torch.ones(2, 2, dtype=torch.bool),
        court_reference_provenance=(provenance, provenance),
    )

    metrics = blcs_tracking_metrics(
        prediction,
        batch,
        [
            (torch.tensor([0]), torch.tensor([0])),
            (torch.tensor([0]), torch.tensor([0])),
        ],
        config=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
        ),
        court_keypoint_contract=provenance.contract,
        reference_view_index=torch.tensor([1, 0], dtype=torch.int64),
    )

    torch.testing.assert_close(metrics["x_error_m"], torch.tensor(1.5))
    torch.testing.assert_close(metrics["y_error_m"], torch.tensor(1.5))
    torch.testing.assert_close(metrics["z_error_m"], torch.tensor(1.5))
    torch.testing.assert_close(metrics["y_sign_accuracy"], torch.tensor(0.75))
    assert set(metrics) >= {
        "reference_index_0_position_error_m",
        "reference_index_1_position_error_m",
    }
