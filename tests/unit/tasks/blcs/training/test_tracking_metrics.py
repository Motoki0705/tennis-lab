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
from src.tasks.base.training.metric_logging import MetricStatisticsAccumulator
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.blcs.model_io import (
    BLCSTrackQueryPrediction,
    BLCSTrackQueryTrainingBatch,
)
from src.tasks.blcs.training.tracking_metrics import (
    blcs_tracking_metrics,
    blcs_tracking_statistics,
)
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
            id_switch_distance=0.05,
        ),
    )

    scale = torch.tensor(COURT_COORD_SCALE_XYZ)
    torch.testing.assert_close(metrics["x_error_m"], scale[0])
    torch.testing.assert_close(metrics["y_error_m"], 2.0 * scale[1])
    torch.testing.assert_close(metrics["z_error_m"], 3.0 * scale[2])
    assert not set(metrics).intersection(
        {"position_mae_x_m", "position_mae_y_m", "position_mae_z_m"}
    )


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
        id_switch_distance=0.05,
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
            id_switch_distance=0.05,
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


def _tracking_stat_fixture(
    errors_x: list[list[float]],
    target_y: list[list[float]],
    prediction_y: list[list[float]],
    frame_valid: list[list[bool]],
) -> tuple[
    BLCSTrackQueryPrediction,
    BLCSTrackQueryTrainingBatch,
    list[tuple[torch.Tensor, torch.Tensor]],
]:
    batch_size = len(errors_x)
    sequence_length = len(errors_x[0])
    prediction_position = torch.zeros(batch_size, sequence_length, 1, 3)
    target_position = torch.zeros_like(prediction_position)
    prediction_position[:, :, 0, 0] = torch.tensor(errors_x)
    prediction_position[:, :, 0, 1] = torch.tensor(prediction_y)
    target_position[:, :, 0, 1] = torch.tensor(target_y)
    logits = torch.full((batch_size, sequence_length, 1), 20.0)
    prediction = BLCSTrackQueryPrediction(
        position=prediction_position,
        presence_logits=logits,
        presence_probability=logits.sigmoid(),
        presence=torch.ones_like(logits, dtype=torch.bool),
    )
    batch = BLCSTrackQueryTrainingBatch(
        call=ModelCall(kwargs={}),
        target_position=target_position,
        target_velocity=torch.zeros_like(target_position),
        target_presence=torch.ones(batch_size, sequence_length, 1, dtype=torch.bool),
        target_instance_id=torch.zeros(
            batch_size, sequence_length, 1, dtype=torch.long
        ),
        target_slot_mask=torch.ones(batch_size, 1, dtype=torch.bool),
        frame_valid=torch.tensor(frame_valid, dtype=torch.bool),
    )
    assignments = [
        (torch.tensor([0]), torch.tensor([0])) for _ in range(batch_size)
    ]
    return prediction, batch, assignments


def test_tracking_statistics_are_partition_and_padding_invariant() -> None:
    full_prediction, full_batch, full_assignments = _tracking_stat_fixture(
        [[1.0, 3.0], [100.0, 200.0]],
        [[2.0, 2.0], [0.0, 0.0]],
        [[3.0, 1.0], [0.0, 0.0]],
        [[True, True], [False, False]],
    )
    one_prediction, one_batch, one_assignments = _tracking_stat_fixture(
        [[1.0, 3.0]],
        [[2.0, 2.0]],
        [[3.0, 1.0]],
        [[True, True]],
    )
    full = blcs_tracking_statistics(
        full_prediction,
        full_batch,
        full_assignments,
        config=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
            id_switch_distance=0.05,
        ),
        reference_view_index=torch.tensor([0, 1], dtype=torch.int64),
    )
    split = MetricStatisticsAccumulator()
    split.update(
        blcs_tracking_statistics(
            one_prediction,
            one_batch,
            one_assignments,
            config=TrackingMetricConfig(
                presence_threshold=0.5,
                duplicate_distance=0.05,
                id_switch_distance=0.05,
            ),
            reference_view_index=torch.tensor([0], dtype=torch.int64),
        )
    )
    padding_prediction, padding_batch, padding_assignments = _tracking_stat_fixture(
        [[100.0, 200.0]],
        [[0.0, 0.0]],
        [[0.0, 0.0]],
        [[False, False]],
    )
    split.update(
        blcs_tracking_statistics(
            padding_prediction,
            padding_batch,
            padding_assignments,
            config=TrackingMetricConfig(
                presence_threshold=0.5,
                duplicate_distance=0.05,
                id_switch_distance=0.05,
            ),
            reference_view_index=torch.tensor([1], dtype=torch.int64),
        )
    )
    full_reduced = MetricStatisticsAccumulator()
    full_reduced.update(full)
    assert split.compute() == pytest.approx(full_reduced.compute())

    scale_x = float(COURT_COORD_SCALE_XYZ[0])
    assert full["x_error_m"].numerator.item() == pytest.approx(4.0 * scale_x)
    assert full["x_error_m"].denominator.item() == pytest.approx(2.0)
    assert full["reference_index_0_position_error_m"].denominator.item() == 1.0
    assert "reference_index_1_position_error_m" not in full

    padded_metrics = blcs_tracking_metrics(
        padding_prediction,
        padding_batch,
        padding_assignments,
        config=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
            id_switch_distance=0.05,
        ),
        reference_view_index=torch.tensor([1], dtype=torch.int64),
    )
    assert all(value.item() == 0.0 for value in padded_metrics.values())


def test_tracking_statistics_keep_dynamic_strata_sample_weighting() -> None:
    prediction, batch, assignments = _tracking_stat_fixture(
        [[1.0, 3.0], [5.0, 1.0]],
        [[2.0, 2.0], [-2.0, -2.0]],
        [[2.0, 2.0], [-2.0, -2.0]],
        [[True, True], [True, True]],
    )
    statistics = blcs_tracking_statistics(
        prediction,
        batch,
        assignments,
        config=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
            id_switch_distance=0.05,
        ),
        reference_view_index=torch.tensor([0, 1], dtype=torch.int64),
    )

    scale_x = float(COURT_COORD_SCALE_XYZ[0])
    assert statistics["reference_index_0_position_error_m"].numerator.item() == (
        pytest.approx(2.0 * scale_x)
    )
    assert statistics["reference_index_1_position_error_m"].numerator.item() == (
        pytest.approx(3.0 * scale_x)
    )
    assert statistics["reference_index_0_position_error_m"].denominator.item() == 1.0
    assert statistics["reference_index_1_position_error_m"].denominator.item() == 1.0
    assert statistics["y_sign_accuracy"].numerator.item() == 4.0
    assert statistics["y_sign_accuracy"].denominator.item() == 4.0


def test_physical_errors_and_dynamic_strata_match_b2_and_b1_plus_b1() -> None:
    combined_prediction, combined_batch, combined_assignments = (
        _tracking_stat_fixture(
            [[1.0, 3.0], [5.0, 100.0]],
            [[2.0, 2.0], [-2.0, -2.0]],
            [[2.0, 2.0], [-2.0, -2.0]],
            [[True, True], [True, False]],
        )
    )
    combined = MetricStatisticsAccumulator()
    combined.update(
        blcs_tracking_statistics(
            combined_prediction,
            combined_batch,
            combined_assignments,
            config=TrackingMetricConfig(
                presence_threshold=0.5,
                duplicate_distance=0.05,
                id_switch_distance=0.05,
            ),
            reference_view_index=torch.tensor([0, 0], dtype=torch.int64),
        )
    )
    split = MetricStatisticsAccumulator()
    for errors, target_y, valid in (
        ([1.0, 3.0], [2.0, 2.0], [True, True]),
        ([5.0, 100.0], [-2.0, -2.0], [True, False]),
    ):
        prediction, batch, assignments = _tracking_stat_fixture(
            [errors],
            [target_y],
            [target_y],
            [valid],
        )
        split.update(
            blcs_tracking_statistics(
                prediction,
                batch,
                assignments,
                config=TrackingMetricConfig(
                    presence_threshold=0.5,
                    duplicate_distance=0.05,
                    id_switch_distance=0.05,
                ),
                reference_view_index=torch.tensor([0], dtype=torch.int64),
            )
        )

    assert split.compute() == pytest.approx(combined.compute())
    scale_x = float(COURT_COORD_SCALE_XYZ[0])
    assert combined.compute()["x_error_m"] == pytest.approx(3.0 * scale_x)
    assert combined.compute()["reference_index_0_position_error_m"] == (
        pytest.approx(3.5 * scale_x)
    )
