"""Court-frame provenance tests for PLCS evaluation metrics."""

from __future__ import annotations

import pytest
import torch

from src.tasks.base.generate_dataset import (
    build_court_view_record,
    build_reference_frame_provenance,
    resolve_court_keypoint_contract,
)
from src.tasks.plcs.training.metrics import (
    PLCSMetrics,
    compute_plcs_reference_metric_evidence,
)
from src.utils.schema.court_normalization import (
    denormalize_court_position,
    normalize_court_position,
)


def _positive_side_provenance():
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


def test_metrics_restore_reference_frame_values_to_physical_court() -> None:
    metrics = PLCSMetrics(
        position_threshold_m=1.0,
        angle_threshold_deg=10.0,
    )

    result = metrics.update(
        pred_position=torch.tensor([[[0.0, 0.0, 0.0]]]),
        pred_rotation=torch.tensor([[[1.0, 0.0]]]),
        target_position=torch.tensor([[[1.0, 0.0, 0.0]]]),
        target_rotation=torch.tensor([[[0.0, 1.0]]]),
        court_reference_provenance=(_positive_side_provenance(),),
    )

    assert result["position_error_m"] == pytest.approx(11.885)
    assert result["x_error_m"] == pytest.approx(11.885)
    assert result["angular_error_deg"] == pytest.approx(90.0)


def test_metrics_reject_provenance_cardinality_mismatch() -> None:
    metrics = PLCSMetrics(
        position_threshold_m=1.0,
        angle_threshold_deg=10.0,
    )
    provenance = _positive_side_provenance()

    with pytest.raises(ValueError, match="cardinality"):
        metrics.update(
            pred_position=torch.zeros(2, 1, 3),
            pred_rotation=torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]]),
            target_position=torch.zeros(2, 1, 3),
            target_rotation=torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]]),
            court_reference_provenance=(provenance, provenance, provenance),
        )


def test_reference_metric_evidence_reports_y_sign_axes_and_local_index() -> None:
    target_position = torch.tensor([[[1.0, 2.0, 0.5]], [[-2.0, -3.0, 1.0]]])
    prediction_position = target_position + torch.tensor([0.1, -0.2, 0.3])

    evidence = compute_plcs_reference_metric_evidence(
        prediction_position,
        target_position,
        torch.tensor([0, 1], dtype=torch.int64),
    )
    flattened = evidence.to_flat_dict()

    assert flattened["y_sign_accuracy"] == pytest.approx(1.0)
    assert flattened["x_error_m"] == pytest.approx(0.1)
    assert flattened["y_error_m"] == pytest.approx(0.2)
    assert flattened["z_error_m"] == pytest.approx(0.3)
    assert "heading_error_deg" not in flattened
    assert flattened["reference_index_0_position_error_m"] == pytest.approx(0.3741657)
    assert flattened["reference_index_1_position_error_m"] == pytest.approx(0.3741657)


def test_reference_metrics_accept_bfloat16_prediction_and_float32_target() -> None:
    target_position = torch.tensor([[[0.0, 0.5, 0.0]]], dtype=torch.float32)
    prediction_position = target_position.to(torch.bfloat16)
    target_heading = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    prediction_heading = target_heading.to(torch.bfloat16)
    prediction_m = denormalize_court_position(prediction_position)
    target_m = denormalize_court_position(target_position)
    assert isinstance(prediction_m, torch.Tensor)
    assert isinstance(target_m, torch.Tensor)
    expected_y_error = float((prediction_m.float() - target_m).abs()[0, 0, 1])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = PLCSMetrics(
            position_threshold_m=1.0,
            angle_threshold_deg=10.0,
        ).update(
            prediction_position,
            prediction_heading,
            target_position,
            target_heading,
            court_reference_provenance=(_positive_side_provenance(),),
            reference_view_index=torch.tensor([0], dtype=torch.int64),
        )

    assert result["x_error_m"] == pytest.approx(0.0)
    assert result["y_error_m"] == pytest.approx(expected_y_error)
    assert result["z_error_m"] == pytest.approx(0.0)
    assert result["y_sign_accuracy"] == pytest.approx(1.0)
    assert "heading_error_deg" not in result


@pytest.mark.parametrize(
    ("position_threshold", "angle_threshold", "expected_keys"),
    [
        (0.5, 15.0, {"position_accuracy_0.5m", "angle_accuracy_15deg"}),
        (0.75, 20.0, {"position_accuracy_0.75m", "angle_accuracy_20deg"}),
    ],
)
def test_accuracy_metrics_encode_thresholds_without_aliases(
    position_threshold: float,
    angle_threshold: float,
    expected_keys: set[str],
) -> None:
    tracker = PLCSMetrics(
        position_threshold_m=position_threshold,
        angle_threshold_deg=angle_threshold,
    )
    tracker.update(
        torch.zeros(1, 1, 3),
        torch.tensor([[[1.0, 0.0]]]),
        torch.zeros(1, 1, 3),
        torch.tensor([[[1.0, 0.0]]]),
    )

    metrics = tracker.compute()

    assert expected_keys <= set(metrics)
    assert "position_accuracy" not in metrics
    assert "angle_accuracy" not in metrics
    if position_threshold == 0.5:
        assert list(metrics).count("position_accuracy_0.5m") == 1
    if angle_threshold == 15.0:
        assert list(metrics).count("angle_accuracy_15deg") == 1


def test_configurable_accuracy_thresholds_cannot_overwrite_fixed_metrics() -> None:
    tracker = PLCSMetrics(
        position_threshold_m=0.5004,
        angle_threshold_deg=15.0004,
    )
    angle = torch.deg2rad(torch.tensor(15.0002))
    prediction_rotation = torch.stack((torch.cos(angle), torch.sin(angle))).reshape(
        1, 1, 2
    )
    tracker.update(
        normalize_court_position(torch.tensor([[[0.5002, 0.0, 0.0]]])),
        prediction_rotation,
        torch.zeros(1, 1, 3),
        torch.tensor([[[1.0, 0.0]]]),
    )

    metrics = tracker.compute()

    assert metrics["position_accuracy_0.5m"] == pytest.approx(0.0)
    assert metrics["position_accuracy_0.5004m"] == pytest.approx(1.0)
    assert metrics["angle_accuracy_15deg"] == pytest.approx(0.0)
    assert metrics["angle_accuracy_15.0004deg"] == pytest.approx(1.0)


def test_compute_rejects_an_epoch_without_valid_positions() -> None:
    tracker = PLCSMetrics(
        position_threshold_m=0.5,
        angle_threshold_deg=15.0,
    )

    with pytest.raises(RuntimeError, match="requires at least one valid position"):
        tracker.compute()

    batch_result = tracker.update(
        torch.ones(1, 2, 3),
        torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
        torch.zeros(1, 2, 3),
        torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
        padding_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    assert batch_result == {}

    with pytest.raises(RuntimeError, match="epoch contained no metric observations"):
        tracker.compute()


def test_mixed_temporal_padding_subsets_reference_samples_and_valid_frames() -> None:
    tracker = PLCSMetrics(
        position_threshold_m=0.5,
        angle_threshold_deg=15.0,
    )
    target_m = torch.tensor(
        [
            [[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, -2.0, 0.0], [0.0, -2.0, 0.0]],
        ]
    )
    prediction_m = target_m + torch.tensor(
        [
            [[1.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
            [[100.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
        ]
    )
    rotation = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )

    batch_result = tracker.update(
        normalize_court_position(prediction_m),
        rotation,
        normalize_court_position(target_m),
        rotation,
        padding_mask=torch.tensor([[False, True], [True, True]]),
        reference_view_index=torch.tensor([1, 0], dtype=torch.int64),
    )
    epoch_result = tracker.compute()

    assert batch_result["position_error_m"] == pytest.approx(1.0)
    assert batch_result["reference_index_1_position_error_m"] == pytest.approx(1.0)
    assert "reference_index_0_position_error_m" not in batch_result
    assert epoch_result["position_error_m"] == pytest.approx(1.0)


def test_frame_profile_padding_excludes_invalid_samples() -> None:
    tracker = PLCSMetrics(
        position_threshold_m=0.5,
        angle_threshold_deg=15.0,
    )
    prediction_m = torch.tensor([[1.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    rotation = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    batch_result = tracker.update(
        normalize_court_position(prediction_m),
        rotation,
        torch.zeros_like(prediction_m),
        rotation,
        padding_mask=torch.tensor([False, True]),
    )

    assert batch_result["position_error_m"] == pytest.approx(1.0)
    assert tracker.compute()["position_error_m"] == pytest.approx(1.0)
