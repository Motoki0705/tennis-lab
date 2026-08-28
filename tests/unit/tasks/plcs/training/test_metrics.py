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
    compute_plcs_reference_transform_consistency,
)
from src.utils.schema.court_normalization import denormalize_court_position


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


def _negative_side_provenance():
    contract = resolve_court_keypoint_contract("camera_view_v2")
    view = build_court_view_record(
        camera_id="camera_negative",
        camera_center_court_m=(2.0, -12.0, 5.0),
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


def test_reference_metric_evidence_reports_y_sign_axes_heading_and_local_index() -> None:
    target_position = torch.tensor(
        [[[1.0, 2.0, 0.5]], [[-2.0, -3.0, 1.0]]]
    )
    prediction_position = target_position + torch.tensor([0.1, -0.2, 0.3])
    target_heading = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    prediction_heading = torch.tensor([[[0.0, 1.0]], [[0.0, 1.0]]])

    evidence = compute_plcs_reference_metric_evidence(
        prediction_position,
        prediction_heading,
        target_position,
        target_heading,
        torch.tensor([0, 1], dtype=torch.int64),
    )
    flattened = evidence.to_flat_dict()

    assert flattened["y_sign_accuracy"] == pytest.approx(1.0)
    assert flattened["x_error_m"] == pytest.approx(0.1)
    assert flattened["y_error_m"] == pytest.approx(0.2)
    assert flattened["z_error_m"] == pytest.approx(0.3)
    assert flattened["heading_error_deg"] == pytest.approx(45.0)
    assert flattened["reference_index_0_position_error_m"] == pytest.approx(
        0.3741657
    )
    assert flattened["reference_index_1_position_error_m"] == pytest.approx(
        0.3741657
    )


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
    assert result["heading_error_deg"] == pytest.approx(0.0)


def test_paired_reference_transform_consistency_restores_position_and_heading() -> None:
    negative = _negative_side_provenance()
    positive = _positive_side_provenance()
    first_position = torch.tensor([[1.0, 2.0, 0.5]], dtype=torch.float64)
    second_position = torch.tensor([[-1.0, -2.0, 0.5]], dtype=torch.float64)
    first_heading = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    second_heading = torch.tensor([[-1.0, 0.0]], dtype=torch.float64)

    consistency = compute_plcs_reference_transform_consistency(
        first_position,
        first_heading,
        negative,
        second_position,
        second_heading,
        positive,
    )

    assert consistency.position_error_m == pytest.approx(0.0, abs=1e-12)
    assert consistency.heading_error_radians == pytest.approx(0.0, abs=1e-12)
