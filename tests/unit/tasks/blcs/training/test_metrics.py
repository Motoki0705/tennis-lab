"""Court-frame contract tests for standard BLCS evaluation metrics."""

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
from src.tasks.blcs.model_io import (
    BLCSTrajectoryPrediction,
    blcs_trajectory_prediction_to_physical,
)
from src.tasks.blcs.training.metrics import BLCSMetrics
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


def test_physical_v1_metrics_use_current_fixed_isotropic_scale() -> None:
    metrics = BLCSMetrics(
        position_threshold_m=0.3,
        endpoint_threshold_m=0.5,
    )

    result = metrics.update(
        pred_position=torch.tensor([[[1.0, 2.0, 3.0]]]),
        target_position=torch.zeros(1, 1, 3),
    )

    assert result["x_error_m"] == pytest.approx(11.885)
    assert result["y_error_m"] == pytest.approx(23.77)
    assert result["z_error_m"] == pytest.approx(35.655)
    assert result["position_error_m"] == pytest.approx(
        (11.885**2 + 23.77**2 + 35.655**2) ** 0.5
    )


def test_camera_view_v2_restores_position_and_velocity_before_physical_metrics() -> (
    None
):
    provenance = _positive_side_provenance()
    target_position_m = torch.tensor([[[2.0, -3.0, 1.0]]])
    target_velocity_mps = torch.tensor([[[-4.0, 5.0, 0.5]]])
    prediction = BLCSTrajectoryPrediction(
        position=target_position_m,
        velocity=target_velocity_mps,
        court_reference_provenance=(provenance,),
        coordinates_in_metres=True,
    )

    physical = blcs_trajectory_prediction_to_physical(prediction)

    torch.testing.assert_close(
        physical.position,
        torch.tensor([[[-2.0, 3.0, 1.0]]]),
    )
    assert physical.velocity is not None
    torch.testing.assert_close(
        physical.velocity,
        torch.tensor([[[4.0, -5.0, 0.5]]]),
    )
    assert physical.court_reference_provenance == (build_physical_court_provenance(),)

    metric_result = BLCSMetrics(
        position_threshold_m=0.3,
        endpoint_threshold_m=0.5,
        court_keypoint_contract=provenance.contract,
    ).update(
        normalize_court_position(target_position_m),
        torch.zeros_like(target_position_m),
        court_reference_provenance=(provenance,),
    )
    assert metric_result["x_error_m"] == pytest.approx(2.0)
    assert metric_result["y_error_m"] == pytest.approx(3.0)
    assert metric_result["z_error_m"] == pytest.approx(1.0)


def test_camera_view_v2_metrics_reject_missing_or_mismatched_provenance() -> None:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    metrics = BLCSMetrics(
        position_threshold_m=0.3,
        endpoint_threshold_m=0.5,
        court_keypoint_contract=contract,
    )
    values = torch.zeros(1, 1, 3)

    with pytest.raises(MissingCourtKeypointMetadataError, match="require explicit"):
        metrics.update(values, values)

    with pytest.raises(CourtKeypointContractMismatchError, match="does not match"):
        metrics.update(
            values,
            values,
            court_reference_provenance=(build_physical_court_provenance(),),
        )

    with pytest.raises(ValueError, match="one record per batch item"):
        metrics.update(
            values,
            values,
            court_reference_provenance=(
                _positive_side_provenance(),
                _positive_side_provenance(),
            ),
        )


@pytest.mark.parametrize("court_selector", ["physical_v1", "camera_view_v2"])
def test_court_contract_is_independent_of_fixed_coordinate_normalization(
    court_selector: str,
) -> None:
    court_contract = resolve_court_keypoint_contract(court_selector)
    error_m = torch.tensor([[[0.25, 0.5, 0.75]]])
    provenance = (
        None if court_selector == "physical_v1" else (_positive_side_provenance(),)
    )

    result = BLCSMetrics(
        position_threshold_m=1.0,
        endpoint_threshold_m=1.0,
        court_keypoint_contract=court_contract,
    ).update(
        normalize_court_position(error_m),
        torch.zeros_like(error_m),
        court_reference_provenance=provenance,
    )

    assert result["x_error_m"] == pytest.approx(0.25)
    assert result["y_error_m"] == pytest.approx(0.5)
    assert result["z_error_m"] == pytest.approx(0.75)
