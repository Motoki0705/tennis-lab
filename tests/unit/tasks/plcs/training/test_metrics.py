"""Court-frame provenance tests for PLCS evaluation metrics."""

from __future__ import annotations

import pytest
import torch

from src.tasks.base.generate_dataset import (
    build_court_view_record,
    build_reference_frame_provenance,
    resolve_court_keypoint_contract,
)
from src.tasks.plcs.training.metrics import PLCSMetrics


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
