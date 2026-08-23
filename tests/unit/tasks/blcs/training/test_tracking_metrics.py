from __future__ import annotations

import pytest
import torch

from src.tasks.base.model_io import ModelCall
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.blcs.model_io import (
    BLCSTrackQueryPrediction,
    BLCSTrackQueryTrainingBatch,
)
from src.tasks.blcs.training.metrics import BLCSMetrics
from src.tasks.blcs.training.tracking_metrics import blcs_tracking_metrics
from src.utils.schema.court import COURT_COORD_SCALE_XYZ
from src.utils.schema.court_normalization import resolve_court_coordinate_normalization


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


def test_standard_and_tracking_metrics_report_v2_errors_in_meters() -> None:
    contract = resolve_court_coordinate_normalization("v2")
    physical_error = torch.tensor([1.0, 2.0, 3.0])
    normalized_error = physical_error / torch.tensor(contract.scale_xyz)

    standard = BLCSMetrics(
        position_threshold_m=0.3,
        endpoint_threshold_m=0.5,
        normalization=contract,
    ).update(
        normalized_error.view(1, 1, 3),
        torch.zeros(1, 1, 3),
    )
    assert standard["x_error_m"] == pytest.approx(1.0)
    assert standard["y_error_m"] == pytest.approx(2.0)
    assert standard["z_error_m"] == pytest.approx(3.0)

    logits = torch.tensor([[[20.0]]])
    prediction = BLCSTrackQueryPrediction(
        position=normalized_error.view(1, 1, 1, 3),
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
    tracking = blcs_tracking_metrics(
        prediction,
        batch,
        [(torch.tensor([0]), torch.tensor([0]))],
        config=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
        ),
        normalization=contract,
    )
    torch.testing.assert_close(
        torch.stack(
            [
                tracking["position_mae_x_m"],
                tracking["position_mae_y_m"],
                tracking["position_mae_z_m"],
            ]
        ),
        physical_error,
    )
