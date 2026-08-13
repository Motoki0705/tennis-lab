from __future__ import annotations

import torch

from src.tasks.base.model_io import ModelCall
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.blcs.model_io import (
    BLCSTrackQueryPrediction,
    BLCSTrackQueryTrainingBatch,
)
from src.tasks.blcs.training.tracking_metrics import blcs_tracking_metrics
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


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
        source_target_position=torch.zeros(1, 1, 1, 3),
        target_velocity=torch.zeros(1, 1, 1, 3),
        source_target_velocity=torch.zeros(1, 1, 1, 3),
        target_presence=torch.ones(1, 1, 1, dtype=torch.bool),
        target_instance_id=torch.zeros(1, 1, 1, dtype=torch.long),
        target_slot_mask=torch.ones(1, 1, dtype=torch.bool),
        frame_mask=torch.ones(1, 1, dtype=torch.bool),
        reference_view_index=torch.zeros(1, dtype=torch.long),
        orientation_sign=torch.ones(1),
    )

    metrics = blcs_tracking_metrics(
        prediction,
        batch,
        [(torch.tensor([0]), torch.tensor([0]))],
        counterfactual_prediction=BLCSTrackQueryPrediction(
            position=prediction.position
            * torch.tensor([1.0, -1.0, 1.0]),
            presence_logits=prediction.presence_logits,
            presence_probability=prediction.presence_probability,
            presence=prediction.presence,
        ),
        counterfactual_assignments=[(torch.tensor([0]), torch.tensor([0]))],
        counterfactual_orientation_sign=torch.tensor([-1.0]),
        config=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
        ),
    )

    scale = torch.tensor(COURT_COORD_SCALE_XYZ)
    torch.testing.assert_close(metrics["position_mae_x_m"], scale[0])
    torch.testing.assert_close(metrics["position_mae_y_m"], 2.0 * scale[1])
    torch.testing.assert_close(metrics["position_mae_z_m"], 3.0 * scale[2])
    torch.testing.assert_close(metrics["reference_consistency_y_m"], torch.tensor(0.0))
    assert "source_frame_position_mae_y_m" in metrics
