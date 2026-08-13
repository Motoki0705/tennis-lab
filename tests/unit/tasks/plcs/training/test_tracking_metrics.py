from __future__ import annotations

import torch

from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.plcs.training.tracking_metrics import plcs_tracking_metrics


def test_tracking_metrics_report_orientation_and_reference_consistency() -> None:
    source_position = torch.tensor([[[[0.0, 0.25, 0.0]]]])
    target_position = source_position.clone()
    target_position[..., 1] *= -1
    source_rotation = torch.tensor([[[[0.6, 0.8]]]])
    target_rotation = torch.tensor([[[[0.6, -0.8]]]])
    prediction = {
        "position": target_position.clone(),
        "rotation": target_rotation.clone(),
        "presence_logits": torch.full((1, 1, 1), 20.0),
    }
    batch = {
        "target_position": target_position,
        "source_target_position": source_position,
        "target_rotation": target_rotation,
        "source_target_rotation": source_rotation,
        "target_presence": torch.ones(1, 1, 1, dtype=torch.bool),
        "target_instance_id": torch.zeros(1, 1, 1, dtype=torch.long),
        "frame_mask": torch.ones(1, 1, dtype=torch.bool),
        "orientation_sign": torch.tensor([-1.0]),
    }

    metrics = plcs_tracking_metrics(
        prediction,
        batch,
        [(torch.tensor([0]), torch.tensor([0]))],
        counterfactual_prediction={
            "position": source_position.clone(),
            "rotation": source_rotation.clone(),
            "presence_logits": prediction["presence_logits"],
        },
        counterfactual_assignments=[(torch.tensor([0]), torch.tensor([0]))],
        counterfactual_orientation_sign=torch.tensor([1.0]),
        config=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
        ),
    )

    torch.testing.assert_close(metrics["position_mae_y_m"], torch.tensor(0.0))
    torch.testing.assert_close(metrics["y_sign_accuracy"], torch.tensor(1.0))
    torch.testing.assert_close(
        metrics["reference_consistency_y_m"], torch.tensor(0.0)
    )
    torch.testing.assert_close(metrics["angular_error_deg"], torch.tensor(0.0))
    torch.testing.assert_close(
        metrics["reference_consistency_heading_deg"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        metrics["source_frame_position_mae_y_m"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        metrics["source_frame_heading_error_deg"], torch.tensor(0.0)
    )
