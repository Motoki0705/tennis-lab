"""Reference-conditioned PLCS tracking metric tests."""

from __future__ import annotations

import pytest
import torch

from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.plcs.training.tracking_metrics import plcs_tracking_metrics


def test_tracking_metrics_report_heading_y_sign_axes_and_local_index() -> None:
    target_position = torch.tensor(
        [[[[0.2, 0.3, 0.1]]], [[[-0.2, -0.3, 0.1]]]]
    )
    prediction = {
        "position": target_position + torch.tensor([0.1, -0.1, 0.2]),
        "rotation": torch.tensor(
            [[[[0.0, 1.0]]], [[[1.0, 0.0]]]],
        ),
        "presence_logits": torch.full((2, 1, 1), 10.0),
    }
    batch = {
        "target_position": target_position,
        "target_rotation": torch.tensor(
            [[[[1.0, 0.0]]], [[[1.0, 0.0]]]],
        ),
        "target_presence": torch.ones(2, 1, 1, dtype=torch.bool),
        "target_instance_id": torch.tensor([[[10]], [[20]]], dtype=torch.int64),
        "padding_mask": torch.zeros(2, 1, 1, dtype=torch.bool),
    }
    assignments = [
        (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        ),
        (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        ),
    ]

    metrics = plcs_tracking_metrics(
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

    assert metrics["y_sign_accuracy"].item() == pytest.approx(1.0)
    assert metrics["heading_error_deg"].item() == pytest.approx(45.0)
    assert metrics["reference_index_0_position_error_m"].item() > 0.0
    assert metrics["reference_index_1_position_error_m"].item() > 0.0
    assert metrics["x_error_m"].item() > 0.0
    assert metrics["y_error_m"].item() > 0.0
    assert metrics["z_error_m"].item() > 0.0
