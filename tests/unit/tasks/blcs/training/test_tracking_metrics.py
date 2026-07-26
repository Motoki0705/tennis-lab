from __future__ import annotations

import torch

from src.tasks.blcs.training.tracking_metrics import blcs_tracking_metrics
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def test_tracking_metrics_report_per_axis_physical_mae() -> None:
    prediction = {
        "position": torch.tensor([[[[1.0, 2.0, 3.0]]]]),
        "presence_logits": torch.tensor([[[20.0]]]),
    }
    batch = {
        "target_position": torch.zeros(1, 1, 1, 3),
        "target_presence": torch.ones(1, 1, 1, dtype=torch.bool),
        "target_instance_id": torch.zeros(1, 1, 1, dtype=torch.long),
        "frame_mask": torch.ones(1, 1, dtype=torch.bool),
    }

    metrics = blcs_tracking_metrics(
        prediction,
        batch,
        [(torch.tensor([0]), torch.tensor([0]))],
    )

    scale = torch.tensor(COURT_COORD_SCALE_XYZ)
    torch.testing.assert_close(metrics["position_mae_x_m"], scale[0])
    torch.testing.assert_close(metrics["position_mae_y_m"], 2.0 * scale[1])
    torch.testing.assert_close(metrics["position_mae_z_m"], 3.0 * scale[2])
