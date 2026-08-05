from __future__ import annotations

import torch
from torch import Tensor, nn

from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.blcs.inference.tracking_predictor import BLCSTrackingPredictor


class _FixedTrackingModel(nn.Module):
    def forward(
        self,
        *,
        ball_uv: Tensor,
        ball_visible: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        frame_mask: Tensor,
        view_mask: Tensor,
    ) -> dict[str, Tensor]:
        del (
            ball_visible,
            court_kp,
            court_vis,
            frame_mask,
            view_mask,
        )
        batch, _, frames = ball_uv.shape[:3]
        return {
            "position": torch.ones(batch, frames, 2, 3, device=ball_uv.device),
            "presence_logits": torch.tensor([-2.0, 2.0], device=ball_uv.device).expand(
                batch, frames, -1
            ),
        }


def test_predictor_returns_cpu_query_presence_and_positions() -> None:
    predictor = BLCSTrackingPredictor(
        model=_FixedTrackingModel(), device=torch.device("cpu")
    )
    shape = (1, 2, 3, 2)

    result = predictor.predict(
        ball_uv=torch.zeros(*shape, 2),
        ball_visible=torch.ones(*shape, dtype=torch.bool),
        court_kp=torch.zeros(1, 2, 3, 14, 2),
        court_vis=torch.ones(1, 2, 3, 14, dtype=torch.bool),
        frame_mask=torch.ones(1, 3, dtype=torch.bool),
        view_mask=torch.ones(1, 2, dtype=torch.bool),
        tracking_metrics=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
        ),
        denormalize=False,
    )

    assert result["position"].shape == (1, 3, 2, 3)
    assert not result["presence"][..., 0].any()
    assert result["presence"][..., 1].all()
    assert all(value.device.type == "cpu" for value in result.values())
