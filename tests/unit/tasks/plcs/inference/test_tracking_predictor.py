from __future__ import annotations

import torch
from torch import Tensor, nn

from src.tasks.plcs.inference.tracking_predictor import PLCSTrackingPredictor


class _FixedTrackingModel(nn.Module):
    def forward(
        self,
        *,
        human_kp: Tensor,
        detection_mask: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        frame_mask: Tensor,
        view_mask: Tensor,
    ) -> dict[str, Tensor]:
        del (
            detection_mask,
            court_kp,
            court_vis,
            frame_mask,
            view_mask,
        )
        batch, _, frames = human_kp.shape[:3]
        rotation = torch.tensor([0.0, 1.0], device=human_kp.device)
        return {
            "position": torch.ones(batch, frames, 2, 3, device=human_kp.device),
            "rotation": rotation.expand(batch, frames, 2, -1),
            "presence_logits": torch.tensor([2.0, -2.0], device=human_kp.device).expand(
                batch, frames, -1
            ),
        }


def test_predictor_returns_cpu_lifecycle_and_yaw_outputs() -> None:
    predictor = PLCSTrackingPredictor(
        model=_FixedTrackingModel(), device=torch.device("cpu")
    )
    shape = (1, 2, 3, 2)

    result = predictor.predict(
        human_kp=torch.zeros(*shape, 17, 2),
        detection_mask=torch.ones(*shape, dtype=torch.bool),
        court_kp=torch.zeros(1, 2, 3, 14, 2),
        court_vis=torch.ones(1, 2, 3, 14, dtype=torch.bool),
        frame_mask=torch.ones(1, 3, dtype=torch.bool),
        view_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    assert result["position_meters"].shape == (1, 3, 2, 3)
    assert result["presence"][..., 0].all()
    assert not result["presence"][..., 1].any()
    torch.testing.assert_close(
        result["yaw_radians"],
        torch.full((1, 3, 2), torch.pi / 2),
    )
    assert all(value.device.type == "cpu" for value in result.values())
