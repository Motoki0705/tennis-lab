from __future__ import annotations

import torch
from torch import Tensor, nn

from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.plcs.inference.tracking_predictor import PLCSTrackingPredictor
from src.tasks.plcs.model_io import PLCSTrackQueryIOAdapter
from src.utils.schema.court_normalization import resolve_court_coordinate_normalization


class _FixedTrackingModel(nn.Module):
    def forward(
        self,
        *,
        human_kp: Tensor,
        human_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
    ) -> dict[str, Tensor]:
        del (
            human_vis,
            court_kp,
            court_vis,
            padding_mask,
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
        model=_FixedTrackingModel(),
        adapter=PLCSTrackQueryIOAdapter(
            model_type=_FixedTrackingModel,
            num_queries=2,
            num_court_tokens=14,
            num_joints=17,
        ),
        device=torch.device("cpu"),
    )
    shape = (1, 2, 3, 2)

    result = predictor.predict(
        human_kp=torch.zeros(*shape, 17, 2),
        human_vis=torch.ones(*shape, 17, dtype=torch.bool),
        court_kp=torch.zeros(1, 2, 3, 14, 2),
        court_vis=torch.ones(1, 2, 3, 14, dtype=torch.bool),
        padding_mask=torch.zeros(1, 2, 3, dtype=torch.bool),
        tracking_metrics=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
        ),
        denormalize=True,
    )

    assert result["position_meters"].shape == (1, 3, 2, 3)
    assert result["presence"][..., 0].all()
    assert not result["presence"][..., 1].any()
    torch.testing.assert_close(
        result["position_meters"],
        torch.tensor([5.485, 11.885, 1.07]).expand(1, 3, 2, 3),
    )
    torch.testing.assert_close(
        result["yaw_radians"],
        torch.full((1, 3, 2), torch.pi / 2),
    )
    assert all(value.device.type == "cpu" for value in result.values())


def test_v2_tracking_predictor_denormalizes_all_query_positions_to_meters() -> None:
    contract = resolve_court_coordinate_normalization("v2")
    predictor = PLCSTrackingPredictor(
        model=_FixedTrackingModel(),
        adapter=PLCSTrackQueryIOAdapter(
            model_type=_FixedTrackingModel,
            num_queries=2,
            num_court_tokens=14,
            num_joints=17,
        ),
        device=torch.device("cpu"),
        court_coordinate_normalization=contract,
    )

    result = predictor.predict(
        human_kp=torch.zeros(1, 1, 2, 2, 17, 2),
        human_vis=torch.ones(1, 1, 2, 2, 17, dtype=torch.bool),
        court_kp=torch.zeros(1, 1, 2, 14, 2),
        court_vis=torch.ones(1, 1, 2, 14, dtype=torch.bool),
        padding_mask=torch.zeros(1, 1, 2, dtype=torch.bool),
        tracking_metrics=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
        ),
        denormalize=True,
    )

    torch.testing.assert_close(
        result["position_meters"],
        torch.tensor(contract.scale_xyz).expand(1, 2, 2, 3),
    )
