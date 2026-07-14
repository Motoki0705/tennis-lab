"""Tests for PLCS inference predictor output conversions."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.tasks.plcs.inference.predictor import PLCSPredictor


class _FixedRotationModel(nn.Module):
    def __init__(self, rotation: Tensor) -> None:
        super().__init__()
        self.register_buffer("rotation", rotation)

    def forward(
        self,
        *,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        human_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        del human_kp, court_kp, human_vis, human_mask, court_vis
        return {
            "position": torch.zeros(
                *self.rotation.shape[:-1],
                3,
                device=self.rotation.device,
                dtype=self.rotation.dtype,
            ),
            "rotation": self.rotation,
        }


class _LineModel(nn.Module):
    court_input_type = "line"

    def forward(
        self,
        *,
        human_kp: Tensor,
        court_lines: Tensor,
        human_vis: Tensor | None = None,
        human_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        del court_lines, human_vis, human_mask
        leading = human_kp.shape[:3]
        return {
            "position": torch.zeros(*leading, 3),
            "rotation": torch.ones(*leading, 2),
        }


def test_yaw_radians_round_trips_dataset_cos_sin_encoding() -> None:
    angles = torch.tensor(
        [[0.0, math.pi / 6.0, -math.pi / 2.0], [2.3, -1.2, 3.0]],
        dtype=torch.float32,
    )
    dataset_encoded_rotation = torch.stack(
        [torch.cos(angles), torch.sin(angles)],
        dim=-1,
    )
    predictor = PLCSPredictor(
        model=_FixedRotationModel(dataset_encoded_rotation),
        device=torch.device("cpu"),
    )

    result = predictor.predict(
        human_kp=torch.zeros(2, 1, 3, 17, 2),
        court_kp=torch.zeros(2, 1, 3, 20, 2),
        denormalize=True,
    )

    assert torch.allclose(result["rotation"], dataset_encoded_rotation)
    assert torch.allclose(result["yaw_radians"], angles, atol=1e-6)


def test_line_predictor_routes_only_court_lines() -> None:
    predictor = PLCSPredictor(_LineModel(), torch.device("cpu"))
    result = predictor.predict(
        human_kp=torch.zeros(1, 2, 4, 17, 2),
        court_lines=torch.zeros(1, 2, 4, 12, 4),
        human_vis=torch.ones(1, 2, 4, 17),
        human_mask=torch.ones(1, 2, 4),
        denormalize=False,
    )
    assert result["position"].shape == (1, 2, 4, 3)
