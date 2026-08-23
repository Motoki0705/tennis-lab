"""Tests for typed standard BLCS inference."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from src.tasks.base.model_io import bind_model_io
from src.tasks.blcs.inference.predictor import BLCSPredictor
from src.tasks.blcs.model_io import (
    SingleTrajectoryModelIOAdapter,
    TrajectoryBoundModelIO,
)
from src.tasks.blcs.models import BLCSModel
from src.utils.schema.court_normalization import resolve_court_coordinate_normalization


class _FixedTrajectoryModel(BLCSModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)

    def forward(
        self,
        ball_uv: Tensor,
        ball_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
    ) -> dict[str, Tensor]:
        del court_kp, ball_vis, court_vis, padding_mask
        shape = (ball_uv.shape[0], ball_uv.shape[1], 3)
        return {
            "position": torch.ones(shape, device=ball_uv.device),
            "velocity": torch.full(shape, 2.0, device=ball_uv.device),
        }


def test_predict_returns_typed_cpu_trajectory_decode() -> None:
    binding = cast(
        "TrajectoryBoundModelIO",
        bind_model_io(
            _FixedTrajectoryModel(),
            SingleTrajectoryModelIOAdapter(
                num_court_tokens=14,
                max_seq_len=8,
                predict_velocity=True,
                input_profile="single",
                max_num_cameras=None,
            ),
        ),
    )
    predictor = BLCSPredictor(model_io=binding, device=torch.device("cpu"))

    prediction = predictor.predict(
        ball_uv=torch.zeros(1, 3, 2),
        court_kp=torch.zeros(1, 14, 2),
        ball_vis=torch.ones(1, 3, dtype=torch.bool),
        padding_mask=torch.zeros(1, 3, dtype=torch.bool),
        court_vis=torch.ones(1, 14, dtype=torch.bool),
        denormalize=False,
    )

    assert prediction.position.shape == (1, 3, 3)
    assert prediction.velocity is not None
    torch.testing.assert_close(prediction.velocity, torch.full((1, 3, 3), 2.0))
    assert prediction.position.device.type == "cpu"
    assert prediction.velocity.device.type == "cpu"


def test_v2_predictor_decodes_position_and_velocity_to_physical_units() -> None:
    binding = cast(
        "TrajectoryBoundModelIO",
        bind_model_io(
            _FixedTrajectoryModel(),
            SingleTrajectoryModelIOAdapter(
                num_court_tokens=14,
                max_seq_len=8,
                predict_velocity=True,
                input_profile="single",
                max_num_cameras=None,
            ),
        ),
    )
    contract = resolve_court_coordinate_normalization("v2")
    predictor = BLCSPredictor(
        model_io=binding,
        device=torch.device("cpu"),
        normalization=contract,
    )

    prediction = predictor.predict(
        ball_uv=torch.zeros(1, 3, 2),
        court_kp=torch.zeros(1, 14, 2),
        ball_vis=torch.ones(1, 3, dtype=torch.bool),
        padding_mask=torch.zeros(1, 3, dtype=torch.bool),
        court_vis=torch.ones(1, 14, dtype=torch.bool),
        denormalize=True,
    )

    expected_position = torch.tensor(contract.scale_xyz).expand(1, 3, 3)
    torch.testing.assert_close(prediction.position, expected_position)
    assert prediction.velocity is not None
    torch.testing.assert_close(prediction.velocity, 2.0 * expected_position)
