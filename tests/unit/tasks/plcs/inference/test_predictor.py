"""Tests for PLCS inference predictor output conversions."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from src.tasks.plcs.inference.predictor import PLCSPredictor
from src.tasks.plcs.model_io import PLCSInputProfile, PLCSModelIOAdapter


class _FixedRotationModel(nn.Module):
    def __init__(self, rotation: Tensor) -> None:
        super().__init__()
        self.register_buffer("rotation", rotation)

    def forward(
        self,
        *,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor,
        human_mask: Tensor,
        court_vis: Tensor,
    ) -> dict[str, Tensor]:
        del human_kp, court_kp, human_vis, human_mask, court_vis
        rotation = cast(Tensor, self.rotation)
        return {
            "position": torch.zeros(
                *rotation.shape[:-1],
                3,
                device=rotation.device,
                dtype=rotation.dtype,
            ),
            "rotation": rotation,
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
        adapter=PLCSModelIOAdapter(
            model_type=_FixedRotationModel,
            profile=PLCSInputProfile.MULTIVIEW,
            num_court_tokens=20,
            output_rank=3,
            predict_canonical_pose=False,
            predict_auxiliary_position=False,
        ),
        device=torch.device("cpu"),
    )

    human_shape = (2, 1, 3)
    result = predictor.predict(
        human_kp=torch.zeros(*human_shape, 17, 2),
        court_kp=torch.zeros(*human_shape, 20, 2),
        human_vis=torch.ones(*human_shape, 17, dtype=torch.bool),
        human_mask=torch.ones(*human_shape, dtype=torch.bool),
        court_vis=torch.ones(*human_shape, 20, dtype=torch.bool),
        denormalize=True,
    )

    assert torch.allclose(result["rotation"], dataset_encoded_rotation)
    assert torch.allclose(result["yaw_radians"], angles, atol=1e-6)

    physical = predictor.predict_multiview_observations(
        human_kp=np.zeros((*human_shape, 17, 2), dtype=np.float32),
        court_kp=np.zeros((1, 3, 20, 2), dtype=np.float32),
        human_vis=np.ones((*human_shape, 17), dtype=np.bool_),
        human_mask=np.ones(human_shape, dtype=np.bool_),
        court_vis=np.ones((1, 3, 20), dtype=np.bool_),
    )
    np.testing.assert_allclose(physical.yaw_radians, angles.numpy(), atol=1e-6)
    assert physical.position_meters.shape == (2, 3, 3)
    assert physical.position_meters.dtype == np.float32
