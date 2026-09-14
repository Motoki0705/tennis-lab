"""Temporal sampling contracts for integrated PLCS and BLCS inference."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.base.generate_dataset import build_physical_court_provenance
from src.tasks.blcs.model_io import BLCSTrajectoryPrediction
from src.tasks.plcs.model_io import PLCSPhysicalPrediction
from src.tennis_scene.pipeline.components.blcs import BLCSModule
from src.tennis_scene.pipeline.components.plcs import PLCSModule
from tests.unit.tennis_scene.pipeline.config_factories import (
    make_blcs_config,
    make_plcs_config,
)


class _SampledPLCS:
    def require_input_profile(self, profile: str) -> None:
        assert profile == "multiview"

    def predict_multiview_observations(self, **kwargs: Any) -> PLCSPhysicalPrediction:
        human = cast("np.ndarray", kwargs["human_kp"])
        values = human[:, 0, :, 0, 0]
        position = np.stack([values, np.zeros_like(values), np.zeros_like(values)], -1)
        return PLCSPhysicalPrediction(
            position_meters=position.astype(np.float32),
            yaw_radians=np.zeros(values.shape, dtype=np.float32),
            court_reference_provenance=(build_physical_court_provenance(),),
        )


class _SampledBLCS:
    input_profile = "multiview"

    def predict_multiview_arrays(self, **kwargs: Any) -> BLCSTrajectoryPrediction:
        ball = cast("np.ndarray", kwargs["ball_uv"])
        values = torch.from_numpy(ball[0, :, 0])
        position = torch.stack([values, torch.zeros_like(values), torch.zeros_like(values)], -1)
        return BLCSTrajectoryPrediction(
            position=position.unsqueeze(0),
            velocity=None,
            court_reference_provenance=(build_physical_court_provenance(),),
            coordinates_in_metres=True,
        )


def test_plcs_restores_sampled_predictions_to_source_timeline(tmp_path: Path) -> None:
    module = PLCSModule(replace(make_plcs_config(tmp_path), sample_stride=2))
    module._predictor = cast(Any, _SampledPLCS())
    human: NDArray[np.float32] = np.zeros(
        (1, 1, 6, 17, 2), dtype=np.float32
    )
    human[0, 0, :, 0, 0] = np.arange(6)
    result = module.process(
        human_kp_2d=human,
        court_kp=np.zeros((1, 6, 14, 2), dtype=np.float32),
        human_kp_vis=np.ones((1, 1, 6, 17), dtype=np.float32),
        court_vis=np.ones((1, 6, 14), dtype=np.float32),
        track_ids=np.array([7], dtype=np.int32),
    )
    np.testing.assert_allclose(result.position[0, :, 0], [0, 1, 2, 3, 4, 4])


def test_blcs_restores_sampled_predictions_to_source_timeline(tmp_path: Path) -> None:
    module = BLCSModule(replace(make_blcs_config(tmp_path), sample_stride=2))
    module._predictor = cast(Any, _SampledBLCS())
    ball: NDArray[np.float32] = np.zeros((1, 6, 2), dtype=np.float32)
    ball[0, :, 0] = np.arange(6)
    result = module.process(
        ball_uv=ball,
        court_kp=np.zeros((1, 6, 14, 2), dtype=np.float32),
        ball_vis=np.ones((1, 6), dtype=np.bool_),
        court_vis=np.ones((1, 6, 14), dtype=np.float32),
    )
    np.testing.assert_allclose(result.ball_3d[:, 0], [0, 1, 2, 3, 4, 4])
