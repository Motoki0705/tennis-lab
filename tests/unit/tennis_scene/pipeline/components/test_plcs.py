"""Tests for tennis_scene PLCS pipeline component."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import Tensor

from src.tasks.plcs.models import PLCSModel, PLCSMultiViewAxialModel
from src.tennis_scene.pipeline.components.plcs import PLCSModule
from tests.unit.tennis_scene.pipeline.config_factories import make_plcs_config


class _FakePLCSPredictor:
    def predict(
        self,
        *,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        human_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        denormalize: bool = True,
    ) -> dict[str, Tensor]:
        del human_vis, court_vis, denormalize
        assert human_kp.shape == (2, 1, 4, 17, 2)
        assert court_kp.shape == (2, 1, 4, 20, 2)
        assert human_mask is not None
        assert human_mask.shape == (2, 1, 4)
        return {
            "position_meters": torch.zeros(2, 4, 3),
            "yaw_radians": torch.zeros(2, 4),
        }


def test_process_passes_single_camera_as_multiview_n_equals_one(tmp_path) -> None:
    module = PLCSModule(make_plcs_config(tmp_path))
    module._predictor = cast(Any, _FakePLCSPredictor())
    result = module.process(
        human_kp_2d=np.zeros((2, 1, 4, 17, 2), dtype=np.float32),
        court_kp=np.zeros((1, 4, 20, 2), dtype=np.float32),
        human_kp_vis=np.ones((2, 1, 4, 17), dtype=np.float32),
        court_vis=np.ones((1, 4, 20), dtype=np.float32),
    )

    assert result.position.shape == (2, 4, 3)
    assert result.yaw.shape == (2, 4)
    np.testing.assert_array_equal(result.track_ids, np.array([0, 1], dtype=np.int32))


def test_validate_pipeline_checkpoint_rejects_frame_model(tmp_path) -> None:
    module = PLCSModule(make_plcs_config(tmp_path))
    module._predictor = cast(
        Any,
        SimpleNamespace(model=object.__new__(PLCSModel)),
    )

    with pytest.raises(ValueError, match="requires a multiview PLCS checkpoint"):
        module._validate_pipeline_checkpoint_profile()


def test_validate_pipeline_checkpoint_accepts_multiview_model(tmp_path) -> None:
    module = PLCSModule(make_plcs_config(tmp_path))
    module._predictor = cast(
        Any,
        SimpleNamespace(
            model=object.__new__(PLCSMultiViewAxialModel)
        ),
    )

    module._validate_pipeline_checkpoint_profile()
