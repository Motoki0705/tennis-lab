"""Tests for tennis_scene BLCS pipeline component."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from src.tasks.blcs.model_io import BLCSTrajectoryPrediction
from src.tennis_scene.pipeline.components.blcs import BLCSModule, BLCSResult
from tests.unit.tennis_scene.pipeline.config_factories import make_blcs_config


class _FakeBLCSPredictor:
    def __init__(self, *, input_profile: str = "multiview") -> None:
        self.input_profile = input_profile
        self.calls: list[dict[str, object]] = []

    def predict_multiview_arrays(
        self,
        *,
        ball_uv: np.ndarray,
        court_kp: np.ndarray,
        ball_vis: np.ndarray,
        court_vis: np.ndarray,
        denormalize: bool,
    ) -> BLCSTrajectoryPrediction:
        self.calls.append(
            {
                "ball_uv": ball_uv,
                "court_kp": court_kp,
                "ball_vis": ball_vis,
                "court_vis": court_vis,
                "denormalize": denormalize,
            }
        )
        return BLCSTrajectoryPrediction(
            position=torch.ones(1, ball_uv.shape[1], 3), velocity=None
        )


def test_process_delegates_multiview_assembly_and_typed_decode(tmp_path: Path) -> None:
    module = BLCSModule(make_blcs_config(tmp_path))
    predictor = _FakeBLCSPredictor()
    module._predictor = cast(Any, predictor)
    visibility = np.array([[True, False, True, True]], dtype=np.bool_)

    result = module.process(
        ball_uv=np.zeros((1, 4, 2), dtype=np.float32),
        court_kp=np.zeros((1, 4, 20, 2), dtype=np.float32),
        ball_vis=visibility,
        court_vis=np.ones((1, 4, 20), dtype=np.float32),
    )

    assert result.ball_3d.shape == (4, 3)
    np.testing.assert_array_equal(result.visibility, visibility[0])
    np.testing.assert_array_equal(result.ball_3d, np.ones((4, 3), dtype=np.float32))
    assert len(predictor.calls) == 1
    call = predictor.calls[0]
    assert call["denormalize"] is True
    assert cast(np.ndarray, call["ball_uv"]).shape == (1, 4, 2)
    np.testing.assert_array_equal(call["ball_vis"], visibility)


def test_load_rejects_non_multiview_profile_before_inference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    predictor = _FakeBLCSPredictor(input_profile="single")

    def fake_load(*args: Any, **kwargs: Any) -> _FakeBLCSPredictor:
        del args, kwargs
        return predictor

    monkeypatch.setattr(
        "src.tasks.blcs.inference.predictor.BLCSPredictor.load_from_checkpoint",
        fake_load,
    )
    module = BLCSModule(make_blcs_config(tmp_path))

    with pytest.raises(ValueError, match="input_profile='multiview'"):
        module.load()


def test_result_validate_allows_inferred_positions_for_invisible_frames() -> None:
    result = BLCSResult(
        ball_3d=np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        ),
        visibility=np.array([True, False], dtype=np.bool_),
    )

    is_valid, errors = result.validate()

    assert is_valid
    assert errors == []


def test_result_rejects_missing_visibility() -> None:
    with pytest.raises(ValueError, match="missing required fields.*visibility"):
        BLCSResult.from_dict({"ball_3d": [[0.0, 0.0, 0.0]]})
