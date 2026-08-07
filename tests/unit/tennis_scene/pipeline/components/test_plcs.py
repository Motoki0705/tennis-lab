"""Tests for tennis_scene PLCS pipeline component."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from src.tasks.plcs.model_io import PLCSPhysicalPrediction
from src.tennis_scene.pipeline.components.plcs import PLCSModule, PLCSResult
from tests.unit.tennis_scene.pipeline.config_factories import make_plcs_config


class _FakePLCSPredictor:
    def __init__(self) -> None:
        self.profiles: list[str] = []
        self.calls: list[dict[str, np.ndarray]] = []

    def require_input_profile(self, profile: str) -> None:
        self.profiles.append(profile)

    def predict_multiview_observations(
        self,
        *,
        human_kp: np.ndarray,
        court_kp: np.ndarray,
        human_vis: np.ndarray,
        human_mask: np.ndarray,
        court_vis: np.ndarray,
    ) -> PLCSPhysicalPrediction:
        self.calls.append(
            {
                "human_kp": human_kp,
                "court_kp": court_kp,
                "human_vis": human_vis,
                "human_mask": human_mask,
                "court_vis": court_vis,
            }
        )
        players, _, frames = human_kp.shape[:3]
        return PLCSPhysicalPrediction(
            position_meters=np.zeros((players, frames, 3), dtype=np.float32),
            yaw_radians=np.zeros((players, frames), dtype=np.float32),
        )


def test_process_delegates_multiview_assembly_and_decode_to_predictor(tmp_path) -> None:
    module = PLCSModule(make_plcs_config(tmp_path))
    predictor = _FakePLCSPredictor()
    module._predictor = cast(Any, predictor)
    confidence: np.ndarray = np.ones((2, 1, 4, 17), dtype=np.float32)
    confidence[0, 0, 0, 0] = 0.01

    result = module.process(
        human_kp_2d=np.zeros((2, 1, 4, 17, 2), dtype=np.float32),
        court_kp=np.zeros((1, 4, 20, 2), dtype=np.float32),
        human_kp_vis=confidence,
        court_vis=np.ones((1, 4, 20), dtype=np.float32),
        track_ids=np.array([5, 9], dtype=np.int32),
    )

    assert result.position.shape == (2, 4, 3)
    assert result.yaw.shape == (2, 4)
    np.testing.assert_array_equal(result.track_ids, np.array([5, 9], dtype=np.int32))
    assert len(predictor.calls) == 1
    call = predictor.calls[0]
    assert call["human_kp"].shape == (2, 1, 4, 17, 2)
    assert call["court_kp"].shape == (1, 4, 20, 2)
    assert call["human_vis"].dtype == np.bool_
    assert not bool(call["human_vis"][0, 0, 0, 0])
    assert call["human_mask"].dtype == np.bool_
    assert call["human_mask"].all()


def test_load_requires_multiview_profile_before_component_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    predictor = _FakePLCSPredictor()

    def fake_load(*args: Any, **kwargs: Any) -> _FakePLCSPredictor:
        del args, kwargs
        return predictor

    monkeypatch.setattr(
        "src.tasks.plcs.inference.predictor.PLCSPredictor.load_from_checkpoint",
        fake_load,
    )
    module = PLCSModule(make_plcs_config(tmp_path))

    module.load()

    assert predictor.profiles == ["multiview"]


def test_process_requires_explicit_track_ids(tmp_path) -> None:
    module = PLCSModule(make_plcs_config(tmp_path))
    module._predictor = cast(Any, _FakePLCSPredictor())

    with pytest.raises(ValueError, match="track_ids must have shape"):
        module.process(
            human_kp_2d=np.zeros((2, 1, 4, 17, 2), dtype=np.float32),
            court_kp=np.zeros((1, 4, 20, 2), dtype=np.float32),
            human_kp_vis=np.ones((2, 1, 4, 17), dtype=np.float32),
            court_vis=np.ones((1, 4, 20), dtype=np.float32),
            track_ids=np.array([], dtype=np.int32),
        )


def test_result_rejects_missing_track_ids() -> None:
    with pytest.raises(ValueError, match="missing required fields.*track_ids"):
        PLCSResult.from_dict(
            {
                "position": np.zeros((1, 2, 3), dtype=np.float32).tolist(),
                "yaw": np.zeros((1, 2), dtype=np.float32).tolist(),
            }
        )
