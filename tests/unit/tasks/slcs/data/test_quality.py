"""Tests for pseudo-label quality filtering."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.slcs.data.quality import (
    QualityConfig,
    ball_label_confidence,
    build_label_masks,
    player_label_confidence,
    window_label_ratio,
)


def _inputs(
    num_players: int = 2, num_cameras: int = 2, num_frames: int = 6
) -> dict[str, np.ndarray]:
    return {
        "human_kp_vis": np.ones((num_players, num_cameras, num_frames, 17), np.float32),
        "ball_vis": np.ones((num_cameras, num_frames), bool),
        "player_position": np.zeros((num_players, num_frames, 3), np.float32),
        "player_yaw": np.zeros((num_players, num_frames), np.float32),
        "ball_3d": np.zeros((num_frames, 3), np.float32),
    }


def test_confidences_are_coverage_means() -> None:
    vis: NDArray[np.float32] = np.zeros((1, 2, 4, 17), np.float32)
    vis[0, 0] = 1.0  # camera 0 sees everything, camera 1 nothing
    assert np.allclose(player_label_confidence(vis), 0.5)

    ball_vis = np.array([[True, False], [True, True]])
    assert np.allclose(ball_label_confidence(ball_vis), [1.0, 0.5])


def test_masks_all_valid_when_fully_observed() -> None:
    masks = build_label_masks(config=QualityConfig(), **_inputs())
    assert masks["player_label_valid"].all()
    assert masks["ball_label_valid"].all()
    assert np.allclose(masks["player_label_weight"], 1.0)
    assert np.allclose(masks["ball_label_weight"], 1.0)


def test_low_confidence_player_frames_masked() -> None:
    inputs = _inputs()
    inputs["human_kp_vis"][0, :, 2, :] = 0.1  # frame 2 of player 0 barely observed
    masks = build_label_masks(
        config=QualityConfig(min_player_confidence=0.3), **inputs
    )
    assert not masks["player_label_valid"][0, 2]
    assert masks["player_label_weight"][0, 2] == 0.0
    assert masks["player_label_valid"][1].all()


def test_ball_camera_threshold() -> None:
    inputs = _inputs(num_cameras=2)
    inputs["ball_vis"][:, 3] = False
    inputs["ball_vis"][1, 4] = False
    masks = build_label_masks(config=QualityConfig(min_ball_cameras=2), **inputs)
    assert not masks["ball_label_valid"][3]  # zero cameras
    assert not masks["ball_label_valid"][4]  # one of two cameras
    assert masks["ball_label_valid"][0]


def test_non_finite_labels_always_invalid() -> None:
    inputs = _inputs()
    inputs["player_position"][0, 1, 0] = np.nan
    inputs["ball_3d"][2, 1] = np.inf
    masks = build_label_masks(config=QualityConfig(), **inputs)
    assert not masks["player_label_valid"][0, 1]
    assert not masks["ball_label_valid"][2]


def test_min_ball_cameras_above_camera_count_is_error() -> None:
    with pytest.raises(ValueError, match="min_ball_cameras"):
        build_label_masks(config=QualityConfig(min_ball_cameras=3), **_inputs())


def test_label_weight_power() -> None:
    inputs = _inputs()
    inputs["human_kp_vis"][...] = 0.5
    masks = build_label_masks(config=QualityConfig(label_weight_power=2.0), **inputs)
    assert np.allclose(masks["player_label_weight"], 0.25)


def test_window_label_ratio() -> None:
    player_valid: NDArray[np.bool_] = np.zeros((2, 8), bool)
    ball_valid: NDArray[np.bool_] = np.zeros(8, bool)
    player_valid[0, 0] = True
    ball_valid[1] = True
    assert window_label_ratio(player_valid, ball_valid, start=0, length=4) == 0.5
    assert window_label_ratio(player_valid, ball_valid, start=4, length=4) == 0.0


def test_quality_config_validation() -> None:
    with pytest.raises(ValueError):
        QualityConfig(min_player_confidence=1.5)
    with pytest.raises(ValueError):
        QualityConfig(min_ball_cameras=0)
    with pytest.raises(ValueError):
        QualityConfig(min_window_label_ratio=-0.1)
