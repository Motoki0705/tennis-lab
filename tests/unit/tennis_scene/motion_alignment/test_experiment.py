import numpy as np
import pytest

from src.tennis_scene.motion_alignment.diagnostics import ankle_motion, reprojection
from src.tennis_scene.motion_alignment.experiment import match_player


def test_player_matching_uses_image_correspondence_not_track_ids():
    source = np.full((10, 17, 3), 50.0)
    candidates = np.stack([np.full((10, 17, 2), 0.8), np.full((10, 17, 2), 0.5)])
    player, errors = match_player(source, np.ones(10, bool), candidates, (100, 100))
    assert player == 1
    assert errors[1] == 0
    candidates[0] = candidates[1]
    with pytest.raises(ValueError, match="Ambiguous"):
        match_player(source, np.ones(10, bool), candidates, (100, 100))


def test_static_similarity_preserves_low_speed_ankle_proxy_up_to_scale():
    source = np.zeros((10, 17, 3))
    source[:, :, 0] = np.arange(10)[:, None] * 0.001
    result = ankle_motion(
        source, source * 1.2 + 9, np.ones(10, bool), np.ones((10, 17)), 60
    )
    assert result["source_low_speed_mps"]["mean"] == pytest.approx(0.06)
    assert result["output_on_same_mask_mps"]["mean"] == pytest.approx(0.072)
    assert result["output_on_same_mask_mps"]["count"] == 18


def test_reprojection_ignores_unobserved_and_behind_camera_points():
    joints = np.ones((2, 17, 3))
    joints[0, 0, 2] = -1
    observations = np.full((1, 2, 17, 2), 0.01)
    observations[0, 1] = 1
    fit = {"R": np.eye(3).tolist(), "t": [0, 0, 0], "K": np.eye(3).tolist()}
    result = reprojection(
        joints,
        observations,
        np.ones((1, 2, 17)),
        [fit],
        (100, 100),
        np.array([True, False]),
    )
    assert result["camera_0"]["count"] == 16
    assert result["camera_0"]["rmse"] == 0
