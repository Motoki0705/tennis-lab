"""Per-point consistency: exact views score zero, wrong or implausible geometry scores one."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.utils.geometry.multiview_consistency import (
    ConsistencyBounds,
    score_multiview_points,
)
from src.utils.geometry.triangulation import PinholeCamera

BOUNDS = ConsistencyBounds((-0.2, 20.))


def look_at(name: str, center: list[float]) -> PinholeCamera:
    position = np.asarray(center, np.float64)
    forward = -position / np.linalg.norm(position)
    right = np.cross(forward, [0, 0, 1.])
    right /= np.linalg.norm(right)
    rotation = np.stack((right, np.cross(forward, right), forward))
    return PinholeCamera(name, np.array([[1000., 0, 640], [0, 1000, 360], [0, 0, 1]]), rotation, -rotation @ position)


CAMERAS = (look_at("a", [-8, -18, 7]), look_at("b", [9, -19, 6]), look_at("c", [0, 20, 9]))


def test_exact_observations_are_consistent_and_a_missing_view_is_ignored() -> None:
    xyz = np.array([[1., 2., 1.], [-3., 5., 2.5]])
    uv = np.stack([c.project(xyz)[0] for c in CAMERAS])
    observed: NDArray[np.bool_] = np.ones((3, 2), bool)
    observed[2, 1] = False
    uv[2, 1] = 1e6  # an unobserved view's pixels never enter the score
    score = score_multiview_points(uv, observed, CAMERAS, threshold_px=10., bounds=BOUNDS)
    np.testing.assert_allclose(score.xyz, xyz, atol=1e-6)
    np.testing.assert_allclose(score.cost, 0., atol=1e-9)
    assert score.support.all() and score.plausible.all()


def test_wrong_camera_pose_and_implausible_points_cost_one() -> None:
    xyz = np.array([[4., 6., 1.], [0., 0., -3.]])
    uv = np.stack([c.project(xyz)[0] for c in CAMERAS])
    wrong = (CAMERAS[0], CAMERAS[1], CAMERAS[2].half_turned(True))
    score = score_multiview_points(uv, np.ones((3, 2), bool), wrong, threshold_px=10., bounds=BOUNDS)
    assert score.cost[0] > .5 and not score.support[0]
    exact = score_multiview_points(uv, np.ones((3, 2), bool), CAMERAS, threshold_px=10., bounds=BOUNDS)
    assert not exact.plausible[1] and exact.cost[1] == 1. and not exact.support[1]  # below the height range


def test_contract_violations_are_errors() -> None:
    uv = np.zeros((3, 1, 2))
    with pytest.raises(ValueError, match="two observing views"):
        score_multiview_points(uv, np.array([[True], [False], [False]]), CAMERAS, threshold_px=10., bounds=BOUNDS)
    with pytest.raises(ValueError, match="height range"):
        ConsistencyBounds((1., 1.))
