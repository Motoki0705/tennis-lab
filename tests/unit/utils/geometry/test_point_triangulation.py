"""Robust per-point triangulation with explicit rejection reasons."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.utils.geometry.triangulation import (
    PinholeCamera,
    PointRejection,
    PointTriangulationConfig,
    TriangulatedPoints,
    reject_excessive_speed,
    triangulate_points,
)

_CONFIG = PointTriangulationConfig(2.0, (-0.5, 4.0))


def camera(name: str, center: list[float]) -> PinholeCamera:
    center_array = np.asarray(center, np.float64)
    forward = np.array([0, 0, 1.0]) - center_array
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    rotation = np.stack((right, np.cross(forward, right), forward))
    intrinsic = np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]])
    return PinholeCamera(name, intrinsic, rotation, -rotation @ center_array)


def scene_cameras() -> tuple[PinholeCamera, ...]:
    return (camera("a", [-8, -16, 9]), camera("b", [8, -16, 9]), camera("c", [5, 16, 9]))


def project(cameras: tuple[PinholeCamera, ...], xyz: NDArray[np.float64]) -> NDArray[np.float32]:
    return np.stack([c.project(xyz)[0] for c in cameras]).astype(np.float32)


def test_recovers_known_points_and_rejects_missing_support() -> None:
    cameras = scene_cameras()
    xyz = np.array([[1.0, -3.0, 1.2], [1.2, -2.8, 1.1], [0.0, -2.0, 1.0]])
    visible: NDArray[np.bool_] = np.ones((3, 3), bool)
    visible[1:, 2] = False
    result = triangulate_points(project(cameras, xyz), visible, cameras, config=_CONFIG)
    np.testing.assert_allclose(result.positions[:2], xyz[:2], atol=2e-5)
    assert result.valid.tolist() == [True, True, False]
    assert not result.positions[2].any()
    assert result.reasons[2] == PointRejection.INSUFFICIENT_VIEWS


def test_pair_consensus_ignores_one_bad_camera() -> None:
    cameras = scene_cameras()
    xyz = np.array([[1.0, -3.0, 1.2]])
    uv = project(cameras, xyz)
    uv[2] += 100
    result = triangulate_points(uv, np.ones((3, 1), bool), cameras, config=_CONFIG)
    assert result.valid[0]
    np.testing.assert_allclose(result.positions[0], xyz[0], atol=2e-5)
    assert result.inliers[:, 0].tolist() == [True, True, False]


def test_points_outside_the_height_bounds_are_rejected() -> None:
    cameras = scene_cameras()
    xyz = np.array([[1.0, -3.0, 6.0]])
    result = triangulate_points(project(cameras, xyz), np.ones((3, 1), bool), cameras, config=_CONFIG)
    assert not result.valid[0]
    assert result.reasons[0] == PointRejection.OUTSIDE_BOUNDS
    assert not result.positions[0].any()


def test_half_turn_is_a_gauge_change_of_the_court_frame() -> None:
    base = scene_cameras()[0]
    turned = base.half_turned(True)
    xyz = np.array([[1.0, -3.0, 1.2]])
    mirrored = xyz * np.array([-1.0, -1.0, 1.0])
    np.testing.assert_allclose(turned.project(mirrored)[0], base.project(xyz)[0], atol=1e-9)
    np.testing.assert_array_equal(base.half_turned(False).rotation, base.rotation)


def test_speed_gate_rejects_both_endpoints_without_bridging_gaps() -> None:
    positions = np.array([[0, 0, 1], [0.1, 0, 1], [9.0, 0, 1], [0, 0, 0], [9.2, 0, 1]], np.float32)
    valid = np.array([True, True, True, False, True])
    reasons = np.where(valid, 0, int(PointRejection.INSUFFICIENT_VIEWS)).astype(np.uint8)
    positions[~valid] = 0
    track = TriangulatedPoints(positions, valid, reasons, np.stack([valid, valid]), np.zeros((2, 5), np.float32))
    gated = reject_excessive_speed(track, fps=30.0, max_speed_mps=60.0)
    # 0.1 m/frame = 3 m/s passes; 8.9 m/frame = 267 m/s rejects frames 1 and 2.
    # Frames 2 -> 4 are not adjacent (frame 3 is missing) and are not compared.
    assert gated.valid.tolist() == [True, False, False, False, True]
    assert gated.reasons[1:3].tolist() == [PointRejection.EXCESSIVE_SPEED] * 2
    assert gated.reasons[3] == PointRejection.INSUFFICIENT_VIEWS
    assert not gated.positions[1:3].any()
    assert not gated.inliers[:, 1:3].any()


def test_rejects_invalid_confidence() -> None:
    cameras = scene_cameras()
    uv = project(cameras, np.array([[1.0, -3.0, 1.2]]))
    with pytest.raises(ValueError, match="confidence"):
        triangulate_points(uv, np.ones((3, 1), bool), cameras, config=_CONFIG, confidence=np.full((3, 1), 1.5))
