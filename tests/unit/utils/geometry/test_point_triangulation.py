"""Robust per-point triangulation with explicit rejection reasons."""

from __future__ import annotations

from itertools import permutations
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

import src.utils.geometry.triangulation as triangulation
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
    center_array: NDArray[np.float64] = np.asarray(center, np.float64)
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


def stereo_cameras(baseline: float = 2.0) -> tuple[PinholeCamera, ...]:
    """Parallel +Z cameras, with the second center at (baseline, 0, 0)."""
    intrinsic = np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]])
    return tuple(
        PinholeCamera(name, intrinsic.copy(), np.eye(3), np.array([-x, 0.0, 0.0]))
        for name, x in (("left", 0.0), ("right", baseline))
    )


def assert_rejected(result: TriangulatedPoints, reason: PointRejection) -> None:
    assert result.valid.dtype == np.bool_
    assert result.inliers.dtype == np.bool_
    assert result.reasons.dtype == np.uint8
    assert not result.valid.any()
    np.testing.assert_array_equal(result.reasons, np.full(result.valid.shape, reason))
    np.testing.assert_array_equal(result.positions, np.zeros_like(result.positions))
    assert not result.inliers.any()


def test_recovers_known_points_and_rejects_missing_support() -> None:
    cameras = scene_cameras()
    xyz = np.array([[1.0, -3.0, 1.2], [1.2, -2.8, 1.1], [0.0, -2.0, 1.0]])
    visible: NDArray[np.bool_] = np.ones((3, 3), bool)
    visible[1:, 2] = False
    result = triangulate_points(project(cameras, xyz), visible, cameras, config=_CONFIG)
    np.testing.assert_allclose(result.positions[:2], xyz[:2], atol=2e-5)
    assert result.valid.tolist() == [True, True, False]
    assert not result.positions[2].any()
    assert result.reasons.tolist() == [PointRejection.VALID, PointRejection.VALID, PointRejection.INSUFFICIENT_VIEWS]
    np.testing.assert_array_equal(result.inliers, [[True, True, False]] * 3)


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_pair_consensus_ignores_one_bad_camera(order: tuple[int, ...]) -> None:
    cameras = scene_cameras()
    xyz = np.array([[1.0, -3.0, 1.2], [-2.0, 1.0, 0.8]])
    uv = project(cameras, xyz)
    uv[2, 0] += 100
    visible: NDArray[np.bool_] = np.ones((3, 2), bool)
    visible[1, 1] = False
    indices = list(order)
    result = triangulate_points(uv[indices], visible[indices], tuple(cameras[i] for i in order), config=_CONFIG)
    assert result.valid.tolist() == [True, True]
    assert result.reasons.tolist() == [PointRejection.VALID] * 2
    np.testing.assert_allclose(result.positions, xyz, atol=2e-5)
    np.testing.assert_array_equal(result.inliers[np.argsort(order)], [[True, True], [True, False], [False, True]])


@pytest.mark.parametrize(
    ("baseline", "pixels"),
    [
        pytest.param(0.0, [[640.0, 360.0], [640.0, 360.0]], id="coincident-cameras"),
        # A 0.5 px disparity at f=900 gives a ray angle far below the 1 degree gate.
        pytest.param(2.0, [[640.0, 360.0], [639.5, 360.0]], id="nearly-parallel-rays"),
    ],
)
def test_degenerate_rays_are_rejected(baseline: float, pixels: list[list[float]]) -> None:
    result = triangulate_points(np.array(pixels)[:, None], np.ones((2, 1), bool), stereo_cameras(baseline), config=_CONFIG)
    assert_rejected(result, PointRejection.DEGENERATE_RAYS)


def test_intersection_behind_the_cameras_is_rejected() -> None:
    # u_left - u_right = -180, so Z = f * baseline / disparity = -10 m.
    uv = np.array([[[640.0, 360.0]], [[820.0, 360.0]]])
    result = triangulate_points(uv, np.ones((2, 1), bool), stereo_cameras(), config=_CONFIG)
    assert_rejected(result, PointRejection.BEHIND_CAMERA)


def test_inconsistent_epipolar_observations_are_rejected() -> None:
    # Rectified cameras must agree in v; a 40 px difference cannot fit within 2 px.
    uv = np.array([[[640.0, 360.0]], [[460.0, 400.0]]])
    result = triangulate_points(uv, np.ones((2, 1), bool), stereo_cameras(), config=_CONFIG)
    assert_rejected(result, PointRejection.REPROJECTION)


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_mixed_pair_failures_have_a_camera_order_independent_reason(order: tuple[int, ...]) -> None:
    stereo = stereo_cameras()
    cameras = (*stereo, PinholeCamera("third", stereo[0].intrinsic, np.eye(3), np.array([-4.0, 0.0, 0.0])))
    # The first pair meets behind the cameras; the other pairs meet in front
    # but fail reprojection. Report the furthest gate reached by any hypothesis.
    uv = np.array([[[640.0, 360.0]], [[820.0, 360.0]], [[300.0, 450.0]]])
    indices = list(order)
    result = triangulate_points(uv[indices], np.ones((3, 1), bool), tuple(cameras[i] for i in order), config=_CONFIG)
    assert_rejected(result, PointRejection.REPROJECTION)


@pytest.mark.parametrize(
    ("success", "point"),
    [
        pytest.param(False, [1.0, -3.0, 1.2], id="not-converged"),
        pytest.param(True, [np.nan, -3.0, 1.2], id="nonfinite-solution"),
    ],
)
def test_solver_failure_is_rejected(monkeypatch: pytest.MonkeyPatch, success: bool, point: list[float]) -> None:
    cameras = scene_cameras()
    xyz = np.array([[1.0, -3.0, 1.2]])
    solver = Mock(return_value=OptimizeResult(success=success, x=np.array(point)))
    monkeypatch.setattr(triangulation, "least_squares", solver)
    result = triangulate_points(project(cameras, xyz), np.ones((3, 1), bool), cameras, config=_CONFIG)
    solver.assert_called_once()
    assert_rejected(result, PointRejection.OPTIMIZATION_FAILED)


def test_points_outside_the_height_bounds_are_rejected() -> None:
    cameras = scene_cameras()
    xyz = np.array([[1.0, -3.0, 6.0]])
    result = triangulate_points(project(cameras, xyz), np.ones((3, 1), bool), cameras, config=_CONFIG)
    assert_rejected(result, PointRejection.OUTSIDE_BOUNDS)


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
    assert gated.reasons.tolist() == [PointRejection.VALID, PointRejection.EXCESSIVE_SPEED, PointRejection.EXCESSIVE_SPEED, PointRejection.INSUFFICIENT_VIEWS, PointRejection.VALID]
    np.testing.assert_array_equal(gated.positions[~gated.valid], np.zeros((3, 3)))
    np.testing.assert_array_equal(gated.positions[gated.valid], positions[gated.valid])
    np.testing.assert_array_equal(gated.inliers, np.stack([gated.valid, gated.valid]))


def test_rejects_invalid_confidence() -> None:
    cameras = scene_cameras()
    uv = project(cameras, np.array([[1.0, -3.0, 1.2]]))
    with pytest.raises(ValueError, match="confidence"):
        triangulate_points(uv, np.ones((3, 1), bool), cameras, config=_CONFIG, confidence=np.full((3, 1), 1.5))


def test_backproject_to_plane_inverts_projection_and_rejects_rays_behind_the_camera() -> None:
    view = camera("a", [-8, -16, 9])
    ground = np.array([[0.0, 0.0, 0.0], [3.0, -5.0, 0.0], [-4.0, 10.0, 0.0]])
    uv, front = view.project(ground)
    assert front.all()
    points, valid = view.backproject_to_plane(uv)
    assert valid.all()
    np.testing.assert_allclose(points, ground, atol=1e-6)
    raised, valid_raised = view.backproject_to_plane(view.project(ground + [0, 0, 1.2])[0], height=1.2)
    assert valid_raised.all()
    np.testing.assert_allclose(raised[:, :2], ground[:, :2], atol=1e-6)
    # A pixel above the horizon meets the ground plane behind the camera.
    sky = view.project(np.array([[40.0, 80.0, 30.0]]))[0]
    points, valid = view.backproject_to_plane(np.concatenate((sky, [[np.nan, 1.0]])))
    assert not valid.any() and (points == 0).all()
