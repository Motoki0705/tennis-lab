"""Known-camera geometry, ambiguous sides, and source timeline reconstruction."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.pipeline.components.camera_geometry import (
    CalibrationSet,
    CameraGeometryConfig,
    LocalCourtCalibration,
    SideEvidence,
    resolve_camera_geometry,
)
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.utilts.timeline import (
    association_frame_indices,
)
from src.utils.geometry.triangulation import (
    PinholeCamera,
    PointTriangulationConfig,
    triangulate_points,
)


def camera(name: str, center: list[float]) -> PinholeCamera:
    center_array = np.asarray(center, np.float64)
    forward = np.array([0, 0, 1.]) - center_array
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.])
    right /= np.linalg.norm(right)
    rotation = np.stack((right, np.cross(forward, right), forward))
    return PinholeCamera(name, np.array([[900., 0, 640], [0, 900, 360], [0, 0, 1]]), rotation, -rotation @ center_array)


def scene_cameras() -> tuple[PinholeCamera, ...]:
    return (camera("a", [-8, -16, 9]), camera("b", [8, -16, 9]), camera("c", [5, 16, 9]))


def test_triangulation_recovers_known_points_and_rejects_missing_support() -> None:
    cameras = scene_cameras()
    xyz = np.array([[1., -3., 1.2], [1.2, -2.8, 1.1], [0., -2., 1.]])
    uv = np.stack([c.project(xyz)[0] for c in cameras]).astype(np.float32)
    visible: NDArray[np.bool_] = np.ones((3, 3), bool)
    visible[1:, 2] = False
    result = triangulate_points(uv, visible, cameras, config=PointTriangulationConfig(2., (-0.5, 4.)))
    np.testing.assert_allclose(result.positions[:2], xyz[:2], atol=2e-5)
    assert result.valid.tolist() == [True, True, False]
    assert not result.positions[2].any()
    assert result.reasons[2] == 1


def test_pair_consensus_ignores_one_bad_camera() -> None:
    cameras = scene_cameras()
    xyz = np.array([[1., -3., 1.2]])
    uv = np.stack([c.project(xyz)[0] for c in cameras]).astype(np.float32)
    uv[2] += 100
    result = triangulate_points(uv, np.ones((3, 1), bool), cameras, config=PointTriangulationConfig(2., (-0.5, 4.)))
    assert result.valid[0]
    np.testing.assert_allclose(result.positions[0], xyz[0], atol=2e-5)
    assert not result.inliers[2, 0]


def side_inputs() -> tuple[CalibrationSet, SideEvidence]:
    cameras = scene_cameras()
    local = (cameras[0], cameras[1], cameras[2].half_turned(True))
    calibration = CalibrationSet(tuple(LocalCourtCalibration(c, i, 0, np.eye(3), 0., (0, 2, 4)) for i, c in enumerate(local)), {})
    xyz = np.tile(np.array([[0., -4., 1.], [.4, -4., 1.], [0., -4., 1.6], [.4, -4., 1.6]]), (10, 1, 1))
    xyz[:, :, 1] += np.arange(10)[:, None] * .1
    uv = np.stack([c.project(xyz)[0] for c in cameras]).astype(np.float32)[None]
    return calibration, SideEvidence("plcs", uv, np.ones(uv.shape[:-1], bool), 5.)


def test_disagreeing_side_is_decided_by_geometry() -> None:
    calibration, evidence = side_inputs()
    result = resolve_camera_geometry(calibration, "a", (np.array([False, False, True]), np.array([False, False, False])), (evidence,), config=CameraGeometryConfig())
    assert result.view_half_turns == (False, False, True)
    assert len(result.document["side_candidates"]) == 2


def test_agreement_cannot_bypass_evidence_or_geometry() -> None:
    calibration, evidence = side_inputs()
    with pytest.raises(ReconstructionUnavailable, match="geometric"):
        resolve_camera_geometry(calibration, "a", (np.zeros(3, bool),), (evidence,), config=CameraGeometryConfig())
    evidence.visibility[:, 1:] = False
    with pytest.raises(ReconstructionUnavailable, match="Too few"):
        resolve_camera_geometry(calibration, "a", (np.zeros(3, bool),), (evidence,), config=CameraGeometryConfig())


def test_full_clip_grid_preserves_duration_without_windowing() -> None:
    indices = association_frame_indices(1607, 59.94006)
    assert len(indices) == 805
    assert indices[0] == 0 and indices[-1] < 1607 and (np.diff(indices) > 0).all()
    with pytest.raises(ReconstructionUnavailable, match="maximum"):
        association_frame_indices(2050, 30.)
