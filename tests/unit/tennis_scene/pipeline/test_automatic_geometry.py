"""Known-camera geometry, ambiguous sides, and the full-clip sampling grid."""

from __future__ import annotations

import numpy as np
import pytest

from src.tennis_scene.pipeline.components.camera_geometry import (
    CalibrationSet,
    CameraGeometryConfig,
    LocalCourtCalibration,
    SideEvidence,
    resolve_camera_geometry,
)
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.frame_sampling import sampled_frame_indices
from src.utils.geometry.triangulation import PinholeCamera


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


def side_inputs() -> tuple[CalibrationSet, SideEvidence]:
    cameras = scene_cameras()
    local = (cameras[0], cameras[1], cameras[2].half_turned(True))
    calibration = CalibrationSet(tuple(LocalCourtCalibration(c, i, 0, np.eye(3), 0.) for i, c in enumerate(local)), {})
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
    indices = sampled_frame_indices(1607, 59.94006, max_frames=1024)
    assert len(indices) == 805
    assert indices[0] == 0 and indices[-1] < 1607 and (np.diff(indices) > 0).all()
    with pytest.raises(ReconstructionUnavailable, match="maximum"):
        sampled_frame_indices(2050, 30., max_frames=1024)


def _static_court(status: str = "ok") -> CourtKPResult:
    from tests.unit.tennis_scene.pipeline.test_auto_pipeline import inputs
    court, _, _ = inputs()
    assert court.diagnostics is not None
    cameras = [{"frames": [{**c["frames"][0], "status": status if v == 1 else "ok"}]} for v, c in enumerate(court.diagnostics["cameras"])]
    return CourtKPResult(court.keypoints[:, :1], court.visibility[:, :1], np.array([0], np.int32),
                         {"output_keypoint_contract": "camera_view_v2", "cameras": cameras})


def test_static_calibration_fits_every_accepted_view_from_frame_zero() -> None:
    from src.tennis_scene.pipeline.components.camera_geometry import (
        calibrate_local_courts,
    )
    calibration = calibrate_local_courts(_static_court(), ("cam0", "cam1", "cam2"), size=(1280, 720), config=CameraGeometryConfig())
    assert calibration.camera_ids == ("cam0", "cam1", "cam2") and not calibration.excluded
    truth = (camera("cam0", [-8, -16, 9]), camera("cam1", [8, -16, 9]), camera("cam2", [5, 16, 9]).half_turned(True))
    for view, expected in zip(calibration.views, truth, strict=True):
        np.testing.assert_allclose(view.camera.center, expected.center, atol=1e-3)


def test_rejected_homography_excludes_the_camera_explicitly() -> None:
    from src.tennis_scene.pipeline.components.camera_geometry import (
        calibrate_local_courts,
    )
    calibration = calibrate_local_courts(_static_court("joint_optimization_failed"), ("cam0", "cam1", "cam2"),
                                         size=(1280, 720), config=CameraGeometryConfig())
    assert calibration.camera_ids == ("cam0", "cam2")
    assert calibration.excluded == {"cam1": "no_accepted_homography"}


def test_static_calibration_requires_exactly_frame_zero() -> None:
    from dataclasses import replace

    from src.tennis_scene.pipeline.components.camera_geometry import (
        calibrate_local_courts,
    )
    court = _static_court()
    two_frames = replace(court, keypoints=np.repeat(court.keypoints, 2, axis=1), visibility=np.repeat(court.visibility, 2, axis=1),
                         frame_indices=np.array([0, 1], np.int32))
    with pytest.raises(ValueError, match="one frame-0"):
        calibrate_local_courts(two_frames, ("cam0", "cam1", "cam2"), size=(1280, 720), config=CameraGeometryConfig())
