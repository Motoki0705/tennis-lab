"""Camera-view pose and semantic canonicalization contract."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.components.camera_view import (
    CAMERA_VIEW_MID_PLANE_TOLERANCE_M,
    AmbiguousCameraRelativeNearFarError,
    CameraViewCanonicalization,
    camera_view_canonicalization,
    validate_finite_camera_view_projection,
)
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    RigidTransform,
    SceneCamera,
)
from src.utils.schema.court import (
    CAMERA_VIEW_HALF_TURN_INDEX,
    HALF_LENGTH,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)


def _court() -> CourtInstance:
    transform = RigidTransform.identity()
    return CourtInstance(
        court_instance_id="court",
        candidate_id="candidate",
        scene_from_court=transform,
        court_from_scene=transform,
        fit_status="accepted",
        fit_metrics={"rms_error_m": 0.01},
        holdout_status="accepted",
        holdout_metrics={"rms_error_m": 0.02},
    )


def _look_at(center: tuple[float, float, float]) -> SceneCamera:
    center_array = np.asarray(center, dtype=np.float64)
    target: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
    forward = target - center_array
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((right, down, forward))
    matrix[:3, 3] = center_array
    return SceneCamera(
        camera_id=f"camera-{center[1]:g}",
        source_frame_index=0,
        width=640,
        height=480,
        intrinsics=(500.0, 0.0, 319.5, 0.0, 500.0, 239.5, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.from_matrix(matrix),
        image_path="generated/camera.png",
    )


@pytest.mark.parametrize(
    ("camera_y", "expected"),
    [(-30.0, tuple(range(14))), (30.0, CAMERA_VIEW_HALF_TURN_INDEX)],
)
def test_side_decision_drives_inventory_and_proper_canonical_pose(
    camera_y: float,
    expected: tuple[int, ...],
) -> None:
    canonical = camera_view_canonicalization(_look_at((0.0, camera_y, 12.0)), _court())

    assert canonical.semantic_to_physical == expected
    assert set(expected) == set(range(14))
    assert np.linalg.det(canonical.canonical_from_court.matrix()[:3, :3]) == pytest.approx(
        1.0
    )
    assert canonical.camera_center_canonical_m == pytest.approx((0.0, -30.0, 12.0))


@pytest.mark.parametrize(
    ("center", "expected_near_physical"),
    [
        ((0.0, -30.0, 12.0), (2, 3)),
        ((0.0, 30.0, 12.0), (1, 0)),
        ((-30.0, -2.0, 12.0), (2, 3)),
        ((30.0, 2.0, 12.0), (1, 0)),
    ],
)
def test_baseline_distance_switch_matches_camera_side_and_full_half_turn(
    center: tuple[float, float, float],
    expected_near_physical: tuple[int, int],
) -> None:
    canonical = camera_view_canonicalization(_look_at(center), _court())
    center_array = np.asarray(center, dtype=np.float64)
    plus_baseline = np.asarray((0.0, HALF_LENGTH, 0.0), dtype=np.float64)
    minus_baseline = np.asarray((0.0, -HALF_LENGTH, 0.0), dtype=np.float64)
    plus_distance_squared = float(np.sum((plus_baseline - center_array) ** 2))
    minus_distance_squared = float(np.sum((minus_baseline - center_array) ** 2))

    assert plus_distance_squared - minus_distance_squared == pytest.approx(
        -4.0 * center[1] * HALF_LENGTH
    )
    assert canonical.semantic_to_physical[2:4] == expected_near_physical
    if center[1] < 0.0:
        assert minus_distance_squared < plus_distance_squared
    else:
        assert plus_distance_squared < minus_distance_squared


@pytest.mark.parametrize(
    ("negative_center", "positive_center"),
    [
        ((0.0, -30.0, 12.0), (0.0, 30.0, 12.0)),
        ((-30.0, -2.0, 12.0), (30.0, 2.0, 12.0)),
    ],
)
def test_opposite_camera_pose_target_focal_and_projection_canonicalize_identically(
    negative_center: tuple[float, float, float],
    positive_center: tuple[float, float, float],
) -> None:
    court = _court()
    negative_camera = _look_at(negative_center)
    positive_camera = _look_at(positive_center)
    negative = camera_view_canonicalization(negative_camera, court)
    positive = camera_view_canonicalization(positive_camera, court)

    np.testing.assert_allclose(
        negative.camera_from_canonical.matrix(),
        positive.camera_from_canonical.matrix(),
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        negative.camera_center_canonical_m,
        positive.camera_center_canonical_m,
        atol=1.0e-12,
        rtol=0.0,
    )
    target_court = np.asarray(((0.0, 0.0, 0.0),), dtype=np.float64)
    np.testing.assert_allclose(
        negative.canonical_from_court.apply(target_court),
        positive.canonical_from_court.apply(target_court),
        atol=0.0,
        rtol=0.0,
    )
    assert negative_camera.intrinsics[0] == positive_camera.intrinsics[0]

    known_court = np.asarray(
        court_keypoints_3d(STANDARD_COURT_CONFIG)[:14].numpy(),
        dtype=np.float64,
    )
    for camera, canonical in (
        (negative_camera, negative),
        (positive_camera, positive),
    ):
        original_camera = camera.camera_to_scene.inverse().apply(
            court.scene_from_court.apply(known_court)
        )
        canonical_camera = canonical.camera_from_canonical.apply(
            canonical.canonical_from_court.apply(known_court)
        )
        np.testing.assert_allclose(
            original_camera,
            canonical_camera,
            atol=1.0e-12,
            rtol=0.0,
        )
        intrinsics = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
        original_pixels = (original_camera @ intrinsics.T)[:, :2] / original_camera[
            :, 2:3
        ]
        canonical_pixels = (canonical_camera @ intrinsics.T)[:, :2] / canonical_camera[
            :, 2:3
        ]
        np.testing.assert_allclose(
            original_pixels,
            canonical_pixels,
            atol=1.0e-12,
            rtol=0.0,
        )


@pytest.mark.parametrize("camera_y", [-1.0e-6, 0.0, 1.0e-6])
def test_mid_plane_is_rejected_inclusively(camera_y: float) -> None:
    assert CAMERA_VIEW_MID_PLANE_TOLERANCE_M == 1.0e-6
    with pytest.raises(AmbiguousCameraRelativeNearFarError):
        camera_view_canonicalization(_look_at((5.0, camera_y, 12.0)), _court())


@pytest.mark.parametrize(
    ("camera_y", "expected"),
    [
        (np.nextafter(-1.0e-6, -np.inf), tuple(range(14))),
        (np.nextafter(1.0e-6, np.inf), CAMERA_VIEW_HALF_TURN_INDEX),
    ],
)
def test_finite_camera_just_outside_mid_plane_is_accepted(
    camera_y: float,
    expected: tuple[int, ...],
) -> None:
    canonical = camera_view_canonicalization(
        _look_at((5.0, camera_y, 12.0)),
        _court(),
    )

    assert canonical.semantic_to_physical == expected


def test_invalid_type_inventory_and_nonfinite_projection_reject() -> None:
    with pytest.raises(TypeError, match="SceneCamera"):
        camera_view_canonicalization(object(), _court())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="identity or the shared"):
        CameraViewCanonicalization(
            semantic_to_physical=tuple(reversed(range(14))),
            canonical_from_court=RigidTransform.identity(),
            camera_from_canonical=RigidTransform.identity(),
            camera_center_canonical_m=(0.0, -30.0, 12.0),
        )

    valid_uv = np.column_stack((np.arange(14, dtype=np.float64), np.zeros(14)))
    valid_uv[4, 0], valid_uv[6, 0] = 4.0, 5.0
    valid_uv[5, 0], valid_uv[7, 0] = 6.0, 7.0
    validate_finite_camera_view_projection(valid_uv)
    tied = valid_uv.copy()
    tied[1, 0] = tied[0, 0]
    validate_finite_camera_view_projection(tied)
    nonfinite = valid_uv.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_finite_camera_view_projection(nonfinite)
    with pytest.raises(ValueError, match=r"\(14, 2\)"):
        validate_finite_camera_view_projection(valid_uv[:13])
