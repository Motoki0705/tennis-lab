"""Strict target-court pose target and decode contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.tasks.court_detection.data.contracts import CourtPoseAuthority
from src.tasks.court_detection.geometry.pose import (
    MIN_PROJECTION_REFERENCE_POINTS,
    POSE10D_RAW_ORDER,
    PROJECTIVE_DEPTH_EPS_M,
    CourtPoseTarget,
    build_pose_target,
    canonical_semantic_court_points,
    decode_pose10d_strict,
    project_canonical_points,
    validate_projection_round_trip,
    validate_square_intrinsics,
)


def _authority(
    camera_y: float = -30.0,
    *,
    center_xyz: tuple[float, float, float] | None = None,
) -> CourtPoseAuthority:
    center = np.asarray(
        center_xyz if center_xyz is not None else (2.0, camera_y, 12.0),
        dtype=np.float64,
    )
    forward = -center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    camera_to_scene = np.eye(4, dtype=np.float64)
    camera_to_scene[:3, :3] = np.column_stack((right, down, forward))
    camera_to_scene[:3, 3] = center
    return CourtPoseAuthority(
        source_schema="canonical_court_dataset_v3",
        camera=SceneCamera(
            camera_id="camera",
            source_frame_index=0,
            width=640,
            height=480,
            intrinsics=(500.0, 0.0, 319.5, 0.0, 500.0, 239.5, 0.0, 0.0, 1.0),
            camera_to_scene=RigidTransform.from_matrix(camera_to_scene),
            image_path="generated/camera.png",
        ),
        target_court=TargetCourtBinding(
            court_instance_id="court",
            candidate_id="candidate",
            scene_from_court=RigidTransform.identity(),
            selection_seed=779,
        ),
    )


def _negative_y_camera_target(camera_y: float) -> CourtPoseTarget:
    """Return a proper pose looking toward canonical negative Y."""
    rotation_canonical_from_camera = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    intrinsics = torch.tensor(
        [[500.0, 0.0, 319.5], [0.0, 500.0, 239.5], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    return CourtPoseTarget(
        translation_m=torch.tensor([0.0, camera_y, 0.0], dtype=torch.float64),
        rotation=rotation_canonical_from_camera,
        log_focal=torch.log(intrinsics[0, 0]),
        intrinsics=intrinsics,
        semantic_to_physical=torch.arange(14),
    )


@pytest.mark.parametrize("camera_y", [-30.0, 30.0])
def test_v3_authority_derives_exact_pose_order_and_projection(camera_y: float) -> None:
    target = build_pose_target(_authority(camera_y))

    assert POSE10D_RAW_ORDER == (
        "tx",
        "ty",
        "tz",
        "a11",
        "a12",
        "a13",
        "a21",
        "a22",
        "a23",
        "logf",
    )
    assert target.raw_values.shape == (10,)
    torch.testing.assert_close(target.raw_values[:3], target.translation_m)
    torch.testing.assert_close(target.raw_values[3:9], target.rotation[:2].reshape(6))
    assert float(target.raw_values[9]) == pytest.approx(math.log(500.0))
    decoded = decode_pose10d_strict(target.raw_values.unsqueeze(0))
    torch.testing.assert_close(decoded.translation_m[0], target.translation_m)
    torch.testing.assert_close(decoded.rotation[0], target.rotation)
    assert float(decoded.focal_px[0]) == pytest.approx(500.0, rel=1.0e-6)

    expected = project_canonical_points(
        target,
        canonical_semantic_court_points(target),
    )
    validate_projection_round_trip(target, expected.float())


def test_isotropic_letterbox_updates_focal_and_principal_point() -> None:
    homography = torch.tensor(
        [[0.4, 0.0, 0.1], [0.0, 0.4, 32.2], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    target = build_pose_target(_authority(), source_to_output=homography)

    assert float(target.intrinsics[0, 0]) == pytest.approx(200.0)
    assert float(target.intrinsics[1, 1]) == pytest.approx(200.0)
    assert float(target.intrinsics[0, 2]) == pytest.approx(127.9, abs=1.0e-5)
    assert float(target.intrinsics[1, 2]) == pytest.approx(128.0, abs=1.0e-5)
    expected = project_canonical_points(
        build_pose_target(_authority()),
        canonical_semantic_court_points(target),
    )
    ones = torch.ones((14, 1), dtype=torch.float64)
    homogeneous = torch.cat((expected, ones), dim=1) @ homography.T
    validate_projection_round_trip(target, homogeneous[:, :2].float())


def test_round_trip_uses_valid_references_from_mixed_camera_depths() -> None:
    target = build_pose_target(_authority(center_xyz=(0.0, -1.0, 2.0)))
    points = canonical_semantic_court_points(target)
    camera_points = (points - target.translation_m) @ target.rotation
    expected = project_canonical_points(target, points)

    assert bool(torch.any(camera_points[:, 2] > PROJECTIVE_DEPTH_EPS_M))
    assert bool(torch.any(camera_points[:, 2] < -PROJECTIVE_DEPTH_EPS_M))
    behind_index = int(
        torch.nonzero(camera_points[:, 2] < -PROJECTIVE_DEPTH_EPS_M)[0, 0]
    )
    expected[behind_index] += 100.0
    validate_projection_round_trip(target, expected.float())


def test_projection_rejects_near_zero_depth_before_reference_filtering() -> None:
    target = _negative_y_camera_target(-11.885)

    with pytest.raises(ValueError, match="zero/near-zero projective depth"):
        validate_projection_round_trip(
            target,
            torch.zeros((14, 2), dtype=torch.float64),
        )


@pytest.mark.parametrize(
    ("camera_y", "error"),
    [
        (-20.0, f"at least {MIN_PROJECTION_REFERENCE_POINTS}"),
        (-10.0, "non-collinear"),
    ],
)
def test_round_trip_rejects_insufficient_positive_depth_evidence(
    camera_y: float,
    error: str,
) -> None:
    target = _negative_y_camera_target(camera_y)
    expected = project_canonical_points(
        target,
        canonical_semantic_court_points(target),
    )

    with pytest.raises(ValueError, match=error):
        validate_projection_round_trip(target, expected)


def test_projection_rejects_nonfinite_points_and_expected_uv() -> None:
    target = build_pose_target(_authority(center_xyz=(0.0, -1.0, 2.0)))
    points = canonical_semantic_court_points(target)
    points[0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        project_canonical_points(target, points)

    expected = project_canonical_points(
        target,
        canonical_semantic_court_points(target),
    )
    clean_points = canonical_semantic_court_points(target)
    camera_points = (clean_points - target.translation_m) @ target.rotation
    behind_index = int(torch.nonzero(camera_points[:, 2] < 0.0)[0, 0])
    expected[behind_index, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        validate_projection_round_trip(target, expected)


def test_round_trip_rejects_mismatch_among_positive_depth_references() -> None:
    target = build_pose_target(_authority(center_xyz=(0.0, -1.0, 2.0)))
    points = canonical_semantic_court_points(target)
    camera_points = (points - target.translation_m) @ target.rotation
    expected = project_canonical_points(target, points).float()
    valid_index = int(torch.nonzero(camera_points[:, 2] > 0.0)[0, 0])
    expected[valid_index, 0] += 1.0e-2

    with pytest.raises(ValueError, match="round-trip exceeds"):
        validate_projection_round_trip(target, expected)


@pytest.mark.parametrize(
    "raw",
    [
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 6.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 6.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0e-7, 0.0, 6.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, float("nan")]]),
    ],
)
def test_pose_decode_rejects_nonfinite_and_predecode_degeneracy(raw: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="finite|norm"):
        decode_pose10d_strict(raw)


@pytest.mark.parametrize(
    "intrinsics",
    [
        torch.tensor([[500.0, 1.1e-6, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
        torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.001, 240.0], [0.0, 0.0, 1.0]]),
        torch.tensor([[float("nan"), 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
    ],
)
def test_square_focal_and_skew_are_strict(intrinsics: torch.Tensor) -> None:
    with pytest.raises((ValueError, TypeError), match="finite|fx=fy|skew"):
        validate_square_intrinsics(intrinsics)
