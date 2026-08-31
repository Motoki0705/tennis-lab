"""Analytic tests for task-agnostic OpenCV camera geometry."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.utils.rendering import (
    camera_coverage_segments,
    camera_frustum_corners,
    camera_frustum_segments,
    camera_trajectory_points,
    camera_trajectory_segments,
    camera_view_direction_segments,
)


def _intrinsics() -> NDArray[np.float64]:
    return np.asarray(
        (
            (2.0, 0.0, 1.0),
            (0.0, 2.0, 1.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _transform(
    *,
    rotation: NDArray[np.float64] | None = None,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> NDArray[np.float64]:
    matrix = np.eye(4, dtype=np.float64)
    if rotation is not None:
        matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def test_identity_frustum_uses_right_down_forward_opencv_axes() -> None:
    corners = camera_frustum_corners(
        _intrinsics(),
        (3, 3),
        _transform(),
        depth=2.0,
    )

    expected = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (-1.0, -1.0, 2.0),
            (1.0, -1.0, 2.0),
            (1.0, 1.0, 2.0),
            (-1.0, 1.0, 2.0),
        ),
        dtype=np.float64,
    )
    assert corners.dtype == np.float64
    np.testing.assert_allclose(corners, expected, atol=1.0e-12, rtol=0.0)


def test_frustum_segment_order_is_rays_then_clockwise_perimeter() -> None:
    corners = camera_frustum_corners(
        _intrinsics(),
        (3, 3),
        _transform(),
        depth=2.0,
    )
    segments = camera_frustum_segments(
        _intrinsics(),
        (3, 3),
        _transform(),
        depth=2.0,
    )

    expected_indices = np.asarray(
        ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1))
    )
    assert segments.shape == (8, 2, 3)
    np.testing.assert_array_equal(segments, corners[expected_indices])


def test_rotated_pose_maps_camera_axes_without_coordinate_conversion() -> None:
    # +90 degrees around world Y: camera right -> world -Z,
    # camera down -> world +Y, camera forward -> world +X.
    rotation = np.asarray(
        (
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    transform = _transform(rotation=rotation, translation=(10.0, 20.0, 30.0))

    corners = camera_frustum_corners(_intrinsics(), (3, 3), transform, depth=2.0)
    directions = camera_view_direction_segments(transform[None, ...], length=4.0)

    expected_corners = np.asarray(
        (
            (10.0, 20.0, 30.0),
            (12.0, 19.0, 31.0),
            (12.0, 19.0, 29.0),
            (12.0, 21.0, 29.0),
            (12.0, 21.0, 31.0),
        )
    )
    np.testing.assert_allclose(corners, expected_corners, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(
        directions,
        np.asarray((((10.0, 20.0, 30.0), (14.0, 20.0, 30.0)),)),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_trajectory_primitives_preserve_input_order_and_do_not_close_path() -> None:
    transforms = np.stack(
        (
            _transform(translation=(2.0, 0.0, 0.0)),
            _transform(translation=(-1.0, 3.0, 0.0)),
            _transform(translation=(5.0, 1.0, 4.0)),
        )
    )

    points = camera_trajectory_points(transforms)
    segments = camera_trajectory_segments(transforms)

    expected_points = np.asarray(((2.0, 0.0, 0.0), (-1.0, 3.0, 0.0), (5.0, 1.0, 4.0)))
    np.testing.assert_array_equal(points, expected_points)
    np.testing.assert_array_equal(
        segments,
        np.stack((expected_points[:-1], expected_points[1:]), axis=1),
    )


def test_one_camera_trajectory_has_explicit_empty_segment_shape() -> None:
    segments = camera_trajectory_segments(_transform()[None, ...])
    assert segments.shape == (0, 2, 3)
    assert segments.dtype == np.float64


def test_coverage_preserves_camera_order_and_requires_per_camera_inputs() -> None:
    transforms = np.stack(
        (
            _transform(translation=(4.0, 0.0, 0.0)),
            _transform(translation=(-3.0, 2.0, 1.0)),
        )
    )
    intrinsics = np.stack((_intrinsics(), _intrinsics()))
    image_sizes = np.asarray(((3, 3), (3, 3)), dtype=np.int64)

    coverage = camera_coverage_segments(
        intrinsics,
        image_sizes,
        transforms,
        depth=2.0,
    )

    assert coverage.shape == (2, 8, 2, 3)
    np.testing.assert_array_equal(
        coverage[0],
        camera_frustum_segments(intrinsics[0], (3, 3), transforms[0], depth=2.0),
    )
    np.testing.assert_array_equal(
        coverage[1],
        camera_frustum_segments(intrinsics[1], (3, 3), transforms[1], depth=2.0),
    )
    np.testing.assert_array_equal(coverage[:, 0, 0], transforms[:, :3, 3])


@pytest.mark.parametrize("depth", [0.0, -1.0, np.nan, np.inf])
def test_frustum_rejects_non_positive_or_non_finite_depth(depth: float) -> None:
    with pytest.raises(ValueError, match="depth must be a positive finite"):
        camera_frustum_corners(_intrinsics(), (3, 3), _transform(), depth=depth)


@pytest.mark.parametrize("image_size", [(0, 3), (3, -1), (3.0, 3), [3, 3]])
def test_frustum_rejects_invalid_image_size(image_size: object) -> None:
    with pytest.raises(ValueError, match="image_size"):
        camera_frustum_corners(
            _intrinsics(),
            image_size,  # type: ignore[arg-type]
            _transform(),
            depth=1.0,
        )


@pytest.mark.parametrize(
    ("intrinsics", "message"),
    [
        (np.eye(2), "finite 3x3"),
        (
            np.asarray(((2.0, 0.0, np.nan), (0.0, 2.0, 1.0), (0.0, 0.0, 1.0))),
            "finite 3x3",
        ),
        (
            np.asarray(((0.0, 0.0, 1.0), (0.0, 2.0, 1.0), (0.0, 0.0, 1.0))),
            "focal lengths",
        ),
        (
            np.asarray(((2.0, 0.0, 3.0), (0.0, 2.0, 1.0), (0.0, 0.0, 1.0))),
            "principal point",
        ),
        (np.asarray(((2.0, 0.0, 1.0), (0.0, 2.0, 1.0), (0.0, 1.0, 1.0))), "bottom row"),
        (np.asarray(((2.0, 2.0, 1.0), (2.0, 2.0, 1.0), (0.0, 0.0, 1.0))), "invertible"),
    ],
)
def test_frustum_rejects_invalid_intrinsics(
    intrinsics: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        camera_frustum_corners(intrinsics, (3, 3), _transform(), depth=1.0)


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (np.eye(3), "finite \\(4, 4\\)"),
        (
            np.asarray(
                (
                    (1.0, 0.0, 0.0, np.nan),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                )
            ),
            "finite \\(4, 4\\)",
        ),
        (np.diag((1.0, 1.0, 1.0, 2.0)), "homogeneous bottom row"),
        (np.diag((2.0, 1.0, 1.0, 1.0)), "orthonormal"),
        (np.diag((-1.0, 1.0, 1.0, 1.0)), "determinant \\+1"),
    ],
)
def test_frustum_rejects_non_se3_transform(
    transform: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        camera_frustum_corners(_intrinsics(), (3, 3), transform, depth=1.0)


def test_ordered_primitives_reject_empty_or_mismatched_camera_batches() -> None:
    empty: NDArray[np.float64] = np.empty((0, 4, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="finite \\(N, 4, 4\\)"):
        camera_trajectory_points(empty)

    transforms = np.stack((_transform(), _transform()))
    with pytest.raises(ValueError, match="intrinsics must be a finite"):
        camera_coverage_segments(
            _intrinsics()[None, ...],
            np.asarray(((3, 3), (3, 3))),
            transforms,
            depth=1.0,
        )
    with pytest.raises(ValueError, match="image_sizes must have shape"):
        camera_coverage_segments(
            np.stack((_intrinsics(), _intrinsics())),
            np.asarray(((3, 3),)),
            transforms,
            depth=1.0,
        )
