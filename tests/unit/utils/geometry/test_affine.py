from __future__ import annotations

from typing import cast

import numpy as np
from torchvision.transforms.functional import _get_inverse_affine_matrix

from src.utils.geometry.affine import (
    build_centered_affine_matrix,
    invert_homogeneous_matrix,
    to_cv2_affine,
    transform_points,
)


def _torchvision_forward_matrix(
    *,
    width: int,
    height: int,
    angle: float,
    translate: tuple[float, float],
    scale: float,
    shear: float,
) -> np.ndarray:
    inv = _get_inverse_affine_matrix(
        [width / 2.0, height / 2.0],
        angle,
        list(translate),
        scale,
        [shear, 0.0],
    )
    inverse_matrix = np.array(
        [[inv[0], inv[1], inv[2]], [inv[3], inv[4], inv[5]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return cast(np.ndarray, np.linalg.inv(inverse_matrix))


def test_centered_affine_matches_torchvision_single_shear_forward_matrix() -> None:
    cases = [
        (25.0, 1.1, 12.0, (10.0, -8.0)),
        (-15.0, 0.9, -10.0, (-12.0, 6.0)),
        (40.0, 1.0, 8.0, (0.0, 0.0)),
    ]
    for angle, scale, shear, translate in cases:
        actual = build_centered_affine_matrix(
            width=240,
            height=240,
            rotation_degrees=angle,
            translate=translate,
            scale=scale,
            shear_degrees=shear,
            shear_mode="torchvision",
        )
        expected = _torchvision_forward_matrix(
            width=240,
            height=240,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_transform_points_accepts_homogeneous_and_cv2_affine_matrices() -> None:
    matrix = build_centered_affine_matrix(
        width=320,
        height=180,
        rotation_degrees=12.0,
        translate=(5.0, -3.0),
        scale=1.05,
        shear_degrees=(4.0, -2.0),
        center=((320 - 1) / 2.0, (180 - 1) / 2.0),
    )
    points = np.array([[100.0, 50.0], [120.0, 90.0]], dtype=np.float32)

    from_homogeneous = transform_points(points, matrix)
    from_cv2 = transform_points(points, to_cv2_affine(matrix))

    np.testing.assert_allclose(from_homogeneous, from_cv2, atol=1e-6)


def test_invert_homogeneous_matrix_round_trips_affine_matrix() -> None:
    matrix = build_centered_affine_matrix(
        width=640,
        height=360,
        rotation_degrees=-7.0,
        translate=(21.0, -9.0),
        scale=0.95,
        shear_degrees=(3.0, 1.0),
    )
    inverse = invert_homogeneous_matrix(matrix)

    np.testing.assert_allclose(inverse @ matrix, np.eye(3), atol=1e-12)
