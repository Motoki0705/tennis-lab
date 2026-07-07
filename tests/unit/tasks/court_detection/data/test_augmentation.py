"""Tests for court-detection joint spatial augmentations.

The keypoint and image/mask affine warps share one ``build_centered_affine_matrix``
per draw (``src.tasks.court_detection.data.augmentation._random_affine_matrix``).
These tests guard against the two failure modes a closed-form point mapping is
prone to: rotating keypoints opposite to the image, and mis-coupling the shear
term between the x and y axes.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms.functional import _get_inverse_affine_matrix

from src.tasks.court_detection.data import augmentation as aug
from src.tasks.court_detection.data.augmentation import KPRandomAffine, SegRandomAffine
from src.utils.geometry.affine import (
    build_centered_affine_matrix,
    invert_homogeneous_matrix,
    to_pil_affine_coefficients,
    transform_points,
)


def _image_point_after_affine(
    px: float,
    py: float,
    *,
    width: int,
    height: int,
    angle: float,
    scale: float,
    shear: float,
    tx: float,
    ty: float,
) -> tuple[float, float]:
    """Warp a bright dot with ``TF.affine`` and return its centroid."""
    img: np.ndarray = np.zeros((height, width), dtype=np.uint8)
    row, col = int(round(py)), int(round(px))
    img[row - 1 : row + 2, col - 1 : col + 2] = 255
    warped = TF.affine(
        Image.fromarray(img),
        angle=angle,
        translate=[tx, ty],
        scale=scale,
        shear=[shear],
        interpolation=TF.InterpolationMode.NEAREST,
    )
    ys, xs = np.where(np.array(warped) > 50)
    assert len(xs) > 0, "warped point fell outside the canvas"
    return float(xs.mean()), float(ys.mean())


def _apply_kp_affine(
    px: float,
    py: float,
    *,
    width: int,
    height: int,
    angle: float,
    scale: float,
    shear: float,
    tx: float,
    ty: float,
) -> tuple[float, float]:
    """Run a single point through ``KPRandomAffine`` with fixed parameters.

    ``KPRandomAffine`` samples its parameters internally via ``random.uniform``
    in the order ``angle, tx, ty, scale, shear``; patch that call to inject
    known values so the transform is fully deterministic.
    """
    transform = KPRandomAffine(
        degrees=abs(angle) + 1.0,
        translate=(0.5, 0.5),
        scale=(min(scale, 1.0), max(scale, 1.0)),
        shear=abs(shear) + 1.0,
    )
    img = Image.new("RGB", (width, height))
    kps = np.array([[px, py]], dtype=np.float32)
    values = iter([angle, tx, ty, scale, shear])

    with patch.object(aug.random, "uniform", side_effect=lambda *_: next(values)):
        _, out = transform(img, kps)
    return float(out[0, 0]), float(out[0, 1])


def test_kp_affine_matches_image_rotation() -> None:
    """A pure rotation must move keypoints the same way it moves the image."""
    w = h = 240
    px, py = 180.0, 120.0  # 60 px right of centre
    for angle in (30.0, -20.0, 45.0):
        img_x, img_y = _image_point_after_affine(
            px, py, width=w, height=h, angle=angle,
            scale=1.0, shear=0.0, tx=0.0, ty=0.0,
        )
        kp_x, kp_y = _apply_kp_affine(
            px, py, width=w, height=h, angle=angle,
            scale=1.0, shear=0.0, tx=0.0, ty=0.0,
        )
        assert math.hypot(kp_x - img_x, kp_y - img_y) < 2.0, (
            f"angle={angle}: kp=({kp_x:.1f},{kp_y:.1f}) image=({img_x:.1f},{img_y:.1f})"
        )


def _torchvision_forward_point(
    px: float,
    py: float,
    *,
    width: int,
    height: int,
    angle: float,
    scale: float,
    shear: float,
    tx: float,
    ty: float,
) -> tuple[float, float]:
    """Exact forward point mapping from torchvision's affine matrix.

    ``_get_inverse_affine_matrix`` returns the output→input matrix used by the
    image warp; inverting it gives the input→output map a keypoint must follow.
    This is a noise-free oracle (unlike thresholding a warped image), so it can
    pin the closed-form coupling of shear and scale precisely. ``shear`` is
    passed as ``[shear, shear]`` (not ``[shear, 0.0]``) because
    ``_random_affine_matrix`` couples shear_x=shear_y to match what
    ``TF.affine(..., shear=[shear_val])`` renders: torchvision's public
    ``affine()`` normalizes a one-element shear list to ``[value, value]``
    before building the matrix, so a single shear knob is coupled, not x-only.
    """
    inv = _get_inverse_affine_matrix(
        [width / 2.0, height / 2.0], angle, [tx, ty], scale, [shear, shear]
    )
    m_inv = np.array([[inv[0], inv[1], inv[2]], [inv[3], inv[4], inv[5]], [0.0, 0.0, 1.0]])
    out = np.linalg.inv(m_inv) @ np.array([px, py, 1.0])
    return float(out[0]), float(out[1])


def test_kp_affine_matches_torchvision_matrix_with_shear_and_scale() -> None:
    """Combined rotation + scale + shear + translation matches torchvision exactly."""
    w = h = 240
    px, py = 150.0, 110.0
    cases = [
        (25.0, 1.1, 12.0, 10.0, -8.0),
        (-15.0, 0.9, -10.0, -12.0, 6.0),
        (40.0, 1.0, 8.0, 0.0, 0.0),
    ]
    for angle, scale, shear, tx, ty in cases:
        oracle_x, oracle_y = _torchvision_forward_point(
            px, py, width=w, height=h, angle=angle,
            scale=scale, shear=shear, tx=tx, ty=ty,
        )
        kp_x, kp_y = _apply_kp_affine(
            px, py, width=w, height=h, angle=angle,
            scale=scale, shear=shear, tx=tx, ty=ty,
        )
        assert math.hypot(kp_x - oracle_x, kp_y - oracle_y) < 1e-3, (
            f"({angle},{scale},{shear}): kp=({kp_x:.3f},{kp_y:.3f}) "
            f"oracle=({oracle_x:.3f},{oracle_y:.3f})"
        )


def test_kp_affine_matches_image_warp_with_shear() -> None:
    """A sheared affine must move keypoints the same way it moves the image.

    Regression test for a coupling bug: the previous closed-form keypoint
    mapping assumed an x-only shear (``shear_y=0``), but the image warp it
    paired with actually applied a coupled ``shear_x=shear_y`` (torchvision
    normalizes a one-element ``shear=[value]`` list that way). That silently
    desynced keypoints from the image whenever ``shear != 0``.
    """
    w = h = 240
    px, py = 170.0, 130.0
    cases = [
        (15.0, 1.0, 10.0, 0.0, 0.0),
        (-10.0, 1.05, -14.0, 6.0, -4.0),
    ]
    for angle, scale, shear, tx, ty in cases:
        img_x, img_y = _image_point_after_affine(
            px, py, width=w, height=h, angle=angle,
            scale=scale, shear=shear, tx=tx, ty=ty,
        )
        kp_x, kp_y = _apply_kp_affine(
            px, py, width=w, height=h, angle=angle,
            scale=scale, shear=shear, tx=tx, ty=ty,
        )
        assert math.hypot(kp_x - img_x, kp_y - img_y) < 2.0, (
            f"angle={angle} shear={shear}: "
            f"kp=({kp_x:.1f},{kp_y:.1f}) image=({img_x:.1f},{img_y:.1f})"
        )


def test_seg_affine_matches_shared_matrix_warp() -> None:
    """Seg image/mask warp must equal a PIL AFFINE warp using the shared matrix."""
    w, h = 137, 91
    rng = np.random.default_rng(0)
    img = Image.fromarray(
        rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8), mode="RGB",
    )
    mask = Image.fromarray(
        rng.integers(0, 5, size=(h, w), dtype=np.uint8), mode="L",
    )

    transform = SegRandomAffine(degrees=25.0, translate=(0.2, 0.2), scale=(0.8, 1.3), shear=15.0)
    angle, tx, ty, scale, shear = 12.0, 5.0, -3.0, 1.1, 8.0
    values = iter([angle, tx, ty, scale, shear])
    with patch.object(aug.random, "uniform", side_effect=lambda *_: next(values)):
        out_img, out_mask = transform(img, mask)

    matrix = build_centered_affine_matrix(
        width=w, height=h, rotation_degrees=angle, translate=(tx, ty), scale=scale,
        shear_degrees=(shear, shear), shear_mode="torchvision",
    )
    coeffs = to_pil_affine_coefficients(invert_homogeneous_matrix(matrix))
    expected_img = img.transform(img.size, Image.AFFINE, coeffs, Image.BILINEAR, fillcolor=0)
    expected_mask = mask.transform(mask.size, Image.AFFINE, coeffs, Image.NEAREST, fillcolor=0)

    np.testing.assert_array_equal(np.array(out_img), np.array(expected_img))
    np.testing.assert_array_equal(np.array(out_mask), np.array(expected_mask))


def test_kp_affine_matches_transform_points_with_shared_matrix() -> None:
    """KP output must equal ``transform_points`` on the shared affine matrix."""
    w = h = 240
    px, py = 150.0, 110.0
    angle, tx, ty, scale, shear = 22.0, 9.0, -4.0, 1.05, 11.0
    kp_x, kp_y = _apply_kp_affine(
        px, py, width=w, height=h, angle=angle,
        scale=scale, shear=shear, tx=tx, ty=ty,
    )

    matrix = build_centered_affine_matrix(
        width=w, height=h, rotation_degrees=angle, translate=(tx, ty), scale=scale,
        shear_degrees=(shear, shear), shear_mode="torchvision",
    )
    expected = transform_points(np.array([[px, py]], dtype=np.float32), matrix)[0]

    np.testing.assert_allclose([kp_x, kp_y], expected, atol=1e-4)
