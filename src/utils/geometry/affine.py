"""Affine matrix helpers shared by image and point augmentations.

The canonical matrix returned here maps input/source coordinates to
output/destination coordinates. OpenCV's ``warpAffine`` accepts that forward
matrix directly (it inverts internally unless ``WARP_INVERSE_MAP`` is set),
while PIL image warps need the inverse matrix coefficients.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

AffineMatrix: TypeAlias = npt.NDArray[np.floating]

__all__ = [
    "AffineMatrix",
    "build_centered_affine_matrix",
    "invert_homogeneous_matrix",
    "to_cv2_affine",
    "to_pil_affine_coefficients",
    "transform_points",
]


def _as_homogeneous_matrix(matrix: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return ``matrix`` as a 3x3 homogeneous affine matrix."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape == (3, 3):
        return arr
    if arr.shape == (2, 3):
        return np.vstack([arr, np.array([[0.0, 0.0, 1.0]], dtype=np.float64)])
    raise ValueError(
        f"affine matrix must have shape (2, 3) or (3, 3), got {arr.shape}."
    )


def build_centered_affine_matrix(
    *,
    width: int,
    height: int,
    rotation_degrees: float = 0.0,
    translate: tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    shear_degrees: float | tuple[float, float] = 0.0,
    shear_mode: Literal["matrix", "torchvision"] = "matrix",
    center: tuple[float, float] | None = None,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.floating]:
    """Build a source→destination affine matrix around an image-space centre.

    Composition order is scale, shear, then rotation, all around ``center``,
    followed by translation. This matches the forward matrix convention used by
    OpenCV ``warpAffine`` and by point/keypoint targets. When ``shear_mode`` is
    ``"torchvision"``, x/y shear signs and coupling match torchvision's
    ``_get_inverse_affine_matrix(..., inverted=False)`` helper. Otherwise,
    ``"matrix"`` mode uses the plain shear matrix ``[[1, tan(x)], [tan(y), 1]]``.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")
    if scale <= 0.0:
        raise ValueError("scale must be positive.")

    if isinstance(shear_degrees, tuple):
        shear_x_degrees, shear_y_degrees = shear_degrees
    else:
        shear_x_degrees = float(shear_degrees)
        shear_y_degrees = 0.0

    cx, cy = center if center is not None else (width / 2.0, height / 2.0)
    tx, ty = translate

    rotation_rad = np.deg2rad(rotation_degrees)
    shear_x_rad = np.deg2rad(shear_x_degrees)
    shear_y_rad = np.deg2rad(shear_y_degrees)
    cos_theta = float(np.cos(rotation_rad))
    sin_theta = float(np.sin(rotation_rad))

    center_to_origin = np.array(
        [[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    if shear_mode == "torchvision":
        cos_sy = float(np.cos(shear_y_rad))
        if np.isclose(cos_sy, 0.0):
            raise ValueError(
                "torchvision-style y-shear is singular near +/- 90 degrees."
            )
        a = float(np.cos(rotation_rad - shear_y_rad) / cos_sy)
        b = float(
            -np.cos(rotation_rad - shear_y_rad)
            * np.tan(shear_x_rad)
            / cos_sy
            - np.sin(rotation_rad)
        )
        c = float(np.sin(rotation_rad - shear_y_rad) / cos_sy)
        d = float(
            -np.sin(rotation_rad - shear_y_rad)
            * np.tan(shear_x_rad)
            / cos_sy
            + np.cos(rotation_rad)
        )
        linear_matrix = np.array(
            [
                [scale * a, scale * b, 0.0],
                [scale * c, scale * d, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    elif shear_mode == "matrix":
        scale_matrix = np.array(
            [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        shear_matrix = np.array(
            [
                [1.0, np.tan(shear_x_rad), 0.0],
                [np.tan(shear_y_rad), 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rotation_matrix = np.array(
            [
                [cos_theta, -sin_theta, 0.0],
                [sin_theta, cos_theta, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        linear_matrix = rotation_matrix @ shear_matrix @ scale_matrix
    else:
        raise ValueError(f"unknown shear_mode: {shear_mode!r}.")

    recenter = np.array(
        [[1.0, 0.0, cx + tx], [0.0, 1.0, cy + ty], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    matrix = recenter @ linear_matrix @ center_to_origin
    return matrix.astype(dtype, copy=False)


def invert_homogeneous_matrix(matrix: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Invert a 2x3 or 3x3 affine matrix and return a 3x3 matrix."""
    return np.asarray(np.linalg.inv(_as_homogeneous_matrix(matrix)), dtype=np.float64)


def to_cv2_affine(
    matrix: npt.ArrayLike,
    *,
    dtype: npt.DTypeLike = np.float32,
) -> npt.NDArray[np.floating]:
    """Return the 2x3 matrix form consumed by ``cv2.warpAffine``."""
    return _as_homogeneous_matrix(matrix)[:2, :].astype(dtype, copy=False)


def to_pil_affine_coefficients(
    matrix: npt.ArrayLike,
) -> tuple[float, float, float, float, float, float]:
    """Return PIL ``Image.transform(..., AFFINE, coeffs)`` coefficients.

    PIL expects an output→input matrix. Pass ``invert_homogeneous_matrix`` of the
    canonical source→destination matrix when warping images with this helper.
    """
    arr = _as_homogeneous_matrix(matrix)
    return (
        float(arr[0, 0]),
        float(arr[0, 1]),
        float(arr[0, 2]),
        float(arr[1, 0]),
        float(arr[1, 1]),
        float(arr[1, 2]),
    )


def transform_points(
    points: npt.ArrayLike,
    matrix: npt.ArrayLike,
    *,
    dtype: npt.DTypeLike = np.float32,
) -> npt.NDArray[np.floating]:
    """Apply a homogeneous affine matrix to ``(..., 2)`` point coordinates."""
    points_arr = np.asarray(points, dtype=np.float64)
    if points_arr.shape[-1:] != (2,):
        raise ValueError(f"points must have shape (..., 2), got {points_arr.shape}.")

    original_shape = points_arr.shape
    flat_points = points_arr.reshape(-1, 2)
    ones = np.ones((flat_points.shape[0], 1), dtype=np.float64)
    homogeneous_points = np.concatenate([flat_points, ones], axis=1)
    transformed = homogeneous_points @ _as_homogeneous_matrix(matrix).T
    denom = transformed[:, 2:3]
    if np.any(np.isclose(denom, 0.0)):
        raise ValueError(
            "affine transform produced points with near-zero homogeneous scale."
        )
    transformed_points = transformed[:, :2] / denom
    return transformed_points.reshape(original_shape).astype(dtype, copy=False)
