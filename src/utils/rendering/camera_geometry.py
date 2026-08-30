"""Pure NumPy geometry for OpenCV pinhole-camera visualizations.

The helpers in this module accept explicit camera intrinsics, image sizes, and
camera-to-target-coordinate transforms.  Camera coordinates always follow the
OpenCV convention: ``+x`` points image-right, ``+y`` points image-down, and
``+z`` points forward.  The supplied SE(3) transform maps those axes into the
caller's target coordinate system; this module never converts coordinate
systems or assigns physical units.

Frustum vertices are ordered as camera centre, top-left, top-right,
bottom-right, bottom-left.  Pixels denote zero-based pixel centres, so the
image-plane corner rays pass through ``(0, 0)``, ``(width - 1, 0)``,
``(width - 1, height - 1)``, and ``(0, height - 1)`` at the requested positive
camera-Z depth.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

_MATRIX_ATOL = 1.0e-6

# Segment order is part of the public array contract: centre-to-corner rays
# first, followed by the image-plane perimeter in clockwise pixel order.
_FRUSTUM_SEGMENT_VERTEX_INDICES: NDArray[np.int64] = np.asarray(
    (
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),
    ),
    dtype=np.int64,
)


def _float_array(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only numeric values.") from error


def _validate_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(image_size, tuple) or len(image_size) != 2:
        raise ValueError("image_size must be a (width, height) tuple.")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in image_size
    ):
        raise ValueError("image_size values must be integers.")
    width, height = (int(value) for value in image_size)
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size values must be positive, got {(width, height)}.")
    return width, height


def _positive_finite(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a positive finite number, got {value!r}."
        ) from error
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    return number


def _validate_intrinsics(
    intrinsics: ArrayLike,
    *,
    image_size: tuple[int, int],
    name: str = "intrinsics",
) -> NDArray[np.float64]:
    matrix = _float_array(intrinsics, name=name)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix, got {matrix.shape}.")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError(f"{name} focal lengths must be positive.")
    if not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=_MATRIX_ATOL, rtol=0.0):
        raise ValueError(f"{name} must have bottom row [0, 0, 1].")
    width, height = image_size
    if not (0.0 <= matrix[0, 2] < width and 0.0 <= matrix[1, 2] < height):
        raise ValueError(f"{name} principal point must lie inside image_size.")
    return matrix


def _validate_transforms(
    camera_to_world: ArrayLike,
    *,
    batched: bool,
) -> NDArray[np.float64]:
    transforms = _float_array(camera_to_world, name="camera_to_world")
    expected_shape = "(N, 4, 4)" if batched else "(4, 4)"
    valid_shape = (
        transforms.ndim == 3
        and transforms.shape[0] > 0
        and transforms.shape[1:] == (4, 4)
        if batched
        else transforms.shape == (4, 4)
    )
    if not valid_shape or not np.isfinite(transforms).all():
        raise ValueError(
            f"camera_to_world must be a finite {expected_shape} array, got {transforms.shape}."
        )

    matrices = transforms if batched else transforms[None, ...]
    expected_bottom = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    if not np.allclose(matrices[:, 3, :], expected_bottom, atol=_MATRIX_ATOL, rtol=0.0):
        raise ValueError(
            "camera_to_world must have homogeneous bottom row [0, 0, 0, 1]."
        )

    rotations = matrices[:, :3, :3]
    gram = np.swapaxes(rotations, 1, 2) @ rotations
    identity = np.broadcast_to(np.eye(3, dtype=np.float64), gram.shape)
    if not np.allclose(gram, identity, atol=_MATRIX_ATOL, rtol=0.0):
        raise ValueError("camera_to_world rotations must be orthonormal.")
    determinants = np.linalg.det(rotations)
    if not np.allclose(determinants, 1.0, atol=_MATRIX_ATOL, rtol=0.0):
        raise ValueError(
            "camera_to_world rotations must be proper with determinant +1."
        )
    return transforms


def camera_frustum_corners(
    intrinsics: ArrayLike,
    image_size: tuple[int, int],
    camera_to_world: ArrayLike,
    *,
    depth: float,
) -> NDArray[np.float64]:
    """Return one camera frustum's five ordered vertices.

    Args:
        intrinsics: Finite OpenCV pinhole matrix of shape ``(3, 3)``. Focal
            lengths must be positive and the principal point must lie inside
            ``image_size``.
        image_size: Positive integer ``(width, height)`` in pixels.
        camera_to_world: Proper SE(3) matrix mapping camera coordinates into
            the caller's target coordinate system. Despite the parameter name,
            the target may be world, scene, or another explicitly chosen frame.
        depth: Positive finite camera-Z depth of the image-plane corners, in
            the same length unit as ``camera_to_world`` translation.

    Returns:
        Float64 array of shape ``(5, 3)`` ordered as camera centre, top-left,
        top-right, bottom-right, bottom-left in the target coordinate system.
    """
    size = _validate_image_size(image_size)
    matrix = _validate_intrinsics(intrinsics, image_size=size)
    transform = _validate_transforms(camera_to_world, batched=False)
    z_depth = _positive_finite(depth, name="depth")

    width, height = size
    pixel_corners = np.asarray(
        (
            (0.0, 0.0, 1.0),
            (float(width - 1), 0.0, 1.0),
            (float(width - 1), float(height - 1), 1.0),
            (0.0, float(height - 1), 1.0),
        ),
        dtype=np.float64,
    )
    try:
        corner_rays = np.linalg.solve(matrix, pixel_corners.T).T
    except np.linalg.LinAlgError as error:
        raise ValueError("intrinsics must be invertible.") from error
    corners_camera = corner_rays * (z_depth / corner_rays[:, 2:3])
    vertices_camera = np.vstack((np.zeros((1, 3), dtype=np.float64), corners_camera))

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return vertices_camera @ rotation.T + translation


def camera_frustum_segments(
    intrinsics: ArrayLike,
    image_size: tuple[int, int],
    camera_to_world: ArrayLike,
    *,
    depth: float,
) -> NDArray[np.float64]:
    """Return the eight ordered line segments of one OpenCV camera frustum.

    Segment order is centre-to-top-left, centre-to-top-right,
    centre-to-bottom-right, centre-to-bottom-left, then top, right, bottom,
    and left image-plane edges. The result has shape ``(8, 2, 3)``.
    """
    vertices = camera_frustum_corners(
        intrinsics,
        image_size,
        camera_to_world,
        depth=depth,
    )
    return vertices[_FRUSTUM_SEGMENT_VERTEX_INDICES]


def camera_trajectory_points(camera_to_world: ArrayLike) -> NDArray[np.float64]:
    """Return camera centres in exactly the supplied transform order.

    Args:
        camera_to_world: Non-empty proper SE(3) array of shape ``(N, 4, 4)``.

    Returns:
        Float64 camera-centre array of shape ``(N, 3)``. No temporal sorting,
        filtering, or coordinate conversion is performed.
    """
    transforms = _validate_transforms(camera_to_world, batched=True)
    return transforms[:, :3, 3].copy()


def camera_trajectory_segments(camera_to_world: ArrayLike) -> NDArray[np.float64]:
    """Connect consecutive camera centres without reordering or closing the path.

    The result has shape ``(N - 1, 2, 3)``. A one-camera trajectory returns an
    explicit empty ``(0, 2, 3)`` array.
    """
    centres = camera_trajectory_points(camera_to_world)
    if centres.shape[0] == 1:
        return np.empty((0, 2, 3), dtype=np.float64)
    return np.stack((centres[:-1], centres[1:]), axis=1)


def camera_view_direction_segments(
    camera_to_world: ArrayLike,
    *,
    length: float,
) -> NDArray[np.float64]:
    """Return ordered centre-to-forward segments for OpenCV cameras.

    Camera ``+z`` is forward. The result has shape ``(N, 2, 3)`` and retains
    the input camera order. ``length`` uses the target coordinate system's
    explicitly chosen unit.
    """
    transforms = _validate_transforms(camera_to_world, batched=True)
    segment_length = _positive_finite(length, name="length")
    centres = transforms[:, :3, 3]
    forward = transforms[:, :3, 2]
    endpoints = centres + segment_length * forward
    return np.stack((centres, endpoints), axis=1)


def camera_coverage_segments(
    intrinsics: ArrayLike,
    image_sizes: ArrayLike,
    camera_to_world: ArrayLike,
    *,
    depth: float,
) -> NDArray[np.float64]:
    """Return ordered frustum segments for an explicit camera collection.

    Args:
        intrinsics: Array of shape ``(N, 3, 3)``.
        image_sizes: Integer array of shape ``(N, 2)`` in ``(width, height)``
            order. Broadcasting one size across cameras is intentionally not
            supported.
        camera_to_world: Proper SE(3) array of shape ``(N, 4, 4)``.
        depth: Shared positive camera-Z depth for every frustum.

    Returns:
        Float64 array of shape ``(N, 8, 2, 3)``. Both camera order and each
        frustum's segment order are preserved exactly.
    """
    transforms = _validate_transforms(camera_to_world, batched=True)
    matrices = _float_array(intrinsics, name="intrinsics")
    if matrices.shape != (transforms.shape[0], 3, 3) or not np.isfinite(matrices).all():
        raise ValueError(
            "intrinsics must be a finite (N, 3, 3) array matching camera_to_world; "
            f"got {matrices.shape}."
        )

    raw_sizes = np.asarray(image_sizes)
    if raw_sizes.shape != (transforms.shape[0], 2):
        raise ValueError(
            "image_sizes must have shape (N, 2) matching camera_to_world; "
            f"got {raw_sizes.shape}."
        )
    if np.issubdtype(raw_sizes.dtype, np.bool_) or not np.issubdtype(
        raw_sizes.dtype, np.integer
    ):
        raise ValueError("image_sizes must contain only integers.")

    z_depth = _positive_finite(depth, name="depth")
    result = np.empty((transforms.shape[0], 8, 2, 3), dtype=np.float64)
    for index in range(transforms.shape[0]):
        size = _validate_image_size(
            (int(raw_sizes[index, 0]), int(raw_sizes[index, 1]))
        )
        _validate_intrinsics(
            matrices[index],
            image_size=size,
            name=f"intrinsics[{index}]",
        )
        result[index] = camera_frustum_segments(
            matrices[index],
            size,
            transforms[index],
            depth=z_depth,
        )
    return result


__all__ = [
    "camera_coverage_segments",
    "camera_frustum_corners",
    "camera_frustum_segments",
    "camera_trajectory_points",
    "camera_trajectory_segments",
    "camera_view_direction_segments",
]
