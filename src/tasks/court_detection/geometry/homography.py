"""Court-plane homography utilities."""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d


def court_template_xy(num_keypoints: int = 14) -> NDArray[np.float32]:
    """Return the canonical court-plane template points in ``(x, y)`` order."""
    if num_keypoints <= 0:
        raise ValueError(f"num_keypoints must be positive, got {num_keypoints}")
    points = (
        court_keypoints_3d(STANDARD_COURT_CONFIG)[:num_keypoints]
        .detach()
        .cpu()
        .numpy()[:, :2]
    )
    return np.asarray(points, dtype=np.float32)


def compute_template_to_image_homography(
    image_keypoints_xy: NDArray[np.floating],
    *,
    template_xy: NDArray[np.floating] | None = None,
    ransac_reproj_threshold: float = 5.0,
) -> NDArray[np.float32] | None:
    """Estimate a homography from court template coordinates to image pixels."""
    image_points = _as_point_array("image_keypoints_xy", image_keypoints_xy)
    template_points = (
        court_template_xy(image_points.shape[0])
        if template_xy is None
        else _as_point_array("template_xy", template_xy)
    )
    if template_points.shape[0] != image_points.shape[0]:
        raise ValueError(
            "template_xy and image_keypoints_xy must have the same point count, "
            f"got {template_points.shape[0]} and {image_points.shape[0]}."
        )
    return estimate_homography(
        template_points,
        image_points,
        method=int(cv2.RANSAC),
        ransac_reproj_threshold=ransac_reproj_threshold,
    )


def estimate_homography(
    src_xy: NDArray[np.floating],
    dst_xy: NDArray[np.floating],
    *,
    method: int = 0,
    ransac_reproj_threshold: float = 5.0,
) -> NDArray[np.float32] | None:
    """Estimate a 3x3 homography from ``src_xy`` to ``dst_xy``."""
    src = _as_point_array("src_xy", src_xy)
    dst = _as_point_array("dst_xy", dst_xy)
    if src.shape[0] != dst.shape[0]:
        raise ValueError(
            f"src_xy and dst_xy must have the same point count, got {src.shape[0]} "
            f"and {dst.shape[0]}."
        )
    if src.shape[0] < 4:
        raise ValueError(f"At least 4 points are required, got {src.shape[0]}.")
    if ransac_reproj_threshold <= 0:
        raise ValueError(
            f"ransac_reproj_threshold must be positive, got {ransac_reproj_threshold}."
        )

    raw_result: Any = cv2.findHomography(
        src,
        dst,
        method,
        float(ransac_reproj_threshold),
    )
    homography_raw, _status = cast("tuple[object | None, object]", raw_result)
    if homography_raw is None:
        return None
    homography = np.asarray(homography_raw, dtype=np.float64)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        return None
    return cast(NDArray[np.float32], homography.astype(np.float32))


def project_points(
    points_xy: NDArray[np.floating],
    homography: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Project ``(N, 2)`` points through a template-to-image homography."""
    points = _as_point_array("points_xy", points_xy)
    matrix = np.asarray(homography, dtype=np.float32)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(
            f"homography must be finite with shape (3, 3), got {matrix.shape}."
        )
    projected = cv2.perspectiveTransform(points.reshape(1, -1, 2), matrix).reshape(
        -1, 2
    )
    if not np.isfinite(projected).all():
        raise ValueError("Projected points contain non-finite coordinates.")
    return cast(NDArray[np.float32], np.asarray(projected, dtype=np.float32))


def _as_point_array(name: str, points: NDArray[np.floating]) -> NDArray[np.float32]:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates.")
    return array
