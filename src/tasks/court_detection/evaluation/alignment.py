"""Court-plane alignment geometry shared by every benchmarked model.

Both models are compared through the same repo-native contract: a RANSAC
homography from the canonical court template to image pixels, dense
reprojection of the regulation court lines, and the clipped doubles polygon.
Nothing here depends on a model's own homography post-processor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from src.utils.schema.court import (
    GROUND_COURT_KP_NAMES,
    NUM_GROUND_COURT_KP,
)

# Court-line segments expressed over the canonical (physical) keypoint order.
# They are exactly the regulation ground lines a court detection model is
# expected to align to; the net structure is excluded because no benchmarked
# head predicts it.
COURT_LINE_SEGMENTS: tuple[tuple[int, int], ...] = (
    (0, 1),  # far doubles baseline
    (2, 3),  # near doubles baseline
    (0, 2),  # left doubles sideline
    (1, 3),  # right doubles sideline
    (4, 8),  # far-left singles sideline
    (8, 10),  # left singles sideline (service box)
    (10, 5),  # near-left singles sideline
    (6, 9),  # far-right singles sideline
    (9, 11),  # right singles sideline (service box)
    (11, 7),  # near-right singles sideline
    (8, 9),  # far service line
    (10, 11),  # near service line
    (12, 13),  # center service line
)

# Doubles court outline in template coordinates (far-left, far-right,
# near-right, near-left).
DOUBLES_POLYGON_INDICES: tuple[int, int, int, int] = (0, 1, 3, 2)

ALIGNMENT_REASON_INSUFFICIENT_POINTS = "insufficient_valid_keypoints"
ALIGNMENT_REASON_RANSAC_FAILED = "ransac_failed"
ALIGNMENT_REASON_DEGENERATE_POLYGON = "degenerate_polygon"
ALIGNMENT_REASON_EMPTY_INTERSECTION = "empty_polygon_intersection"


@dataclass(frozen=True, slots=True)
class HomographyFit:
    """One template-to-image fit with an explicit failure reason."""

    homography: NDArray[np.float64] | None
    inlier_count: int
    reason: str | None

    @property
    def succeeded(self) -> bool:
        return self.homography is not None


def sample_segment_points(
    segments: tuple[tuple[int, int], ...],
    template_xy: NDArray[np.floating],
    *,
    per_segment: int,
) -> NDArray[np.float64]:
    """Sample evenly spaced template points along each canonical segment.

    The result is a flat ``(N * per_segment, 2)`` array, which is what
    point-wise metrics need.  Callers that *draw* the samples must use
    :func:`segment_polylines` instead: treating this flat array as one polyline
    would add a bridge segment between the end of one court line and the start
    of the next.
    """
    return np.concatenate(
        segment_polylines(segments, template_xy, per_segment=per_segment), axis=0
    )


def segment_polylines(
    segments: tuple[tuple[int, int], ...],
    template_xy: NDArray[np.floating],
    *,
    per_segment: int,
) -> tuple[NDArray[np.float64], ...]:
    """Return one independent polyline per canonical court-line segment."""
    if per_segment < 2:
        raise ValueError("per_segment must be at least 2 for line sampling.")
    template = _point_array("template_xy", template_xy)
    weights = np.linspace(0.0, 1.0, per_segment, dtype=np.float64)[:, None]
    chunks: list[NDArray[np.float64]] = []
    for start_index, end_index in segments:
        if (
            not 0 <= start_index < template.shape[0]
            or not 0 <= end_index < template.shape[0]
        ):
            raise ValueError("Court line segments must reference template points.")
        start = template[start_index]
        end = template[end_index]
        chunks.append(start[None, :] * (1.0 - weights) + end[None, :] * weights)
    if not chunks:
        raise ValueError("At least one segment is required for line sampling.")
    return tuple(chunks)


def project_points(
    points_xy: NDArray[np.floating], homography: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Project ``(N, 2)`` template points through a template-to-image H."""
    points = _point_array("points_xy", points_xy)
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Homography must be finite with shape (3, 3).")
    projected = cv2.perspectiveTransform(
        points.reshape(1, -1, 2).astype(np.float64), matrix
    ).reshape(-1, 2)
    if not np.isfinite(projected).all():
        raise ValueError("Projected points contain non-finite coordinates.")
    return cast(NDArray[np.float64], projected)


def fit_template_homography(
    template_xy: NDArray[np.floating],
    image_xy: NDArray[np.floating],
    valid: NDArray[np.bool_],
    *,
    ransac_threshold_px: float,
) -> HomographyFit:
    """Fit one canonical-template to image homography from accepted points.

    Points that the model rejected (or that are not finite) never participate,
    and a fit with fewer than four accepted points is reported as a typed
    failure instead of falling back to a lower-order transform.
    """
    template = _point_array("template_xy", template_xy)
    image = _point_array("image_xy", image_xy)
    if template.shape != image.shape:
        raise ValueError("Template and image keypoints must share their shape.")
    if valid.shape != (template.shape[0],):
        raise ValueError("Homography validity mask must match the keypoint count.")
    if not np.isfinite(ransac_threshold_px) or ransac_threshold_px <= 0.0:
        raise ValueError("ransac_threshold_px must be finite and positive.")

    accepted = np.asarray(valid, dtype=bool) & np.isfinite(image).all(axis=1)
    if int(accepted.sum()) < 4:
        return HomographyFit(
            None, int(accepted.sum()), ALIGNMENT_REASON_INSUFFICIENT_POINTS
        )
    source = template[accepted].astype(np.float64)
    destination = image[accepted].astype(np.float64)
    if _collinear(source) or _collinear(destination):
        return HomographyFit(None, 0, ALIGNMENT_REASON_RANSAC_FAILED)
    raw_matrix, raw_status = cv2.findHomography(
        source,
        destination,
        int(cv2.RANSAC),
        float(ransac_threshold_px),
    )
    if raw_matrix is None or raw_status is None:
        return HomographyFit(None, 0, ALIGNMENT_REASON_RANSAC_FAILED)
    matrix = np.asarray(raw_matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return HomographyFit(None, 0, ALIGNMENT_REASON_RANSAC_FAILED)
    inliers = int(np.asarray(raw_status).reshape(-1).astype(bool).sum())
    if inliers < 4:
        return HomographyFit(None, inliers, ALIGNMENT_REASON_RANSAC_FAILED)
    return HomographyFit(matrix, inliers, None)


def symmetric_line_reprojection_error_px(
    reference_homography: NDArray[np.floating],
    candidate_homography: NDArray[np.floating],
    template_line_points: NDArray[np.floating],
    *,
    include: NDArray[np.bool_] | None = None,
) -> float:
    """Mean pixel distance between two dense reprojections of the same lines.

    ``include`` restricts the average to a subset of the sampled template
    points.  Callers use it to report error inside the image frame separately
    from the full set, since template lines that project far outside the frame
    are extrapolations of the fitted homography.
    """
    reference = project_points(template_line_points, reference_homography)
    candidate = project_points(template_line_points, candidate_homography)
    distances = np.linalg.norm(reference - candidate, axis=1)
    if include is not None:
        mask = np.asarray(include, dtype=bool)
        if mask.shape != (distances.shape[0],):
            raise ValueError("The include mask must match the sampled line points.")
        distances = distances[mask]
        if distances.size == 0:
            raise ValueError("No line samples remain after applying the mask.")
    return float(distances.mean())


def points_inside_image(
    points_xy: NDArray[np.floating], width: int, height: int
) -> NDArray[np.bool_]:
    """Return a mask of points that fall inside the image rectangle."""
    points = _point_array("points_xy", points_xy)
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    return cast(
        NDArray[np.bool_],
        (points[:, 0] >= 0.0)
        & (points[:, 0] < float(width))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < float(height)),
    )


def doubles_polygon_template(template_xy: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return the doubles outline as a closed template-space polygon."""
    template = _point_array("template_xy", template_xy)
    return cast(
        NDArray[np.float64],
        template[list(DOUBLES_POLYGON_INDICES)].astype(np.float64),
    )


def clip_convex_polygon_to_image(
    polygon_xy: NDArray[np.floating], width: int, height: int
) -> NDArray[np.float64] | None:
    """Clip a convex polygon to the image rectangle with Sutherland-Hodgman.

    Returns ``None`` for a non-convex input so callers record an explicit
    undefined reason instead of scoring a silently wrong area.  A convex input
    that lies entirely outside the image yields an empty ``(0, 2)`` polygon.
    """
    polygon = _point_array("polygon_xy", polygon_xy)
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive for polygon clipping.")
    if polygon.shape[0] < 3 or not _is_convex(polygon):
        return None
    rectangle = np.array(
        [
            [0.0, 0.0],
            [float(width), 0.0],
            [float(width), float(height)],
            [0.0, float(height)],
        ],
        dtype=np.float64,
    )
    return _clip_convex(_oriented(polygon), _oriented(rectangle))


def polygon_iou(
    polygon_a: NDArray[np.floating],
    polygon_b: NDArray[np.floating],
    *,
    width: int,
    height: int,
) -> tuple[float | None, str | None]:
    """IoU of two image-space doubles polygons after image clipping."""
    clipped_a = clip_convex_polygon_to_image(polygon_a, width, height)
    clipped_b = clip_convex_polygon_to_image(polygon_b, width, height)
    if clipped_a is None or clipped_b is None:
        return None, ALIGNMENT_REASON_DEGENERATE_POLYGON
    area_a = _polygon_area(clipped_a)
    area_b = _polygon_area(clipped_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return None, ALIGNMENT_REASON_EMPTY_INTERSECTION
    intersection = _clip_convex(clipped_a, clipped_b)
    if intersection.shape[0] < 3:
        return 0.0, None
    area_intersection = _polygon_area(intersection)
    union = area_a + area_b - area_intersection
    if union <= 0.0:
        return None, ALIGNMENT_REASON_EMPTY_INTERSECTION
    return float(area_intersection / union), None


def canonical_keypoint_names() -> tuple[str, ...]:
    """Return the ordered canonical names the template indices refer to."""
    return tuple(GROUND_COURT_KP_NAMES[:NUM_GROUND_COURT_KP])


def _clip_convex(
    subject: NDArray[np.float64], clip: NDArray[np.float64]
) -> NDArray[np.float64]:
    output = _oriented(subject)
    clip = _oriented(clip)
    clip_count = clip.shape[0]
    for index in range(clip_count):
        edge_start = clip[index]
        edge_end = clip[(index + 1) % clip_count]
        input_points = output
        if input_points.shape[0] == 0:
            return input_points
        output = _clip_polygon_by_edge(input_points, edge_start, edge_end)
    return output


def _oriented(polygon: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the polygon with its interior on the positive edge side."""
    points = np.asarray(polygon, dtype=np.float64)
    if _signed_area(points) < 0.0:
        return np.ascontiguousarray(points[::-1])
    return points


def _signed_area(polygon: NDArray[np.float64]) -> float:
    if polygon.shape[0] < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float((np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _clip_polygon_by_edge(
    subject: NDArray[np.float64],
    edge_start: NDArray[np.float64],
    edge_end: NDArray[np.float64],
) -> NDArray[np.float64]:
    edge = edge_end - edge_start

    def side(point: NDArray[np.float64]) -> float:
        delta = point - edge_start
        return float(edge[0] * delta[1] - edge[1] * delta[0])

    result: list[NDArray[np.float64]] = []
    count = subject.shape[0]
    for index in range(count):
        current = subject[index]
        following = subject[(index + 1) % count]
        current_side = side(current)
        following_side = side(following)
        if current_side >= 0.0:
            result.append(current)
        if (current_side > 0.0 > following_side) or (
            current_side < 0.0 < following_side
        ):
            denominator = current_side - following_side
            ratio = current_side / denominator
            result.append(current + ratio * (following - current))
    if not result:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(result, dtype=np.float64)


def _polygon_area(polygon: NDArray[np.float64]) -> float:
    return abs(_signed_area(polygon))


def _is_convex(polygon: NDArray[np.float64]) -> bool:
    count = polygon.shape[0]
    sign = 0
    for index in range(count):
        a = polygon[index]
        b = polygon[(index + 1) % count]
        c = polygon[(index + 2) % count]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) <= 1e-12:
            continue
        current = 1 if cross > 0.0 else -1
        if sign == 0:
            sign = current
        elif sign != current:
            return False
    return sign != 0


def _collinear(points: NDArray[np.float64]) -> bool:
    if points.shape[0] < 3:
        return True
    centered = points - points.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values.size < 2 or singular_values[0] <= 0.0:
        return True
    return bool(singular_values[1] / singular_values[0] < 1e-9)


def _point_array(name: str, values: NDArray[np.floating]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {array.shape}.")
    return array


__all__ = [
    "ALIGNMENT_REASON_DEGENERATE_POLYGON",
    "ALIGNMENT_REASON_EMPTY_INTERSECTION",
    "ALIGNMENT_REASON_INSUFFICIENT_POINTS",
    "ALIGNMENT_REASON_RANSAC_FAILED",
    "COURT_LINE_SEGMENTS",
    "DOUBLES_POLYGON_INDICES",
    "HomographyFit",
    "canonical_keypoint_names",
    "clip_convex_polygon_to_image",
    "doubles_polygon_template",
    "fit_template_homography",
    "points_inside_image",
    "polygon_iou",
    "project_points",
    "sample_segment_points",
    "segment_polylines",
    "symmetric_line_reprojection_error_px",
]
