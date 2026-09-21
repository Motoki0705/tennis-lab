"""Explicit, auditable court completion and short ball interpolation."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .contracts import (
    Annotation,
    Ball,
    ClipManifest,
    CourtPoint,
    CourtSample,
    covered_indices,
)


def _spread(points: NDArray[np.float64]) -> None:
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular[0] <= 1e-9 or singular[1] / singular[0] < 1e-3:
        raise ValueError(
            "homography anchors are coincident, collinear or nearly collinear"
        )
    # No three-point line plus one isolated point can determine an invertible H.
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            direction = points[second] - points[first]
            length = float(np.linalg.norm(direction))
            if length <= 1e-9:
                raise ValueError("duplicate homography anchor positions")
            cross = direction[0] * (points[:, 1] - points[first, 1]) - direction[1] * (
                points[:, 0] - points[first, 0]
            )
            if (
                int(np.count_nonzero(np.abs(cross) / length < singular[0] * 1e-5))
                >= len(points) - 1
            ):
                raise ValueError(
                    "homography anchors do not constrain an invertible projectivity"
                )


def complete_court(
    sample: CourtSample, definition: dict[str, Any], manifest: ClipManifest
) -> CourtSample:
    """Fill missing ground points only. Observed/inferred inputs remain untouched."""
    if sample.orientation != "known":
        raise ValueError("resolve court orientation before homography completion")
    if [p.index for p in sample.points] != list(range(20)):
        raise ValueError("court points must be in CourtKP20 order")
    anchors = [
        p
        for p in sample.points[:15]
        if p.source == "observed"
        and p.visibility == "visible"
        and p.point_px is not None
    ]
    if len(anchors) < 4:
        raise ValueError(
            "at least four visible, directly observed ground anchors required"
        )
    template = np.asarray(definition["points_xyz"], dtype=np.float64)[:, :2]
    indices = [p.index for p in anchors]
    image_points = np.asarray([p.point_px for p in anchors], dtype=np.float64)
    _spread(template[indices])
    _spread(image_points)
    matrix, _ = cv2.findHomography(template[indices], image_points, method=0)
    if (
        matrix is None
        or not np.isfinite(matrix).all()
        or np.linalg.matrix_rank(matrix) != 3
    ):
        raise ValueError("homography is singular or not finite")
    homogeneous = np.column_stack((template[:15], np.ones(15))) @ matrix.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-9):
        raise ValueError("court projection crosses a singularity")
    if np.min(homogeneous[:, 2]) < 0 < np.max(homogeneous[:, 2]):
        raise ValueError("court projection crosses the horizon")
    projected = homogeneous[:, :2] / homogeneous[:, 2, None]
    residual = float(np.max(np.linalg.norm(projected[indices] - image_points, axis=1)))
    tolerance = (
        manifest.policies.homography_max_error_px_at_1080p * manifest.height / 1080
    )
    if residual > tolerance:
        raise ValueError(
            f"homography residual {residual:.3f}px exceeds {tolerance:.3f}px"
        )
    result = sample.model_copy(deep=True)
    for index in range(15):
        if result.points[index].point_px is not None:
            continue
        x, y = map(float, projected[index])
        outside = not (0 <= x < manifest.width and 0 <= y < manifest.height)
        result.points[index] = CourtPoint(
            index=index,
            name=result.points[index].name,
            point_px=[x, y],
            visibility="out_of_frame" if outside else "unassessed",
            source="homography",
            source_frames=[sample.frame_index],
            anchor_indices=indices,
        )
    return cast(CourtSample, result)


def court_mode(annotation: Annotation, manifest: ClipManifest) -> str:
    """A missing motion review or anchor never authorizes static reuse."""
    count = len(manifest.frames)
    if annotation.camera_motion != "none":
        return "dynamic"
    if covered_indices(annotation.camera_review_ranges, count) != set(range(count)):
        return "dynamic"
    if any("cut" in frame.events for frame in annotation.frames):
        return "dynamic"
    if len({frame.shot_id for frame in annotation.frames}) != 1:
        return "dynamic"
    samples = {sample.frame_index: sample for sample in annotation.court_samples}
    required = sorted({0, count // 2, count - 1})
    if not all(index in samples for index in required):
        return "dynamic"
    for index in required:
        sample = samples[index]
        if sample.orientation != "known" or any(
            p.point_px is None for p in sample.points[:15]
        ):
            return "dynamic"
        if (
            len(
                [
                    p
                    for p in sample.points[:15]
                    if p.source == "observed" and p.visibility == "visible"
                ]
            )
            < 4
        ):
            return "dynamic"
    tolerance = manifest.policies.static_tolerance_px_at_1080p * manifest.height / 1080
    # Static references reuse all 20 points, including independently observed
    # posts/net tops. Missing points remain unresolved; known disagreements veto reuse.
    for point_index in range(20):
        coordinates = [
            np.asarray(point, dtype=np.float64)
            for index in required
            if (point := samples[index].points[point_index].point_px) is not None
        ]
        for a in coordinates:
            for b in coordinates:
                if float(np.linalg.norm(a - b)) > tolerance:
                    return "dynamic"
    return "static"


def interpolation_values(
    annotation: Annotation,
    manifest: ClipManifest,
    track_id: str,
    start: int,
    stop: int,
) -> dict[int, list[float]]:
    frames = {frame.frame_index: frame for frame in annotation.frames}
    if stop <= start + 1 or any(
        index not in frames for index in range(start, stop + 1)
    ):
        raise ValueError(
            "interpolation requires two endpoints enclosing consecutive target frames"
        )
    selected = [frames[index] for index in range(start, stop + 1)]
    if len({frame.shot_id for frame in selected}) != 1 or any(
        frame.events for frame in selected
    ):
        raise ValueError(
            "interpolation cannot cross hits, bounces, cuts or play boundaries"
        )
    if any(frame.balls_review != "complete" for frame in selected):
        raise ValueError("inspect every interpolation frame before using the helper")
    balls: list[Ball] = []
    for frame in selected:
        matches = [ball for ball in frame.balls if ball.track_id == track_id]
        if len(matches) != 1:
            raise ValueError(
                "same alive ball must be explicitly present throughout the interval"
            )
        balls.append(matches[0])
    first, last = balls[0], balls[-1]
    if (
        first.status != "visible"
        or last.status != "visible"
        or first.center_px is None
        or last.center_px is None
    ):
        raise ValueError("interpolation endpoints must be directly visible")
    start_pts, stop_pts = (
        manifest.frames[start].clip_pts,
        manifest.frames[stop].clip_pts,
    )
    elapsed = (stop_pts - start_pts) * Fraction(manifest.time_base)
    if elapsed > Fraction(str(manifest.policies.ball_max_gap_seconds)):
        raise ValueError("interpolation gap exceeds configured time limit")
    return {
        index: [
            float(
                a
                + (b - a)
                * Fraction(
                    manifest.frames[index].clip_pts - start_pts, stop_pts - start_pts
                )
            )
            for a, b in zip(first.center_px, last.center_px, strict=True)
        ]
        for index in range(start + 1, stop)
    }


def interpolate_ball(
    annotation: Annotation, manifest: ClipManifest, track_id: str, start: int, stop: int
) -> Annotation:
    values = interpolation_values(annotation, manifest, track_id, start, stop)
    result = annotation.model_copy(deep=True)
    for frame in result.frames:
        if frame.frame_index not in values:
            continue
        ball = next(ball for ball in frame.balls if ball.track_id == track_id)
        if ball.center_px is not None or ball.missing_reason == "out_of_frame":
            raise ValueError(
                "interpolation must not overwrite a position or fill an out-of-frame ball"
            )
        ball.center_px = values[frame.frame_index]
        ball.status = "interpolated"
        ball.missing_reason = None
        ball.source_frames = [start, stop]
    return cast(Annotation, result)
