"""Reusable rendering helpers for tennis pose visualization.

This module provides low-level drawing utilities (court lines, player skeleton,
racket points) that can be used both from scene-based visualization
(`tennis_multi_cam_3d_pose.py`) and from training-time debug rendering.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from src.tennis.geometry.skeleton import COCO_BONES

Color = tuple[int, int, int]


COURT_LINE_SEGMENTS: list[tuple[int, int]] = [
    (0, 1),
    (1, 3),
    (2, 3),
    (0, 2),
    (4, 6),
    (5, 7),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (8, 12),
    (12, 9),
    (10, 13),
    (13, 11),
    (14, 15),
    (15, 16),
    (14, 17),
    (17, 18),
]

RACKET_SEGMENTS: list[tuple[int, int]] = [(0, 1), (1, 2)]


def _to_int_point(pt: Sequence[float]) -> tuple[int, int]:
    return int(round(float(pt[0]))), int(round(float(pt[1])))


def draw_segments(
    image: np.ndarray,
    points: Sequence[Sequence[float]],
    visibility: Sequence[int] | None,
    segments: Iterable[tuple[int, int]],
    color: Color,
    thickness: int,
) -> None:
    """Draw line segments between visible keypoints on the image."""
    h, w = image.shape[:2]
    if visibility is None:
        vis_list: list[int] = [1] * len(points)
    else:
        vis_list = list(visibility)
    for i, j in segments:
        if not (0 <= i < len(points) and 0 <= j < len(points)):
            continue
        if not (vis_list[i] and vis_list[j]):
            continue
        p1 = _to_int_point(points[i])
        p2 = _to_int_point(points[j])
        if not (
            0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h
        ):
            continue
        cv2.line(image, p1, p2, color=color, thickness=thickness)


def draw_points(
    image: np.ndarray,
    points: Sequence[Sequence[float]],
    visibility: Sequence[int] | None,
    color: Color,
    radius: int,
) -> None:
    """Draw visible keypoints as filled circles."""
    h, w = image.shape[:2]
    if visibility is None:
        vis_list: list[int] = [1] * len(points)
    else:
        vis_list = list(visibility)
    for idx, pt in enumerate(points):
        if not vis_list[idx]:
            continue
        x, y = _to_int_point(pt)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), radius, color, thickness=-1)


def render_pose2d_frame(
    width: int,
    height: int,
    court_points: Sequence[Sequence[float]] | np.ndarray,
    court_visibility: Sequence[int] | None,
    player_poses: Sequence[np.ndarray] | Sequence[Sequence[Sequence[float]]],
    player_pose_visibility: Sequence[Sequence[int]] | None,
    racket_points: Sequence[np.ndarray] | Sequence[Sequence[Sequence[float]]] | None,
    racket_visibility: Sequence[Sequence[int]] | None,
    background_color: Color = (30, 80, 30),
) -> np.ndarray:
    """Render court + player skeletons + racket points into an RGB image.

    Args:
        width (int): Output image width in pixels.
        height (int): Output image height in pixels.
        court_points (Sequence[Sequence[float]] | np.ndarray): Court keypoints
            (20x2 expected, but only the first 20 are used).
        court_visibility (Sequence[int] | None): Optional visibility flags for
            court points (20 ints).
        player_poses (Sequence[np.ndarray] | Sequence[Sequence[Sequence[float]]]):
            Iterable of pose-2D arrays shaped ``[J_pose, 2]`` (COCO-style).
        player_pose_visibility (Sequence[Sequence[int]] | None): Optional
            per-player visibility lists shaped ``[J_pose]``.
        racket_points (Sequence[np.ndarray] | Sequence[Sequence[Sequence[float]]] | None):
            Optional iterable of racket-2D arrays ``[3, 2]`` per player.
        racket_visibility (Sequence[Sequence[int]] | None): Optional per-player
            visibility lists shaped ``[3]``.
        background_color (Color): BGR tuple for the background.

    Returns:
        np.ndarray: ``HxWx3`` uint8 BGR image.

    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = background_color

    court_pts = np.asarray(court_points, dtype=np.float32)
    if court_pts.shape[0] >= 20:
        court_pts = court_pts[:20]
    court_vis: list[int] | None = None
    if court_visibility is not None:
        court_vis = list(court_visibility)[: len(court_pts)]
    draw_segments(image, court_pts, court_vis, COURT_LINE_SEGMENTS, (220, 220, 220), 2)
    draw_points(image, court_pts, court_vis, (255, 255, 255), 3)

    poses_list: list[np.ndarray] = [
        np.asarray(p, dtype=np.float32) for p in player_poses
    ]
    vis_list: list[list[int]] = []
    if player_pose_visibility is not None:
        for vis in player_pose_visibility:
            vis_list.append(list(vis))
    else:
        for pts in poses_list:
            vis_list.append([1] * len(pts))

    for pts, vis in zip(poses_list, vis_list, strict=True):
        draw_segments(image, pts, vis, COCO_BONES, (0, 255, 255), 2)
        draw_points(image, pts, vis, (0, 255, 255), 3)

    if racket_points is not None:
        r_pts_list: list[np.ndarray] = [
            np.asarray(p, dtype=np.float32) for p in racket_points
        ]
        r_vis_list: list[list[int]] = []
        if racket_visibility is not None:
            for vis in racket_visibility:
                r_vis_list.append(list(vis))
        else:
            for pts in r_pts_list:
                r_vis_list.append([1] * len(pts))
        for pts, vis in zip(r_pts_list, r_vis_list, strict=True):
            draw_segments(image, pts, vis, RACKET_SEGMENTS, (0, 165, 255), 2)
            draw_points(image, pts, vis, (0, 165, 255), 3)

    return image
