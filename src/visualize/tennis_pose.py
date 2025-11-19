"""Rendering utilities for tennis pose scenes.

This module turns a single scene frame + camera index into an RGB image that
shows projected court lines, player skeleton, and racket keypoints.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from src.tennis.geometry.skeleton import COCO_BONES

Color = tuple[int, int, int]


COURT_LINE_SEGMENTS = [
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


def _to_int_point(pt: Sequence[float]) -> tuple[int, int]:
    return int(round(float(pt[0]))), int(round(float(pt[1])))


def _draw_segments(
    image: np.ndarray,
    points: Sequence[Sequence[float]],
    visibility: Sequence[int],
    segments: Iterable[tuple[int, int]],
    color: Color,
    thickness: int,
) -> None:
    h, w = image.shape[:2]
    for i, j in segments:
        if not (0 <= i < len(points) and 0 <= j < len(points)):
            continue
        if not (visibility[i] and visibility[j]):
            continue
        p1 = _to_int_point(points[i])
        p2 = _to_int_point(points[j])
        if not (
            0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h
        ):
            continue
        cv2.line(image, p1, p2, color=color, thickness=thickness)


def _draw_points(
    image: np.ndarray,
    points: Sequence[Sequence[float]],
    visibility: Sequence[int],
    color: Color,
    radius: int,
) -> None:
    h, w = image.shape[:2]
    for idx, pt in enumerate(points):
        if not visibility[idx]:
            continue
        x, y = _to_int_point(pt)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), radius, color, thickness=-1)


def render_frame(
    scene: Mapping[str, Any],
    frame_idx: int,
    camera_index: int = 0,
    width: int | None = None,
    height: int | None = None,
) -> np.ndarray:
    r"""Render a single RGB frame for the given scene and camera.

    Args:
        scene (Mapping[str, Any]): Parsed scene dictionary.
        frame_idx (int): Frame index to render (0-based).
        camera_index (int): Camera index into ``scene[\"cameras\"]``.
        width (int | None): Optional override for output width (pixels).
        height (int | None): Optional override for output height (pixels).

    Returns:
        np.ndarray: ``HxWx3`` uint8 BGR image.

    Raises:
        IndexError: If frame or camera indices are invalid.

    """
    frames = scene["frames"]
    cameras = scene["cameras"]
    if not (0 <= frame_idx < len(frames)):
        msg = f"frame_idx {frame_idx} out of range (0..{len(frames) - 1})"
        raise IndexError(msg)
    if not (0 <= camera_index < len(cameras)):
        msg = f"camera_index {camera_index} out of range (0..{len(cameras) - 1})"
        raise IndexError(msg)

    cam = cameras[camera_index]
    cam_w, cam_h = cam["image_size"]
    w = int(width or cam_w)
    h = int(height or cam_h)

    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, :] = (30, 80, 30)

    frame = frames[frame_idx]
    cam_key = f"cam_{camera_index}"
    cam_payload = frame[cam_key]

    court_2d = cam_payload["court_keypoints_2d"]["points"]
    court_vis = cam_payload["court_keypoints_2d"]["visibility"]
    players_2d = cam_payload["player_keypoints_2d"]
    rackets_2d = cam_payload["racket_keypoints_2d"]
    player_tracks = players_2d.get("joints", [])
    player_vis_tracks = players_2d.get("visibility", [])
    racket_tracks = rackets_2d.get("points", [])
    racket_vis_tracks = rackets_2d.get("visibility", [])

    _draw_segments(image, court_2d, court_vis, COURT_LINE_SEGMENTS, (220, 220, 220), 2)
    _draw_points(image, court_2d, court_vis, (255, 255, 255), 3)

    for pts, vis in zip(player_tracks, player_vis_tracks, strict=True):
        _draw_segments(image, pts, vis, COCO_BONES, (0, 255, 255), 2)
        _draw_points(image, pts, vis, (0, 255, 255), 3)

    racket_segments = [(0, 1), (1, 2)]
    for pts, vis in zip(racket_tracks, racket_vis_tracks, strict=True):
        _draw_segments(image, pts, vis, racket_segments, (0, 165, 255), 2)
        _draw_points(image, pts, vis, (0, 165, 255), 3)

    return image


def render_video(
    scene: Mapping[str, Any],
    out_path: str,
    camera_index: int = 0,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
) -> None:
    r"""Render an entire scene to a video file for one camera.

    Args:
        scene (Mapping[str, Any]): Scene dictionary produced by the simulator.
        out_path (str): Destination video path (e.g. ``.mp4``).
        camera_index (int): Camera index to render.
        width (int | None): Optional override for width in pixels.
        height (int | None): Optional override for height in pixels.
        fps (int | None): Video FPS; defaults to ``scene[\"fps\"]``.

    Returns:
        None: The function only writes to disk.

    Raises:
        ValueError: If the scene lacks frames/cameras.
        RuntimeError: If the video writer cannot be created.

    """
    frames = scene["frames"]
    cameras = scene["cameras"]
    if not frames:
        raise ValueError("scene has no frames")
    if not cameras:
        raise ValueError("scene has no cameras")

    tmp_frame = render_frame(scene, 0, camera_index, width=width, height=height)
    h, w = tmp_frame.shape[:2]
    out_fps = int(fps or scene.get("fps", 30))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))
    if not writer.isOpened():  # pragma: no cover - defensive
        raise RuntimeError(f"failed to open video writer for {out_path}")

    try:
        for idx in range(len(frames)):
            frame_img = render_frame(scene, idx, camera_index, width=w, height=h)
            writer.write(frame_img)
    finally:
        writer.release()
