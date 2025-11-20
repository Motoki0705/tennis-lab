"""Rendering utilities for tennis pose scenes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from src.visualize.tennis_render import (
    COCO_BONES,
    COURT_LINE_SEGMENTS,
    RACKET_SEGMENTS,
    draw_points,
    draw_segments,
)


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

    court_pts = np.asarray(court_2d, dtype=np.float32)
    court_vis_arr = list(int(v) for v in court_vis)
    draw_segments(
        image, court_pts, court_vis_arr, COURT_LINE_SEGMENTS, (220, 220, 220), 2
    )
    draw_points(image, court_pts, court_vis_arr, (255, 255, 255), 3)

    for pts, vis in zip(player_tracks, player_vis_tracks, strict=True):
        draw_segments(image, pts, vis, COCO_BONES, (0, 255, 255), 2)
        draw_points(image, pts, vis, (0, 255, 255), 3)

    for pts, vis in zip(racket_tracks, racket_vis_tracks, strict=True):
        draw_segments(image, pts, vis, RACKET_SEGMENTS, (0, 165, 255), 2)
        draw_points(image, pts, vis, (0, 165, 255), 3)

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
