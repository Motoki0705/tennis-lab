"""2D overlay rendering on clip media frames.

Draws, per frame of one clip camera:

- the raw 2D observations (court keypoints, player pose keypoints, ball UV),
- the model's predicted player positions and the ball's ground shadow
  ``(x, y, 0)`` projected through the per-frame **ground-plane homography**
  estimated from the detected court keypoints
  (:mod:`src.tasks.slcs.visualization.ground_projection`).

Because the clips carry no calibrated cameras, elevated 3D points cannot be
projected exactly; this overlay therefore shows ground projections only and
labels them as such. Frames whose court detections cannot support a
homography draw observations only, with an explicit on-frame notice (counted
in the return value, never silent).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.slcs.data.dataset import ClipArrays
from src.tasks.slcs.visualization.ground_projection import (
    GroundProjectionError,
    ground_homography_from_court,
    project_ground_points,
)
from src.utils.schema.player import COCO17_SKELETON
from src.utils.video.reader import OpenCVVideoFrameReader
from src.utils.video.writer import save_video_rgb

_PLAYER_COLORS_BGR = ((255, 128, 0), (0, 128, 255))  # per player slot
_BALL_COLOR_BGR = (0, 200, 0)
_COURT_COLOR_BGR = (200, 200, 0)


def render_overlay_video(
    clip: ClipArrays,
    camera_index: int,
    *,
    player_position_m: NDArray[np.float32],  # (P, T, 3) predicted, meters
    player_yaw_rad: NDArray[np.float32],  # (P, T)
    ball_position_m: NDArray[np.float32],  # (T, 3)
    output_path: str | Path,
    court_kp_indices: tuple[int, ...],
    min_homography_points: int,
    court_visibility_threshold: float,
) -> tuple[Path, int]:
    """Render the overlay video; returns ``(path, frames_without_homography)``."""
    manifest = clip.manifest
    camera_id = manifest.camera_ids[camera_index]
    video_path = manifest.media_path(camera_id)
    num_frames = clip.num_frames
    if (
        player_position_m.shape[1] != num_frames
        or ball_position_m.shape[0] != num_frames
    ):
        raise ValueError(
            f"prediction timeline ({player_position_m.shape[1]} / "
            f"{ball_position_m.shape[0]} frames) does not match clip length {num_frames}."
        )

    width, height = manifest.width, manifest.height
    rendered: list[NDArray[np.uint8]] = []
    frames_without_homography = 0

    for packet in OpenCVVideoFrameReader(video_path):
        t = packet.index
        if t >= num_frames:
            break
        frame = packet.frame.copy()
        _draw_observations(frame, clip, camera_index, t, width, height)
        try:
            homography = ground_homography_from_court(
                clip.court_kp[camera_index, t],
                clip.court_vis[camera_index, t],
                width=width,
                height=height,
                court_kp_indices=court_kp_indices,
                min_points=min_homography_points,
                vis_threshold=court_visibility_threshold,
            )
        except GroundProjectionError:
            frames_without_homography += 1
            cv2.putText(
                frame,
                "no ground homography (court occluded)",
                (10, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            _draw_ground_predictions(
                frame,
                homography,
                player_position_m=player_position_m[:, t],
                player_yaw_rad=player_yaw_rad[:, t],
                ball_position_m=ball_position_m[t],
            )
        rendered.append(np.ascontiguousarray(frame[..., ::-1]))

    if len(rendered) != num_frames:
        raise RuntimeError(
            f"{manifest.clip_id}/{camera_id}: decoded {len(rendered)} frames, "
            f"manifest declares {num_frames}."
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video_rgb(np.stack(rendered), output_path, fps=clip.fps)
    return output_path, frames_without_homography


def _draw_observations(
    frame: NDArray[np.uint8],
    clip: ClipArrays,
    cam: int,
    t: int,
    width: int,
    height: int,
) -> None:
    for k in range(clip.court_kp.shape[2]):
        if clip.court_vis[cam, t, k] > 0.5:
            u, v = clip.court_kp[cam, t, k]
            cv2.circle(
                frame, (int(u * width), int(v * height)), 3, _COURT_COLOR_BGR, -1
            )
    for p in range(clip.human_kp_2d.shape[0]):
        color = _PLAYER_COLORS_BGR[p % 2]
        kp = clip.human_kp_2d[p, cam, t]
        vis = clip.human_kp_vis[p, cam, t]
        for a, b in COCO17_SKELETON:
            if vis[a] > 0.5 and vis[b] > 0.5:
                pa = (int(kp[a, 0] * width), int(kp[a, 1] * height))
                pb = (int(kp[b, 0] * width), int(kp[b, 1] * height))
                cv2.line(frame, pa, pb, color, 1, cv2.LINE_AA)
        for j in range(kp.shape[0]):
            if vis[j] > 0.5:
                cv2.circle(
                    frame, (int(kp[j, 0] * width), int(kp[j, 1] * height)), 2, color, -1
                )
    if clip.ball_vis[cam, t]:
        u, v = clip.ball_uv[cam, t]
        cv2.circle(frame, (int(u * width), int(v * height)), 5, _BALL_COLOR_BGR, 2)


def _draw_ground_predictions(
    frame: NDArray[np.uint8],
    homography: NDArray[np.float64],
    *,
    player_position_m: NDArray[np.float32],  # (P, 3)
    player_yaw_rad: NDArray[np.float32],  # (P,)
    ball_position_m: NDArray[np.float32],  # (3,)
) -> None:
    num_players = player_position_m.shape[0]
    arrow_len = 1.0  # meters on the ground plane
    for p in range(num_players):
        color = _PLAYER_COLORS_BGR[p % 2]
        base = player_position_m[p, :2][None, :]
        tip = base + arrow_len * np.array(
            [[np.cos(player_yaw_rad[p]), np.sin(player_yaw_rad[p])]], dtype=np.float32
        )
        base_px, tip_px = project_ground_points(homography, np.concatenate([base, tip]))
        center = (int(base_px[0]), int(base_px[1]))
        cv2.drawMarker(frame, center, color, cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.arrowedLine(
            frame,
            center,
            (int(tip_px[0]), int(tip_px[1])),
            color,
            2,
            cv2.LINE_AA,
            tipLength=0.3,
        )
        cv2.putText(
            frame,
            f"P{p} ground",
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    shadow_px = project_ground_points(homography, ball_position_m[None, :2])[0]
    center = (int(shadow_px[0]), int(shadow_px[1]))
    cv2.circle(frame, center, 6, _BALL_COLOR_BGR, 2)
    cv2.putText(
        frame,
        f"ball shadow (z={float(ball_position_m[2]):.2f}m)",
        (center[0] + 8, center[1] + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        _BALL_COLOR_BGR,
        1,
        cv2.LINE_AA,
    )


__all__ = ["render_overlay_video"]
