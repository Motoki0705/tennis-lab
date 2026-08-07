"""Render per-task visualizations from a saved SceneResult run directory.

The tennis-scene pipeline consolidates every stage output into a single
``SceneResult`` npz. This script reads that npz (plus the source video from the
metadata sidecar) and renders one video per task so each stage can be inspected
in isolation, writing the results back into the same run directory:

- ``ball_detection`` -> 2D ball position overlaid on the source video.
- ``court_kp``       -> 2D court keypoints overlaid on the source video.
- ``gvhmr``          -> per-player COCO-17 2D pose skeleton on the source video.
- ``plcs``           -> court top-view of player positions + heading (yaw).
- ``blcs``           -> court top/side view of the 3D ball trajectory.

Usage:
    python -m src.tennis_scene.scripts.visualize_tasks
    python -m src.tennis_scene.scripts.visualize_tasks \
        scene_path=tennis_scene/tennis_clip.npz
    python -m src.tennis_scene.scripts.visualize_tasks tasks='[plcs,blcs]'

Notes:
    - Hydra loads configuration from `src/tennis_scene/configs/visualize_tasks.yaml`.
    - Scene and source-video paths are explicit role-relative configuration values.
    - 2D overlays are written with OpenCV (mp4v); the court top-view animations
      are written with matplotlib + ffmpeg.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from omegaconf import DictConfig

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.configuration import validate_visualize_tasks_boundary
from src.utils.hydra import hydra_main, register_boundary_validator
from src.utils.schema.player import COCO17_SKELETON

if TYPE_CHECKING:
    import cv2
    from numpy.typing import NDArray

    from src.tennis_scene.schema import SceneResult

LOGGER = logging.getLogger(__name__)
_BOUNDARY = "tennis_scene.visualize_tasks"
register_boundary_validator(_BOUNDARY, validate_visualize_tasks_boundary)

# Per-player colors (BGR for OpenCV) matching the 3D renderer palette.
_PLAYER_BGR: list[tuple[int, int, int]] = [
    (81, 111, 231),  # #E76F51 orange
    (143, 157, 42),  # #2A9D8F teal
    (106, 196, 233),  # #E9C46A yellow
    (83, 70, 38),  # #264653 dark
]
_BALL_BGR: tuple[int, int, int] = (0, 220, 255)  # amber
_COURT_KP_BGR: tuple[int, int, int] = (60, 255, 60)  # green


def _player_bgr(idx: int) -> tuple[int, int, int]:
    return _PLAYER_BGR[idx % len(_PLAYER_BGR)]


def _to_rgb01(bgr: tuple[int, int, int]) -> tuple[float, float, float]:
    b, g, r = bgr
    return (r / 255.0, g / 255.0, b / 255.0)


def _denorm(uv: NDArray[np.float32], width: int, height: int) -> NDArray[np.float32]:
    """Scale normalized [0, 1] uv coordinates to pixel coordinates."""
    out = uv.astype(np.float32).copy()
    out[..., 0] *= float(width)
    out[..., 1] *= float(height)
    return out


def _load_run(scene_path: Path) -> SceneResult:
    """Load a canonical SceneResult archive and its mandatory metadata."""
    if not scene_path.is_file():
        raise FileNotFoundError(f"SceneResult npz not found: {scene_path}")
    return load_scene_result(scene_path)


def _require_array(scene: SceneResult, field: str) -> NDArray:
    value = getattr(scene, field)
    if not isinstance(value, np.ndarray):
        raise ValueError(f"SceneResult field {field!r} is required for visualization")
    return value


def _read_frames(video_path: Path, num_frames: int) -> list[NDArray[np.uint8]]:
    """Read exactly ``num_frames`` BGR frames from the source video."""
    import cv2

    if not video_path.exists():
        raise FileNotFoundError(f"source video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    frames: list[NDArray[np.uint8]] = []
    while len(frames) < num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cast("NDArray[np.uint8]", frame))
    cap.release()
    if len(frames) != num_frames:
        raise RuntimeError(
            f"video frame count {len(frames)} != SceneResult num_frames {num_frames} "
            f"({video_path})"
        )
    return frames


def _open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    import cv2

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {path}")
    return writer


def _render_ball_detection(
    frames: list[NDArray[np.uint8]],
    scene: SceneResult,
    out_path: Path,
    *,
    fps: float,
    frame_range: range,
    trail_length: int,
) -> None:
    import cv2

    h, w = frames[0].shape[:2]
    ball_uv = _denorm(_require_array(scene, "ball_uv")[0], w, h)  # (T, 2)
    vis = _require_array(scene, "ball_vis")[0].astype(bool)  # (T,)

    writer = _open_writer(out_path, fps, w, h)
    for t in frame_range:
        frame = frames[t].copy()
        # Trail of recent visible detections.
        pts = [
            (int(ball_uv[s, 0]), int(ball_uv[s, 1]))
            for s in range(max(frame_range.start, t - trail_length), t)
            if vis[s]
        ]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], _BALL_BGR, 1, cv2.LINE_AA)
        if vis[t]:
            x, y = int(ball_uv[t, 0]), int(ball_uv[t, 1])
            cv2.circle(frame, (x, y), 6, _BALL_BGR, 2, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 1, _BALL_BGR, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"ball_detection  frame {t}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()
    LOGGER.info("wrote %s", out_path)


def _render_court_kp(
    frames: list[NDArray[np.uint8]],
    scene: SceneResult,
    out_path: Path,
    *,
    fps: float,
    frame_range: range,
) -> None:
    import cv2

    h, w = frames[0].shape[:2]
    kp = _denorm(scene.court_kp[0], w, h)  # (T, K, 2)
    vis = scene.court_vis[0]  # (T, K)
    num_kp = kp.shape[1]

    writer = _open_writer(out_path, fps, w, h)
    for t in frame_range:
        frame = frames[t].copy()
        for k in range(num_kp):
            if vis[t, k] <= 0.0:
                continue
            x, y = int(kp[t, k, 0]), int(kp[t, k, 1])
            cv2.circle(frame, (x, y), 4, _COURT_KP_BGR, -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                str(k),
                (x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                _COURT_KP_BGR,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            f"court_kp ({num_kp})  frame {t}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()
    LOGGER.info("wrote %s", out_path)


def _render_gvhmr_pose(
    frames: list[NDArray[np.uint8]],
    scene: SceneResult,
    out_path: Path,
    *,
    fps: float,
    frame_range: range,
    conf_threshold: float,
) -> None:
    import cv2

    h, w = frames[0].shape[:2]
    # human_kp_2d: (P, C, T, 17, 2) normalized; single camera -> C index 0.
    kp = _denorm(_require_array(scene, "human_kp_2d")[:, 0], w, h)
    conf = _require_array(scene, "human_kp_vis")[:, 0]  # (P, T, 17)
    track_ids = [
        int(v) for v in _require_array(scene, "player_track_ids").tolist()
    ]
    num_players = kp.shape[0]

    writer = _open_writer(out_path, fps, w, h)
    for t in frame_range:
        frame = frames[t].copy()
        for p in range(num_players):
            color = _player_bgr(p)
            good = conf[p, t] >= conf_threshold
            for a, b in COCO17_SKELETON:
                if good[a] and good[b]:
                    pa = (int(kp[p, t, a, 0]), int(kp[p, t, a, 1]))
                    pb = (int(kp[p, t, b, 0]), int(kp[p, t, b, 1]))
                    cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)
            for j in range(kp.shape[2]):
                if good[j]:
                    cv2.circle(
                        frame,
                        (int(kp[p, t, j, 0]), int(kp[p, t, j, 1])),
                        3,
                        color,
                        -1,
                        cv2.LINE_AA,
                    )
            # Label near the nose (joint 0) when confident.
            if good[0]:
                cv2.putText(
                    frame,
                    f"P{track_ids[p]}",
                    (int(kp[p, t, 0, 0]) + 6, int(kp[p, t, 0, 1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        cv2.putText(
            frame,
            f"gvhmr 2D pose  frame {t}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()
    LOGGER.info("wrote %s", out_path)


def _court_limits() -> tuple[float, float, float, float]:
    from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH

    return (-HALF_DOUBLES_WIDTH, HALF_DOUBLES_WIDTH, -HALF_LENGTH, HALF_LENGTH)


def _render_plcs(
    scene: SceneResult,
    out_path: Path,
    *,
    fps: float,
    frame_range: range,
    dpi: int,
    trail_length: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    from src.utils.rendering.court_renderer import CourtRenderer

    pos = scene.player_position  # (P, T, 3)
    yaw = scene.player_yaw  # (P, T)
    track_ids = [
        int(v) for v in _require_array(scene, "player_track_ids").tolist()
    ]
    num_players = pos.shape[0]

    court = CourtRenderer()
    xmin, xmax, ymin, ymax = _court_limits()
    x_lo = min(xmin, float(pos[..., 0].min())) - 2.0
    x_hi = max(xmax, float(pos[..., 0].max())) + 2.0
    y_lo = min(ymin, float(pos[..., 1].min())) - 2.0
    y_hi = max(ymax, float(pos[..., 1].max())) + 2.0

    fig, ax = plt.subplots(figsize=(7, 9))

    def update(t: int) -> list:
        ax.clear()
        court.render_2d(ax, set_limits=False)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")
        for p in range(num_players):
            rgb = _to_rgb01(_player_bgr(p))
            s = max(frame_range.start, t - trail_length)
            ax.plot(
                pos[p, s : t + 1, 0],
                pos[p, s : t + 1, 1],
                color=rgb,
                alpha=0.5,
                linewidth=1.5,
            )
            x, y, hd = pos[p, t, 0], pos[p, t, 1], yaw[p, t]
            ax.scatter([x], [y], color=rgb, s=60, zorder=5)
            ax.arrow(
                x,
                y,
                np.sin(hd) * 1.5,
                np.cos(hd) * 1.5,
                color=rgb,
                width=0.06,
                head_width=0.35,
                zorder=6,
            )
            ax.text(x + 0.3, y + 0.3, f"P{track_ids[p]}", color=rgb, fontsize=9)
        ax.set_title(f"plcs (position + yaw)  frame {t}")
        ax.set_xlabel("court x [m]")
        ax.set_ylabel("court y [m]")
        return []

    anim = FuncAnimation(fig, update, frames=frame_range, interval=1000.0 / fps)
    anim.save(str(out_path), writer=FFMpegWriter(fps=int(round(fps))), dpi=dpi)
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


def _render_blcs(
    scene: SceneResult,
    out_path: Path,
    *,
    fps: float,
    frame_range: range,
    dpi: int,
    trail_length: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    from src.utils.rendering.court_renderer import CourtRenderer

    ball = _require_array(scene, "ball_3d")  # (T, 3)
    finite = np.isfinite(ball).all(axis=-1)

    court = CourtRenderer()
    xmin, xmax, ymin, ymax = _court_limits()
    fin = ball[finite]
    x_lo = min(xmin, float(fin[:, 0].min())) - 2.0
    x_hi = max(xmax, float(fin[:, 0].max())) + 2.0
    y_lo = min(ymin, float(fin[:, 1].min())) - 2.0
    y_hi = max(ymax, float(fin[:, 1].max())) + 2.0
    z_hi = max(4.0, float(fin[:, 2].max())) + 0.5

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(13, 8))
    ball_rgb = _to_rgb01(_BALL_BGR)

    def update(t: int) -> list:
        ax_top.clear()
        ax_side.clear()
        # Top view (x-y) with court.
        court.render_2d(ax_top, set_limits=False)
        ax_top.set_xlim(x_lo, x_hi)
        ax_top.set_ylim(y_lo, y_hi)
        ax_top.set_aspect("equal")
        # Side view (y-z): ground + net line.
        ax_side.axhline(0.0, color="#555555", linewidth=1.0)
        ax_side.plot([0.0, 0.0], [0.0, 0.914], color="#404040", linewidth=3.0)
        ax_side.set_xlim(y_lo, y_hi)
        ax_side.set_ylim(-0.2, z_hi)

        s = max(frame_range.start, t - trail_length)
        seg = ball[s : t + 1]
        seg_ok = np.isfinite(seg).all(axis=-1)
        if seg_ok.sum() > 1:
            ax_top.plot(
                seg[seg_ok, 0], seg[seg_ok, 1], color=ball_rgb, alpha=0.6, linewidth=1.5
            )
            ax_side.plot(
                seg[seg_ok, 1], seg[seg_ok, 2], color=ball_rgb, alpha=0.6, linewidth=1.5
            )
        if finite[t]:
            ax_top.scatter(
                [ball[t, 0]],
                [ball[t, 1]],
                color=ball_rgb,
                s=50,
                zorder=5,
                edgecolors="k",
            )
            ax_side.scatter(
                [ball[t, 1]],
                [ball[t, 2]],
                color=ball_rgb,
                s=50,
                zorder=5,
                edgecolors="k",
            )

        ax_top.set_title(f"blcs top view (x-y)  frame {t}")
        ax_top.set_xlabel("court x [m]")
        ax_top.set_ylabel("court y [m]")
        ax_side.set_title("blcs side view (y-z, height)")
        ax_side.set_xlabel("court y [m]")
        ax_side.set_ylabel("height z [m]")
        return []

    anim = FuncAnimation(fig, update, frames=frame_range, interval=1000.0 / fps)
    anim.save(str(out_path), writer=FFMpegWriter(fps=int(round(fps))), dpi=dpi)
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)


_VIDEO_TASKS = {"ball_detection", "court_kp", "gvhmr"}
_PLOT_TASKS = {"plcs", "blcs"}


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="visualize_tasks",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:
    """Render the requested per-task visualizations for one run directory."""
    from src.tennis_scene.configuration import parse_visualize_tasks_config

    runtime = parse_visualize_tasks_config(cfg)
    tasks = list(runtime.tasks)
    scene = _load_run(runtime.scene_path)
    video_path = runtime.video_paths[0]
    num_frames = scene.num_frames
    fps = runtime.fps if runtime.fps is not None else scene.fps

    start = runtime.start_frame
    end = runtime.end_frame if runtime.end_frame is not None else num_frames
    frame_range = range(start, end)
    trail_length = runtime.trail_length
    output_dir = runtime.output_directory
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "scene=%s  frames=%d  fps=%.1f  tasks=%s",
        runtime.scene_path,
        num_frames,
        fps,
        tasks,
    )

    if _VIDEO_TASKS & set(tasks):
        LOGGER.info("reading source video %s", video_path)
        frames = _read_frames(video_path, num_frames)
        if "ball_detection" in tasks:
            _render_ball_detection(
                frames,
                scene,
                output_dir / "ball_detection_viz.mp4",
                fps=fps,
                frame_range=frame_range,
                trail_length=trail_length,
            )
        if "court_kp" in tasks:
            _render_court_kp(
                frames,
                scene,
                output_dir / "court_kp_viz.mp4",
                fps=fps,
                frame_range=frame_range,
            )
        if "gvhmr" in tasks:
            _render_gvhmr_pose(
                frames,
                scene,
                output_dir / "gvhmr_viz.mp4",
                fps=fps,
                frame_range=frame_range,
                conf_threshold=runtime.kp_conf_threshold,
            )

    if "plcs" in tasks:
        _render_plcs(
            scene,
            output_dir / "plcs_viz.mp4",
            fps=fps,
            frame_range=frame_range,
            dpi=runtime.dpi,
            trail_length=trail_length,
        )
    if "blcs" in tasks:
        _render_blcs(
            scene,
            output_dir / "blcs_viz.mp4",
            fps=fps,
            frame_range=frame_range,
            dpi=runtime.dpi,
            trail_length=trail_length,
        )

    LOGGER.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
