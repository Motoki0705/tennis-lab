"""Debug video renderers for tennis scene intermediate outputs."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.utils.rendering.court_renderer import CourtLines
from src.utils.schema.court import COURT_SKELETON, HALF_DOUBLES_WIDTH, HALF_LENGTH

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.tennis_scene.io import SceneResult

LOGGER = logging.getLogger(__name__)


_COCO17_SKELETON = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)

_PLAYER_COLORS_BGR = (
    (81, 111, 231),
    (143, 157, 42),
    (106, 196, 233),
    (83, 70, 38),
    (97, 161, 244),
    (172, 129, 94),
)


@dataclass
class DebugVisualizationConfig:
    """Configuration for saving tennis scene intermediate debug videos."""

    output_dir: str | Path
    save_court_kp: bool = True
    save_ball_2d: bool = True
    save_blcs_input: bool = True
    save_blcs_result: bool = True
    save_human_kp: bool = True
    save_plcs_result: bool = True
    fps: float | None = None
    codec: str = "mp4v"
    max_frames: int | None = None
    result_trail_length: int = 30
    court_view_width: int = 720
    court_view_height: int = 960


@dataclass
class DebugVisualizationManifest:
    """Saved and skipped debug visualization outputs."""

    saved: dict[str, Path]
    skipped: dict[str, str]
    manifest_path: Path


@dataclass
class _VisualizationOutput:
    name: str
    path: Path
    draw: Callable[[NDArray[np.uint8], int], NDArray[np.uint8]]


def save_intermediate_visualizations(
    scene: SceneResult,
    video_path: str | Path,
    config: DebugVisualizationConfig,
) -> DebugVisualizationManifest:
    """Save human-readable videos for tennis scene intermediate outputs."""
    video_path = Path(video_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(scene.width)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(scene.height)
        video_fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(scene.fps)
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = _resolve_num_frames(scene, video_frames, config.max_frames)
        fps = float(config.fps or scene.fps or video_fps or 30.0)

        outputs, skipped = _build_outputs(scene, config, output_dir, width, height)
        saved = {output.name: output.path for output in outputs}
        writers = {
            output.name: _open_writer(
                output.path,
                fps=fps,
                width=_output_width(output.name, width, config),
                height=_output_height(output.name, height, config),
                codec=config.codec,
            )
            for output in outputs
        }

        try:
            for frame_idx in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                for output in outputs:
                    rendered = output.draw(frame.copy(), frame_idx)
                    writers[output.name].write(rendered)
        finally:
            for writer in writers.values():
                writer.release()
    finally:
        cap.release()

    manifest_path = output_dir / "debug_visualizations.json"
    _write_manifest(
        manifest_path,
        video_path=video_path,
        num_frames=num_frames,
        saved=saved,
        skipped=skipped,
    )
    LOGGER.info(f"Saved debug visualization manifest to {manifest_path}")
    return DebugVisualizationManifest(
        saved=saved,
        skipped=skipped,
        manifest_path=manifest_path,
    )


def _resolve_num_frames(
    scene: SceneResult,
    video_frames: int,
    max_frames: int | None,
) -> int:
    candidates = [int(scene.num_frames)]
    if video_frames > 0:
        candidates.append(video_frames)
    if max_frames is not None:
        candidates.append(int(max_frames))
    num_frames = min(candidates)
    if num_frames <= 0:
        raise ValueError(f"No frames available for debug visualization: {num_frames}")
    return num_frames


def _open_writer(
    path: Path,
    *,
    fps: float,
    width: int,
    height: int,
    codec: str,
) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def _output_width(name: str, video_width: int, config: DebugVisualizationConfig) -> int:
    if name.endswith("_court_view"):
        return config.court_view_width
    return video_width


def _output_height(name: str, video_height: int, config: DebugVisualizationConfig) -> int:
    if name.endswith("_court_view"):
        return config.court_view_height
    return video_height


def _build_outputs(
    scene: SceneResult,
    config: DebugVisualizationConfig,
    output_dir: Path,
    width: int,
    height: int,
) -> tuple[list[_VisualizationOutput], dict[str, str]]:
    outputs: list[_VisualizationOutput] = []
    skipped: dict[str, str] = {}

    def add_or_skip(
        name: str,
        enabled: bool,
        available: bool,
        reason: str,
        draw: Callable[[NDArray[np.uint8], int], NDArray[np.uint8]],
    ) -> None:
        if not enabled:
            skipped[name] = "disabled"
            return
        if not available:
            skipped[name] = reason
            return
        outputs.append(_VisualizationOutput(name, output_dir / f"{name}.mp4", draw))

    add_or_skip(
        "court_kp_overlay",
        config.save_court_kp,
        scene.court_kp is not None,
        "scene.court_kp is missing",
        lambda frame, idx: _draw_court_kp_overlay(scene, frame, idx, width, height),
    )
    add_or_skip(
        "ball_2d_overlay",
        config.save_ball_2d,
        scene.ball_uv is not None,
        "scene.ball_uv is missing",
        lambda frame, idx: _draw_ball_overlay(scene, frame, idx, width, height),
    )
    add_or_skip(
        "blcs_input_overlay",
        config.save_blcs_input,
        scene.ball_uv is not None and scene.court_kp is not None,
        "scene.ball_uv or scene.court_kp is missing",
        lambda frame, idx: _draw_blcs_input_overlay(scene, frame, idx, width, height),
    )
    add_or_skip(
        "blcs_result_court_view",
        config.save_blcs_result,
        scene.ball_3d is not None,
        "scene.ball_3d is missing",
        lambda _frame, idx: _draw_blcs_result_court_view(scene, idx, config),
    )
    add_or_skip(
        "human_kp_overlay",
        config.save_human_kp,
        scene.human_kp_2d is not None,
        "scene.human_kp_2d is missing",
        lambda frame, idx: _draw_human_kp_overlay(scene, frame, idx, width, height),
    )
    add_or_skip(
        "plcs_result_court_view",
        config.save_plcs_result,
        scene.player_position is not None and scene.player_yaw is not None,
        "scene.player_position or scene.player_yaw is missing",
        lambda _frame, idx: _draw_plcs_result_court_view(scene, idx, config),
    )
    return outputs, skipped


def _draw_court_kp_overlay(
    scene: SceneResult,
    frame: NDArray[np.uint8],
    frame_idx: int,
    width: int,
    height: int,
) -> NDArray[np.uint8]:
    points = _select_frame_points(scene.court_kp, frame_idx, scene.num_frames)
    visibility = _select_frame_visibility(scene.court_vis, frame_idx, scene.num_frames, points.shape[0])
    pixel_points = _points_to_pixels(points, width, height)
    _draw_court_points(frame, pixel_points, visibility)
    _draw_label(frame, "court_kp", frame_idx)
    return frame


def _draw_ball_overlay(
    scene: SceneResult,
    frame: NDArray[np.uint8],
    frame_idx: int,
    width: int,
    height: int,
) -> NDArray[np.uint8]:
    _draw_ball_trace(scene, frame, frame_idx, width, height)
    _draw_label(frame, "ball_2d", frame_idx)
    return frame


def _draw_blcs_input_overlay(
    scene: SceneResult,
    frame: NDArray[np.uint8],
    frame_idx: int,
    width: int,
    height: int,
) -> NDArray[np.uint8]:
    frame = _draw_court_kp_overlay(scene, frame, frame_idx, width, height)
    _draw_ball_trace(scene, frame, frame_idx, width, height)
    _draw_label(frame, "blcs_input", frame_idx)
    return frame


def _draw_human_kp_overlay(
    scene: SceneResult,
    frame: NDArray[np.uint8],
    frame_idx: int,
    width: int,
    height: int,
) -> NDArray[np.uint8]:
    if scene.human_kp_2d is None:
        return frame

    keypoints = _select_human_points(scene.human_kp_2d, frame_idx, scene.num_frames)
    visibility = _select_human_visibility(
        scene.human_kp_vis,
        frame_idx,
        scene.num_frames,
        keypoints.shape[:2],
    )
    track_ids = _track_ids(scene, keypoints.shape[0])
    for player_idx in range(keypoints.shape[0]):
        color = _PLAYER_COLORS_BGR[player_idx % len(_PLAYER_COLORS_BGR)]
        pixel_points = _points_to_pixels(keypoints[player_idx], width, height)
        player_vis = visibility[player_idx] > 0.5
        for start, end in _COCO17_SKELETON:
            if start >= pixel_points.shape[0] or end >= pixel_points.shape[0]:
                continue
            if player_vis[start] and player_vis[end]:
                cv2.line(
                    frame,
                    tuple(pixel_points[start]),
                    tuple(pixel_points[end]),
                    color,
                    2,
                    cv2.LINE_AA,
                )
        for point_idx, point in enumerate(pixel_points):
            if player_vis[point_idx]:
                cv2.circle(frame, tuple(point), 3, color, -1, cv2.LINE_AA)
        visible_points = pixel_points[player_vis]
        if visible_points.size:
            label_xy = tuple(visible_points[0] + np.array([6, -6], dtype=np.int32))
            cv2.putText(
                frame,
                f"P{track_ids[player_idx]}",
                label_xy,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    _draw_label(frame, "human_kp", frame_idx)
    return frame


def _draw_blcs_result_court_view(
    scene: SceneResult,
    frame_idx: int,
    config: DebugVisualizationConfig,
) -> NDArray[np.uint8]:
    canvas = _make_court_canvas(config)
    _draw_ball_3d_trajectory(canvas, scene, frame_idx, config.result_trail_length)
    _draw_label(canvas, "blcs_result_court_view", frame_idx)
    return canvas


def _draw_plcs_result_court_view(
    scene: SceneResult,
    frame_idx: int,
    config: DebugVisualizationConfig,
) -> NDArray[np.uint8]:
    canvas = _make_court_canvas(config)

    positions = scene.player_position[:, frame_idx]
    yaws = scene.player_yaw[:, frame_idx]
    track_ids = _track_ids(scene, positions.shape[0])
    for player_idx, position in enumerate(positions):
        if not np.isfinite(position[:2]).all():
            continue
        color = _PLAYER_COLORS_BGR[player_idx % len(_PLAYER_COLORS_BGR)]
        _draw_player_trail(
            canvas,
            scene.player_position[player_idx],
            frame_idx,
            color,
            config.result_trail_length,
        )
        xy = _court_to_canvas(position[0], position[1], canvas.shape[1], canvas.shape[0])
        yaw = yaws[player_idx]
        direction = np.array([np.sin(yaw), np.cos(yaw)], dtype=np.float32)
        arrow_end = _court_to_canvas(
            position[0] + direction[0] * 1.2,
            position[1] + direction[1] * 1.2,
            canvas.shape[1],
            canvas.shape[0],
        )
        cv2.circle(canvas, xy, 8, color, -1, cv2.LINE_AA)
        cv2.arrowedLine(canvas, xy, arrow_end, color, 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(
            canvas,
            f"P{track_ids[player_idx]}",
            (xy[0] + 10, xy[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    if scene.ball_3d is not None and frame_idx < scene.ball_3d.shape[0]:
        ball = np.asarray(scene.ball_3d[frame_idx], dtype=np.float32)
        if np.isfinite(ball[:2]).all():
            ball_xy = _court_to_canvas(ball[0], ball[1], canvas.shape[1], canvas.shape[0])
            cv2.circle(canvas, ball_xy, 5, (0, 255, 255), -1, cv2.LINE_AA)

    _draw_label(canvas, "plcs_result_court_view", frame_idx)
    return canvas


def _make_court_canvas(config: DebugVisualizationConfig) -> NDArray[np.uint8]:
    canvas = np.full(
        (config.court_view_height, config.court_view_width, 3),
        fill_value=(43, 125, 50),
        dtype=np.uint8,
    )
    _draw_court_topdown(canvas)
    return canvas


def _draw_ball_3d_trajectory(
    canvas: NDArray[np.uint8],
    scene: SceneResult,
    frame_idx: int,
    trail_length: int,
) -> None:
    if scene.ball_3d is None:
        return

    positions = np.asarray(scene.ball_3d, dtype=np.float32)
    max_idx = min(frame_idx, positions.shape[0] - 1)
    start_idx = max(0, max_idx - trail_length)
    previous: tuple[int, int] | None = None
    for idx in range(start_idx, max_idx + 1):
        if not _is_ball_3d_visible(scene, idx):
            previous = None
            continue
        position = positions[idx]
        if not np.isfinite(position).all():
            previous = None
            continue
        xy = _court_to_canvas(position[0], position[1], canvas.shape[1], canvas.shape[0])
        color = _height_color(float(position[2]))
        radius = 5 if idx == max_idx else 3
        if previous is not None:
            cv2.line(canvas, previous, xy, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, xy, radius, color, -1, cv2.LINE_AA)
        previous = xy

    if _is_ball_3d_visible(scene, max_idx) and np.isfinite(positions[max_idx]).all():
        xy = _court_to_canvas(
            positions[max_idx, 0],
            positions[max_idx, 1],
            canvas.shape[1],
            canvas.shape[0],
        )
        cv2.circle(canvas, xy, 10, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"z={positions[max_idx, 2]:.2f}m",
            (xy[0] + 12, xy[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _draw_player_trail(
    canvas: NDArray[np.uint8],
    positions: NDArray[np.float32],
    frame_idx: int,
    color: tuple[int, int, int],
    trail_length: int,
) -> None:
    max_idx = min(frame_idx, positions.shape[0] - 1)
    start_idx = max(0, max_idx - trail_length)
    previous: tuple[int, int] | None = None
    for idx in range(start_idx, max_idx + 1):
        position = positions[idx]
        if not np.isfinite(position[:2]).all():
            previous = None
            continue
        xy = _court_to_canvas(position[0], position[1], canvas.shape[1], canvas.shape[0])
        if previous is not None:
            cv2.line(canvas, previous, xy, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, xy, 3, color, -1, cv2.LINE_AA)
        previous = xy


def _is_ball_3d_visible(scene: SceneResult, frame_idx: int) -> bool:
    if scene.ball_visibility is None:
        return True
    return _select_ball_visibility(scene.ball_visibility, frame_idx)


def _height_color(height_meters: float) -> tuple[int, int, int]:
    clipped = float(np.clip(height_meters, 0.0, 3.0)) / 3.0
    blue = int(255 * (1.0 - clipped))
    green = int(80 + 175 * clipped)
    red = int(255 * clipped)
    return blue, green, red


def _draw_court_points(
    frame: NDArray[np.uint8],
    pixel_points: NDArray[np.int32],
    visibility: NDArray[np.float32],
) -> None:
    visible = visibility > 0.5
    for start, end in COURT_SKELETON:
        if start >= pixel_points.shape[0] or end >= pixel_points.shape[0]:
            continue
        if visible[start] and visible[end]:
            cv2.line(
                frame,
                tuple(pixel_points[start]),
                tuple(pixel_points[end]),
                (64, 220, 255),
                2,
                cv2.LINE_AA,
            )
    for idx, point in enumerate(pixel_points):
        if not visible[idx]:
            continue
        cv2.circle(frame, tuple(point), 4, (64, 220, 255), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            str(idx),
            (int(point[0]) + 5, int(point[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _draw_ball_trace(
    scene: SceneResult,
    frame: NDArray[np.uint8],
    frame_idx: int,
    width: int,
    height: int,
) -> None:
    if scene.ball_uv is None:
        return
    start_idx = max(0, frame_idx - 8)
    previous: tuple[int, int] | None = None
    for idx in range(start_idx, frame_idx + 1):
        point = _select_ball_point(scene.ball_uv, idx, scene.num_frames)
        visible = _select_ball_visibility(scene.ball_visibility, idx)
        if not visible:
            previous = None
            continue
        pixel = _points_to_pixels(point[None, :], width, height)[0]
        alpha = (idx - start_idx + 1) / (frame_idx - start_idx + 1)
        radius = 3 if idx < frame_idx else 6
        color = (0, int(180 + 75 * alpha), 255)
        if previous is not None:
            cv2.line(frame, previous, tuple(pixel), color, 2, cv2.LINE_AA)
        cv2.circle(frame, tuple(pixel), radius, color, -1, cv2.LINE_AA)
        previous = tuple(pixel)


def _draw_court_topdown(canvas: NDArray[np.uint8]) -> None:
    height, width = canvas.shape[:2]
    lines = CourtLines()
    for (x1, y1), (x2, y2) in lines.lines:
        cv2.line(
            canvas,
            _court_to_canvas(x1, y1, width, height),
            _court_to_canvas(x2, y2, width, height),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    (nx1, ny1), (nx2, ny2) = lines.net_line
    cv2.line(
        canvas,
        _court_to_canvas(nx1, ny1, width, height),
        _court_to_canvas(nx2, ny2, width, height),
        (80, 80, 80),
        3,
        cv2.LINE_AA,
    )


def _draw_label(frame: NDArray[np.uint8], name: str, frame_idx: int) -> None:
    text = f"{name} | frame {frame_idx}"
    rect_width = min(frame.shape[1] - 8, max(260, 22 + len(text) * 11))
    cv2.rectangle(frame, (8, 8), (rect_width, 40), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _points_to_pixels(
    points: NDArray[np.float32],
    width: int,
    height: int,
) -> NDArray[np.int32]:
    arr = np.asarray(points, dtype=np.float32).copy()
    if arr.size and np.nanmax(np.abs(arr)) <= 2.0:
        arr[..., 0] *= width
        arr[..., 1] *= height
    arr[..., 0] = np.clip(arr[..., 0], 0, width - 1)
    arr[..., 1] = np.clip(arr[..., 1], 0, height - 1)
    return np.rint(arr).astype(np.int32)


def _select_frame_points(
    points: NDArray[np.float32],
    frame_idx: int,
    num_frames: int,
) -> NDArray[np.float32]:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] == num_frames:
            return arr[min(frame_idx, arr.shape[0] - 1)]
        return arr[0]
    if arr.ndim == 4:
        if arr.shape[1] == num_frames:
            return arr[0, min(frame_idx, arr.shape[1] - 1)]
        if arr.shape[0] == num_frames:
            return arr[min(frame_idx, arr.shape[0] - 1), 0]
        return arr[0, 0]
    if arr.ndim == 5:
        return arr[0, 0, min(frame_idx, arr.shape[2] - 1)]
    raise ValueError(f"Unsupported frame point shape: {arr.shape}")


def _select_frame_visibility(
    visibility: NDArray[np.float32] | None,
    frame_idx: int,
    num_frames: int,
    num_points: int,
) -> NDArray[np.float32]:
    if visibility is None:
        return np.ones(num_points, dtype=np.float32)
    arr = np.asarray(visibility, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[0] == num_frames:
            return arr[min(frame_idx, arr.shape[0] - 1)]
        return arr[0]
    if arr.ndim == 3:
        if arr.shape[1] == num_frames:
            return arr[0, min(frame_idx, arr.shape[1] - 1)]
        if arr.shape[0] == num_frames:
            return arr[min(frame_idx, arr.shape[0] - 1), 0]
        return arr[0, 0]
    if arr.ndim == 4:
        return arr[0, 0, min(frame_idx, arr.shape[2] - 1)]
    raise ValueError(f"Unsupported frame visibility shape: {arr.shape}")


def _select_ball_point(
    ball_uv: NDArray[np.float32],
    frame_idx: int,
    num_frames: int,
) -> NDArray[np.float32]:
    arr = np.asarray(ball_uv, dtype=np.float32)
    if arr.ndim == 2:
        return arr[min(frame_idx, arr.shape[0] - 1)]
    if arr.ndim == 3:
        if arr.shape[1] == num_frames:
            return arr[0, min(frame_idx, arr.shape[1] - 1)]
        return arr[min(frame_idx, arr.shape[0] - 1), 0]
    if arr.ndim == 4:
        return arr[0, 0, min(frame_idx, arr.shape[2] - 1)]
    raise ValueError(f"Unsupported ball_uv shape: {arr.shape}")


def _select_ball_visibility(
    visibility: NDArray[np.bool_] | None,
    frame_idx: int,
) -> bool:
    if visibility is None:
        return True
    arr = np.asarray(visibility)
    if arr.ndim == 1:
        return bool(arr[min(frame_idx, arr.shape[0] - 1)])
    if arr.ndim == 2:
        return bool(arr[0, min(frame_idx, arr.shape[1] - 1)])
    if arr.ndim == 3:
        return bool(arr[0, 0, min(frame_idx, arr.shape[2] - 1)])
    raise ValueError(f"Unsupported ball visibility shape: {arr.shape}")


def _select_human_points(
    human_kp_2d: NDArray[np.float32],
    frame_idx: int,
    num_frames: int,
) -> NDArray[np.float32]:
    arr = np.asarray(human_kp_2d, dtype=np.float32)
    if arr.ndim == 4:
        return arr[:, min(frame_idx, arr.shape[1] - 1)]
    if arr.ndim == 5:
        if arr.shape[2] == num_frames:
            return arr[:, 0, min(frame_idx, arr.shape[2] - 1)]
        return arr[:, min(frame_idx, arr.shape[1] - 1), 0]
    raise ValueError(f"Unsupported human_kp_2d shape: {arr.shape}")


def _select_human_visibility(
    human_kp_vis: NDArray[np.float32] | None,
    frame_idx: int,
    num_frames: int,
    shape: tuple[int, int],
) -> NDArray[np.float32]:
    if human_kp_vis is None:
        return np.ones(shape, dtype=np.float32)
    arr = np.asarray(human_kp_vis, dtype=np.float32)
    if arr.ndim == 3:
        return arr[:, min(frame_idx, arr.shape[1] - 1)]
    if arr.ndim == 4:
        if arr.shape[2] == num_frames:
            return arr[:, 0, min(frame_idx, arr.shape[2] - 1)]
        return arr[:, min(frame_idx, arr.shape[1] - 1), 0]
    raise ValueError(f"Unsupported human_kp_vis shape: {arr.shape}")


def _court_to_canvas(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    margin = 36
    x_extent = HALF_DOUBLES_WIDTH + 3.0
    y_extent = HALF_LENGTH + 3.0
    px = margin + (x + x_extent) / (2 * x_extent) * (width - 2 * margin)
    py = margin + (y_extent - y) / (2 * y_extent) * (height - 2 * margin)
    return int(np.clip(round(px), 0, width - 1)), int(np.clip(round(py), 0, height - 1))


def _track_ids(scene: SceneResult, num_players: int) -> list[int]:
    if scene.player_track_ids is None:
        return list(range(num_players))
    return [int(track_id) for track_id in scene.player_track_ids[:num_players].tolist()]


def _write_manifest(
    path: Path,
    *,
    video_path: Path,
    num_frames: int,
    saved: dict[str, Path],
    skipped: dict[str, str],
) -> None:
    data = {
        "video_path": str(video_path),
        "num_frames": num_frames,
        "saved": {name: str(saved_path) for name, saved_path in sorted(saved.items())},
        "skipped": dict(sorted(skipped.items())),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)