"""CPU-only settings and source clock for the standalone Blender rays shot.

Imported by filename in Blender and as a package in repository tests. No bpy
or repository imports: the bundle remains the only source of scene geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RayShot:
    fps: int = 30
    width: int = 1920
    height: int = 1080
    samples: int = 64
    camera_id: str | None = None
    time_mode: str = "continuous"
    two_d_seconds: float | None = None
    pullback_seconds: float = 3.0
    playback_seconds: float = 5.0
    frustum_depth_m: float = 1.5
    ray_radius_m: float = 0.012
    ball_radius_m: float = 0.05
    trail_count: int = 16
    trail_stride: int = 2
    camera_colors: tuple[str, ...] = (
        "#4CC9F0",
        "#F4A261",
        "#8AC926",
        "#C77DFF",
        "#FFCD70",
    )

    def __post_init__(self) -> None:
        if self.time_mode not in ("continuous", "replay"):
            raise ValueError("time_mode must be continuous or replay")
        for name in (
            "fps",
            "width",
            "height",
            "samples",
            "trail_count",
            "trail_stride",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "two_d_seconds",
            "pullback_seconds",
            "playback_seconds",
            "frustum_depth_m",
            "ray_radius_m",
            "ball_radius_m",
        ):
            value = getattr(self, name)
            if value is None and name == "two_d_seconds":
                continue
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            self.time_mode == "replay"
            and self.two_d_seconds is not None
            and self.two_d_seconds != self.playback_seconds
        ):
            raise ValueError("Replay requires equal 2D/playback seconds at 1x")
        if self.camera_id is not None and not self.camera_id:
            raise ValueError("camera_id must be nonempty or None")
        if self.frustum_depth_m <= 0.005:
            raise ValueError("frustum_depth_m must leave room for the detection ring")


@dataclass(frozen=True)
class Stage:
    name: str
    start: int
    stop: int
    source_start: int
    source_stop: int
    source_rate_fps: float

    def to_dict(self, fps: int, source_fps: float) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_frames": [self.start, self.stop],
            "seconds": (self.stop - self.start) / fps,
            "source_frames": [self.source_start, self.source_stop],
            "source_speed": self.source_rate_fps / source_fps,
        }


@dataclass(frozen=True)
class Timeline:
    source_frames: np.ndarray
    stages: tuple[Stage, ...]

    @property
    def pullback_start(self) -> int:
        return self.stages[1].start

    @property
    def playback_start(self) -> int:
        return self.stages[2].start


def source_window(source_fps: float, shot: RayShot) -> dict[str, Any]:
    """Derive an explicit source-frame budget; no stage can retime the video."""
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Require positive finite source fps")
    two_d = shot.two_d_seconds
    if two_d is None:
        two_d = 3.0 if shot.time_mode == "continuous" else shot.playback_seconds
    seconds = {
        "two_d": two_d,
        "pullback": shot.pullback_seconds,
        "playback": shot.playback_seconds,
    }
    counts = {name: math.ceil(value * source_fps) for name, value in seconds.items()}
    if shot.time_mode == "replay":
        counts["pullback"] = 0
        # Both 2D and 3D consume the same input interval.
        window_frames = counts["playback"]
    else:
        window_frames = sum(counts.values())
    return {
        "time_mode": shot.time_mode,
        "source_fps": source_fps,
        "requested_seconds": seconds,
        "source_frame_counts": counts,
        "window_frames": window_frames,
        "rounding_rule": "ceil(seconds * source_fps) for each advancing stage",
        "source_speed": 1.0,
        "pullback_source_policy": "advance_1x"
        if shot.time_mode == "continuous"
        else "hold_last_frame",
    }


def plan_timeline(manifest: dict[str, Any], shot: RayShot) -> Timeline:
    """Sample at 1x, using seconds to partition an explicitly sized interval.

    Continuous uses one native-speed clock across all three stages. Replay
    samples the whole interval twice and holds the final 2D sample while the
    camera moves. A short/long input interval is rejected, never retimed.
    """
    start, stop = manifest["frames"]
    source_fps = float(manifest["fps"])
    budget = source_window(source_fps, shot)
    window_frames = budget["window_frames"]
    if stop - start != window_frames:
        raise ValueError(
            f"1x shot requires window_frames={window_frames} at {source_fps}fps "
            f"for the configured seconds; got {stop - start}. Retiming is not allowed."
        )
    count = math.ceil(window_frames * shot.fps / source_fps)
    native = start + np.floor(np.arange(count) * source_fps / shot.fps + 0.5).astype(
        np.int64
    )
    native = np.minimum(native, stop - 1)
    if shot.time_mode == "continuous":
        counts = budget["source_frame_counts"]
        a = start + counts["two_d"]
        b = a + counts["pullback"]
        # Assign each nearest source sample to its source span without resetting
        # the clock or repeating samples at phase boundaries.
        first = int(np.searchsorted(native, a))
        second = int(np.searchsorted(native, b))
        if first < 1 or second - first < 2 or count - second < 1:
            raise ValueError(
                "Require visible 2D/3D stages and at least two pullback frames"
            )
        stages = (
            Stage("two_d", 0, first, start, a, source_fps),
            Stage("pullback", first, second, a, b, source_fps),
            Stage("playback", second, count, b, stop, source_fps),
        )
        return Timeline(native, stages)
    pullback_count = math.ceil(shot.pullback_seconds * shot.fps)
    if pullback_count < 2:
        raise ValueError("Pullback requires at least two output frames")
    samples = np.concatenate(
        (native, np.full(pullback_count, stop - 1, np.int64), native)
    )
    return Timeline(
        samples,
        (
            Stage("two_d", 0, count, start, stop, source_fps),
            Stage("pullback", count, count + pullback_count, stop - 1, stop, 0.0),
            Stage(
                "playback",
                count + pullback_count,
                len(samples),
                start,
                stop,
                source_fps,
            ),
        ),
    )


def select_camera(
    manifest: dict[str, Any], detection_uv: np.ndarray, camera_id: str | None
) -> tuple[int, dict[str, Any]]:
    ids = [camera["id"] for camera in manifest["cameras"]]
    if detection_uv.shape != (
        len(ids),
        manifest["frames"][1] - manifest["frames"][0],
        2,
    ):
        raise ValueError("detection_uv shape does not match bundle cameras/frames")
    counts = np.isfinite(detection_uv).all(axis=2).sum(axis=1)
    if camera_id is None:
        # Stable tie break by camera ID, independent of bundle view ordering.
        view = min(range(len(ids)), key=lambda i: (-int(counts[i]), ids[i]))
        policy = "most_visible_frames"
    else:
        if camera_id not in ids:
            raise ValueError(f"Unknown camera_id {camera_id!r}; available: {ids}")
        view = ids.index(camera_id)
        policy = "explicit_camera_id"
    if counts[view] == 0:
        raise ValueError(f"Selected camera {ids[view]!r} has no visible ball frames")
    return view, {
        "camera_id": ids[view],
        "policy": policy,
        "visible_frames": int(counts[view]),
        "interval_frames": int(detection_uv.shape[1]),
        "visibility_counts": dict(zip(ids, counts.tolist(), strict=True)),
        "tie_break": "camera_id ascending",
    }


def calibrated_pose(
    camera: dict[str, Any], shot: RayShot
) -> tuple[np.ndarray, np.ndarray, float]:
    """The pipeline's centered, square-pixel pinhole -> Blender (-Z, +Y up).

    Match export.camera.to_blender_camera without importing repository code in
    Blender. Unsupported intrinsics/aspect are rejected instead of cropped.
    """
    K, R, t = (np.asarray(camera[name], np.float64) for name in ("K", "R", "t"))
    width, height = camera["width"], camera["height"]
    expected_K = [[K[0, 0], 0, width / 2], [0, K[0, 0], height / 2], [0, 0, 1]]
    if (
        not np.isfinite(K).all()
        or K[0, 0] <= 0
        or not np.allclose(K, expected_K, atol=1e-8, rtol=0)
    ):
        raise ValueError(
            "2D camera requires centered, zero-skew square-pixel intrinsics"
        )
    if shot.width * height != shot.height * width:
        raise ValueError("Shot aspect ratio must match the selected 2D camera")
    if (
        not np.isfinite(R).all()
        or not np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        or np.linalg.det(R) <= 0
    ):
        raise ValueError("Camera R must be a proper rotation")
    center = -R.T @ t
    if not np.isfinite(center).all() or not np.allclose(
        center, camera["center"], atol=1e-5, rtol=0
    ):
        raise ValueError("Camera center disagrees with R/t")
    return center, R.T @ np.diag([1.0, -1.0, -1.0]), float(K[0, 0] * 36.0 / width)
