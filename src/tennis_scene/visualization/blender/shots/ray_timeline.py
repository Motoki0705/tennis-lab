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
    samples: int = 32
    camera_id: str | None = None
    time_mode: str = "continuous"
    two_d_seconds: float | None = None
    pullback_seconds: float = 3.0
    playback_seconds: float | None = None
    continuous_two_d_fraction: float = 0.5
    continuous_pullback_fraction: float = 0.25
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
            if value is None and name in ("two_d_seconds", "playback_seconds"):
                continue
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        a, b = self.continuous_two_d_fraction, self.continuous_pullback_fraction
        if not (0 < a < 1 and 0 < b < 1 and a + b < 1):
            raise ValueError(
                "Continuous source fractions must be positive and sum to <1"
            )
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


def plan_timeline(manifest: dict[str, Any], shot: RayShot) -> Timeline:
    """Nearest timestamps, with explicit phase durations and no gap filling.

    Continuous partitions the source interval without rewinding. Replay holds
    the final 2D sample during the pullback, then restarts at the first sample.
    None durations mean native source speed; an explicit duration retimes that
    stage. Both the source speed and quantized output duration are recorded.
    """
    start, stop = manifest["frames"]
    source_fps = float(manifest["fps"])
    if not math.isfinite(source_fps) or source_fps <= 0 or stop <= start:
        raise ValueError("Require a nonempty source interval and positive fps")
    if shot.time_mode == "continuous":
        count = stop - start
        split_a = start + round(count * shot.continuous_two_d_fraction)
        split_b = start + round(
            count * (shot.continuous_two_d_fraction + shot.continuous_pullback_fraction)
        )
        if not start < split_a < split_b < stop:
            raise ValueError("Continuous source interval is too short for three stages")
        spans = [(start, split_a), (split_a, split_b), (split_b, stop)]
    else:
        spans = [(start, stop), (stop - 1, stop), (start, stop)]
    durations = (shot.two_d_seconds, shot.pullback_seconds, shot.playback_seconds)
    stages, samples = [], []
    output_start = 0
    for name, (left, right), seconds in zip(
        ("two_d", "pullback", "playback"), spans, durations, strict=True
    ):
        native_speed = seconds is None
        if native_speed:
            seconds = (right - left) / source_fps
        assert seconds is not None
        output_count = math.ceil(seconds * shot.fps)
        if name == "pullback" and output_count < 2:
            raise ValueError("Pullback requires at least two output frames")
        hold = shot.time_mode == "replay" and name == "pullback"
        rate = (
            0.0 if hold else (source_fps if native_speed else (right - left) / seconds)
        )
        source = left + np.floor(np.arange(output_count) * rate / shot.fps + 0.5)
        samples.append(np.minimum(source.astype(np.int64), right - 1))
        stages.append(
            Stage(name, output_start, output_start + output_count, left, right, rate)
        )
        output_start += output_count
    return Timeline(np.concatenate(samples), tuple(stages))


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
