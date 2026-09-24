"""CPU rendering of measured 2-D, raw triangulation and residual predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.utils.geometry.triangulation import project_multiview
from src.utils.video import probe_video_info

_BACKGROUND = "#101924"
_FOREGROUND = "#e7edf5"
_RAW = "#97a6b8"
_COLORS = ("#29d4ba", "#ffaf61")
_MIN_SCORE = 0.3
_WIDTH, _HEIGHT = 1440, 960


@dataclass(frozen=True)
class ResidualComparison:
    observations: NDArray[np.float64]
    scores: NDArray[np.float64]
    initial: NDArray[np.float64]
    prediction: NDArray[np.float64]
    initial_valid: NDArray[np.bool_]
    initial_px: NDArray[np.float64]
    prediction_px: NDArray[np.float64]
    initial_projected_valid: NDArray[np.bool_]
    prediction_projected_valid: NDArray[np.bool_]
    camera_ids: tuple[str, ...]
    videos: tuple[Path, ...]
    width: int
    height: int
    fps: float
    clip_id: str

    @property
    def task(self) -> str:
        return "PLCS"

    @property
    def frames(self) -> int:
        return int(self.initial.shape[1])


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_residual_comparison(
    clip_dir: Path, predictions_path: Path
) -> ResidualComparison:
    manifest = json.loads((clip_dir / "clip.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("clip.json must contain an object")
    width = _positive_int(manifest.get("width"), "clip.width")
    height = _positive_int(manifest.get("height"), "clip.height")
    frames = _positive_int(manifest.get("num_frames"), "clip.num_frames")
    clip_fps = manifest.get("fps")
    if isinstance(clip_fps, bool) or not isinstance(clip_fps, (int, float)):
        raise ValueError("clip.fps must be a positive finite number")
    if not np.isfinite(clip_fps) or clip_fps <= 0:
        raise ValueError("clip.fps must be a positive finite number")
    cameras = manifest.get("camera_ids")
    videos = manifest.get("video_paths")
    if (
        not isinstance(cameras, list)
        or len(cameras) < 2
        or any(not isinstance(camera, str) or not camera for camera in cameras)
        or len(cameras) != len(set(cameras))
        or not isinstance(videos, list)
        or len(videos) != len(cameras)
        or any(not isinstance(video, str) or not video for video in videos)
    ):
        raise ValueError(
            "Clip requires ordered unique cameras and matching video paths"
        )
    with np.load(predictions_path, allow_pickle=False) as archive:
        required = {
            "observations_px",
            "scores",
            "initial_world_m",
            "pred_world_m",
            "initial_valid",
            "projection_matrices",
            "fps",
        }
        if not required.issubset(archive.files):
            raise ValueError(
                f"Prediction archive is missing {required - set(archive.files)}"
            )
        observations = np.asarray(archive["observations_px"], dtype=np.float64)
        scores = np.asarray(archive["scores"], dtype=np.float64)
        initial = np.asarray(archive["initial_world_m"], dtype=np.float64)
        prediction = np.asarray(archive["pred_world_m"], dtype=np.float64)
        initial_valid = archive["initial_valid"]
        matrices = np.asarray(archive["projection_matrices"], dtype=np.float64)
        if archive["fps"].shape != ():
            raise ValueError("Prediction fps must be a scalar")
        fps = float(archive["fps"])
    if (
        initial.ndim != 4
        or initial.shape[1] != frames
        or initial.shape[-1] != 3
        or (initial.shape[0], initial.shape[2]) not in {(1, 17), (2, 17)}
        or prediction.shape != initial.shape
        or initial_valid.shape != initial.shape[:-1]
        or initial_valid.dtype != np.bool_
        or not np.isfinite(initial).all()
        or not np.isfinite(prediction).all()
    ):
        raise ValueError(
            "Expected finite (P,T,J,3) seeds/predictions and boolean initial_valid"
        )
    people, _, joints, _ = initial.shape
    expected = (people, len(cameras), frames, joints)
    if observations.shape != (*expected, 2) or scores.shape != expected:
        raise ValueError(
            "Expected observations/scores in clip camera order: (P,V,T,J,2)/(P,V,T,J)"
        )
    if (
        matrices.shape != (len(cameras), 3, 4)
        or not np.isfinite(matrices).all()
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or ((scores > 0) & ~np.isfinite(observations).all(-1)).any()
        or not np.isfinite(fps)
        or fps <= 0
        or not np.isclose(fps, clip_fps, rtol=0, atol=1e-5)
    ):
        raise ValueError("Invalid calibration, observation weights, or clip FPS")
    video_paths = tuple(clip_dir / video for video in videos)
    for path in video_paths:
        info = probe_video_info(path)
        if (info.width, info.height, info.frame_count) != (width, height, frames):
            raise ValueError(
                f"Source video dimensions/timeline disagree with clip: {path}"
            )
        if not np.isclose(info.fps, fps, rtol=0, atol=1e-4):
            raise ValueError(f"Source video FPS disagrees with predictions: {path}")
    projected_initial, initial_depth = project_multiview(initial, matrices)
    projected_prediction, prediction_depth = project_multiview(prediction, matrices)
    initial_px = projected_initial.transpose(0, 3, 1, 2, 4)
    prediction_px = projected_prediction.transpose(0, 3, 1, 2, 4)
    return ResidualComparison(
        observations,
        scores,
        initial,
        prediction,
        initial_valid,
        initial_px,
        prediction_px,
        initial_valid[:, None]
        & np.isfinite(initial_px).all(-1)
        & (initial_depth.transpose(0, 3, 1, 2) > 1e-6),
        np.isfinite(prediction_px).all(-1)
        & (prediction_depth.transpose(0, 3, 1, 2) > 1e-6),
        tuple(cameras),
        video_paths,
        width,
        height,
        fps,
        str(manifest.get("clip_id", clip_dir.name)),
    )
