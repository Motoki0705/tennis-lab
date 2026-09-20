"""Synchronized reviewed 2-D ball observations, with no claimed 3-D ground truth."""

from __future__ import annotations

import hashlib
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.base.triangulation_residual.contracts import RealResidualScene
from src.tasks.base.triangulation_residual.real_clip import load_clip_calibration
from src.tasks.blcs.triangulation_residual.data import (
    _finite_number,
    _positive_integer,
    _read_object,
)

_STATUS_WEIGHTS = {
    "observed": 1.0,
    "interpolated": 0.5,
    "occlusion_estimated": 0.25,
    "unresolved": 0.0,
}


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _read_annotations(
    path: Path,
    video_path: Path,
    *,
    frames: int,
    fps: float,
    width: int,
    height: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str], dict[str, Any]]:
    annotation = _read_object(path)
    if annotation.get("schema_version") != "video_ball_annotation.v2":
        raise ValueError(f"{path}: unsupported ball annotation schema")
    source = annotation.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"{path}: source must be an object")
    dimensions = tuple(
        _positive_integer(source.get(key), f"source.{key}")
        for key in ("width", "height", "frame_count")
    )
    if dimensions != (width, height, frames):
        raise ValueError(f"{path}: source image dimensions/frame count mismatch")
    if source.get("file_name") != video_path.name:
        raise ValueError(f"{path}: source camera video name mismatch")
    if source.get("sha256") != _sha256(video_path):
        raise ValueError(f"{path}: source video SHA256 mismatch")
    numerator = _positive_integer(source.get("fps_numerator"), "fps_numerator")
    denominator = _positive_integer(source.get("fps_denominator"), "fps_denominator")
    source_fps = Fraction(numerator, denominator)
    if not np.isclose(float(source_fps), fps, rtol=0, atol=1e-6):
        raise ValueError(f"{path}: source FPS does not match synchronized clip")
    if source.get("constant_pts_step") is not True:
        raise ValueError(f"{path}: constant source PTS step is required")
    pts_step = _positive_integer(source.get("pts_step"), "pts_step")
    time_base_raw = source.get("time_base")
    if not isinstance(time_base_raw, str):
        raise ValueError(f"{path}: time_base must be a rational string")
    try:
        time_base = Fraction(time_base_raw)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{path}: invalid source time_base") from error
    if pts_step * time_base != 1 / source_fps:
        raise ValueError(f"{path}: source PTS step and FPS disagree")
    if source.get("rotation") != 0:
        raise ValueError(f"{path}: rotated source pixels are unsupported")
    coordinates = annotation.get("coordinate_system")
    if not isinstance(coordinates, dict) or any(
        coordinates.get(key) != expected
        for key, expected in (
            ("origin", "top_left"),
            ("x_axis", "right"),
            ("y_axis", "down"),
            ("frame_index", "zero_based"),
            (
                "pixel_coordinates",
                "Original decoded source pixels; no crop or rescaling.",
            ),
        )
    ):
        raise ValueError(f"{path}: original decoded pixel coordinates are required")
    target = annotation.get("target")
    if not isinstance(target, dict) or target.get("label") != "tennis_ball":
        raise ValueError(f"{path}: target must be the tennis ball")
    track_id = _positive_integer(target.get("track_id"), "target.track_id")
    records = annotation.get("frames")
    if not isinstance(records, list) or len(records) != frames:
        raise ValueError(f"{path}: annotation timeline length mismatch")

    observations: NDArray[np.float64] = np.full((frames, 2), np.nan, dtype=np.float64)
    scores: NDArray[np.float64] = np.zeros(frames, dtype=np.float64)
    states: list[str] = []
    for index, frame in enumerate(records):
        if (
            not isinstance(frame, dict)
            or type(frame.get("frame_index")) is not int
            or frame["frame_index"] != index
        ):
            raise ValueError(f"{path}: frame IDs must be complete, unique and ordered")
        if type(frame.get("pts")) is not int or frame["pts"] != index * pts_step:
            raise ValueError(f"{path}:{index}: frame PTS is not synchronized")
        timestamp = _finite_number(frame.get("timestamp_seconds"), "timestamp_seconds")
        if not np.isclose(timestamp, index / float(source_fps), rtol=0, atol=1e-7):
            raise ValueError(f"{path}:{index}: frame timestamp is not synchronized")
        if frame.get("track_id") != track_id or frame.get("label") != "tennis_ball":
            raise ValueError(f"{path}:{index}: frame target identity mismatch")
        status = frame.get("status")
        if not isinstance(status, str) or status not in _STATUS_WEIGHTS:
            raise ValueError(f"{path}:{index}: unsupported ball status {status!r}")
        states.append(status)
        if status == "unresolved":
            if (
                frame.get("center_px") is not None
                or frame.get("center_normalized") is not None
            ):
                raise ValueError(f"{path}:{index}: unresolved coordinates must be null")
            continue
        center, normalized = frame.get("center_px"), frame.get("center_normalized")
        if (
            not isinstance(center, dict)
            or not isinstance(normalized, dict)
            or set(center) != {"x", "y"}
            or set(normalized) != {"x", "y"}
        ):
            raise ValueError(
                f"{path}:{index}: localized center coordinates are required"
            )
        xy = np.asarray(
            [_finite_number(center[key], f"center_px.{key}") for key in ("x", "y")],
            dtype=np.float64,
        )
        uv = np.asarray(
            [
                _finite_number(normalized[key], f"center_normalized.{key}")
                for key in ("x", "y")
            ],
            dtype=np.float64,
        )
        if (xy < 0).any() or (xy >= [width, height]).any():
            raise ValueError(
                f"{path}:{index}: ball pixels lie outside the decoded frame"
            )
        if not np.allclose(xy / [width, height], uv, rtol=0, atol=1e-7):
            raise ValueError(f"{path}:{index}: normalized and original pixels disagree")
        observations[index] = xy
        scores[index] = _STATUS_WEIGHTS[status]
    provenance = {
        "path": str(path),
        "sha256": _sha256(path),
        "source_video_sha256": source["sha256"],
        "review": annotation.get("review"),
    }
    return observations, scores, states, provenance


def load_real_clip(clip_dir: Path) -> RealResidualScene:
    """Load original pixels in clip camera order and preserve unresolved points.

    Status weights are explicit heuristic observation weights, not probabilities
    or image scores. Occlusion estimates remain available for separate reporting;
    their 0.25 weight is below the shared triangulator's default 0.3 threshold.
    """
    rig, court_px, court_scores, fps, metadata = load_clip_calibration(clip_dir)
    clip = _read_object(clip_dir / "clip.json")
    camera_ids = metadata["camera_ids"]
    video_paths = clip.get("video_paths")
    if not isinstance(video_paths, list) or len(video_paths) != len(camera_ids):
        raise ValueError("Each clip camera requires one source video path")
    frames, width, height = (
        _positive_integer(metadata.get(key), key)
        for key in ("num_frames", "width", "height")
    )
    observations, weights, statuses = [], [], []
    sources = {}
    for camera_id, relative_video in zip(camera_ids, video_paths, strict=True):
        if not isinstance(relative_video, str):
            raise ValueError("Source video path must be a string")
        video_path = clip_dir / relative_video
        if not video_path.resolve().is_relative_to(clip_dir.resolve()):
            raise ValueError("Source video path must remain inside the clip directory")
        pixels, scores, states, provenance = _read_annotations(
            clip_dir / "outsource" / f"{camera_id}_annotations.json",
            video_path,
            frames=frames,
            fps=fps,
            width=width,
            height=height,
        )
        observations.append(pixels)
        weights.append(scores)
        statuses.append(states)
        sources[camera_id] = provenance
    return RealResidualScene(
        observations_px=np.asarray(observations, dtype=np.float64)[:, :, None, :],
        scores=np.asarray(weights, dtype=np.float64)[:, :, None],
        court_px=court_px,
        court_scores=court_scores,
        rig=rig,
        fps=fps,
        metadata={
            **metadata,
            "has_ground_truth_3d": False,
            "evaluation_reference": "reviewed_2d_observations",
            "observation_source": "video_ball_annotation.v2",
            "annotation_sources": sources,
            "frame_status": statuses,
            "status_counts": {
                camera_id: dict(Counter(states))
                for camera_id, states in zip(camera_ids, statuses, strict=True)
            },
            "status_weights": dict(_STATUS_WEIGHTS),
            "score_semantics": "fixed status weights; not calibrated detector probabilities",
            "missing_observations": "NaN pixels with score=0; no interpolation added",
            "triangulation_default_min_score": 0.3,
            "excluded_at_default_min_score": ["occlusion_estimated", "unresolved"],
        },
    )
