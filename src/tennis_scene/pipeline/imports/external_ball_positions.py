"""Import the external clip-level pixel coordinates on the clip.json frame axis."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.generate_dataset.manifest import ClipManifest

STATUSES = ("observed", "interpolated", "candidate_supported", "estimated")
REPORTED_VISIBILITY = ("visible", "not_visible", "out_of_frame", "unverified")
SCHEMA = "external_ball_positions"


@dataclass(frozen=True)
class ExternalBallPositions:
    camera_ids: tuple[str, ...]
    image_size: tuple[int, int]
    uv_px: NDArray[
        np.float64
    ]  # original coordinates, including rejected out-of-frame points
    status: NDArray[np.uint8]  # indexes into STATUSES, even for rejected points
    reported_visibility: NDArray[np.uint8]  # indexes into REPORTED_VISIBILITY
    input_files: tuple[str, ...]
    sha256: str

    def __post_init__(self) -> None:
        if (
            self.uv_px.ndim != 3
            or self.uv_px.shape[0] != len(self.camera_ids)
            or self.uv_px.shape[-1] != 2
            or len(self.input_files) != len(self.camera_ids)
            or self.uv_px.dtype != np.float64
            or not np.isfinite(self.uv_px).all()
            or min(self.image_size) < 1
        ):
            raise ValueError(
                "External ball positions must contain finite camera/frame/pixel axes"
            )
        for codes, labels in (
            (self.status, STATUSES),
            (self.reported_visibility, REPORTED_VISIBILITY),
        ):
            if (
                codes.shape != self.uv_px.shape[:2]
                or codes.dtype != np.uint8
                or (codes >= len(labels)).any()
            ):
                raise ValueError(
                    "External ball position attributes disagree with their codebook or axes"
                )

    @property
    def usable(self) -> NDArray[np.bool_]:
        # Compare in the precision of the original JSON. Rounding near the
        # edge of the frame must not change whether a position is accepted.
        return ((self.uv_px >= 0) & (self.uv_px < self.image_size)).all(axis=-1)


def _visibility_code(value: Any) -> int:
    if value is True or value in ("yes", "verified", "visible"):
        return REPORTED_VISIBILITY.index("visible")
    if value is False or value == "no":
        return REPORTED_VISIBILITY.index("not_visible")
    if value in ("out_of_frame", "unverified"):
        return REPORTED_VISIBILITY.index(value)
    raise ValueError(f"Unknown visible_in_frame value: {value!r}")


def _check_number(value: Any, expected: int | float, field: str) -> None:
    types = (float, int) if type(expected) is float else (int,)
    if type(value) not in types or value != expected:
        raise ValueError(f"{field} differs from clip.json: {value!r} != {expected}")


def read_external_ball_positions(
    path: Path, clip: ClipManifest
) -> ExternalBallPositions:
    if path.name != f"{clip.clip_name}_ball_positions.json":
        raise ValueError(
            f"External ball filename disagrees with clip.json: {path.name}"
        )
    content = path.read_bytes()
    document = json.loads(content)
    metadata, cameras = document["metadata"], document["cameras"]
    keys = tuple(f"{clip.clip_name}_{camera}" for camera in clip.camera_ids)
    if set(cameras) != set(keys):
        raise ValueError("External ball clip or camera IDs differ from clip.json")
    _check_number(metadata["total_cameras"], len(keys), "total_cameras")
    _check_number(metadata["frames_per_camera"], clip.num_frames, "frames_per_camera")
    _check_number(metadata["fps"], clip.fps, "metadata FPS")
    shape = (len(keys), clip.num_frames)
    pixels = np.empty((*shape, 2), np.float64)
    status = np.empty(shape, np.uint8)
    visibility = np.empty(shape, np.uint8)
    files = []
    for i, key in enumerate(keys):
        stream = cameras[key]
        filename = stream["input_file"]
        if filename not in (f"{key}.mp4", f"{key}(1).mp4"):
            raise ValueError(
                f"{key}: input filename disagrees with the clip and camera ID"
            )
        files.append(filename)
        for field, expected in (
            ("width", clip.width),
            ("height", clip.height),
            ("fps", clip.fps),
            ("frame_count", clip.num_frames),
        ):
            _check_number(stream[field], expected, f"{key} {field}")
        rows = stream["frames"]
        _check_number(len(rows), clip.num_frames, f"{key} frame array length")
        for frame, row in enumerate(rows):
            if type(row["frame"]) is not int or row["frame"] != frame:
                raise ValueError(f"{key}: expected consecutive frame index {frame}")
            xy = row["x"], row["y"]
            if any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in xy
            ):
                raise ValueError(
                    f"{key} frame {frame}: expected finite pixel coordinates"
                )
            pixels[i, frame] = xy
            status[i, frame] = STATUSES.index(row["status"])
            visibility[i, frame] = _visibility_code(row["visible_in_frame"])
    return ExternalBallPositions(
        clip.camera_ids,
        (clip.width, clip.height),
        pixels,
        status,
        visibility,
        tuple(files),
        sha256(content).hexdigest(),
    )
