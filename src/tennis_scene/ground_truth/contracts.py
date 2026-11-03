"""Evaluation-only reference types, independent of reconstruction components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ._validation import digest, integer, number, string

ReferenceKind = Literal[
    "observed", "estimated", "excluded_extrapolation", "occluded", "unknown"
]


@dataclass(frozen=True)
class ReferenceIdentity:
    """Caller-owned clip identity and native axes; hashes come from trusted inputs."""

    clip_id: str
    camera_ids: tuple[str, ...]
    num_frames: int
    fps: float
    source_size_wh: tuple[int, int]
    video_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        string(self.clip_id, "clip_id")
        if not isinstance(self.camera_ids, tuple) or not self.camera_ids:
            raise ValueError("camera_ids: expected a nonempty tuple")
        for camera in self.camera_ids:
            string(camera, "camera_id")
        if len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError("camera_ids: duplicate camera")
        integer(self.num_frames, "num_frames", minimum=1)
        if number(self.fps, "fps") <= 0:
            raise ValueError("fps: expected a positive value")
        if not isinstance(self.source_size_wh, tuple) or len(self.source_size_wh) != 2:
            raise ValueError("source_size_wh: expected (width, height)")
        for dimension in self.source_size_wh:
            integer(dimension, "source_size_wh", minimum=1)
        if set(self.video_sha256) != set(self.camera_ids):
            raise ValueError("video_sha256: camera keys mismatch")
        for value in self.video_sha256.values():
            digest(value, "video_sha256")
        object.__setattr__(
            self, "video_sha256", MappingProxyType(dict(self.video_sha256))
        )


@dataclass(frozen=True)
class ReferencePosition:
    camera_id: str
    frame_index: int
    source_status: str
    reference_kind: ReferenceKind
    valid_for_scoring: bool
    uv_px: tuple[float, float] | None
    source_file: str
    estimation: Mapping[str, str] | None


@dataclass(frozen=True)
class ReferenceSelection:
    """Explicit (camera, frame) axes. Missing UVs are NaN and never scoreable."""

    camera_ids: tuple[str, ...]
    frame_indices: tuple[int, ...]
    uv_px: NDArray[np.float64]  # V,T,2, native pixels
    valid_for_scoring: NDArray[np.bool_]  # V,T
    reference_kind: NDArray[np.str_]  # V,T; estimates remain distinct


@dataclass(frozen=True)
class UserReference:
    """Validated scoring truth. Never a BallDetectionOutput or model evidence."""

    identity: ReferenceIdentity
    sha256: str
    positions: Mapping[tuple[str, int], ReferencePosition]
    events: tuple[Mapping[str, Any], ...]
    estimation_segments: tuple[Mapping[str, str], ...]
    metadata: Mapping[str, Any]

    def position(self, camera_id: str, frame_index: int) -> ReferencePosition:
        string(camera_id, "camera_id")
        integer(frame_index, "frame_index")
        if camera_id not in self.identity.camera_ids:
            raise ValueError(f"Unknown camera_id: {camera_id}")
        if frame_index >= self.identity.num_frames:
            raise ValueError(f"frame_index outside reference timeline: {frame_index}")
        return self.positions[camera_id, frame_index]

    def select(
        self, *, camera_ids: Sequence[str], frame_indices: Sequence[int]
    ) -> ReferenceSelection:
        """Return requested axes in caller order, without resampling or filling."""
        cameras, frames = tuple(camera_ids), tuple(frame_indices)
        if not cameras or not frames:
            raise ValueError("Selection requires nonempty camera and frame axes")
        for camera in cameras:
            self.position(camera, 0)
        for frame in frames:
            self.position(cameras[0], frame)
        if len(set(cameras)) != len(cameras) or len(set(frames)) != len(frames):
            raise ValueError("Selection axes must not contain duplicates")
        uv: NDArray[np.float64] = np.full(
            (len(cameras), len(frames), 2), np.nan, dtype=np.float64
        )
        valid = np.empty(uv.shape[:2], dtype=np.bool_)
        kinds = np.empty(uv.shape[:2], dtype="U22")
        for v, camera in enumerate(cameras):
            for t, frame in enumerate(frames):
                row = self.position(camera, frame)
                if row.uv_px is not None:
                    uv[v, t] = row.uv_px
                valid[v, t] = row.valid_for_scoring
                kinds[v, t] = row.reference_kind
        for result in (uv, valid, kinds):
            result.setflags(write=False)
        return ReferenceSelection(cameras, frames, uv, valid, kinds)
