"""Camera-local detector observations; interpolated boxes are never observations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ObjectObservations:
    camera_ids: tuple[str, ...]
    size: tuple[int, int]  # width, height
    fps: float
    uv_px: NDArray[np.float32]  # (V,T,D,J,2)
    confidence: NDArray[np.float32]  # (V,T,D,J)
    observed: NDArray[np.bool_]  # (V,T,D), actual detector support
    local_track_ids: NDArray[np.int64]  # (V,D), provenance only
    boxes_xys: NDArray[np.float32] | None = None  # (V,T,D,3)

    def __post_init__(self) -> None:
        if self.uv_px.ndim != 5 or self.uv_px.shape[-1] != 2 or self.confidence.shape != self.uv_px.shape[:-1]:
            raise ValueError("Object observations require (V,T,D,J,2) UV and matching confidence")
        v, t, d = self.uv_px.shape[:3]
        if t < 1 or v != len(self.camera_ids) or len(set(self.camera_ids)) != v:
            raise ValueError("Object camera IDs or timeline disagree")
        if self.observed.shape != (v, t, d) or self.observed.dtype != np.bool_ or self.local_track_ids.shape != (v, d):
            raise ValueError("Invalid detection support / local track provenance")
        if min(self.size) <= 0 or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Invalid observation image size / fps")
        if self.uv_px.dtype != np.float32 or self.confidence.dtype != np.float32 or self.local_track_ids.dtype != np.int64:
            raise TypeError("Observations require float32 coordinates/confidence and int64 local IDs")
        if not np.isfinite(self.uv_px).all() or not np.isfinite(self.confidence).all() or (self.confidence < 0).any():
            raise ValueError("Object observations must be finite with nonnegative confidence")
        if self.boxes_xys is not None and (self.boxes_xys.shape != (v, t, d, 3) or not np.isfinite(self.boxes_xys).all()):
            raise ValueError("Object boxes must match (V,T,D,3)")

    @property
    def num_frames(self) -> int:
        return int(self.uv_px.shape[1])

    def visibility(self, threshold: float) -> NDArray[np.bool_]:
        width, height = self.size
        inside = ((self.uv_px[..., 0] >= 0) & (self.uv_px[..., 0] < width) & (self.uv_px[..., 1] >= 0) & (self.uv_px[..., 1] < height))
        return inside & self.observed[..., None] & (self.confidence >= threshold)

    def normalized(self, threshold: float) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        visible = self.visibility(threshold)
        uv = self.uv_px / np.asarray(self.size, np.float32)
        return np.where(visible[..., None], uv, 0).astype(np.float32), visible

    def select_views(self, indices: tuple[int, ...]) -> ObjectObservations:
        rows = list(indices)
        return ObjectObservations(
            tuple(self.camera_ids[i] for i in rows), self.size, self.fps,
            self.uv_px[rows], self.confidence[rows], self.observed[rows],
            self.local_track_ids[rows], None if self.boxes_xys is None else self.boxes_xys[rows],
        )


@dataclass(frozen=True)
class GroupedObservations:
    identities: NDArray[np.int64]  # (P,)
    uv_px: NDArray[np.float32]  # (P,V,T,J,2)
    confidence: NDArray[np.float32]
    visibility: NDArray[np.bool_]
    raw_indices: NDArray[np.int64]  # (P,V,T), -1 absent


def group_observations(
    observations: ObjectObservations, object_ids: NDArray[np.int64], *, threshold: float
) -> GroupedObservations:
    if object_ids.shape != observations.uv_px.shape[:3] or object_ids.dtype != np.int64:
        raise ValueError("Object IDs must match raw (V,T,D) observations")
    identities = np.unique(object_ids[object_ids >= 0])
    views, frames, _, joints, _ = observations.uv_px.shape
    uv = np.zeros((len(identities), views, frames, joints, 2), np.float32)
    confidence = np.zeros(uv.shape[:-1], np.float32)
    visible = np.zeros(uv.shape[:-1], bool)
    raw = np.full((len(identities), views, frames), -1, np.int64)
    visibility = observations.visibility(threshold)
    for row, identity in enumerate(identities):
        chosen = object_ids == identity
        if (chosen.sum(-1) > 1).any():
            raise ValueError("An ID occupies multiple raw detections in one camera/frame")
        v, t, d = np.nonzero(chosen)
        uv[row, v, t] = observations.uv_px[v, t, d]
        confidence[row, v, t] = np.clip(observations.confidence[v, t, d], 0, 1)
        visible[row, v, t] = visibility[v, t, d]
        raw[row, v, t] = d
    uv[~visible] = 0
    confidence[~visible] = 0
    return GroupedObservations(identities, uv, confidence, visible, raw)
