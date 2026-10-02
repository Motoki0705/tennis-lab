"""Generated, camera-local context in the decoded JPEG's pixel coordinates."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import ClipRecord
from src.tasks.ball_refiner.data.context import (
    COURT_KEYPOINTS,
    POSE_JOINTS,
    CourtContext,
    PipelineContext,
    PoseContext,
)


@dataclass(frozen=True)
class ContextArrays:
    """Raw COCO17 retains heatmap peaks; absent crops are explicitly zero.

    All model coordinates use stored JPEG pixels. Camera IDs are optional
    store metadata, never invented for single-camera sources. Track IDs are
    provenance only; the model consumes an unordered set of people.
    """

    frame_index: NDArray[np.int32]
    pts: NDArray[np.int64]
    detection_count: NDArray[np.int32]
    track_ids: NDArray[np.int64]
    boxes_xys: NDArray[np.float32]  # T,P,3; only observed boxes are retained
    track_observed: NDArray[np.bool_]  # T,P
    keypoints: NDArray[np.float32]  # T,P,17,3 (x,y,raw heatmap peak)
    court_points: NDArray[np.float32]  # 14,2; frame 0
    court_valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        n, p = len(self.frame_index), len(self.track_ids)
        for name, shape, dtype in (
            ("frame_index", (n,), np.int32), ("pts", (n,), np.int64),
            ("detection_count", (n,), np.int32), ("track_ids", (p,), np.int64),
            ("boxes_xys", (n, p, 3), np.float32), ("track_observed", (n, p), np.bool_),
            ("keypoints", (n, p, 17, 3), np.float32),
            ("court_points", (COURT_KEYPOINTS, 2), np.float32),
            ("court_valid", (COURT_KEYPOINTS,), np.bool_),
        ):
            value = getattr(self, name)
            if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError(f"Invalid context array {name}")
        if n < 1 or not np.array_equal(self.frame_index, np.arange(n)) or (np.diff(self.pts) <= 0).any():
            raise ValueError("Context must cover a dense, increasing source timeline")
        if (self.detection_count < 0).any() or (self.track_observed.sum(axis=1) > self.detection_count).any():
            raise ValueError("Observed tracks must be backed by per-frame detections")
        if (self.track_ids < 0).any() or (np.diff(self.track_ids) <= 0).any():
            raise ValueError("Track IDs must be nonnegative, unique and sorted")
        if p and not self.track_observed.any(axis=0).all():
            raise ValueError("Every track needs at least one observed crop")
        if (self.boxes_xys[..., 2][self.track_observed] <= 0).any():
            raise ValueError("Observed boxes need positive sizes")
        if (self.keypoints[~self.track_observed] != 0).any() or (self.boxes_xys[~self.track_observed] != 0).any():
            raise ValueError("Unobserved crops must be zero, not interpolated pose observations")

    def arrays(self) -> dict[str, NDArray[np.generic]]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_arrays(cls, values: dict[str, NDArray[np.generic]]) -> ContextArrays:
        if set(values) != {field.name for field in fields(cls)}:
            raise ValueError("Context array keys do not match the versioned contract")
        return cls(**values)

    def model_context(self, clip: ClipRecord, *, pose_threshold: float, provenance: dict[str, Any]) -> PipelineContext:
        """Convert once to source endpoint UV, preserving outside-image poses."""
        if len(self.frame_index) != clip.frame_count or min(clip.width, clip.height, clip.source_width, clip.source_height) <= 1:
            raise ValueError("Context/source size or timeline mismatch")
        if not np.isfinite(pose_threshold) or not 0 <= pose_threshold <= 1:
            raise ValueError("pose_threshold must be in [0,1]")
        denominator = np.asarray((clip.source_width - 1, clip.source_height - 1), np.float32)
        joints = np.take(self.keypoints, POSE_JOINTS, axis=2)
        raw = joints[..., 2]
        confidence = np.clip(raw, np.float32(0), np.float32(1))
        valid = self.track_observed[..., None] & (confidence > 0) & (confidence >= pose_threshold)
        pose = PoseContext(joints[..., :2] / np.float32(clip.scale) / denominator, confidence, valid)
        court = CourtContext(self.court_points / np.float32(clip.scale) / denominator,
                             self.court_valid.astype(np.float32), self.court_valid.copy())
        return PipelineContext(pose, court, {
            **provenance, "pose_status": "generated", "court_status": "generated",
            "pose_threshold": pose_threshold, "pose_frames": int(valid.any(axis=(1, 2)).sum()),
            "court_keypoints": int(self.court_valid.sum()),
            "pose_confidence_transform": "finite_heatmap_peak_clip_zero_one.v2",
            "pose_confidence_total_slots": int(raw.size),
            "pose_confidence_saturated_slots": int((raw > 1).sum()),
            "pose_confidence_negative_slots": int((raw < 0).sum()),
            "pose_confidence_raw_max": float(raw.max(initial=0)),
        })


@dataclass(frozen=True)
class GeneratedContext:
    arrays: ContextArrays
    execution: dict[str, Any]

    def __post_init__(self) -> None:
        """Execution receipts distinguish no observations from skipped models."""
        n = len(self.arrays.frame_index)
        expected = {
            "person_detection_frames": n, "tracking_frames": n,
            "pose_crops": int(self.arrays.track_observed.sum()),
            "court_frame_indices": [0], "person_region_policy": "full_frame",
            "status": "complete",
        }
        if any(type(self.execution.get(key)) is not type(value) or self.execution[key] != value for key, value in expected.items()):
            raise ValueError("Context execution receipt is incomplete or inconsistent")
        if not isinstance(self.execution.get("court_diagnostics"), dict):
            raise ValueError("Court execution needs explicit geometry diagnostics")
