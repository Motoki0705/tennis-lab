"""Scoring-only native-pixel labels with explicit review states and provenance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.utils.checksum import dual_sha256

from .bundle import FixedInputBundle
from .contracts import MethodInput, readonly_arrays
from .storage import read_json


class ObservationState(IntEnum):
    UNREVIEWED = 0
    OBSERVED = 1
    OCCLUDED = 2
    OUT_OF_FRAME = 3
    UNKNOWN = 4


@dataclass(frozen=True)
class EventLabel:
    frame_index: int
    kind: str
    evidence: str

    def __post_init__(self) -> None:
        if type(self.frame_index) is not int or self.frame_index < 0 or self.kind not in {"bounce", "hit"} or not self.evidence:
            raise ValueError("Events require a source frame, bounce/hit kind and review evidence")


@dataclass(frozen=True)
class Ball2DAnnotations:
    camera_ids: tuple[str, ...]
    frame_indices: NDArray[np.int64]
    source_size_wh: tuple[int, int]
    fps: float
    uv_px: NDArray[np.float64]  # V,T,2; zero except observed
    states: NDArray[np.uint8]  # V,T
    provenance: dict[str, Any]
    events: tuple[EventLabel, ...] = ()

    def __post_init__(self) -> None:
        v, t = len(self.camera_ids), len(self.frame_indices)
        if (len(set(self.camera_ids)) != v or v < 1 or t < 1
                or self.frame_indices.dtype != np.int64
                or not np.array_equal(self.frame_indices, np.arange(t))
                or not math.isfinite(self.fps) or self.fps <= 0):
            raise ValueError("Annotation camera/frame/fps axis mismatch")
        if len(self.source_size_wh) != 2 or any(type(n) is not int or n < 2 for n in self.source_size_wh):
            raise ValueError("Annotation source size must be positive native pixels")
        if (self.uv_px.shape != (v, t, 2) or self.uv_px.dtype != np.float64
                or self.states.shape != (v, t) or self.states.dtype != np.uint8
                or np.any(self.states > int(ObservationState.UNKNOWN))):
            raise ValueError("Invalid annotation array schema")
        observed = self.observed
        if (not np.isfinite(self.uv_px).all() or np.any(self.uv_px[~observed] != 0)
                or np.any(self.uv_px[observed] < 0)
                or np.any(self.uv_px[observed] > np.asarray(self.source_size_wh) - 1)):
            raise ValueError("Only observed labels may have finite in-image native pixel coordinates")
        if any(event.frame_index >= t for event in self.events):
            raise ValueError("Event outside source frame axis")
        readonly_arrays(self)

    @property
    def observed(self) -> NDArray[np.bool_]:
        return self.states == ObservationState.OBSERVED

    def state_counts(self) -> dict[str, dict[str, int]]:
        return {camera: {state.name.lower(): int((self.states[i] == state).sum())
                         for state in ObservationState} for i, camera in enumerate(self.camera_ids)}

    def validate_for_inputs(self, inputs: MethodInput) -> None:
        if (self.camera_ids != inputs.camera_ids or self.fps != inputs.fps
                or self.source_size_wh != inputs.source_size_wh
                or not np.array_equal(self.frame_indices, inputs.frame_indices)):
            raise ValueError("Scoring annotation camera/frame/coordinate axes disagree")


def load_annotations(path: Path, bundle: FixedInputBundle) -> Ball2DAnnotations:
    """Missing records remain UNREVIEWED. No interpolation or detector fallback."""
    doc, inputs = read_json(path), bundle.inputs
    required = {"schema", "clip_id", "camera_ids", "num_frames", "fps", "source_size_wh",
                "video_sha256", "provenance", "observations"}
    if not required <= set(doc) or set(doc) - required - {"events"} or doc["schema"] != "tennis_lab_ball2d_annotations.v1":
        raise ValueError("Unsupported scoring annotation schema")
    if (doc["clip_id"] != inputs.clip_id or doc["camera_ids"] != list(inputs.camera_ids)
            or doc["num_frames"] != inputs.num_frames or doc["fps"] != inputs.fps
            or doc["source_size_wh"] != list(inputs.source_size_wh)
            or doc["video_sha256"] != {v.camera_id: v.sha256 for v in bundle.source.videos}):
        raise ValueError("Annotation source hash/camera/frame/fps mismatch")
    provenance = doc["provenance"]
    if (provenance.get("kind") != "direct_video_review" or not provenance.get("model")
            or not provenance.get("effort")):
        raise ValueError("Scoring labels require direct-video review provenance, never detector/test fixtures")
    uv: NDArray[np.float64] = np.zeros((len(inputs.views), inputs.num_frames, 2), np.float64)
    states = np.zeros(uv.shape[:2], np.uint8)
    seen: set[tuple[str, int]] = set()
    state_names = {state.name.lower(): state for state in ObservationState}
    for record in doc["observations"]:
        camera, frame = record["camera_id"], record["frame_index"]
        if (camera not in inputs.camera_ids or type(frame) is not int or not 0 <= frame < inputs.num_frames
                or (camera, frame) in seen or record["status"] not in state_names):
            raise ValueError("Annotation has duplicate/unknown camera, frame or state")
        seen.add((camera, frame))
        state = state_names[record["status"]]
        if state != ObservationState.UNREVIEWED and (record.get("reviewed") is not True or not record.get("evidence")):
            raise ValueError("Reviewed states require explicit image-review evidence")
        if "time_seconds" in record and not math.isclose(record["time_seconds"], frame / inputs.fps, abs_tol=.5 / inputs.fps):
            raise ValueError("Annotation frame/time mismatch")
        i = inputs.camera_ids.index(camera)
        states[i, frame] = state
        if state == ObservationState.OBSERVED:
            point = np.asarray(record.get("uv_px"), np.float64)
            if point.shape != (2,):
                raise ValueError("Observed label needs one native-pixel point")
            uv[i, frame] = point
        elif record.get("uv_px") is not None:
            raise ValueError("Nonobserved annotation must have uv_px=null or omit coordinates")
    events = tuple(EventLabel(**event) for event in doc.get("events", []))
    return Ball2DAnnotations(inputs.camera_ids, inputs.frame_indices, inputs.source_size_wh, inputs.fps,
                             uv, states, {**provenance, "file_sha256": dual_sha256(path)}, events)
