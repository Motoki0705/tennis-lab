"""Associate stored detections without invoking a person detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.submodules.models import (
    BotSortAssociator,
    PersonDetectionResult,
    TrackRequest,
    select_and_complete_tracks,
)
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort, SourceVideo
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.utils.video import OpenCVVideoFrameReader


@dataclass(frozen=True)
class PersonTrackingInput:
    video: SourceVideo
    detections: PersonDetectionOutput


@dataclass(frozen=True)
class PersonTrackingOutput:
    camera_id: str
    track_ids: NDArray[np.int64]
    boxes_xyxy: NDArray[np.float32]  # P,T,4; interpolation is explicitly masked
    observed: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if self.track_ids.ndim != 1 or self.track_ids.dtype != np.int64 or len(np.unique(self.track_ids)) != len(self.track_ids):
            raise ValueError("Track IDs must be unique int64 values")
        if len(self.track_ids) > 4:
            raise ReconstructionUnavailable("person_capacity_exceeded", "More than four cumulative camera-local tracks; IDs cannot be recycled or silently merged")
        if self.boxes_xyxy.ndim != 3 or self.boxes_xyxy.shape[0] != len(self.track_ids) or self.boxes_xyxy.shape[-1] != 4 or self.observed.shape != self.boxes_xyxy.shape[:2]:
            raise ValueError("Invalid track shapes")
        if self.observed.dtype != np.bool_ or self.boxes_xyxy.dtype != np.float32 or not np.isfinite(self.boxes_xyxy).all():
            raise ValueError("Invalid track values")


class PersonTrackingModule:
    io = ComponentIO("person_tracking", PersonTrackingInput, PersonTrackingOutput,
        {"detections": InputPort("person_detections")}, "person_tracks")

    def process(self, inputs: PersonTrackingInput) -> PersonTrackingOutput:
        if inputs.detections.camera_id != inputs.video.camera_id or len(inputs.detections.frame_offsets) != inputs.video.num_frames + 1:
            raise ValueError("Tracking input camera/timeline mismatch")
        if not len(inputs.detections.confidence):
            return PersonTrackingOutput(inputs.video.camera_id, np.empty(0, np.int64),
                np.empty((0, inputs.video.num_frames, 4), np.float32), np.empty((0, inputs.video.num_frames), bool))
        tracker = BotSortAssociator()
        history: list[list[dict[str, Any]]] = []
        for packet in OpenCVVideoFrameReader(inputs.video.path, max_frames=inputs.video.num_frames):
            start, end = inputs.detections.frame_offsets[packet.index:packet.index + 2]
            detection = PersonDetectionResult(inputs.detections.boxes_xyxy[start:end], inputs.detections.confidence[start:end])
            history.append(tracker.update(detection, packet.frame))
        if len(history) != inputs.video.num_frames:
            raise ValueError("Tracking did not decode the complete source timeline")
        result = select_and_complete_tracks(history, TrackRequest(inputs.video.path, None, False, max_frames=inputs.video.num_frames), inputs.video.num_frames)
        ids = result.track_ids
        boxes = np.stack([result.tracks[i].numpy() for i in ids]) if ids else np.zeros((0, inputs.video.num_frames, 4), np.float32)
        observed = np.stack([result.observed_mask(i).numpy() for i in ids]) if ids else np.zeros((0, inputs.video.num_frames), bool)
        return PersonTrackingOutput(inputs.video.camera_id, np.asarray(ids, np.int64), boxes.astype(np.float32), observed)
