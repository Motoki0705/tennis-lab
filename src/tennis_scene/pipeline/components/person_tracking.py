"""Associate stored detections without invoking a person detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.submodules.models import PersonDetectionResult
from src.tasks.person_tracking.all_person import AllPersonAssociator
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.tracking_identity import TrackletLink
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort, SourceVideo
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
    source_track_ids: tuple[tuple[int, ...], ...]
    tracklet_links: tuple[TrackletLink, ...]

    def __post_init__(self) -> None:
        if self.track_ids.ndim != 1 or self.track_ids.dtype != np.int64 or len(np.unique(self.track_ids)) != len(self.track_ids):
            raise ValueError("Track IDs must be unique int64 values")
        if self.boxes_xyxy.ndim != 3 or self.boxes_xyxy.shape[0] != len(self.track_ids) or self.boxes_xyxy.shape[-1] != 4 or self.observed.shape != self.boxes_xyxy.shape[:2]:
            raise ValueError("Invalid track shapes")
        if self.observed.dtype != np.bool_ or self.boxes_xyxy.dtype != np.float32 or not np.isfinite(self.boxes_xyxy).all():
            raise ValueError("Invalid track values")
        if len(self.source_track_ids) != len(self.track_ids) or any(not group or len(set(group)) != len(group) for group in self.source_track_ids):
            raise ValueError("Each stable ID requires unique source tracklet IDs")
        if any(group[0] != int(track_id) for track_id, group in zip(self.track_ids, self.source_track_ids, strict=True)):
            raise ValueError("Stable ID must be the first source tracklet ID")
        if len({item for group in self.source_track_ids for item in group}) != sum(map(len, self.source_track_ids)):
            raise ValueError("A source tracklet belongs to exactly one stable ID")


class PersonTrackingModule:
    """All-person motion baseline; court selection alone owns the candidate cap."""

    io = ComponentIO("person_tracking", PersonTrackingInput, PersonTrackingOutput,
        {"detections": InputPort("person_detections")}, "person_tracks", version=4)

    def process(self, inputs: PersonTrackingInput) -> PersonTrackingOutput:
        video, detections = inputs.video, inputs.detections
        if detections.camera_id != video.camera_id or len(detections.frame_offsets) != video.num_frames + 1:
            raise ValueError("Tracking input camera/timeline mismatch")
        tracker = AllPersonAssociator()
        history = []
        for packet in OpenCVVideoFrameReader(video.path, max_frames=video.num_frames):
            start, end = detections.frame_offsets[packet.index:packet.index + 2]
            detection = PersonDetectionResult(detections.boxes_xyxy[start:end], detections.confidence[start:end])
            ids, rows = tracker.update(detection, packet.frame)
            history.append((ids, detection.boxes_xyxy[rows]))
        if len(history) != video.num_frames:
            raise ValueError("Tracking did not decode the complete source timeline")
        ids = np.unique(np.concatenate([item[0] for item in history]))
        boxes: NDArray[np.float32] = np.zeros((len(ids), video.num_frames, 4), np.float32)
        observed: NDArray[np.bool_] = np.zeros((len(ids), video.num_frames), bool)
        for frame, (frame_ids, frame_boxes) in enumerate(history):
            rows = np.asarray(np.searchsorted(ids, frame_ids), dtype=np.int64)
            boxes[rows, frame] = frame_boxes
            observed[rows, frame] = True
        return PersonTrackingOutput(video.camera_id, ids, boxes, observed,
                                    tuple((int(i),) for i in ids), ())
