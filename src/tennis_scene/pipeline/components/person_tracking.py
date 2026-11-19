"""Associate stored detections without invoking a person detector."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.sequence import (
    TrackEvidence,
    TrackingConfig,
    track_sequence,
)
from src.tasks.person_tracking.strongsort_offline import AFLink, ReconstructedTracks
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_features import (
    APPEARANCE_FEATURES_PORT,
    PersonAppearanceFeaturesOutput,
    PersonPoseFeaturesOutput,
    appearance_feature_settings,
    pose_feature_settings,
    require_alignment,
    require_feature_identity,
)
from src.tennis_scene.pipeline.components.tracking_identity import TrackletLink
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort, SourceVideo
from src.tennis_scene.pipeline.inference_runtime import ModelSpec


class _FrameProgress:
    """Report completed online frames without emitting per-frame redirected logs."""

    def __init__(self, camera_id: str, total: int, *, enabled: bool) -> None:
        self.label = f"person_tracking/{camera_id}"
        self.total = total
        self.count = 0
        self.enabled = enabled
        self.stream = sys.stderr
        self.bar: Any = None
        self.last_reported = 0
        self.last_update = 0.0
        if enabled and self.stream.isatty():
            self.bar = tqdm(total=total, desc=self.label, unit="frame", file=self.stream)
        elif enabled:
            self._write()
            self.last_update = monotonic()

    def _write(self) -> None:
        print(f"{self.label}: {self.count}/{self.total} frames", file=self.stream, flush=True)
        self.last_reported = self.count

    def advance(self) -> None:
        self.count += 1
        if self.bar is not None:
            self.bar.update(1)
        elif self.enabled:
            now = monotonic()
            if self.count == self.total or now - self.last_update >= 30:
                self._write()
                self.last_update = now

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
        elif self.enabled and self.count != self.last_reported:
            self._write()


@dataclass(frozen=True)
class PersonTrackingInput:
    video: SourceVideo
    detections: PersonDetectionOutput
    poses: PersonPoseFeaturesOutput
    appearance: PersonAppearanceFeaturesOutput


@dataclass(frozen=True)
class PersonTrackingOutput:
    camera_id: str
    track_ids: NDArray[np.int64]
    boxes_xyxy: NDArray[np.float32]  # P,T,4; interpolation is explicitly masked
    observed: NDArray[np.bool_]
    source_track_ids: tuple[tuple[int, ...], ...]
    tracklet_links: tuple[TrackletLink, ...]
    evidence: TrackEvidence | None = None  # None for disabled or imported tracks
    reconstruction: ReconstructedTracks | None = None
    offline_link_candidates: tuple[dict[str, Any], ...] = ()

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
        if self.evidence is not None and not np.array_equal(self.evidence.detection_rows >= 0, self.observed):
            raise ValueError('Detection evidence must identify exactly the real observations')
        if self.reconstruction is not None:
            r = self.reconstruction
            if r.boxes.shape != self.boxes_xyxy.shape or r.interpolated.shape != self.observed.shape \
                    or r.interpolated.dtype != np.bool_ or not np.isfinite(r.boxes).all() \
                    or not np.array_equal(r.observed, self.observed) or (r.interpolated & self.observed).any():
                raise ValueError('GSI synthetic boxes must be separate from real observations')


class PersonTrackingModule:
    """CPU tracking on stored features; no media decoding or inference models."""

    io = ComponentIO("person_tracking", PersonTrackingInput, PersonTrackingOutput,
        {"detections": InputPort("person_detections", 2), "poses": InputPort("person_pose_features", 2),
         "appearance": APPEARANCE_FEATURES_PORT}, "person_tracks", version=7)

    def __init__(self, config: TrackingConfig, *, aflink_checkpoint: Path | None = None,
                 expected_pose: ModelSpec[Any] | None = None, expected_encoder: ModelSpec[Any] | None = None,
                 pose_preprocessing: Mapping[str, Any] | None = None, appearance_preprocessing: Mapping[str, Any] | None = None,
                 enabled: bool = True, show_progress: bool = True) -> None:
        self.config = config
        self.expected_pose, self.expected_encoder = expected_pose, expected_encoder
        self.pose_preprocessing = dict(pose_feature_settings(config.features) if pose_preprocessing is None else pose_preprocessing)
        self.appearance_preprocessing = dict(appearance_feature_settings(config.features) if appearance_preprocessing is None else appearance_preprocessing)
        self.aflink_checkpoint, self.enabled = aflink_checkpoint, enabled
        if type(show_progress) is not bool:
            raise TypeError('show_progress must be a bool')
        self.show_progress = show_progress
        if enabled and aflink_checkpoint is None:
            raise ValueError('AFLink/GSI tracking requires an explicit checkpoint')
        if enabled and (expected_pose is None or expected_encoder is None):
            raise ValueError('CPU tracking requires explicit expected pose and appearance declarations')

    def process(self, inputs: PersonTrackingInput) -> PersonTrackingOutput:
        video, detections = inputs.video, inputs.detections
        rows = require_alignment(video, detections, inputs.poses, inputs.appearance)
        if inputs.appearance.encoder != self.config.encoder:
            raise ValueError('Tracking encoder differs from the configured CLIP profile')
        if not self.enabled:
            return PersonTrackingOutput(video.camera_id, np.empty(0, np.int64),
                np.zeros((0, video.num_frames, 4), np.float32), np.zeros((0, video.num_frames), bool), (), ())
        require_feature_identity(inputs.poses.model_identity, inputs.poses.preprocessing_identity,
                                 self.expected_pose, self.pose_preprocessing)
        require_feature_identity(inputs.appearance.encoder_identity, inputs.appearance.preprocessing_identity,
                                 self.expected_encoder, self.appearance_preprocessing)
        assert self.aflink_checkpoint is not None
        af = AFLink(self.aflink_checkpoint)
        progress = _FrameProgress(video.camera_id, video.num_frames, enabled=self.show_progress)

        def frames() -> Iterator[DetectionFeatures]:
            for frame in range(video.num_frames):
                start, end = detections.frame_offsets[frame:frame + 2]
                yield DetectionFeatures(frame, rows[start:end], detections.boxes_xyxy[start:end],
                    detections.confidence[start:end], inputs.poses.keypoints[start:end],
                    inputs.appearance.embeddings[start:end], inputs.appearance.appearance_valid[start:end])
                progress.advance()

        try:
            result = track_sequence(frames(), fps=video.fps, config=self.config, aflink=af)
        finally:
            progress.close()
        return PersonTrackingOutput(video.camera_id, result.track_ids, result.boxes, result.observed,
            result.source_track_ids, (), replace(result.evidence, encoder_identity=inputs.appearance.encoder_identity),
            result.reconstruction, result.link_candidates)
