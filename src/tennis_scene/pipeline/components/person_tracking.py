"""Associate stored detections without invoking a person detector."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.submodules.models import ViTPosePose2D
from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.features import FeatureExtractor, UnpromptedEncoder
from src.tasks.person_tracking.sequence import (
    TrackEvidence,
    TrackingConfig,
    track_sequence,
)
from src.tasks.person_tracking.strongsort_offline import AFLink, ReconstructedTracks
from src.tasks.player_association.appearance.encoders import AppearanceEncoder
from src.tennis_scene.pipeline.components.base import release_inference_memory
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.tracking_identity import TrackletLink
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort, SourceVideo
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig
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
    """Shared feature tracking; court selection alone owns the candidate cap."""

    io = ComponentIO("person_tracking", PersonTrackingInput, PersonTrackingOutput,
        {"detections": InputPort("person_detections", 2)}, "person_tracks", version=5)

    def __init__(self, config: TrackingConfig, *, people: PeopleModelConfig | None = None,
                 encoder: Callable[[], AppearanceEncoder] | None = None,
                 aflink_checkpoint: Path | None = None, enabled: bool = True) -> None:
        self.config, self.people, self.encoder = config, people, encoder
        self.aflink_checkpoint, self.enabled = aflink_checkpoint, enabled
        if enabled and (people is None or encoder is None):
            raise ValueError('Feature tracking requires explicit pose assets and CLIP encoder')
        if enabled and aflink_checkpoint is None:
            raise ValueError('AFLink/GSI tracking requires an explicit checkpoint')

    def process(self, inputs: PersonTrackingInput) -> PersonTrackingOutput:
        video, detections = inputs.video, inputs.detections
        if detections.camera_id != video.camera_id or len(detections.frame_offsets) != video.num_frames + 1:
            raise ValueError("Tracking input camera/timeline mismatch")
        if not self.enabled:
            return PersonTrackingOutput(video.camera_id, np.empty(0, np.int64),
                np.zeros((0, video.num_frames, 4), np.float32), np.zeros((0, video.num_frames), bool), (), ())
        return self._features(inputs)

    def _features(self, inputs: PersonTrackingInput) -> PersonTrackingOutput:
        video, detections = inputs.video, inputs.detections
        if detections.source_rows is None:
            raise ValueError('Feature tracking requires person_detections v2 source rows; regenerate the artifact')
        assert self.people is not None and self.encoder is not None and self.aflink_checkpoint is not None
        people = self.people
        af = AFLink(self.aflink_checkpoint)
        pose = ViTPosePose2D(people.vitpose_checkpoint, device=people.runtime.device,
            flip_test=people.runtime.vitpose.flip_test, batch_size=people.runtime.vitpose.batch_size,
            head_config=people.runtime.vitpose.head, precision='float32')
        encoder = None
        try:
            encoder = self.encoder()
            if encoder.name != self.config.encoder:
                raise ValueError('Tracking encoder differs from the configured CLIP profile')
            extractor = FeatureExtractor(pose, UnpromptedEncoder(encoder, dimension=1280), self.config.features)

            def frames() -> Iterator[DetectionFeatures]:
                assert detections.source_rows is not None
                count = 0
                for packet in OpenCVVideoFrameReader(video.path, max_frames=video.num_frames):
                    if packet.index != count:
                        raise ValueError('Tracking video frame order changed')
                    start, end = detections.frame_offsets[packet.index:packet.index + 2]
                    yield extractor.extract(packet.index, packet.frame, detections.source_rows[start:end],
                                            detections.boxes_xyxy[start:end], detections.confidence[start:end])
                    count += 1
                if count != video.num_frames:
                    raise ValueError('Tracking did not decode the complete source timeline')

            result = track_sequence(frames(), fps=video.fps, config=self.config, aflink=af)
        finally:
            pose.unload()
            del encoder
            release_inference_memory(people.runtime.device)
        return PersonTrackingOutput(video.camera_id, result.track_ids, result.boxes, result.observed,
            result.source_track_ids, (), result.evidence, result.reconstruction, result.link_candidates)
