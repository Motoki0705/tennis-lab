"""Court-coordinate membership before pose and cross-camera identities.

The rule is defined once in person_tracking.court_linking. ``selected`` keeps
every original observation of selected fragments, including wide runs and
handoff duplicates. ``tracks`` is the one-box/group/frame adapter needed by
the v3 identity safety net; ``origin_rows`` records that reduction explicitly.
Its IDs are group IDs, not raw tracker IDs. Raw boxes remain in the declared
person_tracks dependency and are addressed by raw_track_ids + selected.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.person_tracking.court_linking import (
    LinkingConfig,
    select_linked_candidates,
)
from src.tasks.person_tracking.feature_tracks import evidence_appearance
from src.tasks.person_tracking.linked_timeline import linked_timeline
from src.tasks.player_association.appearance.encoders import AppearanceEncoder
from src.tasks.player_association.appearance.sampling import (
    CropSamplingConfig,
    embed_tracks,
)
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.geometry.footpoints import FootpointConfig
from src.tennis_scene.pipeline.components.base import release_inference_memory
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort, SourceVideo


@dataclass(frozen=True)
class PlayerSelectionInput:
    video: SourceVideo
    calibration: CourtCalibrationOutput
    tracks: PersonTrackingOutput


@dataclass(frozen=True)
class PlayerSelectionOutput:
    camera_id: str
    raw_track_ids: NDArray[np.int64]
    selected: NDArray[np.bool_]
    tracks: PersonTrackingOutput
    origin_rows: NDArray[np.int64]
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if self.camera_id != self.tracks.camera_id or self.raw_track_ids.dtype != np.int64 \
                or self.raw_track_ids.ndim != 1 or len(np.unique(self.raw_track_ids)) != len(self.raw_track_ids):
            raise ValueError('Selection requires aligned camera and unique raw track IDs')
        if self.selected.dtype != np.bool_ or self.selected.shape != (len(self.raw_track_ids), self.tracks.observed.shape[1]):
            raise ValueError('Selection mask must preserve the raw track/frame axes')
        if self.origin_rows.dtype != np.int64 or self.origin_rows.shape != self.tracks.observed.shape \
                or (self.origin_rows < -1).any() or (self.origin_rows >= len(self.raw_track_ids)).any() \
                or not np.array_equal(self.origin_rows >= 0, self.tracks.observed):
            raise ValueError('Group origins must match observed group boxes')
        groups, frames = np.nonzero(self.tracks.observed)
        if not self.selected[self.origin_rows[groups, frames], frames].all():
            raise ValueError('Group timeline contains an unselected observation')


class PlayerSelectionModule:
    io = ComponentIO('player_selection', PlayerSelectionInput, PlayerSelectionOutput,
        {'tracks': InputPort('person_tracks', 5), 'calibration': InputPort('local_court_calibration')},
        'selected_player_tracks', version=2)

    def __init__(self, config: LinkingConfig, *, footpoints: FootpointConfig,
                 sampling: CropSamplingConfig, encoder: Callable[[], AppearanceEncoder] | None,
                 encoder_name: str | None, device: str, enabled: bool) -> None:
        if (encoder is None) != (encoder_name is None):
            raise ValueError('Selection encoder and name must be provided together')
        self.config, self.footpoints, self.sampling = config, footpoints, sampling
        self.encoder, self.encoder_name, self.device, self.enabled = encoder, encoder_name, device, enabled

    def process(self, inputs: PlayerSelectionInput) -> PlayerSelectionOutput:
        video, raw = inputs.video, inputs.tracks
        if raw.camera_id != video.camera_id or raw.observed.shape[1] != video.num_frames:
            raise ValueError('Selection input camera/timeline mismatch')
        cameras = {v.camera.camera_id: v.camera for v in inputs.calibration.calibration.views}
        if not self.enabled or video.camera_id not in cameras:
            empty = PersonTrackingOutput(video.camera_id, np.empty(0, np.int64),
                np.zeros((0, video.num_frames, 4), np.float32), np.zeros((0, video.num_frames), bool), (), (),
                None if raw.evidence is None else raw.evidence.regroup(np.full((0, video.num_frames), -1, np.int64)))
            return PlayerSelectionOutput(video.camera_id, raw.track_ids, np.zeros_like(raw.observed), empty,
                np.full((0, video.num_frames), -1, np.int64),
                {'reason': 'disabled' if not self.enabled else 'camera_not_calibrated'})
        scale = math.hypot(video.width, video.height) / math.hypot(1920, 1080)
        sampling = replace(self.sampling, min_height_px=self.sampling.min_height_px * scale,
                           border_px=self.sampling.border_px * scale)
        footpoints = replace(self.footpoints, bottom_border_px=self.footpoints.bottom_border_px * scale)
        appearances = None
        appearance_record: dict[str, Any] = {'enabled': self.encoder is not None}
        if raw.evidence is not None and self.encoder_name == raw.evidence.encoder:
            appearances = evidence_appearance(raw.boxes_xyxy, raw.observed, raw.evidence,
                                              (video.width, video.height), sampling)
            appearance_record.update(encoder=raw.evidence.encoder, sampling=asdict(sampling),
                                     source='tracked_detection_features')
        elif self.encoder is not None and raw.observed.any():
            encoder = self.encoder()
            try:
                if encoder.name != self.encoder_name:
                    raise ValueError('Selection encoder differs from configured encoder')
                appearances, samples = embed_tracks(video.path, raw.boxes_xyxy, raw.observed,
                    (video.width, video.height), encoder, sampling)
                appearance_record.update(encoder=encoder.name, sampling=asdict(sampling),
                    samples=[{'frames': s.frames.tolist(), 'rejected': s.rejected} for s in samples])
            finally:
                del encoder
                release_inference_memory(self.device)
        camera_tracks = CameraTracks(cameras[video.camera_id], (video.width, video.height),
            raw.track_ids, raw.boxes_xyxy, raw.observed, appearances)
        selected, diagnostic = select_linked_candidates(camera_tracks, video.fps, self.config, footpoints)
        grouped, origins = linked_timeline(camera_tracks, diagnostic)
        tracks = PersonTrackingOutput(video.camera_id, grouped.track_ids, grouped.boxes_xyxy, grouped.observed,
            tuple((int(i),) for i in grouped.track_ids), (),
            None if raw.evidence is None else raw.evidence.regroup(origins))
        return PlayerSelectionOutput(video.camera_id, raw.track_ids, selected, tracks, origins,
            {'rule': asdict(self.config), 'appearance': appearance_record, 'linking': diagnostic})
