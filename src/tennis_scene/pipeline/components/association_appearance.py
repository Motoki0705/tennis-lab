"""Appearance for the existing association samples, with per-row reuse evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from itertools import islice
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.person_tracking.sequence import CLIP_ENCODER
from src.tasks.player_association.appearance.encoders import AppearanceEncoder
from src.tasks.player_association.appearance.sampling import (
    DEFAULT_BATCH_SIZE,
    CropSamplingConfig,
    TrackAppearance,
    crop,
    iter_sampled_frames,
    sample_tracks,
)
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_features import (
    PersonAppearanceFeaturesOutput,
    appearance_image_digest,
    matches_feature_identity,
    require_alignment,
)
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.contracts import SourceVideo
from src.tennis_scene.pipeline.inference_runtime import ModelSpec
from src.utils.video.reader import OpenCVDecodeIdentity


@dataclass(frozen=True)
class SavedAssociationAppearance:
    """Features and their exact detection dependency from the clip store."""

    detections: PersonDetectionOutput
    features: PersonAppearanceFeaturesOutput


def _reuse_rows(video: SourceVideo, tracks: PersonTrackingOutput, saved: SavedAssociationAppearance | None,
                encoder: ModelSpec[AppearanceEncoder], preprocessing: Mapping[str, Any] | None,
                name: str) -> tuple[NDArray[np.int64] | None, str]:
    if preprocessing is None:
        return None, 'reuse_disabled'
    if name != CLIP_ENCODER:
        return None, 'encoder_not_clip'
    if saved is None or tracks.evidence is None:
        return None, 'missing_detection_evidence'
    if not matches_feature_identity(saved.features.encoder_identity, saved.features.preprocessing_identity, encoder, preprocessing):
        return None, 'feature_identity_mismatch'
    try:
        return require_alignment(video, saved.detections, saved.features), ''
    except ValueError:
        return None, 'detection_alignment_mismatch'


def _sample_feature(saved: SavedAssociationAppearance, detection_rows: NDArray[np.int64], frame: int,
                    row: int, box: NDArray[np.float32]) -> tuple[NDArray[np.float32] | None, str]:
    start, end = saved.detections.frame_offsets[frame:frame + 2]
    index = start + np.searchsorted(detection_rows[start:end], row)
    if index >= end or detection_rows[index] != row:
        return None, 'detection_row_mismatch'
    if not np.array_equal(box, saved.detections.boxes_xyxy[index]):
        return None, 'box_mismatch'
    if not saved.features.appearance_valid[index]:
        return None, 'appearance_invalid'
    return saved.features.embeddings[index], ''


def camera_track_appearances(
    video: SourceVideo, tracks: PersonTrackingOutput, sampling: CropSamplingConfig, *,
    saved: SavedAssociationAppearance | None, preprocessing: Mapping[str, Any] | None,
    encoder: ModelSpec[AppearanceEncoder], name: str, borrow_encoder: Callable[[], AppearanceEncoder],
    expected_decoder: OpenCVDecodeIdentity | None = None,
) -> tuple[tuple[TrackAppearance, ...], list[dict[str, Any]], Counter[str]]:
    samples = sample_tracks(tracks.boxes_xyxy, tracks.observed, (video.width, video.height), sampling)
    embeddings: list[dict[int, NDArray[np.float32]]] = [{} for _ in samples]
    reused: list[list[int]] = [[] for _ in samples]
    recomputed: list[dict[str, list[int]]] = [{} for _ in samples]
    reasons: Counter[str] = Counter()
    detection_rows, camera_reason = _reuse_rows(video, tracks, saved, encoder, preprocessing, name)

    def missing_crops() -> Iterator[tuple[int, int, NDArray[np.float32]]]:
        for packet, track_rows in iter_sampled_frames(video.path, samples, expected_decoder=expected_decoder):
            frame, image = packet.index, packet.frame
            reason_for_frame = camera_reason
            if detection_rows is not None:
                assert saved is not None
                frame_sha256 = saved.features.frame_sha256[frame]
                if not frame_sha256.any():
                    reason_for_frame = 'image_evidence_missing'
                elif (image.shape != (video.height, video.width, 3) or image.dtype != np.uint8
                      or not np.array_equal(appearance_image_digest(image), frame_sha256)):
                    reason_for_frame = 'image_mismatch'
            for track in track_rows:
                box = tracks.boxes_xyxy[track, frame]
                value, reason = None, reason_for_frame
                if not reason:
                    assert detection_rows is not None and saved is not None and tracks.evidence is not None
                    value, reason = _sample_feature(saved, detection_rows, frame, int(tracks.evidence.detection_rows[track, frame]), box)
                if value is not None:
                    embeddings[track][frame] = value
                    reused[track].append(frame)
                else:
                    reasons[reason] += 1
                    recomputed[track].setdefault(reason, []).append(frame)
                    yield track, frame, crop(image, box, borrow_encoder().input_size)

    misses = missing_crops()
    batch_size = int(encoder.settings.get("batch_size", DEFAULT_BATCH_SIZE))
    while batch := list(islice(misses, batch_size)):
        pixels = torch.from_numpy(np.stack([patch for _, _, patch in batch]))
        encoded = borrow_encoder().embed(pixels).detach().cpu().numpy().astype(np.float32)
        for (track, frame, _), value in zip(batch, encoded, strict=True):
            embeddings[track][frame] = value

    appearances = tuple(TrackAppearance(sample.frames,
        np.stack([embeddings[i][frame] for frame in sample.frames]) if len(sample.frames) else np.zeros((0, 0), np.float32))
        for i, sample in enumerate(samples))
    diagnostics = [dict(track_id=int(track_id), samples=len(sample.frames), rejected=sample.rejected,
                        reused_frames=hit, recomputed_frames=miss)
                   for track_id, sample, hit, miss in zip(tracks.track_ids, samples, reused, recomputed, strict=True)]
    return appearances, diagnostics, reasons
