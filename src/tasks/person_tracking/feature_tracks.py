"""Exact detection-row scatter and sampled appearance for tracker comparisons."""

import numpy as np

from src.tasks.person_tracking.contracts import DetectionFeatures, TrackAssignments
from src.tasks.player_association.appearance.parts import NativeParts
from src.tasks.player_association.appearance.sampling import (
    CropSamplingConfig,
    TrackAppearance,
    sample_tracks,
)
from src.tasks.player_association.association.associate import CameraTracks
from src.utils.geometry.triangulation import PinholeCamera


def scatter_tracks(camera: PinholeCamera, frames: list[DetectionFeatures], assignments: list[TrackAssignments],
                   image_size: tuple[int, int]) -> tuple[CameraTracks, np.ndarray]:
    if len(frames) != len(assignments) or not frames:
        raise ValueError('Tracking/feature timelines must agree and be nonempty')
    ids = np.unique(np.concatenate([a.track_ids for a in assignments]))
    shape = (len(ids), len(frames))
    boxes: np.ndarray = np.zeros((*shape, 4), np.float32)
    origins: np.ndarray = np.full(shape, -1, np.int64)
    for f, (features, result) in enumerate(zip(frames, assignments, strict=True)):
        if features.frame != f or result.frame != f:
            raise ValueError('Tracking frame order differs from source features')
        lookup = {int(row): i for i, row in enumerate(features.rows)}
        for row, identity in zip(result.detection_rows, result.track_ids, strict=True):
            if row not in lookup:
                raise ValueError('Tracker emitted a detection row absent from this frame')
            track = int(np.searchsorted(ids, identity))
            boxes[track, f] = features.boxes[lookup[row]]
            origins[track, f] = row
    return CameraTracks(camera, image_size, ids, boxes, origins >= 0), origins


def sampled_appearance(tracks: CameraTracks, origins: np.ndarray, frames: list[DetectionFeatures]) -> list[TrackAppearance]:
    if origins.shape != tracks.observed.shape or not np.array_equal(origins >= 0, tracks.observed) \
            or len(frames) != tracks.observed.shape[1]:
        raise ValueError('Feature origins must identify every real observation')
    samples = sample_tracks(tracks.boxes_xyxy, tracks.observed, tracks.image_size, CropSamplingConfig())
    result = []
    for row, sample in enumerate(samples):
        vectors, indices = [], []
        parts = []
        for f in sample.frames:
            feature = frames[f]
            found = np.flatnonzero(feature.rows == origins[row, f])
            if len(found) != 1 or not np.array_equal(feature.boxes[found[0]], tracks.boxes_xyxy[row, f]):
                raise ValueError('Sample does not refer to the exact source detection box')
            index = found[0]
            if feature.parts is not None:
                indices.append(f)
                parts.append(feature.parts.take(np.asarray([index], np.int64)))
            elif feature.appearance_valid[index]:
                indices.append(f)
                vectors.append(feature.embeddings[index])
        native = None
        if frames[0].parts is not None:
            native = NativeParts(np.concatenate([p.embeddings for p in parts]) if parts else frames[0].parts.embeddings[:0],
                                 np.concatenate([p.visible for p in parts]) if parts else frames[0].parts.visible[:0])
        result.append(TrackAppearance(np.asarray(indices, np.int64),
                                     np.stack(vectors) if vectors else np.empty((len(indices), 0), np.float32), native))
    return result
