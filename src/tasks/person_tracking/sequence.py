"""Shared production/context tracking on exact detection features, without media I/O.

The named profiles freeze the run-10 settings. Consumers can supply features
from decoded video, a JPEG context store, or the immutable evaluation archive.
Court selection and its cap occur after this uncapped camera-local sequence.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.features import FeatureConfig
from src.tasks.person_tracking.strongsort import StrongSort, StrongSortConfig
from src.tasks.person_tracking.strongsort_offline import (
    AFLink,
    ReconstructedTracks,
    gaussian_interpolation,
)

CLIP_ENCODER = 'clipreid_vitb16_market1501'
ADOPTED_METHOD = 'strongsort_pp_pose'


@dataclass(frozen=True)
class TrackingConfig:
    method: str = ADOPTED_METHOD
    encoder: str = CLIP_ENCODER
    features: FeatureConfig = FeatureConfig()

    def __post_init__(self) -> None:
        if self.method != ADOPTED_METHOD:
            raise ValueError(f'Production tracking requires {ADOPTED_METHOD!r}; '
                             'other methods use tests/benchmarks/person_tracking_features.py')
        if self.encoder != CLIP_ENCODER:
            raise ValueError('Production tracking profile requires explicit CLIP-ReID; other encoders use comparison adapters')

    def online_config(self) -> StrongSortConfig:
        return StrongSortConfig(pose_weight=.15)

    @property
    def requires_pose(self) -> bool:
        return True

    def identity(self) -> dict[str, Any]:
        return {**asdict(self), 'online_config': asdict(self.online_config()),
                'offline_profile': 'i964_run9_aflink_gsi'}


@dataclass(frozen=True)
class DatasetTrackingConfig(TrackingConfig):
    """Dataset-only profile: identity review precedes all pose inference."""

    method: str = 'strongsort_pp_appearance'

    def __post_init__(self) -> None:
        if self.method != 'strongsort_pp_appearance' or self.encoder != CLIP_ENCODER:
            raise ValueError('Dataset tracking requires the pose-free CLIP-ReID profile')

    @property
    def requires_pose(self) -> bool:
        return False

    def online_config(self) -> StrongSortConfig:
        return StrongSortConfig(pose_weight=0.)


@dataclass(frozen=True)
class TrackEvidence:
    detection_rows: np.ndarray
    poses: np.ndarray | None
    embeddings: np.ndarray
    appearance_valid: np.ndarray
    encoder: str

    def __post_init__(self) -> None:
        shape = self.detection_rows.shape
        if len(shape) != 2 or self.detection_rows.dtype != np.int64 or (self.detection_rows < -1).any() \
                or (self.poses is not None and (self.poses.shape != (*shape, 17, 3) or self.poses.dtype != np.float32)) \
                or self.embeddings.ndim != 3 or self.embeddings.shape[:2] != shape or self.embeddings.dtype != np.float32 \
                or self.appearance_valid.shape != shape or self.appearance_valid.dtype != np.bool_ \
                or not self.encoder or (self.poses is not None and not np.isfinite(self.poses).all()) or not np.isfinite(self.embeddings).all():
            raise ValueError('Invalid tracked detection evidence')
        absent = self.detection_rows < 0
        if self.appearance_valid[absent].any() or (self.poses is not None and (self.poses[absent] != 0).any()) \
                or (self.embeddings[~self.appearance_valid] != 0).any() \
                or not np.allclose(np.linalg.norm(self.embeddings[self.appearance_valid], axis=1), 1., atol=1e-4):
            raise ValueError('Missing observations/features must remain explicitly masked')

    def require_poses(self) -> np.ndarray:
        if self.poses is None:
            raise ValueError('This consumer requires inferred track pose evidence')
        return self.poses

    def regroup(self, origins: np.ndarray) -> TrackEvidence:
        """Project raw track rows selected for each group/frame; never fill gaps."""
        if origins.ndim != 2 or origins.shape[1] != self.detection_rows.shape[1] \
                or origins.dtype != np.int64 or (origins < -1).any() or (origins >= len(self.detection_rows)).any():
            raise ValueError('Invalid group source track rows')
        rows = np.full(origins.shape, -1, np.int64)
        poses = None if self.poses is None else np.zeros((*origins.shape, 17, 3), np.float32)
        embeddings = np.zeros((*origins.shape, self.embeddings.shape[-1]), np.float32)
        valid = np.zeros(origins.shape, bool)
        g, f = np.nonzero(origins >= 0)
        source = origins[g, f]
        if (self.detection_rows[source, f] < 0).any():
            raise ValueError('Group mapping refers to a synthetic/missing observation')
        rows[g, f] = self.detection_rows[source, f]
        if poses is not None:
            poses[g, f] = self.require_poses()[source, f]
        embeddings[g, f], valid[g, f] = self.embeddings[source, f], self.appearance_valid[source, f]
        return TrackEvidence(rows, poses, embeddings, valid, self.encoder)


@dataclass(frozen=True)
class TrackingSequence:
    track_ids: np.ndarray
    boxes: np.ndarray
    evidence: TrackEvidence
    source_track_ids: tuple[tuple[int, ...], ...]
    reconstruction: ReconstructedTracks | None
    link_candidates: tuple[dict[str, Any], ...]

    @property
    def observed(self) -> np.ndarray:
        observed: np.ndarray = self.evidence.detection_rows >= 0
        return observed


def track_sequence(frames: Iterable[DetectionFeatures], *, fps: float,
                   config: TrackingConfig, aflink: AFLink | None) -> TrackingSequence:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Tracking requires positive finite fps')
    if aflink is None:
        raise ValueError('AFLink must be supplied for the adopted AFLink/GSI profile')
    tracker = StrongSort(config.online_config())
    features, assignments = [], []
    seen: set[int] = set()
    dimension = None
    for index, frame in enumerate(frames):
        if (frame.poses is not None) != config.requires_pose:
            raise ValueError('Feature pose presence differs from the explicit tracking profile')
        if frame.frame != index or seen.intersection(frame.rows.tolist()) or frame.parts is not None:
            raise ValueError('Production features need contiguous frames and globally unique detection rows, without native parts')
        seen.update(frame.rows.tolist())
        if dimension is not None and frame.embeddings.shape[1] != dimension:
            raise ValueError('Appearance dimension changed during tracking')
        dimension = frame.embeddings.shape[1]
        features.append(frame)
        assignments.append(tracker.update(frame))
    if not features:
        raise ValueError('Tracking requires a nonempty timeline, including empty detection frames')
    ids = np.unique(np.concatenate([a.track_ids for a in assignments]))
    shape = (len(ids), len(features))
    assert dimension is not None
    boxes: np.ndarray = np.zeros((*shape, 4), np.float32)
    origins: np.ndarray = np.full(shape, -1, np.int64)
    poses: np.ndarray | None = np.zeros((*shape, 17, 3), np.float32) if config.requires_pose else None
    embeddings: np.ndarray = np.zeros((*shape, dimension), np.float32)
    valid: np.ndarray = np.zeros(shape, bool)
    for frame, result in zip(features, assignments, strict=True):
        lookup = {int(row): i for i, row in enumerate(frame.rows)}
        for detection, identity in zip(result.detection_rows, result.track_ids, strict=True):
            if int(detection) not in lookup:
                raise ValueError('Tracker emitted an absent source detection row')
            i, p, f = lookup[int(detection)], int(np.searchsorted(ids, identity)), frame.frame
            boxes[p, f], origins[p, f] = frame.boxes[i], detection
            if poses is not None:
                poses[p, f] = frame.require_poses()[i]
            embeddings[p, f], valid[p, f] = frame.embeddings[i], frame.appearance_valid[i]
    evidence = TrackEvidence(origins, poses, embeddings, valid, config.encoder)
    source_ids: tuple[tuple[int, ...], ...] = tuple((int(i),) for i in ids)
    output = TrackingSequence(ids, boxes, evidence, source_ids, None, ())
    roots, candidates = aflink.links(boxes, output.observed)
    groups = sorted(set(roots.values()))
    merged: np.ndarray = np.zeros((len(groups), len(features), 4), np.float32)
    mapping = np.full(merged.shape[:2], -1, np.int64)
    for source, root in roots.items():
        destination, at = groups.index(root), output.observed[source]
        if (mapping[destination, at] >= 0).any():
            raise ValueError('AFLink created overlapping observations')
        merged[destination, at], mapping[destination, at] = boxes[source, at], source
    source_ids = tuple((int(ids[root]), *(int(ids[i]) for i in sorted(roots) if roots[i] == root and i != root)) for root in groups)
    evidence = evidence.regroup(mapping)
    return replace(output, track_ids=ids[groups], boxes=merged, evidence=evidence, source_track_ids=source_ids,
                   reconstruction=gaussian_interpolation(merged, evidence.detection_rows >= 0), link_candidates=tuple(candidates))
