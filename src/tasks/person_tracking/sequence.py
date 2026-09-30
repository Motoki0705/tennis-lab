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

from src.tasks.person_tracking.botsort_pose import BotSortPoseConfig
from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.deep_ocsort_pose import DeepOCSortPoseConfig
from src.tasks.person_tracking.features import FeatureConfig
from src.tasks.person_tracking.methods import build_tracker
from src.tasks.person_tracking.strongsort import StrongSortConfig
from src.tasks.person_tracking.strongsort_offline import (
    AFLink,
    ReconstructedTracks,
    gaussian_interpolation,
)

CLIP_ENCODER = 'clipreid_vitb16_market1501'
METHODS = ('strongsort_pp_pose', 'strongsort_pp', 'deep_ocsort_pose',
           'deep_ocsort_pose_aflink_gsi', 'botsort_pose', 'all_person_botsort')


@dataclass(frozen=True)
class TrackingConfig:
    method: str = 'strongsort_pp_pose'
    encoder: str = CLIP_ENCODER
    features: FeatureConfig = FeatureConfig()

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f'Unknown tracking method {self.method!r}; available: {METHODS}')
        if self.encoder != CLIP_ENCODER:
            raise ValueError('Production tracking profile requires explicit CLIP-ReID; other encoders use comparison adapters')

    @property
    def offline(self) -> bool:
        return self.method in ('strongsort_pp_pose', 'strongsort_pp', 'deep_ocsort_pose_aflink_gsi')

    @property
    def online(self) -> str:
        if self.method.startswith('strongsort'):
            return 'strongsort'
        return 'deep_ocsort_pose' if self.method.startswith('deep_ocsort') else self.method

    def online_config(self) -> StrongSortConfig | DeepOCSortPoseConfig | BotSortPoseConfig:
        if self.online == 'strongsort':
            return StrongSortConfig(pose_weight=.15 if self.method == 'strongsort_pp_pose' else 0.)
        if self.online == 'deep_ocsort_pose':
            return DeepOCSortPoseConfig()
        if self.online == 'botsort_pose':
            return BotSortPoseConfig()
        raise ValueError('Motion-only baseline consumes decoded frames, not detection features')

    def identity(self) -> dict[str, Any]:
        return {**asdict(self), 'online_config': asdict(self.online_config())
                if self.method != 'all_person_botsort' else None,
                'offline_profile': 'i964_run9_aflink_gsi' if self.offline else None}


@dataclass(frozen=True)
class TrackEvidence:
    detection_rows: np.ndarray
    poses: np.ndarray
    embeddings: np.ndarray
    appearance_valid: np.ndarray
    encoder: str

    def __post_init__(self) -> None:
        shape = self.detection_rows.shape
        if len(shape) != 2 or self.detection_rows.dtype != np.int64 or (self.detection_rows < -1).any() \
                or self.poses.shape != (*shape, 17, 3) or self.poses.dtype != np.float32 \
                or self.embeddings.ndim != 3 or self.embeddings.shape[:2] != shape or self.embeddings.dtype != np.float32 \
                or self.appearance_valid.shape != shape or self.appearance_valid.dtype != np.bool_ \
                or not self.encoder or not np.isfinite(self.poses).all() or not np.isfinite(self.embeddings).all():
            raise ValueError('Invalid tracked detection evidence')
        absent = self.detection_rows < 0
        if self.appearance_valid[absent].any() or (self.poses[absent] != 0).any() \
                or (self.embeddings[~self.appearance_valid] != 0).any() \
                or not np.allclose(np.linalg.norm(self.embeddings[self.appearance_valid], axis=1), 1., atol=1e-4):
            raise ValueError('Missing observations/features must remain explicitly masked')

    def regroup(self, origins: np.ndarray) -> TrackEvidence:
        """Project raw track rows selected for each group/frame; never fill gaps."""
        if origins.ndim != 2 or origins.shape[1] != self.detection_rows.shape[1] \
                or origins.dtype != np.int64 or (origins < -1).any() or (origins >= len(self.detection_rows)).any():
            raise ValueError('Invalid group source track rows')
        rows = np.full(origins.shape, -1, np.int64)
        poses = np.zeros((*origins.shape, 17, 3), np.float32)
        embeddings = np.zeros((*origins.shape, self.embeddings.shape[-1]), np.float32)
        valid = np.zeros(origins.shape, bool)
        g, f = np.nonzero(origins >= 0)
        source = origins[g, f]
        if (self.detection_rows[source, f] < 0).any():
            raise ValueError('Group mapping refers to a synthetic/missing observation')
        rows[g, f], poses[g, f] = self.detection_rows[source, f], self.poses[source, f]
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
    if config.offline != (aflink is not None):
        raise ValueError('AFLink must be supplied exactly for an AFLink/GSI profile')
    tracker = build_tracker(config.online, fps=fps, config=config.online_config())
    features, assignments = [], []
    seen: set[int] = set()
    dimension = None
    for index, frame in enumerate(frames):
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
    poses: np.ndarray = np.zeros((*shape, 17, 3), np.float32)
    embeddings: np.ndarray = np.zeros((*shape, dimension), np.float32)
    valid: np.ndarray = np.zeros(shape, bool)
    for frame, result in zip(features, assignments, strict=True):
        lookup = {int(row): i for i, row in enumerate(frame.rows)}
        for detection, identity in zip(result.detection_rows, result.track_ids, strict=True):
            if int(detection) not in lookup:
                raise ValueError('Tracker emitted an absent source detection row')
            i, p, f = lookup[int(detection)], int(np.searchsorted(ids, identity)), frame.frame
            boxes[p, f], origins[p, f], poses[p, f] = frame.boxes[i], detection, frame.poses[i]
            embeddings[p, f], valid[p, f] = frame.embeddings[i], frame.appearance_valid[i]
    evidence = TrackEvidence(origins, poses, embeddings, valid, config.encoder)
    source_ids: tuple[tuple[int, ...], ...] = tuple((int(i),) for i in ids)
    output = TrackingSequence(ids, boxes, evidence, source_ids, None, ())
    if aflink is None:
        return output
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
