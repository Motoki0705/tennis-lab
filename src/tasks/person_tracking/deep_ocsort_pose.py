"""Deep OC-SORT feature adapter, with shared pose cost (see vendor NOTICE).

Uses upstream observation-centric re-update, direction consistency, confidence
adaptive EMA and ambiguity-adaptive appearance weights. Fixed-camera CMC/grid
are off. Both association rounds receive the same confidence-masked pose cost.
Only source detection rows are emitted; internal virtual observations stay private.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from src.tasks.person_tracking.contracts import DetectionFeatures, TrackAssignments
from src.tasks.person_tracking.deep_ocsort_vendor.affinity import (
    compute_aw_new_metric,
    iou_batch,
    speed_direction_batch,
)
from src.tasks.person_tracking.deep_ocsort_vendor.track import (
    KalmanBoxTracker,
    k_previous_obs,
)
from src.tasks.person_tracking.pose_distance import local_pose, pose_distance


@dataclass(frozen=True)
class DeepOCSortPoseConfig:
    detection_threshold: float = .3
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = .3
    delta_t: int = 3
    inertia: float = .2
    appearance_weight: float = .75
    alpha: float = .95
    adaptive_weight: float = .5
    pose_weight: float = .15

    def __post_init__(self) -> None:
        values = (self.detection_threshold, self.iou_threshold, self.inertia, self.appearance_weight,
                  self.alpha, self.adaptive_weight, self.pose_weight)
        if not np.isfinite(values).all() or not all(0 <= x <= 1 for x in values) \
                or self.detection_threshold >= 1 or min(self.max_age, self.min_hits, self.delta_t) < 1:
            raise ValueError('Invalid Deep OC-SORT configuration')


@dataclass
class _Track:
    identity: int
    state: Any  # isolated upstream Kalman state
    pose: NDArray[np.float32]
    has_appearance: bool
    detection_row: int


DEFAULT_CONFIG = DeepOCSortPoseConfig()


class DeepOCSortPose:
    def __init__(self, fps: float, config: DeepOCSortPoseConfig = DEFAULT_CONFIG) -> None:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Tracking requires positive finite source FPS')
        self.config = config
        self.tracks: list[_Track] = []
        self.frame = -1
        self.next_id = 1
        self.dimension: int | None = None
        self.seen_rows: set[int] = set()

    def _pose_cost(self, features: DetectionFeatures, rows: NDArray[np.int64], tracks: list[_Track]) -> NDArray[np.float64]:
        result: NDArray[np.float64] = np.zeros((len(rows), len(tracks)), np.float64)
        for i, row in enumerate(rows):
            pose = local_pose(features.boxes[row], features.poses[row])
            for j, track in enumerate(tracks):
                distance = pose_distance(pose, track.pose)
                if distance is not None:
                    result[i, j] = self.config.pose_weight * distance
        return result

    def _match(self, features: DetectionFeatures, rows: NDArray[np.int64], tracks: list[_Track],
               boxes: NDArray[np.float64], *, first: bool) -> list[tuple[int, _Track]]:
        if not len(rows) or not tracks:
            return []
        cfg = self.config
        detections = np.column_stack((features.boxes[rows], features.scores[rows]))
        iou = iou_batch(detections, boxes)
        similarity = iou - self._pose_cost(features, rows, tracks)
        if first:
            previous = np.asarray([k_previous_obs(t.state.observations, t.state.age, cfg.delta_t) for t in tracks])
            velocities = np.asarray([t.state.velocity if t.state.velocity is not None else [0., 0.] for t in tracks])
            dy, dx = speed_direction_batch(detections, previous)
            angle = np.arccos(np.clip(velocities[:, 0, None] * dy + velocities[:, 1, None] * dx, -1, 1))
            direction = ((np.pi / 2 - np.abs(angle)) / np.pi) * (previous[:, 4, None] >= 0)
            similarity += direction.T * cfg.inertia * features.scores[rows, None]
            embedded = np.asarray([t.state.emb for t in tracks])
            cosine = features.embeddings[rows] @ embedded.T
            present = features.appearance_valid[rows, None] & np.asarray([t.has_appearance for t in tracks])[None]
            cosine = np.where(present, cosine, 0.)
            similarity += cosine * compute_aw_new_metric(cosine, cfg.appearance_weight, cfg.adaptive_weight)
        # Upstream accepts an assignment only if its IoU also passes. Pose does
        # not turn a spatially impossible pair into a match.
        di, ti = linear_sum_assignment(-similarity)
        return [(int(rows[d]), tracks[t]) for d, t in zip(di, ti, strict=True) if iou[d, t] >= cfg.iou_threshold]

    def update(self, features: DetectionFeatures) -> TrackAssignments:
        if features.frame != self.frame + 1:
            raise ValueError('Tracking requires every frame once, starting at zero')
        if self.dimension is not None and self.dimension != features.embeddings.shape[1]:
            raise ValueError('Appearance dimension changed within a camera')
        if self.seen_rows.intersection(features.rows.tolist()):
            raise ValueError('Detection row reused across frames')
        self.frame = features.frame
        self.dimension = features.embeddings.shape[1]
        self.seen_rows.update(features.rows.tolist())
        predicted = np.asarray([t.state.predict()[0] for t in self.tracks], np.float64).reshape(-1, 4)
        if not np.isfinite(predicted).all():
            raise ValueError('Deep OC-SORT predicted a nonfinite box; no silent track deletion')
        rows = np.flatnonzero(features.scores >= self.config.detection_threshold)
        matched = self._match(features, rows, self.tracks, predicted, first=True)
        seen = {r for r, _ in matched}
        used = {t.identity for _, t in matched}
        remaining = [t for t in self.tracks if t.identity not in used]
        remaining_rows = np.asarray([r for r in rows if r not in seen], np.int64)
        last = np.asarray([t.state.last_observation for t in remaining], np.float64).reshape(-1, 5)
        matched += self._match(features, remaining_rows, remaining, last, first=False)
        seen = {r for r, _ in matched}
        used = {t.identity for _, t in matched}
        for row, track in matched:
            observation = np.r_[features.boxes[row], features.scores[row]]
            track.state.update(observation)
            if features.appearance_valid[row]:
                trust = (float(features.scores[row]) - self.config.detection_threshold) / (1 - self.config.detection_threshold)
                alpha = self.config.alpha + (1 - self.config.alpha) * (1 - trust)
                if track.has_appearance:
                    track.state.update_emb(features.embeddings[row], alpha=alpha)
                else:
                    track.state.emb = features.embeddings[row].copy()
                track.has_appearance = True
            track.pose = local_pose(features.boxes[row], features.poses[row])
            track.detection_row = int(features.rows[row])
        for track in self.tracks:
            if track.identity not in used:
                track.state.update(None)
        for row in rows:
            if row in seen:
                continue
            state = KalmanBoxTracker(np.r_[features.boxes[row], features.scores[row]], delta_t=self.config.delta_t,
                                     emb=features.embeddings[row].copy(), new_kf=False)
            self.tracks.append(_Track(self.next_id, state, local_pose(features.boxes[row], features.poses[row]),
                                      bool(features.appearance_valid[row]), int(features.rows[row])))
            self.next_id += 1
        emitted = sorted((t for t in self.tracks if t.state.time_since_update < 1
                          and (t.state.hit_streak >= self.config.min_hits or self.frame + 1 <= self.config.min_hits)),
                         key=lambda t: t.detection_row)
        self.tracks = [t for t in self.tracks if t.state.time_since_update <= self.config.max_age]
        return TrackAssignments(self.frame, np.asarray([t.detection_row for t in emitted], np.int64),
                                np.asarray([t.identity for t in emitted], np.int64))
