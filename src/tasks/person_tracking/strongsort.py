"""Paper-based StrongSORT inference with explicit precomputed appearance.

No upstream GPL implementation is vendored. See strongsort_NOTICE.md for the
paper, numerical defaults, deliberate fixed-camera/common-detection adaptations,
and the separate AFLink weight terms. Offline AFLink/GSI live in their own module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from src.tasks.person_tracking.contracts import DetectionFeatures, TrackAssignments
from src.tasks.person_tracking.pose_distance import local_pose, pose_distance
from src.tasks.player_association.appearance.parts import (
    NativeParts,
    part_distance,
    update_parts,
)
from src.utils.geometry.bbox import pairwise_iou


@dataclass(frozen=True)
class StrongSortConfig:
    max_age: int = 30
    n_init: int = 3
    max_cost: float = .45
    max_iou_distance: float = .7
    motion_gate: float = 9.4877
    appearance_weight: float = .98
    ema_alpha: float = .9
    pose_weight: float = 0.

    def __post_init__(self) -> None:
        if not np.isfinite(self.pose_weight) or not 0 <= self.pose_weight <= 1:
            raise ValueError('StrongSORT pose weight must be finite and in [0, 1]')


class InvalidPrediction(RuntimeError):
    """A predicted state cannot describe a box; caller must record a failed camera."""


def measurement(box: NDArray[np.float32]) -> NDArray[np.float64]:
    width, height = box[2:] - box[:2]
    return np.asarray([*(box[:2] + box[2:]) / 2, width / height, height], np.float64)


@dataclass
class MotionState:
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]

    @classmethod
    def initiate(cls, box: NDArray[np.float32]) -> MotionState:
        value = measurement(box)
        h = value[3]
        std = np.asarray([h / 10, h / 10, .01, h / 10, h / 16, h / 16, .00001, h / 16])
        return cls(np.r_[value, np.zeros(4)], np.diag(std ** 2))

    def predict(self) -> None:
        transition = np.eye(8)
        transition[:4, 4:] = np.eye(4)
        h = self.mean[3]
        std = np.asarray([h / 20, h / 20, .01, h / 20, h / 160, h / 160, .00001, h / 160])
        self.mean = transition @ self.mean
        self.covariance = transition @ self.covariance @ transition.T + np.diag(std ** 2)
        # A missed track may extrapolate through zero extent before max_age.
        # This latent state is not an emitted box; reference Kalman prediction
        # also retains it. Only actual detections ever reach TrackAssignments.
        if not np.isfinite(self.mean).all() or not np.isfinite(self.covariance).all():
            raise InvalidPrediction(f'StrongSORT predicted an invalid box: xyah={self.mean[:4].tolist()}')

    def project(self, confidence: float = 0.) -> NDArray[np.float64]:
        h = self.mean[3]
        std = np.asarray([h / 20, h / 20, .1, h / 20])
        return self.covariance[:4, :4] + (1 - confidence) * np.diag(std ** 2)

    def update(self, box: NDArray[np.float32], confidence: float) -> None:
        covariance = self.project(confidence)
        gain = np.linalg.solve(covariance, self.covariance[:4]).T
        self.mean += gain @ (measurement(box) - self.mean[:4])
        self.covariance -= gain @ covariance @ gain.T

    def distances(self, boxes: NDArray[np.float32]) -> NDArray[np.float64]:
        delta = np.stack([measurement(box) for box in boxes]) - self.mean[:4]
        result: NDArray[np.float64] = np.einsum('ni,in->n', delta, np.linalg.solve(self.project(), delta.T))
        return result

    def box(self) -> NDArray[np.float64]:
        size = np.asarray([self.mean[2] * self.mean[3], self.mean[3]])
        result: NDArray[np.float64] = np.r_[self.mean[:2] - size / 2, self.mean[:2] + size / 2]
        return result


@dataclass
class _Track:
    identity: int
    motion: MotionState
    embedding: NDArray[np.float32]
    valid: bool
    parts: NativeParts | None
    pose: NDArray[np.float32] | None
    hits: int = 1
    age: int = 0
    confirmed: bool = False
    detection_row: int = -1


DEFAULT_CONFIG = StrongSortConfig()


class StrongSort:
    def __init__(self, config: StrongSortConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.tracks: list[_Track] = []
        self.frame = -1
        self.next_id = 1
        self.seen_rows: set[int] = set()
        self.feature_shape: tuple[int, ...] | None = None

    def _appearance(self, features: DetectionFeatures, tracks: list[_Track]) -> NDArray[np.float64]:
        if features.parts is not None:
            stored = [t.parts for t in tracks]
            if any(p is None for p in stored):
                raise ValueError('StrongSORT native appearance state is missing')
            parts = NativeParts(np.concatenate([p.embeddings for p in stored if p is not None]),
                                np.concatenate([p.visible for p in stored if p is not None]))
            distance, valid = part_distance(parts, features.parts)
        else:
            distance = (1 - np.asarray([t.embedding for t in tracks]) @ features.embeddings.T).astype(np.float64)
            valid = np.asarray([t.valid for t in tracks])[:, None] & features.appearance_valid[None]
        # No appearance evidence in the appearance stage; IoU stage is explicit.
        return np.where(valid, distance, np.inf)

    def _pose_cost(self, features: DetectionFeatures, rows: NDArray[np.int64], tracks: list[_Track]) -> NDArray[np.float64]:
        """Same local, last-observed pose evidence as Deep OC-SORT; track x row."""
        result: NDArray[np.float64] = np.zeros((len(tracks), len(rows)), np.float64)
        if self.config.pose_weight == 0:
            return result
        poses = features.require_poses()
        for j, row in enumerate(rows):
            pose = local_pose(features.boxes[row], poses[row])
            for i, track in enumerate(tracks):
                assert track.pose is not None
                distance = pose_distance(pose, track.pose)
                if distance is not None:
                    result[i, j] = self.config.pose_weight * distance
        return result

    @staticmethod
    def _assign(cost: NDArray[np.float64], maximum: float) -> list[tuple[int, int]]:
        if not cost.size:
            return []
        left, right = linear_sum_assignment(np.minimum(cost, maximum + 1e-5))
        return [(int(a), int(b)) for a, b in zip(left, right, strict=True) if cost[a, b] <= maximum]

    def update(self, features: DetectionFeatures) -> TrackAssignments:
        poses = features.require_poses() if self.config.pose_weight else None
        shape = features.embeddings.shape[1:] if features.parts is None else features.parts.embeddings.shape[1:]
        if features.frame != self.frame + 1 or self.seen_rows.intersection(features.rows.tolist()):
            raise ValueError('StrongSORT requires unique rows and sequential complete frames')
        if self.feature_shape is not None and shape != self.feature_shape:
            raise ValueError('StrongSORT appearance representation changed')
        self.feature_shape = shape
        self.frame = features.frame
        self.seen_rows.update(features.rows.tolist())
        for track in self.tracks:
            try:
                track.motion.predict()
            except InvalidPrediction as error:
                raise InvalidPrediction(f'frame={self.frame} track={track.identity} age={track.age}: {error}') from error
            track.age += 1
        confirmed = [t for t in self.tracks if t.confirmed]
        matched: list[tuple[_Track, int]] = []
        if confirmed and len(features.rows):
            distance = np.stack([t.motion.distances(features.boxes) for t in confirmed])
            cost = self.config.appearance_weight * self._appearance(features, confirmed) + (1 - self.config.appearance_weight) * distance
            cost += self._pose_cost(features, np.arange(len(features.rows), dtype=np.int64), confirmed)
            cost[distance > self.config.motion_gate] = np.inf
            matched = [(confirmed[t], r) for t, r in self._assign(cost, self.config.max_cost)]
        used = {t.identity for t, _ in matched}
        seen = {r for _, r in matched}
        remaining = [t for t in self.tracks if t.identity not in used and (not t.confirmed or t.age == 1)]
        rows = np.asarray([r for r in range(len(features.rows)) if r not in seen], np.int64)
        if remaining and len(rows):
            boxes = np.asarray([t.motion.box() for t in remaining])
            cost = 1 - pairwise_iou(boxes, features.boxes[rows])
            cost += self._pose_cost(features, rows, remaining)
            matched.extend((remaining[t], int(rows[r])) for t, r in self._assign(cost, self.config.max_iou_distance))
        used = {t.identity for t, _ in matched}
        seen = {r for _, r in matched}
        for track, row in matched:
            track.motion.update(features.boxes[row], float(features.scores[row]))
            track.age = 0
            track.hits += 1
            track.confirmed = track.hits >= self.config.n_init
            track.detection_row = int(features.rows[row])
            track.pose = None if poses is None else local_pose(features.boxes[row], poses[row])
            if features.parts is not None:
                if track.parts is None:
                    raise ValueError('StrongSORT native appearance state is missing')
                track.parts = update_parts(track.parts, features.parts.take(np.asarray([row], np.int64)), self.config.ema_alpha)
            elif features.appearance_valid[row]:
                vector = features.embeddings[row].copy()
                if track.valid:
                    vector = self.config.ema_alpha * track.embedding + (1 - self.config.ema_alpha) * vector
                norm = np.linalg.norm(vector)
                if norm <= 1e-12:
                    raise ValueError('StrongSORT appearance EMA cancelled to zero')
                track.embedding, track.valid = vector / norm, True
        self.tracks = [t for t in self.tracks if t.identity in used or (t.confirmed and t.age <= self.config.max_age)]
        for row in range(len(features.rows)):
            if row not in seen:
                self.tracks.append(_Track(self.next_id, MotionState.initiate(features.boxes[row]),
                                          features.embeddings[row].copy(), bool(features.appearance_valid[row]),
                                          None if features.parts is None else features.parts.take(np.asarray([row], np.int64)),
                                          None if poses is None else local_pose(features.boxes[row], poses[row]),
                                          detection_row=int(features.rows[row])))
                self.next_id += 1
        emitted = sorted((t for t in self.tracks if t.confirmed and t.age == 0), key=lambda t: t.detection_row)
        return TrackAssignments(self.frame, np.asarray([t.detection_row for t in emitted], np.int64),
                                np.asarray([t.identity for t in emitted], np.int64))
