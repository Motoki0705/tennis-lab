"""Fixed-camera BoT-SORT variant with explicit appearance and pose gates.

Uses Ultralytics' XYWH Kalman filter, high/low confidence association and EMA
appearance. Unlike vanilla BoT-SORT, valid appearance disagreement vetoes an
IoU match in BOTH stages. Pose is a confidence-masked, box-normalized joint
distance. This is a project variant, not a paper reproduction: no camera
motion compensation (fixed cameras), no IoU-only duplicate suppression, and
high-score births are emitted immediately. Thresholds await Meiji calibration.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYWH

from src.tasks.person_tracking.contracts import (
    DetectionFeatures,
    TrackAssignments,
    TrackCapacityExceeded,
)
from src.utils.geometry.bbox import pairwise_iou


@dataclass(frozen=True)
class BotSortPoseConfig:
    high_score: float = .3
    low_score: float = .1
    min_iou: float = .05
    max_cosine_distance: float = .25
    pose_confidence: float = .3
    min_pose_joints: int = 4
    pose_scale: float = .25
    appearance_weight: float = .35
    pose_weight: float = .15
    ema: float = .9
    max_gap_s: float = 1.
    max_tracks: int = 6

    def __post_init__(self) -> None:
        probabilities = (self.low_score, self.high_score, self.min_iou, self.max_cosine_distance,
                         self.pose_confidence, self.appearance_weight, self.pose_weight, self.ema)
        if not np.isfinite([*probabilities, self.pose_scale, self.max_gap_s]).all() \
                or not all(0 <= v <= 1 for v in probabilities) or not self.low_score < self.high_score \
                or self.appearance_weight + self.pose_weight >= 1 or self.pose_scale <= 0 \
                or self.max_gap_s <= 0 or not 1 <= self.min_pose_joints <= 17 or self.max_tracks < 1:
            raise ValueError("Invalid BoT-SORT appearance/pose configuration")


@dataclass
class _Track:
    identity: int
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]
    last_frame: int
    pose: NDArray[np.float32]
    embedding: NDArray[np.float32] | None

    @property
    def box(self) -> NDArray[np.float32]:
        centre, size = self.mean[:2], self.mean[2:4]
        return np.concatenate((centre - size / 2, centre + size / 2)).astype(np.float32)


def _xywh(box: NDArray[np.float32]) -> NDArray[np.float64]:
    return np.concatenate(((box[:2] + box[2:]) / 2, box[2:] - box[:2])).astype(np.float64)


def _local_pose(box: NDArray[np.float32], pose: NDArray[np.float32]) -> NDArray[np.float32]:
    normalized = pose.copy()
    normalized[:, :2] = (pose[:, :2] - box[:2]) / (box[2:] - box[:2])
    return normalized


DEFAULT_CONFIG = BotSortPoseConfig()


class BotSortPose:
    def __init__(self, fps: float, config: BotSortPoseConfig = DEFAULT_CONFIG) -> None:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Tracking requires positive finite source FPS")
        self.config, self.fps = config, fps
        self.filter = KalmanFilterXYWH()
        self.tracks: list[_Track] = []
        self.next_id = 1
        self.frame = -1
        self.dimension: int | None = None
        self.seen_rows: set[int] = set()

    def _cost(self, tracks: list[_Track], features: DetectionFeatures, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        cfg = self.config
        iou = pairwise_iou(np.stack([track.box for track in tracks]), features.boxes[rows])
        cost = np.full(iou.shape, 1e6, np.float64)
        for i, track in enumerate(tracks):
            for j, row in enumerate(rows):
                if iou[i, j] < cfg.min_iou:
                    continue
                weight = 1 - cfg.appearance_weight - cfg.pose_weight
                total = weight * (1 - iou[i, j])
                if track.embedding is not None and features.appearance_valid[row]:
                    appearance = float(np.clip(1 - track.embedding @ features.embeddings[row], 0, 2))
                    if appearance > cfg.max_cosine_distance:
                        continue  # high IoU cannot erase reliable contradictory appearance
                    total += cfg.appearance_weight * appearance
                    weight += cfg.appearance_weight
                pose = _local_pose(features.boxes[row], features.poses[row])
                confident = (track.pose[:, 2] >= cfg.pose_confidence) & (pose[:, 2] >= cfg.pose_confidence)
                if int(confident.sum()) >= cfg.min_pose_joints:
                    joint_weights = np.minimum(track.pose[confident, 2], pose[confident, 2])
                    distance = np.linalg.norm(track.pose[confident, :2] - pose[confident, :2], axis=1)
                    total += cfg.pose_weight * min(1., float(np.average(distance, weights=joint_weights)) / cfg.pose_scale)
                    weight += cfg.pose_weight
                cost[i, j] = total / weight
        return cost

    def _match(self, tracks: list[_Track], features: DetectionFeatures, rows: NDArray[np.int64]) -> list[tuple[_Track, int]]:
        if not tracks or not len(rows):
            return []
        cost = self._cost(tracks, features, rows)
        ti, di = linear_sum_assignment(cost)
        return [(tracks[t], int(rows[d])) for t, d in zip(ti, di, strict=True) if cost[t, d] < 1e6]

    def update(self, features: DetectionFeatures) -> TrackAssignments:
        if features.frame != self.frame + 1:
            raise ValueError("Tracking must receive every frame once, including empty frames, starting at zero")
        if self.dimension is not None and self.dimension != features.embeddings.shape[1]:
            raise ValueError("Appearance dimension changed within a camera")
        if self.seen_rows.intersection(features.rows.tolist()):
            raise ValueError("Detection row reused across frames")
        self.dimension = features.embeddings.shape[1]
        self.seen_rows.update(features.rows.tolist())
        self.frame = features.frame
        cfg = self.config
        self.tracks = [track for track in self.tracks if self.frame - track.last_frame <= cfg.max_gap_s * self.fps]
        for track in self.tracks:
            if track.last_frame != self.frame - 1:
                track.mean[6:8] = 0  # BoT-SORT's lost-track width/height velocity policy
            track.mean, track.covariance = self.filter.predict(track.mean, track.covariance)
        high = np.flatnonzero(features.scores >= cfg.high_score)
        low = np.flatnonzero((features.scores >= cfg.low_score) & (features.scores < cfg.high_score))
        matches = self._match(self.tracks, features, high)
        matched_tracks = {track.identity for track, _ in matches}
        remaining_active = [track for track in self.tracks if track.identity not in matched_tracks and track.last_frame == self.frame - 1]
        matches.extend(self._match(remaining_active, features, low))
        assigned = {row for _, row in matches}
        births = [int(row) for row in high if row not in assigned]
        if self.next_id - 1 + len(births) > cfg.max_tracks:
            raise TrackCapacityExceeded(f"Cumulative camera IDs would exceed {cfg.max_tracks} at frame {self.frame}; no IDs recycled")
        for track, row in matches:
            track.mean, track.covariance = self.filter.update(track.mean, track.covariance, _xywh(features.boxes[row]))
            track.last_frame = self.frame
            track.pose = _local_pose(features.boxes[row], features.poses[row])
            # Low-confidence observations can keep a track but do not contaminate its appearance EMA.
            if features.scores[row] >= cfg.high_score and features.appearance_valid[row]:
                embedded = features.embeddings[row]
                updated = embedded.copy() if track.embedding is None else cfg.ema * track.embedding + (1 - cfg.ema) * embedded
                norm = np.linalg.norm(updated)
                if norm <= 1e-8:
                    raise ValueError("Appearance EMA cancelled to zero")
                track.embedding = (updated / norm).astype(np.float32)
        for row in births:
            mean, covariance = self.filter.initiate(_xywh(features.boxes[row]))
            created = _Track(self.next_id, mean, covariance, self.frame,
                             _local_pose(features.boxes[row], features.poses[row]),
                             features.embeddings[row].copy() if features.appearance_valid[row] else None)
            self.next_id += 1
            self.tracks.append(created)
            matches.append((created, row))
        matches.sort(key=lambda pair: pair[1])
        return TrackAssignments(self.frame, np.asarray([features.rows[row] for _, row in matches], np.int64),
                                np.asarray([track.identity for track, _ in matches], np.int64))
