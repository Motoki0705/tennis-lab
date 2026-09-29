"""Agreement with reviewed old-detector boxes (Meiji #944), not detection recall.

The labels contain only boxes produced by the old tracker, with possible
duplicates. A unit is one labelled person in one camera/frame. Its overlap
with a detection is the maximum IoU over its reviewed boxes. Matching first
maximizes the number of units reached at ``min_iou``, then total IoU. A single
detection cannot recover two people. Extra detections of an already matched
unit are duplicates. Known people take precedence over ambiguous labels.

An unmatched prediction may be a newly found player, a non-player or a false
positive: report it as unlabelled, never as an FP. These counts support known
player-box agreement and known non-player hit rate, NOT detection recall or
full-scene AP/precision. The reference boxes come from the COCO detector being
compared, so the comparison is circular and favours that detector. Lower
non-player agreement is expected from a player-specific detector.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from src.tasks.player_association.evaluation.labels import AMBIGUOUS, CameraLabels
from src.utils.geometry.bbox import pairwise_iou

_COUNTS = (
    "frames", "labelled_frames", "detections", "known_player_units", "known_non_player_units",
    "matched_player_units", "matched_non_player_units", "duplicate_player_detections",
    "duplicate_non_player_detections", "ambiguous_detections", "unlabelled_detections",
)


def _unit_matches(iou: NDArray[np.float64], min_iou: float) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    # More weight than the sum of all IoUs makes cardinality the first objective.
    valid = iou >= min_iou
    cost = valid * (min(iou.shape) + 1.0) + np.where(valid, iou, 0.0)
    detections, units = linear_sum_assignment(-cost)
    keep = valid[detections, units]
    return detections[keep].astype(np.int64), units[keep].astype(np.int64)


@dataclass
class PartialDetectionMetrics:
    """Accumulate labelled camera/frames; all inputs use source-image pixels."""

    min_iou: float = 0.5
    score_threshold: float = 0.3
    _counts: Counter[str] = field(default_factory=lambda: Counter(dict.fromkeys(_COUNTS, 0)), init=False)
    _player_iou_sum: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not 0 < self.min_iou <= 1 or not 0 <= self.score_threshold <= 1:
            raise ValueError("Expected min_iou in (0, 1] and score_threshold in [0, 1]")

    def update(
        self, boxes: NDArray[np.float32], scores: NDArray[np.float32], *,
        labels: CameraLabels, roles: NDArray[np.str_], frame: int,
    ) -> None:
        if boxes.shape != (len(scores), 4) or scores.ndim != 1:
            raise ValueError("Detection arrays must be (N,4) boxes and (N,) scores")
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all() \
                or ((scores < 0) | (scores > 1)).any() or (boxes[:, 2:] < boxes[:, :2]).any():
            raise ValueError("Detection boxes/scores must be finite valid rectangles/probabilities")
        if frame < 0 or roles.ndim != 1 or not np.isin(roles, ["player", "non_player"]).all():
            raise ValueError("Invalid frame or labelled person roles")
        selection = labels.at(frame)
        people = labels.person_index[selection]
        if ((people < AMBIGUOUS) | (people >= len(roles))).any():
            raise ValueError("Label references an unknown person")
        label_boxes = labels.boxes_xyxy[selection]
        boxes = boxes[scores >= self.score_threshold]
        units = np.unique(people[people != AMBIGUOUS])
        unit_roles = roles[units]
        counts = self._counts
        counts["frames"] += 1
        counts["labelled_frames"] += int(bool(len(people)))
        counts["detections"] += len(boxes)
        for role in ("player", "non_player"):
            counts[f"known_{role}_units"] += int((unit_roles == role).sum())
        # Each column is one person, even when the old tracker had duplicate boxes.
        overlaps = pairwise_iou(boxes, label_boxes)
        iou = np.column_stack([overlaps[:, people == person].max(axis=1) for person in units]) \
            if len(units) else np.empty((len(boxes), 0), np.float64)
        assigned_detections, assigned_units = _unit_matches(iou, self.min_iou)
        assigned: set[int] = set(assigned_detections.tolist())
        for detection, unit in zip(assigned_detections, assigned_units, strict=True):
            role = str(unit_roles[unit])
            counts[f"matched_{role}_units"] += 1
            if role == "player":
                self._player_iou_sum += float(iou[detection, unit])
        for detection in range(len(boxes)):
            if detection in assigned:
                continue
            if len(assigned_units) and float(iou[detection, assigned_units].max()) >= self.min_iou:
                unit = assigned_units[int(iou[detection, assigned_units].argmax())]
                counts[f"duplicate_{unit_roles[unit]}_detections"] += 1
            elif (overlaps[detection, people == AMBIGUOUS] >= self.min_iou).any():
                counts["ambiguous_detections"] += 1
            else:
                counts["unlabelled_detections"] += 1

    def merge(self, other: PartialDetectionMetrics) -> None:
        if (self.min_iou, self.score_threshold) != (other.min_iou, other.score_threshold):
            raise ValueError("Cannot aggregate different evaluation thresholds")
        self._counts.update(other._counts)
        self._player_iou_sum += other._player_iou_sum

    def compute(self) -> dict[str, int | float | None]:
        counts = self._counts
        return {
            **counts,
            "known_player_box_agreement": counts["matched_player_units"] / counts["known_player_units"]
                if counts["known_player_units"] else None,
            "known_non_player_hit_rate": counts["matched_non_player_units"] / counts["known_non_player_units"]
                if counts["known_non_player_units"] else None,
            "mean_matched_player_iou": self._player_iou_sum / counts["matched_player_units"]
                if counts["matched_player_units"] else None,
        }
