"""Explicit person-box postprocessing; greedy, class agnostic and row preserving."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DuplicateMerge:
    frame: int
    kept_row: int
    dropped_row: int
    kept_score: float
    dropped_score: float
    iou: float

    def __post_init__(self) -> None:
        if min(self.frame, self.kept_row, self.dropped_row) < 0 or self.kept_row == self.dropped_row \
                or not np.isfinite([self.kept_score, self.dropped_score, self.iou]).all() \
                or not 0 <= self.dropped_score <= self.kept_score <= 1 or not .8 <= self.iou <= 1 \
                or (self.kept_score == self.dropped_score and self.kept_row > self.dropped_row):
            raise ValueError('Invalid duplicate person-box merge record')


def merge_person_boxes(frame: int, rows: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
                       *, enabled: bool) -> tuple[np.ndarray, tuple[DuplicateMerge, ...]]:
    """Return input indices in source-row order and one audit record per drop.

    IoU >= .8 is fixed by the user decision; suppressed rows do not suppress
    further rows. Boxes/scores are retained exactly, with no averaging.
    """
    if frame < 0 or rows.dtype != np.int64 or rows.ndim != 1 or scores.shape != rows.shape \
            or boxes.shape != (len(rows), 4) or (rows < 0).any() or len(np.unique(rows)) != len(rows) \
            or not np.isfinite(boxes).all() or not np.isfinite(scores).all() \
            or (boxes[:, 2:] <= boxes[:, :2]).any() or (scores < 0).any() or (scores > 1).any():
        raise ValueError('Merge requires unique source rows, positive finite boxes and probability scores')
    if not enabled:
        return np.arange(len(rows), dtype=np.int64), ()
    pending = np.lexsort((rows, -scores))
    kept, records = [], []
    values: np.ndarray = boxes.astype(np.float64)
    areas = np.prod(values[:, 2:] - values[:, :2], axis=1)
    while len(pending):
        first, remaining = int(pending[0]), pending[1:]
        kept.append(first)
        extent = np.maximum(0., np.minimum(values[first, 2:], values[remaining, 2:])
                            - np.maximum(values[first, :2], values[remaining, :2]))
        intersection = np.prod(extent, axis=1)
        iou = intersection / (areas[first] + areas[remaining] - intersection)
        for other, overlap in zip(remaining[iou >= .8], iou[iou >= .8], strict=True):
            records.append(DuplicateMerge(frame, int(rows[first]), int(rows[other]),
                                           float(scores[first]), float(scores[other]), float(overlap)))
        pending = remaining[iou < .8]
    selected = np.asarray(kept, dtype=np.int64)
    return selected[np.argsort(rows[selected], kind='stable')], tuple(records)
