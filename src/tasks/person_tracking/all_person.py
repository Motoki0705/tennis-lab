"""Run-6 all-person BoT-SORT input policy, before court player selection.

Only the detector applies a score threshold. Keep source boxes and scores;
never use a Kalman prediction as a real observation or cap raw person IDs.
This is the common motion baseline, not the final appearance/pose winner.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.submodules.models import PersonDetectionResult

ALL_PERSON_BOTSORT_SETTINGS = dict(
    track_high_thresh=0., track_low_thresh=0., new_track_thresh=0., track_buffer=30,
    match_thresh=.8, fuse_score=False, gmc_method='sparseOptFlow', proximity_thresh=.5,
    appearance_thresh=.8, with_reid=False, model='auto',
)


class AllPersonAssociator:
    def __init__(self) -> None:
        from ultralytics.trackers.bot_sort import BOTSORT

        self.tracker = BOTSORT(SimpleNamespace(**ALL_PERSON_BOTSORT_SETTINGS))

    def update(self, detections: PersonDetectionResult, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return unique track IDs and exact frame-local detection rows."""
        from ultralytics.engine.results import Boxes

        data = np.column_stack((detections.boxes_xyxy, detections.scores, np.zeros(len(detections.scores), np.float32)))
        output = self.tracker.update(Boxes(data, orig_shape=frame.shape[:2]), frame)
        if not len(output):
            return np.empty(0, np.int64), np.empty(0, np.int64)
        if output.shape[1] != 8 or not np.isfinite(output).all():
            raise ValueError('BoT-SORT must return finite boxes, IDs and detection rows')
        ids, rows = output[:, 4], output[:, 7]
        if (ids != np.floor(ids)).any() or (ids < 0).any() or (rows != np.floor(rows)).any() \
                or (rows < 0).any() or (rows >= len(detections.scores)).any() \
                or len(np.unique(ids)) != len(ids) or len(np.unique(rows)) != len(rows):
            raise ValueError('BoT-SORT did not return unique exact source detection rows / IDs')
        return ids.astype(np.int64), rows.astype(np.int64)
