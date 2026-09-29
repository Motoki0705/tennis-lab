"""Incomplete reviewed labels must not make newly detected people false positives."""

from __future__ import annotations

import numpy as np
import pytest

from src.tasks.player_association.evaluation.labels import CameraLabels
from src.tasks.player_detection.evaluation.partial_labels import (
    PartialDetectionMetrics,
    _unit_matches,
)


def test_duplicate_labels_are_one_person_and_unknown_detections_are_not_false_positives() -> None:
    labels = CameraLabels(np.zeros(5, np.int64), np.asarray([0, 0, 1, 2, -1], np.int64),
        np.asarray([[0, 0, 10, 10], [1, 0, 11, 10], [20, 0, 30, 10], [40, 0, 50, 10], [60, 0, 70, 10]], np.float64))
    boxes = np.asarray([[0, 0, 10, 10], [1, 0, 11, 10], [40, 0, 50, 10], [60, 0, 70, 10],
                        [80, 0, 90, 10], [20, 0, 30, 10]], np.float32)
    metrics = PartialDetectionMetrics()
    metrics.update(boxes, np.asarray([.9, .8, .9, .9, .9, .2], np.float32), labels=labels,
                   roles=np.asarray(["player", "player", "non_player"]), frame=0)
    result = metrics.compute()
    assert result["detections"] == 5
    assert result["known_player_units"] == 2 and result["matched_player_units"] == 1
    assert result["known_player_recall"] == .5
    assert result["duplicate_player_detections"] == 1
    assert result["known_non_player_hit_rate"] == 1.
    assert result["ambiguous_detections"] == 1 and result["unlabelled_detections"] == 1
    assert result["mean_matched_player_iou"] == 1.
    assert not any("precision" in key or "false_positive" in key for key in result)


def test_one_prediction_cannot_recover_two_people() -> None:
    labels = CameraLabels(np.zeros(2, np.int64), np.asarray([0, 1], np.int64),
        np.asarray([[0, 0, 10, 10], [5, 0, 15, 10]], np.float64))
    metrics = PartialDetectionMetrics()
    metrics.update(np.asarray([[0, 0, 15, 10]], np.float32), np.ones(1, np.float32),
                   labels=labels, roles=np.asarray(["player", "player"]), frame=0)
    assert metrics.compute()["known_player_recall"] == .5


def test_assignment_maximizes_valid_matches_before_overlap() -> None:
    # Maximizing raw IoU chooses the off-diagonal (1 + .49) and loses a match.
    detections, units = _unit_matches(np.asarray([[.51, 1.], [.49, .51]], np.float64), .5)
    assert list(zip(detections.tolist(), units.tolist(), strict=True)) == [(0, 0), (1, 1)]


def test_known_labels_take_priority_over_ambiguous_boxes() -> None:
    labels = CameraLabels(np.zeros(2, np.int64), np.asarray([-1, 0], np.int64),
        np.asarray([[0, 0, 10, 10], [1, 0, 11, 10]], np.float64))
    metrics = PartialDetectionMetrics()
    metrics.update(np.asarray([[0, 0, 10, 10]], np.float32), np.ones(1, np.float32),
                   labels=labels, roles=np.asarray(["player"]), frame=0)
    assert metrics.compute()["known_player_recall"] == 1.
    assert metrics.compute()["ambiguous_detections"] == 0


def test_empty_frames_keep_missing_labels_and_unlabelled_predictions_distinct() -> None:
    labels = CameraLabels(np.asarray([0], np.int64), np.asarray([0], np.int64),
                          np.asarray([[0, 0, 10, 10]], np.float64))
    first, second = PartialDetectionMetrics(), PartialDetectionMetrics()
    first.update(np.empty((0, 4), np.float32), np.empty(0, np.float32),
                 labels=labels, roles=np.asarray(["player"]), frame=0)
    second.update(np.asarray([[0, 0, 10, 10]], np.float32), np.ones(1, np.float32),
                  labels=labels, roles=np.asarray(["player"]), frame=1)
    first.merge(second)
    result = first.compute()
    assert result["frames"] == 2 and result["labelled_frames"] == 1
    assert result["known_player_recall"] == 0 and result["unlabelled_detections"] == 1
    assert result["known_non_player_hit_rate"] is None
    assert result["mean_matched_player_iou"] is None
    with pytest.raises(ValueError, match="thresholds"):
        first.merge(PartialDetectionMetrics(min_iou=.7))


@pytest.mark.parametrize("threshold", [0., -1., float("nan"), 1.1])
def test_invalid_iou_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError, match="min_iou"):
        PartialDetectionMetrics(min_iou=threshold)
