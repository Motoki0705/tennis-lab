import numpy as np
import pytest

from src.submodules.models import PersonDetectionResult, filter_detections_by_footpoint
from src.tasks.player_association.evaluation.labels import CameraLabels
from src.tasks.player_detection.evaluation.far_player import (
    add_counts,
    far_tiles,
    merge_extra,
    threshold,
    unit_rows,
)


def test_low_score_diagnosis_uses_duplicate_unit_boxes_and_nonplayer_competition() -> None:
    labels = CameraLabels(np.zeros(4, np.int64), np.array([0, 0, 1, 2], np.int64),
        np.array([[0, 0, 10, 10], [1, 0, 11, 10], [0, 20, 10, 40], [20, 0, 30, 10]], np.float64))
    prediction = PersonDetectionResult(np.array([[1, 0, 11, 10], [0, 20, 10, 40], [20, 0, 30, 10]], np.float32),
                                        np.array([.02, .9, .1], np.float32))
    roles = np.array(['player', 'player', 'non_player'])
    raw, baseline = unit_rows(prediction, labels, roles, 0), unit_rows(threshold(prediction, .3), labels, roles, 0)
    assert len(raw) == 3 and [r['near_far'] for r in raw] == ['far', 'near', 'far']
    assert raw[0]['best_score_iou03'] == pytest.approx(.02) and baseline[0]['best_score_iou03'] is None
    counts: dict[str, int] = {}
    for row, ref in zip(raw, baseline, strict=True):
        add_counts(counts, row, ref)
    assert counts['player_units'] == 2 and counts['added_player_units'] == 1
    assert counts['added_non_player_units'] == 1
    # Same single box cannot count as both a player and a non-player hit.
    competing = CameraLabels(np.zeros(2, np.int64), np.array([0, 1], np.int64),
                              np.array([[0, 0, 10, 10], [0, 0, 10, 10]], np.float64))
    rows = unit_rows(PersonDetectionResult(np.array([[0, 0, 10, 10]], np.float32), np.array([.9], np.float32)),
                     competing, np.array(['player', 'non_player']), 0)
    assert sum(r['matched'] for r in rows) == 1
    assert all(r['near_far'] == 'unknown' for r in rows)


def test_coco_union_is_small_roi_only_and_does_not_replace_ft_or_duplicate_tiles() -> None:
    primary = PersonDetectionResult(np.array([[0, 0, 10, 20]], np.float32), np.array([.4], np.float32))
    coco = PersonDetectionResult(np.array([[0, 0, 10, 20], [20, 0, 30, 30], [20, 0, 30, 30],
        [40, 0, 50, 100], [200, 0, 210, 20]], np.float32), np.array([.99, .9, .8, .9, .9], np.float32))
    roi = ((0., 0.), (100., 0.), (100., 200.), (0., 200.))
    result = merge_extra(primary, filter_detections_by_footpoint(coco, roi), max_height=64)
    np.testing.assert_array_equal(result.boxes_xyxy, [[0, 0, 10, 20], [20, 0, 30, 30]])
    assert result.scores.tolist() == pytest.approx([.4, .9])
    assert far_tiles(1920, 1080) == ((0, 0, 1056, 648), (864, 0, 1920, 648))
