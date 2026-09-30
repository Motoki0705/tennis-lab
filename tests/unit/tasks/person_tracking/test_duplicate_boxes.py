import numpy as np
import pytest

from src.tasks.person_tracking.duplicate_boxes import merge_person_boxes


def test_inclusive_iou_tie_order_and_off_preserves_every_row() -> None:
    rows = np.array([9, 2, 15], np.int64)
    boxes = np.array([[0, 0, 10, 10], [0, 0, 8, 10], [30, 0, 40, 10]], np.float32)
    scores = np.array([.7, .7, .9], np.float32)
    before = boxes.copy()
    keep, audit = merge_person_boxes(5, rows, boxes, scores, enabled=True)
    assert rows[keep].tolist() == [2, 15]
    assert len(audit) == 1 and audit[0].kept_row == 2 and audit[0].dropped_row == 9
    assert audit[0].iou == .8 and audit[0].frame == 5
    off, records = merge_person_boxes(5, rows, boxes, scores, enabled=False)
    assert off.tolist() == [0, 1, 2] and not records
    np.testing.assert_array_equal(boxes, before)


def test_higher_score_wins_and_suppressed_boxes_do_not_merge_transitively() -> None:
    rows = np.array([0, 1, 2], np.int64)
    # A/B and B/C overlap >=.8; A/C does not.
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [2, 0, 12, 10]], np.float32)
    keep, records = merge_person_boxes(0, rows, boxes, np.array([.9, .8, .7]), enabled=True)
    assert keep.tolist() == [0, 2] and len(records) == 1
    keep, records = merge_person_boxes(0, rows, boxes, np.array([.8, .9, .7]), enabled=True)
    assert keep.tolist() == [1] and {r.dropped_row for r in records} == {0, 2}


def test_empty_frame_and_invalid_input() -> None:
    keep, records = merge_person_boxes(0, np.empty(0, np.int64), np.empty((0, 4)), np.empty(0), enabled=True)
    assert not len(keep) and not records
    with pytest.raises(ValueError, match='positive finite'):
        merge_person_boxes(0, np.array([0], np.int64), np.zeros((1, 4)), np.array([.9]), enabled=False)
