import numpy as np

from src.tasks.player_association.evaluation.labels import CameraLabels
from src.tasks.player_detection.evaluation.disagreements import (
    player_unit_rows,
    summarize,
)


def test_duplicate_old_boxes_and_nonplayer_competition_are_preserved() -> None:
    labels = CameraLabels(np.zeros(4, np.int64), np.asarray([0, 0, 1, 2], np.int64),
                          np.asarray([[0, 0, 10, 10], [1, 0, 11, 10], [0, 20, 10, 40], [0, 0, 10, 10]], np.float64))
    rows = player_unit_rows(np.asarray([[1, 0, 11, 10]], np.float32), labels,
                            np.asarray(["player", "player", "non_player"]), 0, min_iou=.5)
    assert len(rows) == 2
    assert rows[0]["matched"] and rows[0]["best_iou"] == 1
    assert not rows[1]["matched"] and rows[1]["best_iou"] == 0
    assert [row["near_far"] for row in rows] == ["far", "near"]
    summary = summarize(rows)
    assert summary["units"] == 2 and summary["unmatched"] == 1
    assert sum(summary["best_iou_histogram"]) == sum(summary["height_px_histogram"]) == 1


def test_no_prediction_and_single_player_are_not_missing_rows_or_invented_sides() -> None:
    labels = CameraLabels(np.asarray([0], np.int64), np.asarray([0], np.int64),
                          np.asarray([[0, 0, 10, 10]], np.float64))
    rows = player_unit_rows(np.empty((0, 4), np.float32), labels, np.asarray(["player"]), 0, min_iou=.5)
    assert rows[0]["near_far"] == "unknown" and not rows[0]["matched"]
    assert summarize(rows)["zero_overlap"] == 1
