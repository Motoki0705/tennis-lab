"""Keep unknown labels out of BCE and assign peaks in original frames."""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.ball_refiner_3d.post_training.labels import Annotations, Event
from src.tasks.ball_refiner_3d.post_training.motion import MotionFeatures
from src.tasks.ball_refiner_3d.post_training.scoring import report, score_clip


def test_loss_mask_is_per_class_and_targets_include_outside_point() -> None:
    coverage: NDArray[np.bool_] = np.ones(14, dtype=np.bool_)
    coverage[10:12] = False
    labels = Annotations(
        (
            Event(3, "shot", "uncertain", False, None),
            Event(4, "shot", "probable", False, None),
            Event(4, "bounce", "certain", True, None),
            Event(10, "net", "certain", True, None),
        ),
        coverage,
    )
    target, mask = labels.targets(2.0)
    np.testing.assert_array_equal(np.argwhere(target == 1), [[4, 0], [4, 1], [10, 2]])
    np.testing.assert_array_equal(np.flatnonzero(~mask[:10, 0]), [1, 2, 3, 5])
    assert mask[:10, 1:].all() and not mask[10:12].any()
    assert target[6, 0] == pytest.approx(np.exp(-0.5))
    assert mask[9, 2] and target[9, 2] == pytest.approx(np.exp(-0.125))
    assert mask[12, 2] and target[12, 2] == pytest.approx(np.exp(-0.5))


def test_scoring_maximizes_pairs_and_tags_masked_and_point_events() -> None:
    size = 23
    coverage: NDArray[np.bool_] = np.ones(size, dtype=np.bool_)
    coverage[20] = False
    labels = Annotations(
        (
            Event(2, "shot", "certain", True, "serve"),
            Event(5, "shot", "probable", True, "volley"),
            Event(10, "bounce", "certain", True, None),
            Event(14, "net", "uncertain", False, None),
        ),
        coverage,
    )
    probability: NDArray[np.float32] = np.full((size, 3), 0.01, dtype=np.float32)
    probability[[4, 7, 17, 20], 0] = 0.9
    probability[8, 1] = 0.8
    probability[15, 2] = 0.8
    valid = coverage.copy()
    valid[5] = False
    motion = MotionFeatures(
        np.zeros((size, 3), dtype=np.float32),
        valid,
        np.empty((0, 0, 0), np.float32),
        np.zeros((size, 4), dtype=np.bool_),
        np.zeros((size, 4), dtype=np.bool_),
        (8, None, None, None),
    )
    rows = score_clip(probability, labels, 50, motion)
    assert [
        (d.prediction_frame, d.signed_offset_frames) for d in rows if d.outcome == "tp"
    ] == [(4, 2), (7, 2), (8, -2)]
    assert {d.outcome for d in rows} == {
        "tp",
        "fp",
        "near_uncertain",
        "outside_coverage",
        "uncertain_label",
    }
    metrics = report(rows)
    shots = metrics["all"]["by_class"]["shot"]
    assert (
        shots["tp"],
        shots["fp"],
        shots["fn"],
        shots["f1"],
        shots["time_mae_ms"],
    ) == (2, 1, 0, 0.8, 40)
    assert metrics["in_point"]["by_class"]["shot"]["f1"] == 1.0
    assert metrics["conditions"]["ball_missing"]["by_class"]["shot"]["tp"] == 1
    assert metrics["conditions"]["volley"]["by_class"]["shot"]["recall"] == 1.0
    assert metrics["conditions"]["volley"]["by_class"]["shot"]["precision"] is None
    assert metrics["all"]["by_class"]["net"]["f1"] is None


def test_known_events_are_scored_inside_unlabelable_frames() -> None:
    coverage: NDArray[np.bool_] = np.ones(22, dtype=np.bool_)
    coverage[3:6] = coverage[9:12] = coverage[13] = coverage[16:19] = False
    labels = Annotations(
        (
            Event(4, "shot", "probable", False, None),
            Event(10, "shot", "certain", True, None),
            Event(14, "bounce", "certain", True, None),
        ),
        coverage,
    )
    probability: NDArray[np.float32] = np.zeros((22, 3), dtype=np.float32)
    probability[[5, 17, 20], 0] = 0.9
    probability[13, 1] = 0.8
    rows = score_clip(probability, labels, 60)
    assert [
        (r.outcome, r.prediction_frame, None if r.label is None else r.label["frame"])
        for r in rows
    ] == [
        ("tp", 5, 4),
        ("outside_coverage", 17, None),
        ("fp", 20, None),
        ("fn", None, 10),
        ("tp", 13, 14),
    ]
    metrics = report(rows)
    shots = metrics["all"]["by_class"]["shot"]
    assert (shots["events"], shots["tp"], shots["fp"], shots["fn"]) == (2, 1, 1, 1)
    assert shots["excluded_predictions"] == 1 and shots["excluded_labels"] == 0
    assert metrics["in_point"]["by_class"]["shot"]["fn"] == 1
