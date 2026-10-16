from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tasks.player_association.evaluation import (
    AMBIGUOUS,
    CameraPrediction,
    ClipLabels,
    ReviewedTrack,
    evaluate,
    materialize,
)

FRAMES = 20
PEOPLE = {"A": {"role": "player", "description": "a"}, "B": {"role": "player", "description": "b"},
          "X": {"role": "non_player", "description": "adjacent court"}}


def _track(track_id: int, x: float, frames: range = range(FRAMES)) -> ReviewedTrack:
    boxes: np.ndarray = np.zeros((FRAMES, 4), np.float32)
    boxes[:] = (x, 10, x + 20, 60)
    observed: np.ndarray = np.zeros(FRAMES, bool)
    observed[list(frames)] = True
    return ReviewedTrack(track_id, boxes, observed)


def _scene() -> tuple[dict[str, list[ReviewedTrack]], dict[str, dict[int, Any]]]:
    """cam0: A, B, X, and track 4 that follows X until frame 10 and then B (an ID switch)."""
    tracks = {"cam0": [_track(1, 0), _track(2, 100), _track(3, 200), _track(4, 300)],
              "cam1": [_track(1, 0), _track(2, 100)]}
    review: dict[str, dict[int, Any]] = {"cam0": {1: "A", 2: "B", 3: "X", 4: [[0, 10, "X"], [10, 12, None], [12, FRAMES, "B"]]},
              "cam1": {1: "B", 2: "A"}}
    return tracks, review


def _labels() -> ClipLabels:
    tracks, review = _scene()
    return materialize("clip", FRAMES, {"people": PEOPLE, "tracks": review}, tracks, {"source": "test"})


def _prediction(tracks: list[ReviewedTrack], ids: list[int] | np.ndarray) -> CameraPrediction:
    """``ids`` per track (constant over time) or per track and frame."""
    player_ids = np.asarray(ids, np.int64)
    if player_ids.ndim == 1:
        player_ids = np.repeat(player_ids[:, None], FRAMES, 1)
    return CameraPrediction(np.asarray([t.track_id for t in tracks], np.int64), np.stack([t.boxes_xyxy for t in tracks]).astype(np.float64),
                            np.stack([t.observed for t in tracks]), player_ids)


def test_materialize_labels_every_observed_box_and_round_trips(tmp_path: Path) -> None:
    labels = _labels()
    cam0 = labels.cameras["cam0"]
    assert len(cam0.frames) == 4 * FRAMES
    switch_rows = cam0.boxes_xyxy[:, 0] == 300
    persons = cam0.person_index[switch_rows]
    assert [labels.people[i].person_id for i in persons[:10]] == ["X"] * 10
    assert (persons[10:12] == AMBIGUOUS).all()
    assert [labels.people[i].person_id for i in persons[12:]] == ["B"] * 8
    labels.save(tmp_path / "v/c.json")
    loaded = ClipLabels.load(tmp_path / "v/c.json")
    assert loaded.to_json() == labels.to_json()


@pytest.mark.parametrize("change", ["missing_track", "unknown_track", "gap", "unknown_person", "unused_person"])
def test_materialize_refuses_incomplete_reviews(change: str) -> None:
    tracks, review = _scene()
    people = dict(PEOPLE)
    if change == "missing_track":
        del review["cam1"][2]
    elif change == "unknown_track":
        review["cam1"][9] = "A"
    elif change == "gap":
        review["cam0"][4] = [[0, 10, "X"], [11, FRAMES, "B"]]
    elif change == "unknown_person":
        review["cam1"][2] = "Z"
    else:
        people["Y"] = {"role": "non_player", "description": "unused"}
    with pytest.raises(ValueError):
        materialize("clip", FRAMES, {"people": people, "tracks": review}, tracks, {})


def test_perfect_prediction_scores_one() -> None:
    labels = _labels()
    tracks, _ = _scene()
    track4 = np.full(FRAMES, 1)
    track4[:10] = -1  # X, then B (ID 1) after the switch
    predictions = {"cam0": _prediction(tracks["cam0"], np.stack([np.full(FRAMES, 0), np.full(FRAMES, 1), np.full(FRAMES, -1), track4])),
                   "cam1": _prediction(tracks["cam1"], [1, 0])}
    result = evaluate(labels, predictions)
    assert result["pairs"]["f1"] == 1
    assert result["group_accuracy"]["accuracy"] == 1
    assert result["exclusion"]["precision"] == result["exclusion"]["recall"] == 1
    assert result["id_switch"]["true"] == 1 and result["id_switch"]["recall"] == 1 and result["id_switch"]["precision"] == 1
    assert result["failed_frame_runs"] == []


def test_swapped_camera_and_unhandled_switch_are_counted() -> None:
    labels = _labels()
    tracks, _ = _scene()
    # cam1 swapped (A<->B) and track 4 kept whole as excluded: the switch is not detected.
    predictions = {"cam0": _prediction(tracks["cam0"], [0, 1, -1, -1]), "cam1": _prediction(tracks["cam1"], [0, 1])}
    result = evaluate(labels, predictions)
    pairs = result["pairs"]
    # Per frame: both true pairs (A-A, B-B) are missed and both cross pairs (A-B) are linked. From frame 12 track 4 is a
    # duplicate box of B in cam0; it joins B's unit, which still carries ID 1 through track 2.
    assert pairs["tp"] == 0 and pairs["fp"] == 2 * FRAMES and pairs["fn"] == 2 * FRAMES
    assert result["group_accuracy"]["frames_correct"] == 0
    # Track 4 excluded while it shows B: B's unit still carries ID 1 through track 2, so no false exclusion.
    assert result["exclusion"]["fp"] == 0 and result["exclusion"]["tp"] == FRAMES + 10
    assert result["id_switch"]["recall"] == 0 and result["id_switch"]["predicted"] == 0


def test_non_player_linked_and_player_excluded() -> None:
    labels = _labels()
    tracks, _ = _scene()
    predictions = {"cam0": _prediction(tracks["cam0"], [0, -1, 1, -1]), "cam1": _prediction(tracks["cam1"], [1, 0])}
    result = evaluate(labels, predictions)
    exclusion = result["exclusion"]
    # X (track 3) gets ID 1 in all frames; track 4 shows X for 10 frames and is excluded (true exclusion).
    assert exclusion["fn"] == FRAMES and exclusion["tp"] == 10
    # No box of B in cam0 carries an ID in any frame (track 2 and, from frame 12, track 4 are both -1).
    assert exclusion["fp"] == FRAMES
    # X (ID 1) links with B in cam1 (ID 1): a false pair in every frame.
    assert result["pairs"]["fp"] == FRAMES and result["pairs"]["tp"] == FRAMES
    assert result["group_accuracy"]["frames_correct"] == 0


def test_unmatched_boxes_are_coverage_not_association_errors() -> None:
    labels = _labels()
    tracks, _ = _scene()
    stray = _track(9, 600)
    predictions = {"cam0": _prediction([*tracks["cam0"][:2], stray], [0, 1, 5]), "cam1": _prediction(tracks["cam1"], [1, 0])}
    result = evaluate(labels, predictions)
    assert result["coverage"]["cam0"]["unmatched_predicted_boxes_with_id"] == FRAMES
    assert result["coverage"]["cam0"]["matched_label_boxes"] == 2 * FRAMES
    assert result["pairs"]["f1"] == 1 and result["group_accuracy"]["accuracy"] == 1
