"""Regression checks for counts on sparse, camera-synchronized references."""

import numpy as np

from src.tasks.person_tracking.comparison import (
    TrackingPrediction,
    pool_scores,
    score_clip,
)
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.evaluation.labels import (
    CameraLabels,
    ClipLabels,
    LabelledPerson,
)
from src.utils.geometry.triangulation import PinholeCamera

PLAYER = LabelledPerson("P1", "player", "")
OTHER = LabelledPerson("P2", "player", "")
NONPLAYER = LabelledPerson("audience", "non_player", "")
A = [10., 20., 30., 70.]
B = [45., 55., 65., 105.]


def view(name: str, track_ids: list[int], boxes: np.ndarray, observed: np.ndarray) -> CameraTracks:
    camera = PinholeCamera(name, np.eye(3), np.eye(3), np.zeros(3))
    return CameraTracks(camera, (1920, 1080), np.array(track_ids, dtype=np.int64), boxes, observed)


def test_sparse_stratum_restarts_switch_and_fragment_spans() -> None:
    labels = ClipLabels(
        "video/clip", 4, (PLAYER, NONPLAYER),
        {"left": CameraLabels(np.array([0, 1, 2, 2]), np.array([0, 0, 0, 1]), np.asarray([A, A, A, B]))},
        {"pseudo_annotation": {"schema_version": 1, "annotated_frames": [0, 1, 2]}},
    )
    raw = view("left", [10, 20, 30], np.asarray([[A] * 4, [A] * 4, [B] * 4]),
               np.asarray([[True, False, False, True], [False, False, True, False], [False, False, True, True]]))
    grouped = view("left", [3, 4], np.asarray([[A] * 4, [B] * 4]),
                   np.asarray([[True, False, True, True], [False, False, True, True]]))
    prediction = TrackingPrediction({"left": raw}, {"left": raw.observed}, {"left": grouped},
                                    {"left": np.full(raw.observed.shape, -1, np.int64)}, {})
    counts = score_clip(labels, prediction, {"doubt": [0, 1, 2], "sparse": [0, 2], "control": [1]})
    doubt, sparse, control = (pool_scores([counts[name]]) for name in ("doubt", "sparse", "control"))
    assert (doubt["raw"]["idtp"], doubt["raw"]["idfp"], doubt["raw"]["idfn"]) == (1, 2, 2)
    assert (doubt["raw"]["id_switches"], doubt["raw"]["fragments"]) == (1, 1)
    assert (doubt["group"]["id_switches"], doubt["group"]["fragments"]) == (0, 1)
    assert (doubt["selection_precision"], doubt["selection_recall"]) == (2 / 3, 2 / 3)
    assert (sparse["raw"]["id_switches"], sparse["raw"]["fragments"]) == (0, 0)
    assert doubt["raw"]["unmatched_selected_boxes"] == 0
    assert control["selection_recall"] == 0


def test_one_3d_id_cannot_cover_two_labelled_people() -> None:
    labels = ClipLabels(
        "video/clip", 1, (PLAYER, OTHER),
        {cam: CameraLabels(np.array([0, 0]), np.array([0, 1]), np.asarray([A, B])) for cam in ("left", "right")}, {},
    )
    raw = {cam: view(cam, [10, 20], np.asarray([[A], [B]]), np.ones((2, 1), bool)) for cam in labels.cameras}
    observed = {cam: t.observed for cam, t in raw.items()}
    ids: dict[str, np.ndarray] = {cam: np.full((2, 1), 5, np.int64) for cam in labels.cameras}
    prediction = TrackingPrediction(raw, observed, raw, ids, {5: np.array([True])})
    score = pool_scores([score_clip(labels, prediction, {"all": [0]})["all"]])
    assert score["pairs"] == {"tp": 2, "fp": 2, "fn": 0, "f1": 2 / 3}
    assert (score["valid_player_frames"], score["labelled_player_frames"], score["coverage_3d"]) == (1, 2, .5)


def test_stopped_clip_keeps_reference_pairs_in_denominator() -> None:
    labels = ClipLabels(
        "video/clip", 1, (PLAYER,),
        {name: CameraLabels(np.array([0]), np.array([0]), np.asarray([A])) for name in ("left", "right")}, {},
    )
    raw = {
        "left": view("left", [5], np.asarray([[A]]), np.ones((1, 1), bool)),
        "right": view("right", [], np.empty((0, 1, 4)), np.empty((0, 1), bool)),
    }
    observed = {cam: tracks.observed for cam, tracks in raw.items()}
    outcome = TrackingPrediction(raw, observed, raw, {k: np.full(m.shape, -1, np.int64) for k, m in observed.items()},
                                 {}, "undecided")
    score = pool_scores([score_clip(labels, outcome, {"stopped": [0]})["stopped"]])
    assert (score["stopped_clips"], score["labelled_player_frames"], score["raw"]["idfn"]) == (1, 1, 1)
    assert (score["pairs"]["f1"], score["pairs"]["fn"], score["matched_decided_pairs"]["f1"]) == (0., 1, None)
    assert score["coverage_3d"] == 0.
