"""Small CPU checks of the run-specific selection and score contracts."""

import numpy as np
from numpy.typing import NDArray
from scores import box_counts, box_rates, pool_pairs, prediction, project_ids
from selection import LINKING, image_candidates, selected_rows

from src.tasks.person_tracking.sequence import TrackEvidence
from src.tasks.player_association.evaluation.labels import (
    AMBIGUOUS,
    CameraLabels,
    ClipLabels,
    LabelledPerson,
)
from src.tasks.player_association.evaluation.metrics import evaluate, match_to_labels
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput


def tracks(
    camera: str, ids: list[int], boxes: list[list[int]], observed: NDArray[np.bool_]
) -> PersonTrackingOutput:
    count, frames = observed.shape
    xyxy = np.broadcast_to(
        np.array(boxes, np.float32)[:, None, :], (count, frames, 4)
    ).copy()
    rows = np.where(observed, np.arange(count * frames).reshape(count, frames), -1)
    poses = np.zeros((count, frames, 17, 3), np.float32)
    poses[..., 0, 0] = np.where(observed, rows + 100, 0)
    evidence = TrackEvidence(
        rows, poses, observed[..., None].astype(np.float32), observed, "clip"
    )
    return PersonTrackingOutput(
        camera,
        np.array(ids, np.int64),
        xyxy,
        observed,
        tuple((i,) for i in ids),
        (),
        evidence,
    )


def test_image_rank_uses_size_center_real_dwell_cap_and_raw_id() -> None:
    observed = np.array([[True] * 4 + [False] * 4] * 4 + [[True] + [False] * 7])
    raw = tracks(
        "cam0",
        [8, 9, 4, 2, 1],
        [
            [20, 100, 120, 300],
            [450, 150, 550, 350],
            [450, 170, 550, 220],
            [450, 400, 550, 600],
            [450, 180, 550, 800],
        ],
        observed,
    )
    chosen = image_candidates(raw, (1000, 1000), LINKING)
    assert raw.track_ids[chosen].tolist() == [2, 9, 4, 8]

    # More equal eligible tracks than the six-carrier cap.
    duplicate = tracks(
        "cam0",
        list(reversed(range(8))),
        [[450, 150, 550, 350]] * 8,
        np.ones((8, 8), bool),
    )
    capped = image_candidates(duplicate, (1000, 1000), LINKING)
    assert duplicate.track_ids[capped].tolist() == list(range(6))


def test_poses_and_assigned_ids_follow_original_track_rows() -> None:
    raw = tracks(
        "cam0",
        [11, 20],
        [[10, 10, 20, 25], [70, 40, 80, 95]],
        np.array([[True, False], [False, True]]),
    )
    chosen = selected_rows(raw, np.array([1, 0], np.int64))
    evidence = chosen.tracks.evidence
    assert evidence is not None
    assert evidence.detection_rows.tolist() == [[-1, 3], [0, -1]]
    assert evidence.require_poses()[..., 0, 0].tolist() == [[0, 103], [100, 0]]
    # The MILP labels a segment across gaps. Only real boxes receive IDs.
    result = project_ids(prediction(raw), chosen, np.array([[2, 2], [1, 1]], np.int64))
    assert result.player_ids.tolist() == [[1, -1], [-1, 2]]


def test_unmatched_and_ambiguous_do_not_change_role_or_stop_denominators() -> None:
    one = [10, 10, 30, 50]
    two = [50, 10, 70, 50]
    three = [90, 90, 120, 140]
    tracks0 = tracks("cam0", [1, 2, 3], [one, two, three], np.ones((3, 1), bool))
    tracks1 = tracks("cam1", [11, 12], [one, two], np.ones((2, 1), bool))
    labels = ClipLabels(
        "clip",
        1,
        (LabelledPerson("p", "player", ""), LabelledPerson("n", "non_player", "")),
        {
            "cam0": CameraLabels(
                np.zeros(3, np.int64),
                np.array([0, 1, AMBIGUOUS]),
                np.array([one, two, three], float),
            ),
            "cam1": CameraLabels(
                np.zeros(2, np.int64), np.array([0, 1]), np.array([one, three], float)
            ),
        },
        {},
    )
    raw = {"cam0": prediction(tracks0), "cam1": prediction(tracks1)}
    matches = match_to_labels(labels, raw, min_iou=0.5)
    choices = {
        "cam0": selected_rows(tracks0, np.array([0, 2])),
        "cam1": selected_rows(tracks1, np.array([1])),
    }
    counts = box_counts(labels, raw, matches, choices)
    rates = box_rates(counts)
    assert counts["selected_ambiguous"] == counts["selected_unmatched"] == 1
    assert rates == {
        "player_precision": 1.0,
        "player_recall": 0.5,
        "nonplayer_precision": 0.5,
        "nonplayer_recall": 0.5,
        "nonplayer_recall_matched": 1.0,
    }

    stopped = pool_pairs([evaluate(labels, raw)["pairs"]])
    assert (stopped["tp"], stopped["fn"], stopped["f1"]) == (0, 1, 0)
