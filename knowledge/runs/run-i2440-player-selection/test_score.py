import numpy as np
from score import raw_prediction, score_frames

from src.tasks.player_association.evaluation.labels import (
    CameraLabels,
    ClipLabels,
    LabelledPerson,
)


def test_scoring_counts_untracked_players_and_ignores_predictions_outside_scored_frames() -> None:
    player = [0., 0., 10., 10.]
    bystander = [20., 0., 30., 10.]
    ambiguous = [40., 0., 50., 10.]
    untracked = [60., 0., 70., 10.]
    labels = ClipLabels(
        "video/clip", 3,
        (LabelledPerson("p", "player", ""), LabelledPerson("non", "non_player", "")),
        {"cam0": CameraLabels(
            np.zeros(4, np.int64), np.array([0, 1, -1, 0]), np.asarray([player, bystander, ambiguous, untracked]),
        )},
        {},
    )
    raw = raw_prediction({
        "track_ids": np.arange(3), "boxes_xyxy": np.repeat(np.asarray([player, bystander, ambiguous])[:, None], 3, axis=1),
        "observed": np.ones((3, 3), bool),
    })
    result = score_frames(labels, {"cam0": raw}, {"candidate": {"cam0": raw.observed}}, (0, 2))
    assert result["candidate"] == {
        "frames": 2, "player_boxes": 2, "non_player_boxes": 1, "tracked_player": 1, "tracked_non_player": 1,
        "tp": 1, "fp": 1, "fn": 1, "selected_ambiguous": 1, "selected_unmatched": 3,
    }
