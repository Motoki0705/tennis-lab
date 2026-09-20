"""Selection must preserve real detection provenance across entrants and gaps."""

from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.tennis_scene.dataset_pipeline.people import (
    _people_cache_settings,
    select_court_halves,
)
from src.tennis_scene.dataset_pipeline.person_association import (
    PersonAssociation,
    associate_single_person,
    association_settings,
)


def det(raw_id: int, x: float = 0, width: float = 1) -> dict[str, Any]:
    return {
        "id": raw_id,
        "bbx_xyxy": np.array([x, -12, x + width, -10], dtype=np.float32),
    }


def test_larger_entrant_cannot_steal_incumbent() -> None:
    history = [[det(10)], [det(20, 0.1), det(21, 3, 3)], [det(30, 0.2), det(31, 2, 4)]]
    selected = associate_single_person(history, PersonAssociation(0, 1))
    assert [f[0]["source_track_id"] for f in selected] == [10, 20, 30]
    assert all(f[0]["id"] == 0 for f in selected)
    assert history[0][0]["id"] == 10


def test_short_and_long_gaps_never_reset_to_distant_bystander() -> None:
    history = [[det(10)], [det(20, 5, 4)], [], [det(30, 0.1)]]
    history += [[det(40, 5, 4)]] * 100
    history += [[det(50, 0.2), det(51, 5, 4)]]
    selected = associate_single_person(history, PersonAssociation(0, 1))
    assert [i for i, frame in enumerate(selected) if frame] == [0, 3, 104]
    assert selected[-1][0]["source_track_id"] == 50


def test_iou_gate_is_explicit_and_center_normalization_uses_incumbent() -> None:
    history = [[det(10)], [det(20, 1.1)], [det(30, 2, 20)]]
    selected = associate_single_person(history, PersonAssociation(0, 1))
    assert selected[1][0]["source_track_id"] == 20  # no overlap, bounded movement
    assert selected[2] == []  # entrant cannot use its huge diagonal to qualify
    strict = associate_single_person(history, PersonAssociation(0.01, 1))
    assert strict[1:] == [[], []]


def test_temporal_selection_preserves_masks_ids_and_legacy_default() -> None:
    history = [
        [det(10), {"id": 11, "bbx_xyxy": np.array([0, 10, 1, 12], dtype=np.float32)}],
        [],
        [
            det(20, 0.1),
            det(21, 3, 3),
            {"id": 22, "bbx_xyxy": np.array([0, 10, 1, 12], dtype=np.float32)},
        ],
    ]
    kwargs: dict[str, Any] = dict(
        half_width=6.5,
        half_length=18,
        sample_indices=np.array([0, 2]),
        min_coverage=1.0,
        max_gap_frames=1,
    )
    _, legacy = select_court_halves(history, np.eye(3), **kwargs)
    tracks, temporal = select_court_halves(
        history, np.eye(3), association=PersonAssociation(0, 1), **kwargs
    )
    np.testing.assert_array_equal(legacy[0], [10, -1, 21])
    np.testing.assert_array_equal(temporal[0], [10, -1, 20])
    np.testing.assert_array_equal(tracks.observed_mask(0), [True, False, True])


@pytest.mark.parametrize(
    "raw",
    [
        {"selection_policy": "typo"},
        {"association": {"min_iou": 0, "max_center_distance": 1}},
        {"selection_policy": "temporal_continuity"},
        {"selection_policy": "temporal_continuity", "association": {"min_iou": 0}},
        {
            "selection_policy": "temporal_continuity",
            "association": {"min_iou": 0, "max_center_distance": 1, "unknown": 2},
        },
    ],
)
def test_invalid_policy_settings_rejected(raw: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="people"):
        association_settings(raw)


@pytest.mark.parametrize(
    "iou,distance",
    [
        (float("nan"), 1),
        (0, float("inf")),
        (True, 1),
        (0, False),
        (-0.1, 1),
        (1.1, 1),
        (0, 0),
        (0, 1.1),
        ("0", 1),
    ],
)
def test_invalid_gate_settings_rejected(iou: Any, distance: Any) -> None:
    with pytest.raises(ValueError, match="people.association"):
        PersonAssociation(iou, distance)


def test_cache_identity_preserves_legacy_and_changes_for_every_setting() -> None:
    old = {"long_gap_policy": "mask", "detection_stride": 4}
    assert (
        _people_cache_settings(OmegaConf.create({**old, "selection_policy": "largest"}))
        == old
    )
    temporal: dict[str, Any] = {
        **old,
        "selection_policy": "temporal_continuity",
        "association": {"min_iou": 0, "max_center_distance": 1},
    }
    identity = _people_cache_settings(OmegaConf.create(temporal))
    assert identity != old
    for name, value in [("min_iou", 0.1), ("max_center_distance", 0.5)]:
        changed = {**temporal, "association": {**temporal["association"], name: value}}
        assert _people_cache_settings(OmegaConf.create(changed)) != identity


def test_prediction_uses_actual_frame_intervals_and_requires_two_observations() -> None:
    history: list[list[dict[str, Any]]] = [[] for _ in range(21)]
    history[0] = [det(10, 0, 2)]
    history[4] = [det(20, 1, 2)]
    history[12] = [det(30, 3, 2)]
    history[20] = [det(40, 5, 2)]
    selected = associate_single_person(history, PersonAssociation(0, 1, 60, 1))
    assert [selected[t][0]["source_track_id"] for t in (0, 4, 12, 20)] == [
        10,
        20,
        30,
        40,
    ]
    history[4] = []
    history[12] = []
    assert associate_single_person(history, PersonAssociation(0, 1, 60, 1))[20] == []


def test_prediction_is_bounded_after_disappearance_without_adopting_bystander() -> None:
    history: list[list[dict[str, Any]]] = [[] for _ in range(201)]
    history[0] = [det(10, 0, 2)]
    history[4] = [det(20, 1, 2)]
    history[20] = [det(30, 20, 10)]
    history[200] = [det(40, 4, 2)]
    selected = associate_single_person(history, PersonAssociation(0, 1, 60, 1))
    assert selected[20] == []
    assert selected[200][0]["source_track_id"] == 40
    # A bounded predictor stays local even after hundreds of missing frames.
    history[200] = [det(50, 50, 2)]
    assert associate_single_person(history, PersonAssociation(0, 1, 60, 1))[200] == []


def test_prediction_frame_limit_and_distance_limit_are_independent() -> None:
    history: list[list[dict[str, Any]]] = [[] for _ in range(21)]
    history[0] = [det(10, 0, 2)]
    history[4] = [det(20, 1, 2)]
    history[20] = [det(30, 5, 2)]
    assert associate_single_person(history, PersonAssociation(0, 1, 60, 1))[20]
    assert associate_single_person(history, PersonAssociation(0, 1, 4, 1))[20] == []
    assert associate_single_person(history, PersonAssociation(0, 1, 60, 0.1))[20] == []


@pytest.mark.parametrize(
    "frames,distance",
    [
        (-1, 1),
        (1.5, 1),
        (True, 1),
        (60, float("nan")),
        (60, "1"),
        (60, True),
        (60, 1.1),
        (0, 1),
        (60, 0),
    ],
)
def test_invalid_prediction_settings_rejected(frames: Any, distance: Any) -> None:
    with pytest.raises(ValueError, match="people.association"):
        PersonAssociation(0, 1, frames, distance)


def test_prediction_settings_are_explicit_and_in_cache_identity() -> None:
    base: dict[str, Any] = {
        "selection_policy": "temporal_continuity",
        "association": {"min_iou": 0, "max_center_distance": 1},
    }
    original = _people_cache_settings(OmegaConf.create(base))
    base["association"]["max_prediction_frames"] = 60
    with pytest.raises(ValueError, match="paired prediction"):
        association_settings(base)
    base["association"]["max_prediction_distance"] = 1
    assert association_settings(base) == PersonAssociation(0, 1, 60, 1)
    assert _people_cache_settings(OmegaConf.create(base)) != original


def test_stale_velocity_cannot_remove_match_to_last_actual_box() -> None:
    history: list[list[dict[str, Any]]] = [[] for _ in range(201)]
    history[0] = [det(10, 0, 2)]
    history[4] = [det(20, 2, 2)]
    # Reentry on the other side of the last anchor is outside the predicted gate.
    history[200] = [det(30, 0, 2)]
    selected = associate_single_person(history, PersonAssociation(0, 1, 60, 1))
    assert selected[200][0]["source_track_id"] == 30
