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


def test_v4_scene_annotations_support_detailed_subtypes_and_mask_unknown_kind() -> None:
    from src.tennis_scene.ball_event_schema import (
        EVENT_CLASSES_V4,
        EVENT_SUBTYPES_V4,
        ball_event_metadata,
    )
    from src.tennis_scene.schema import SceneResult

    frames = 18
    valid: NDArray[np.bool_] = np.ones(frames, dtype=np.bool_)
    joints_valid: NDArray[np.bool_] = np.zeros((1, frames, 17), dtype=np.bool_)
    scene = SceneResult(
        frames,
        60.0,
        640,
        360,
        np.zeros((2, frames, 14, 2), np.float32),
        np.ones((2, frames, 14), np.float32),
        np.zeros((1, frames, 3), np.float32),
        np.zeros((1, frames), np.float32),
        ball_uv=np.zeros((2, frames, 2), np.float32),
        ball_vis=np.ones((2, frames), dtype=np.bool_),
        ball_3d=np.zeros((frames, 3), np.float32),
        ball_3d_valid=valid,
        ball_rejection_code=np.zeros(frames, np.uint8),
        human_kp_2d=np.zeros((1, 2, frames, 17, 2), np.float32),
        human_kp_vis=np.ones((1, 2, frames, 17), np.float32),
        player_kp_3d=np.zeros((1, frames, 17, 3), np.float32),
        player_kp_3d_vis=joints_valid,
        player_kp_3d_rejection_code=np.ones((1, frames, 17), np.uint8),
        player_observed=np.ones((1, frames), dtype=np.bool_),
        player_valid=np.ones((1, frames), dtype=np.bool_),
        player_rejection_code=np.zeros((1, frames), np.uint8),
        player_heading_valid=np.zeros((1, frames), dtype=np.bool_),
        player_smpl_valid=np.zeros((1, frames), dtype=np.bool_),
        metadata={
            "scene_schema_version": 4,
            "court_reference": {},
            "ball_events": ball_event_metadata(
                "annotation",
                {
                    "code_commit": "a" * 40,
                    "annotation_sha256": "b" * 64,
                    "labeler": "codex",
                    "labeler_original": "Codex",
                },
                scene_version=4,
            ),
        },
        ball_event_frame=np.array([2, 6, 10, 14], np.int32),
        ball_event_kind=np.array(
            [EVENT_CLASSES_V4.index(k) for k in ("shot", "shot", "bounce", "unknown")],
            np.uint8,
        ),
        ball_event_coverage=np.ones(frames, dtype=np.bool_),
        ball_event_status=np.array([0, 0, 0, 2], np.uint8),
        ball_event_frame_uncertainty=np.array([1, 1, 1, 2], np.uint8),
        ball_event_subtype=np.array(
            [
                EVENT_SUBTYPES_V4.index(s)
                for s in ("serve_overhead", "volley_center", "bounce", "unknown")
            ],
            np.uint8,
        ),
        ball_event_side=np.array([-1, 1, 0, 0], np.int8),
    )

    labels = Annotations.from_scene(scene)
    assert [e.subtype for e in labels.events] == [
        "serve_overhead",
        "volley_center",
        "bounce",
    ]
    assert all(e.in_point for e in labels.events)
    assert labels.unknown_frames == (14,)

    # Unknown frame masks +-2 across all three channels
    _, mask = labels.targets(2.0)
    assert not mask[12:17, :].any()
    assert mask[11, :].all() and mask[17, :].all()

    # Volley condition matches v4 volley_center subtype; serve_overhead splits point spans
    probability: NDArray[np.float32] = np.full((frames, 3), 0.01, dtype=np.float32)
    probability[[2, 6], 0] = 0.9
    probability[10, 1] = 0.9
    probability[14, 0] = 0.9  # near unknown_frame -> near_uncertain, not FP
    rows = score_clip(probability, labels, 60.0)
    metrics = report(rows)
    assert metrics["conditions"]["volley"]["by_class"]["shot"]["recall"] == 1.0
    assert metrics["all"]["by_class"]["shot"]["fp"] == 0
    assert metrics["all"]["by_class"]["shot"]["excluded_predictions"] == 1
