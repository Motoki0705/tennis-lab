from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.analytics.hitter import (
    HitterScene,
    HitterThresholds,
    estimate_hitter,
)
from src.tennis_scene.analytics.hitter_eval import VisualLabel, summarize
from src.tennis_scene.ball_events.observations import PlayerEvidence
from src.utils.schema.player import COCO_KP_IDX

CONTACT = 10
LEFT = COCO_KP_IDX["left_wrist"]
RIGHT = COCO_KP_IDX["right_wrist"]


@pytest.fixture
def scene() -> HitterScene:
    frames = np.arange(21)
    ball = np.column_stack(
        (np.zeros(21), -6 + 0.125 * abs(frames - CONTACT), np.full(21, 1.5))
    ).astype(np.float32)
    roots: NDArray[np.float32] = np.empty((2, 21, 3), dtype=np.float32)
    roots[0] = (4.0, -6.0, 0.9)
    roots[1] = (0.0, -6.9, 0.9)
    joint_valid: NDArray[np.bool_] = np.ones((2, 21, 17), dtype=np.bool_)
    joints = np.broadcast_to(roots[:, :, None, :], (2, 21, 17, 3)).copy()
    joints[1, :, LEFT] = ball  # A perfectly overlapping masked wrist must never win.
    joint_valid[1, :, LEFT] = False
    joints[1, :, RIGHT] = (0.5, -6.0, 1.5)
    return HitterScene(
        ball,
        np.ones(21, dtype=np.bool_),
        PlayerEvidence.from_pose(
            roots,
            np.ones((2, 21), dtype=np.bool_),
            joints,
            joint_valid,
            np.array([41, 7], dtype=np.int32),
        ),
        fps=60.0,
    )


def test_wrist_hit_does_not_confuse_masked_joint_index_or_track_id(
    scene: HitterScene,
) -> None:
    estimate = estimate_hitter(scene, frame=CONTACT, frame_uncertainty=1, court_side=-1)
    assert estimate.resolution_status == "resolved"
    assert (estimate.local_player_index, estimate.local_player_id) == (1, 7)
    assert estimate.selected_landmark == "right_wrist"
    best, other = estimate.candidates
    assert best.contact is not None and other.contact is not None
    assert best.contact.frame == CONTACT
    assert best.contact.distance_m == pytest.approx(0.5)
    assert best.contact.side_alignment == "match"
    assert best.valid_frames_by_landmark["left_wrist"] == []
    assert best.contact.velocity_before_mps == pytest.approx([0, -7.5, 0])
    assert best.contact.velocity_after_mps == pytest.approx([0, 7.5, 0])
    assert best.contact.velocity_change_xy_mps == pytest.approx(15)
    assert best.score is not None and other.score is not None
    assert estimate.margin_to_second == pytest.approx(best.score - other.score)
    repeated = estimate_hitter(scene, frame=CONTACT, frame_uncertainty=1, court_side=-1)
    assert json.dumps(estimate.to_record(), allow_nan=False) == json.dumps(
        repeated.to_record(), allow_nan=False
    )


def test_two_same_side_players_with_identical_evidence_are_ambiguous(
    scene: HitterScene,
) -> None:
    scene.players.positions[0] = scene.players.positions[1]
    scene.players.valid[0] = scene.players.valid[1]
    estimate = estimate_hitter(scene, frame=CONTACT, frame_uncertainty=0, court_side=-1)
    assert estimate.resolution_status == "ambiguous"
    assert estimate.rejection_reason == "same_side_competing_players"
    assert (
        estimate.local_player_index,
        estimate.local_player_id,
        estimate.selected_landmark,
    ) == (None, None, None)
    assert [candidate.local_player_id for candidate in estimate.candidates] == [41, 7]
    assert estimate.margin_to_second == 0


def test_opposing_near_net_players_can_compete_despite_side_annotation(
    scene: HitterScene,
) -> None:
    scene.ball[:, 1] += 6.05
    scene.players.positions[:, :, 0] = np.array([(0, 0.8, 0.9), (0, -0.7, 0.9)])[
        :, None
    ]
    scene.players.positions[0, :, 1:] = (0.55, 0.3, 1.5)
    scene.players.positions[1, :, 2] = (0.45, -0.2, 1.5)
    estimate = estimate_hitter(scene, frame=CONTACT, frame_uncertainty=1, court_side=-1)
    assert estimate.resolution_status == "ambiguous"
    assert estimate.rejection_reason == "net_area_ambiguous"
    assert estimate.local_player_id is None
    contacts = [candidate.contact for candidate in estimate.candidates]
    assert {contact.side_alignment for contact in contacts if contact is not None} == {
        "match",
        "mismatch",
    }


@pytest.mark.parametrize(
    ("missing", "reason"),
    [
        ("ball", "ball_3d_invalid"),
        ("players", "no_valid_player"),
        ("wrists", "root_only_low_confidence"),
        ("roots", "player_side_unavailable"),
    ],
)
def test_missing_observation_keeps_status_and_candidate_evidence(
    scene: HitterScene, missing: str, reason: str
) -> None:
    if missing == "ball":
        scene.ball_valid[:] = False
    elif missing == "players":
        scene.players.valid[:] = False
    elif missing == "wrists":
        scene.players.valid[:, :, 1:] = False
    else:
        scene.players.valid[:, :, 0] = False
    estimate = estimate_hitter(scene, frame=CONTACT, frame_uncertainty=1, court_side=-1)
    assert estimate.resolution_status == "unresolved"
    assert estimate.rejection_reason == reason
    assert estimate.local_player_id is None
    assert len(estimate.candidates) == 2
    best = estimate.candidates[0]
    if missing in ("ball", "players"):
        assert best.score is None and best.contact is None
    elif missing == "wrists":
        assert best.contact is not None and best.contact.landmark == "root"
    else:
        assert best.contact is not None and best.contact.root_y_m is None


def test_court_side_conflict_cannot_select_a_player(scene: HitterScene) -> None:
    estimate = estimate_hitter(scene, frame=CONTACT, frame_uncertainty=1, court_side=1)
    assert estimate.resolution_status == "unresolved"
    assert estimate.rejection_reason == "side_mismatch"
    assert estimate.local_player_id is None
    assert estimate.evidence_json["best_contact"]["player_side"] == -1
    estimate = estimate_hitter(scene, frame=CONTACT, frame_uncertainty=1, court_side=0)
    assert estimate.local_player_id == 7


def test_uncertainty_window_recovers_later_contact_but_wide_uncertainty_abstains(
    scene: HitterScene,
) -> None:
    later = CONTACT + 5
    scene.players.valid[:] = False
    scene.players.valid[1, later, [0, 2]] = True
    scene.players.positions[1, later, 2] = scene.ball[later] + (0.5, 0, 0)
    narrow = estimate_hitter(scene, frame=CONTACT, frame_uncertainty=1, court_side=-1)
    assert narrow.rejection_reason == "no_valid_player"
    wide = estimate_hitter(scene, frame=CONTACT, frame_uncertainty=5, court_side=-1)
    assert wide.local_player_id == 7
    assert wide.evidence_json["best_contact"]["frame"] == later
    assert wide.evidence_json["best_contact"]["frame_offset"] == 5
    assert wide.evidence_json["ball_valid_frames"] == list(range(5, 16))
    uncertain = estimate_hitter(
        scene, frame=CONTACT, frame_uncertainty=7, court_side=-1
    )
    assert uncertain.resolution_status == "ambiguous"
    assert uncertain.rejection_reason == "high_time_uncertainty"


def test_ball_motion_stops_at_adjacent_invalid_frame(scene: HitterScene) -> None:
    scene.ball_valid[CONTACT - 1] = False
    estimate = estimate_hitter(scene, frame=CONTACT, frame_uncertainty=0, court_side=-1)
    contact = estimate.candidates[0].contact
    assert contact is not None and contact.frame == CONTACT
    assert contact.velocity_before_mps is None
    assert contact.velocity_after_mps == pytest.approx([0, 7.5, 0])
    assert contact.velocity_change_xy_mps is None
    assert CONTACT - 2 in estimate.evidence_json["motion_observed_frames"]
    assert CONTACT - 1 not in estimate.evidence_json["motion_observed_frames"]


def test_wrist_reach_boundary_is_inclusive(scene: HitterScene) -> None:
    scene.players.positions[1, :, 0] = (0, -9, 0.9)
    rules = HitterThresholds(wrist_reach_m=1.0)
    scene.players.positions[1, :, 2] = (1, -6, 1.5)
    at_limit = estimate_hitter(
        scene, frame=CONTACT, frame_uncertainty=0, court_side=-1, thresholds=rules
    )
    assert at_limit.local_player_id == 7
    scene.players.positions[1, :, 2, 0] = np.nextafter(np.float32(1), np.float32(2))
    beyond_limit = estimate_hitter(
        scene, frame=CONTACT, frame_uncertainty=0, court_side=-1, thresholds=rules
    )
    assert beyond_limit.rejection_reason == "too_far"
    assert beyond_limit.local_player_id is None


def test_metrics_keep_unknown_truth_and_abstentions_in_the_right_denominators(
    scene: HitterScene,
) -> None:
    estimate = estimate_hitter(scene, frame=CONTACT, frame_uncertainty=1, court_side=-1)
    label = VisualLabel("v/c", "e1", CONTACT, "smash", "val", 7, 1, "confirmed", "cam0")
    wrong = replace(
        label, source_event_id="e2", gt_local_player_id=41, gt_local_player_index=0
    )
    unknown = replace(
        label,
        source_event_id="e3",
        gt_local_player_id=None,
        gt_local_player_index=None,
        visibility_status="unidentifiable",
    )
    abstained = replace(label, source_event_id="e4")
    scene.ball_valid[:] = False
    no_ball = estimate_hitter(scene, frame=CONTACT, frame_uncertainty=1, court_side=-1)
    result = summarize(
        (
            (label, estimate),
            (wrong, estimate),
            (unknown, estimate),
            (abstained, no_ball),
        )
    )
    assert result["total_shots"] == 4
    assert result["evaluable_shots"] == 3
    assert result["coverage"] == 3 / 4
    assert result["resolved_accuracy"] == 1 / 2
    assert result["overall_accuracy"] == 1 / 3
    assert result["false_resolution_rate"] == 1 / 3
    assert result["false_resolution_rate_among_evaluable_resolved"] == 1 / 2
    assert result["rejection_reasons"] == {"ball_3d_invalid": 1}
    [failure] = result["false_resolutions"]
    assert failure["source_event_id"] == "e2"
    assert failure["estimated_local_player_id"] == 7
    assert failure["gt_candidate"]["local_player_index"] == 0
