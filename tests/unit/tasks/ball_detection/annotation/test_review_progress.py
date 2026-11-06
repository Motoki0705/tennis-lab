"""Clip-wide review coverage that keeps a page boundary from ending the work."""

from __future__ import annotations

import pytest

from src.tasks.ball_detection.annotation.review.contracts import (
    BallFrame,
    BallLabels,
    BallObservation,
    FrameDecision,
)
from src.tasks.ball_detection.annotation.review.progress import (
    ProofState,
    clip_progress,
    first_scope,
    progress_action,
)
from src.tasks.ball_detection.annotation.review.worker import attempt_time_limit
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    FrameMap,
    FrameRange,
    content_digest,
)

TARGET = FrameRange(start=10, stop=150)
EVIDENCE = content_digest("image")


def _ref(name: str) -> ArtifactRef:
    return ArtifactRef(
        role="annotation",
        path=f"fixture/{name}",
        digest=content_digest(name),
        size_bytes=1,
    )


def _decision(kind: str) -> FrameDecision:
    if kind == "unreviewed":
        return FrameDecision(
            reviewed=False,
            presence="unknown",
            observations=(),
            evidence=(),
            notes=(),
            boundaries=(),
        )
    if kind == "unknown":
        return FrameDecision(
            reviewed=True,
            presence="unknown",
            observations=(),
            evidence=(EVIDENCE,),
            notes=(),
            boundaries=(),
        )
    return FrameDecision(
        reviewed=True,
        presence="present",
        observations=(
            BallObservation(
                track_id="ball", state="visible", center=(4.0, 5.0), origin="manual"
            ),
        ),
        evidence=(EVIDENCE,),
        notes=(),
        boundaries=(),
    )


def _labels(decisions: dict[int, str], frames: int = 160) -> BallLabels:
    return BallLabels(
        clip_id=content_digest("clip"),
        prediction=_ref("prediction"),
        policy=_ref("policy"),
        images=(),
        frames=tuple(
            BallFrame(
                frame=FrameMap(
                    frame_index=index,
                    source_frame_index=index,
                    source_pts=index * 10,
                    clip_pts=index * 10,
                    duration_pts=10,
                ),
                target=TARGET.start <= index < TARGET.stop,
                decision=_decision(decisions.get(index, "unreviewed")),
                review_actor=None
                if decisions.get(index, "unreviewed") == "unreviewed"
                else _ref("launch"),
            )
            for index in range(frames)
        ),
    )


def _states(
    labels: BallLabels, overrides: dict[int, ProofState]
) -> dict[int, ProofState]:
    return {
        frame.frame.frame_index: overrides.get(
            frame.frame.frame_index,
            "verified" if frame.decision.reviewed else "unreviewed",
        )
        for frame in labels.frames
    }


def test_progress_counts_target_frames_and_reconfirmations_as_remaining_work() -> None:
    reviewed = {index: "visible" for index in range(10, 74)}
    reviewed.update({5: "visible", 74: "unknown", 75: "visible"})
    labels = _labels(reviewed)
    # Frame 20's decision lost its rollout proof; frame 75 belongs to a live attempt.
    states = _states(labels, {20: "missing", 75: "pending_current_attempt"})

    progress = clip_progress(labels, states, TARGET)

    assert (progress.target_frames, progress.reviewed_frames) == (140, 66)
    assert progress.verified_frames == 64  # 10..73 minus 20, plus unresolved 74
    assert (progress.missing_proof_frames, progress.pending_proof_frames) == (1, 1)
    assert progress.unresolved_frames == 1
    # The missing proof is the first remaining review, before unreviewed frame 76.
    assert progress.next_review_scope == FrameRange(start=20, stop=84)
    assert progress.next_unresolved_scope == FrameRange(start=74, stop=138)
    assert first_scope(progress) == FrameRange(start=20, stop=84)


def test_first_scope_moves_past_verified_work_and_stops_at_the_target_end() -> None:
    labels = _labels({index: "visible" for index in range(10, 140)})
    progress = clip_progress(labels, _states(labels, {}), TARGET)

    assert progress.next_review_scope == FrameRange(start=140, stop=150)
    assert first_scope(progress) == FrameRange(start=140, stop=150)

    done = _labels({**{index: "visible" for index in range(10, 150)}, 30: "unknown"})
    finished = clip_progress(done, _states(done, {}), TARGET)
    assert finished.next_review_scope is None
    assert first_scope(finished) == FrameRange(start=30, stop=94)
    resolved = _labels({index: "visible" for index in range(10, 150)})
    complete = clip_progress(resolved, _states(resolved, {}), TARGET)
    assert first_scope(complete) == FrameRange(start=10, stop=74)


def test_progress_action_names_the_whole_interval_and_next_scope() -> None:
    labels = _labels({index: "visible" for index in range(10, 74)})
    progress = clip_progress(labels, _states(labels, {}), TARGET)

    action = progress_action(progress)

    assert action.startswith(
        "Clip-wide target [10,150): 64/140 reviewed, 0 need re-confirmation, "
        "0 unresolved."
    )
    assert "First frames still needing review: review_handoff scope [74,138)." in action
    assert action.endswith("This page is a view, not the unit of work.")
    assert len(action) <= 500
    finished = _labels({index: "visible" for index in range(10, 150)})
    done = progress_action(clip_progress(finished, _states(finished, {}), TARGET))
    assert "Every target frame has a decision" in done and "scope [" not in done


def test_progress_rejects_a_target_that_differs_from_the_labels() -> None:
    labels = _labels({})
    with pytest.raises(ValueError, match="target"):
        clip_progress(labels, _states(labels, {}), FrameRange(start=11, stop=150))
    with pytest.raises(ValueError, match="target"):
        clip_progress(labels, _states(labels, {}), FrameRange(start=10, stop=149))


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(14_400.0, (240.0, 216.0)), (1_800.0, (30.0, 25.0)), (20.0, (20 / 60, 10 / 60))],
)
def test_time_limit_reserves_time_to_write_the_result(
    seconds: float, expected: tuple[float, float]
) -> None:
    assert attempt_time_limit(seconds) == pytest.approx(expected)


@pytest.mark.parametrize("seconds", [0.0, -1.0, float("inf"), float("nan")])
def test_time_limit_rejects_unbounded_or_empty_limits(seconds: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        attempt_time_limit(seconds)
