"""Clip-wide review coverage from saved decisions and their image-proof state.

Workers see bounded handoff pages; this summary keeps the whole target interval
visible so a page boundary is never mistaken for the end of the work.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from src.tasks.base.annotation.contracts import Document, FrameRange

from .contracts import BallLabels
from .validation import resolved

ProofState = Literal["unreviewed", "verified", "pending_current_attempt", "missing"]

# Mirrors the handoff page bound; a scope is a view, not a unit of work.
HANDOFF_FRAMES = 64


class ClipReviewProgress(Document):
    """Counts cover target frames only; context frames never need review."""

    target: FrameRange
    target_frames: int = Field(ge=1)
    reviewed_frames: int = Field(ge=0)
    verified_frames: int = Field(ge=0)
    pending_proof_frames: int = Field(ge=0)
    missing_proof_frames: int = Field(ge=0)
    unresolved_frames: int = Field(ge=0)
    next_review_scope: FrameRange | None
    next_unresolved_scope: FrameRange | None


def _scope(start: int, target: FrameRange) -> FrameRange:
    return FrameRange(start=start, stop=min(target.stop, start + HANDOFF_FRAMES))


def clip_progress(
    labels: BallLabels, states: Mapping[int, ProofState], target: FrameRange
) -> ClipReviewProgress:
    """Count the review state of every target frame.

    A decision whose image proof is ``missing`` still needs review: it cannot be
    adopted until the frame is viewed again and its decision replaced.
    """
    frames = labels.frames[target.start : target.stop]
    if target.stop > len(labels.frames) or any(not frame.target for frame in frames):
        raise ValueError("progress target differs from the labelled target frames")
    if any(frame.target for frame in labels.frames[: target.start]) or any(
        frame.target for frame in labels.frames[target.stop :]
    ):
        raise ValueError("labels mark target frames outside the progress target")
    needs_review = [
        f.frame.frame_index
        for f in frames
        if not f.decision.reviewed or states[f.frame.frame_index] == "missing"
    ]
    unresolved = [
        f.frame.frame_index
        for f in frames
        if f.decision.reviewed
        and states[f.frame.frame_index] != "missing"
        and not resolved(f)
    ]
    counts = {
        state: sum(states[f.frame.frame_index] == state for f in frames)
        for state in ("verified", "pending_current_attempt", "missing")
    }
    return ClipReviewProgress(
        target=target,
        target_frames=len(frames),
        reviewed_frames=sum(f.decision.reviewed for f in frames),
        verified_frames=counts["verified"],
        pending_proof_frames=counts["pending_current_attempt"],
        missing_proof_frames=counts["missing"],
        unresolved_frames=len(unresolved),
        next_review_scope=_scope(needs_review[0], target) if needs_review else None,
        next_unresolved_scope=_scope(unresolved[0], target) if unresolved else None,
    )


def first_scope(progress: ClipReviewProgress) -> FrameRange:
    """Start a worker where work remains instead of at the reviewed beginning."""
    return (
        progress.next_review_scope
        or progress.next_unresolved_scope
        or _scope(progress.target.start, progress.target)
    )


def progress_action(progress: ClipReviewProgress) -> str:
    """One bounded handoff action that keeps the whole target interval in view."""
    target = progress.target
    text = (
        f"Clip-wide target [{target.start},{target.stop}): "
        f"{progress.reviewed_frames}/{progress.target_frames} reviewed, "
        f"{progress.missing_proof_frames} need re-confirmation, "
        f"{progress.unresolved_frames} unresolved. "
    )
    scope = progress.next_review_scope
    if scope is not None:
        text += (
            f"First frames still needing review: review_handoff scope "
            f"[{scope.start},{scope.stop}). Continue scope by scope until the whole "
            "target interval satisfies the frozen thresholds. "
        )
    else:
        text += "Every target frame has a decision; check the frozen thresholds. "
    return text + "This page is a view, not the unit of work."
