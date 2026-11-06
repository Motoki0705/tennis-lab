"""Measured ball-review coverage and cost, and explicit continuation of stopped reviews.

Coverage counts only target frames whose decision is bound to rollout-verified
image bytes, so a worker's own claims never count as progress. Rates come from
ended attempts of the same campaign; projections extrapolate them and are not
promises.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from src.tasks.ball_detection.annotation.review.contracts import (
    BallLabels,
    BallReviewPolicy,
)
from src.tasks.ball_detection.annotation.review.progress import (
    ClipReviewProgress,
    clip_progress,
)
from src.tasks.ball_detection.annotation.review.worker import (
    BallReviewServices,
    ImageProof,
    frame_proof_states,
    image_proof,
)
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    ClipManifest,
    Digest,
    Document,
)
from src.tasks.base.annotation.editing import ReviewContext, Revision, RevisionStore
from src.tasks.base.annotation.handoff import read_handoff
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan
from src.tasks.base.annotation.telemetry.capture import verify_report
from src.tasks.base.annotation.workers.contracts import (
    AttemptEvidence,
    ExitReceipt,
    Launch,
    WorkRequest,
)
from src.tasks.base.annotation.workers.ledger import WorkerLedger
from src.tasks.base.annotation.workers.review.contracts import ReviewWorkerInput

from .contracts import ContinueReview, RecoveryInput
from .reviews import DECISION_STATES, requests

if TYPE_CHECKING:
    from .controller import Controller

TOKEN_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)
# A later attempt may continue saved progress only after these stops; failures,
# validation errors and code-update cancellations keep their explicit retry path.
CONTINUABLE = frozenset({"deferred", "timed_out"})


class AttemptMeasure(Document):
    attempt: str = Field(min_length=1)
    number: int = Field(ge=1)
    state: str
    reason: str | None
    # Null without an exit receipt, e.g. an attempt cancelled before it started.
    elapsed_seconds: float | None = Field(ge=0)
    # Null without telemetry; a value is null when its total is incomplete.
    tokens: dict[str, int | None] | None
    compactions: int | None = Field(ge=0)
    verified_before: int = Field(ge=0)
    verified_after: int = Field(ge=0)

    @property
    def verified_gain(self) -> int:
        return self.verified_after - self.verified_before


class ClipReviewStatus(Document):
    clip_id: Digest
    request: ArtifactRef
    state: str
    reason: str | None
    current_attempt: str | None
    adopted: bool
    target_frames: int = Field(ge=1)
    required_reviewed_frames: int = Field(ge=0)
    # Null until the first attempt initializes labels; nothing is reviewed then.
    progress: ClipReviewProgress | None
    attempts: tuple[AttemptMeasure, ...]

    @property
    def verified_frames(self) -> int:
        return 0 if self.progress is None else self.progress.verified_frames

    @property
    def remaining_frames(self) -> int:
        if self.adopted:
            return 0
        return max(0, self.required_reviewed_frames - self.verified_frames)


class ReviewThroughput(Document):
    """Rates over attempts with an exit receipt; a total is null, not partial."""

    measured_attempts: int = Field(ge=0)
    worker_seconds: float = Field(ge=0)
    verified_frames_gained: int
    tokens: dict[str, int | None]
    attempts_missing_tokens: dict[str, int]
    verified_frames_per_worker_hour: float | None
    tokens_per_verified_frame: dict[str, float | None]


class ReviewProjection(Document):
    remaining_target_frames: int = Field(ge=0)
    worker_hours: float | None
    tokens: dict[str, float | None]


class ReviewProgressReport(Document):
    schema_version: Literal["annotation.review-progress.v1"] = (
        "annotation.review-progress.v1"
    )
    plan: ArtifactRef
    clips: tuple[ClipReviewStatus, ...]
    target_frames: int = Field(ge=0)
    verified_frames: int = Field(ge=0)
    adopted_clips: int = Field(ge=0)
    throughput: ReviewThroughput
    projection: ReviewProjection


class ContinuationDecision(Document):
    clip_id: Digest
    request: ArtifactRef
    state: str
    after_attempt: str | None
    action: Literal["continue", "hold"]
    reason: str = Field(min_length=1)
    recovery: ArtifactRef | None = None


def required_frames(policy: BallReviewPolicy, target_frames: int) -> int:
    """Smallest reviewed count whose fraction meets the frozen policy."""
    result: int = math.ceil(
        round(policy.minimum_reviewed_target_fraction * target_frames, 9)
    )
    return result


def _labels(layout: AnnotationLayout, revision: ArtifactRef) -> BallLabels:
    return read_document(
        layout, read_document(layout, revision, Revision).payload, BallLabels
    )


def _start_revision(layout: AnnotationLayout, attempt: AttemptEvidence) -> ArtifactRef:
    launch = read_document(layout, attempt.launch, Launch)
    body = read_document(
        layout,
        annotation_reference(layout, Path(launch.directory) / "request.json"),
        ReviewWorkerInput,
    )
    if body.work != launch.request:
        raise ValueError("attempt review input belongs to another worker request")
    revision: ArtifactRef = read_handoff(layout, body.initial_handoff).revision
    return revision


def _measure(
    layout: AnnotationLayout,
    attempt: AttemptEvidence,
    *,
    before: int,
    after: int,
) -> AttemptMeasure:
    launch = read_document(layout, attempt.launch, Launch)
    elapsed = (
        None
        if attempt.execution is None
        else read_document(layout, attempt.execution, ExitReceipt).elapsed_seconds
    )
    tokens: dict[str, int | None] | None = None
    compactions = None
    if attempt.telemetry is not None:
        report = verify_report(layout, attempt.telemetry)
        tokens = {key: report.totals.get(key) for key in TOKEN_KEYS}
        compactions = report.compactions
    return AttemptMeasure(
        attempt=launch.attempt,
        number=attempt.number,
        state=attempt.state,
        reason=attempt.reason,
        elapsed_seconds=elapsed,
        tokens=tokens,
        compactions=compactions,
        verified_before=before,
        verified_after=after,
    )


def clip_status(
    layout: AnnotationLayout,
    plan: CampaignPlan,
    request_ref: ArtifactRef,
) -> ClipReviewStatus:
    """Measure one clip from its immutable attempt files and current revision."""
    request = read_document(layout, request_ref, WorkRequest)
    context = read_document(layout, request.payload, ReviewContext)
    manifest = read_document(layout, context.manifest, ClipManifest)
    task = next(task for task in plan.tasks if task.task == "ball")
    if task.review is None:
        raise ValueError("review progress requires the frozen ball review policy")
    policy = read_document(layout, task.review.policy, BallReviewPolicy)
    target = manifest.target_range
    target_frames = target.stop - target.start
    ledger = WorkerLedger(layout)
    row = ledger.task(request.work_id)
    history = ledger.history(plan, request)
    progress = None
    adopted = False
    measures: list[AttemptMeasure] = []
    if history:
        head, adoption = RevisionStore(layout).current(
            request.payload, BallReviewServices()
        )
        adopted = adoption is not None
        proof: ImageProof = image_proof(layout, context, manifest.clip_id)

        def verified(revision: ArtifactRef) -> ClipReviewProgress:
            labels = _labels(layout, revision)
            return clip_progress(labels, frame_proof_states(labels, proof), target)

        starts = [_start_revision(layout, attempt) for attempt in history]
        ends = [*starts[1:], head]
        counts = {
            revision: verified(revision).verified_frames for revision in {*starts, head}
        }
        progress = verified(head)
        measures = [
            _measure(
                layout,
                attempt,
                before=counts[start],
                after=counts[end],
            )
            for attempt, start, end in zip(history, starts, ends, strict=True)
        ]
    return ClipReviewStatus(
        clip_id=manifest.clip_id,
        request=request_ref,
        state=row["state"],
        reason=row["reason"],
        current_attempt=row["current_attempt"],
        adopted=adopted,
        target_frames=target_frames,
        required_reviewed_frames=required_frames(policy, target_frames),
        progress=progress,
        attempts=tuple(measures),
    )


def _throughput(clips: tuple[ClipReviewStatus, ...]) -> ReviewThroughput:
    measured = [
        attempt
        for clip in clips
        for attempt in clip.attempts
        if attempt.elapsed_seconds is not None
    ]
    seconds = sum(attempt.elapsed_seconds or 0.0 for attempt in measured)
    gained = sum(attempt.verified_gain for attempt in measured)
    missing = {
        key: sum(
            attempt.tokens is None or attempt.tokens.get(key) is None
            for attempt in measured
        )
        for key in TOKEN_KEYS
    }
    tokens: dict[str, int | None] = {
        key: sum((attempt.tokens or {}).get(key) or 0 for attempt in measured)
        if measured and not missing[key]
        else None
        for key in TOKEN_KEYS
    }
    return ReviewThroughput(
        measured_attempts=len(measured),
        worker_seconds=seconds,
        verified_frames_gained=gained,
        tokens=tokens,
        attempts_missing_tokens=missing,
        verified_frames_per_worker_hour=gained / (seconds / 3600)
        if seconds > 0 and gained > 0
        else None,
        tokens_per_verified_frame={
            key: value / gained if value is not None and gained > 0 else None
            for key, value in tokens.items()
        },
    )


def review_progress(
    layout: AnnotationLayout, plan: CampaignPlan, plan_ref: ArtifactRef
) -> ReviewProgressReport:
    clips = tuple(
        clip_status(layout, plan, ref) for ref in requests(layout, plan, plan_ref)
    )
    throughput = _throughput(clips)
    remaining = sum(clip.remaining_frames for clip in clips)
    rate = throughput.verified_frames_per_worker_hour
    return ReviewProgressReport(
        plan=plan_ref,
        clips=clips,
        target_frames=sum(clip.target_frames for clip in clips),
        verified_frames=sum(clip.verified_frames for clip in clips),
        adopted_clips=sum(clip.adopted for clip in clips),
        throughput=throughput,
        projection=ReviewProjection(
            remaining_target_frames=remaining,
            worker_hours=remaining / rate if rate else None,
            tokens={
                key: remaining * value if value is not None else None
                for key, value in throughput.tokens_per_verified_frame.items()
            },
        ),
    )


def save_review_progress(controller: Controller) -> tuple[ArtifactRef, dict[str, Any]]:
    """Persist the full measurement and return a bounded summary for the caller."""
    if controller.production is None:
        raise ValueError("review progress requires a production campaign")
    report = review_progress(
        controller.layout, controller.plan, controller.runtime.plan
    )
    reference = write_document(
        controller.layout,
        controller.layout.path(
            "campaigns",
            controller.plan.campaign_id,
            "reports",
            "review-progress",
            report.sha256 + ".json",
        ),
        report,
    )
    attention = [
        {
            "clip_id": clip.clip_id,
            "state": clip.state,
            "reason": clip.reason,
            "attempts": len(clip.attempts),
            "verified_frames": clip.verified_frames,
            "required_reviewed_frames": clip.required_reviewed_frames,
            "latest_verified_gain": clip.attempts[-1].verified_gain
            if clip.attempts
            else None,
        }
        for clip in report.clips
        if clip.state in DECISION_STATES
    ]
    summary = {
        "clips": len(report.clips),
        "adopted_clips": report.adopted_clips,
        "target_frames": report.target_frames,
        "verified_frames": report.verified_frames,
        "throughput": report.throughput.model_dump(mode="json"),
        "projection": report.projection.model_dump(mode="json"),
        "needs_decision": attention,
    }
    return reference, summary


def continuation(clip: ClipReviewStatus, max_attempts: int) -> tuple[bool, str]:
    """Allow another attempt only after a stop that verified new coverage."""
    if clip.state not in CONTINUABLE:
        return False, f"{clip.state} requires an explicit decision"
    if not clip.attempts:
        raise ValueError("a stopped review has no registered attempt")
    if len(clip.attempts) >= max_attempts:
        return False, f"attempt limit {max_attempts} reached"
    latest = clip.attempts[-1]
    if latest.verified_gain <= 0:
        return False, (
            f"attempt {latest.attempt} verified no new target frames; "
            "read its reason before deciding"
        )
    return True, (
        f"attempt {latest.attempt} verified {latest.verified_gain} new target frames "
        f"({clip.verified_frames}/{clip.required_reviewed_frames} required)"
    )


def continue_reviews(
    controller: Controller,
    *,
    reason: str,
    max_attempts: int,
    dry_run: bool,
) -> tuple[ContinuationDecision, ...]:
    """Record one continuation per eligible stopped review; never start processes.

    Each continuation is an ordinary recovery request. The scheduler later starts
    the authorized attempts within the shared capacity.
    """
    from .recovery import recover

    if controller.production is None:
        raise ValueError("review continuation requires a production campaign")
    if not reason.strip() or max_attempts < 2:
        raise ValueError("continuation needs a reason and an attempt limit above one")
    report = review_progress(
        controller.layout, controller.plan, controller.runtime.plan
    )
    decisions = []
    for clip in report.clips:
        if clip.state not in DECISION_STATES:
            continue
        allowed, detail = continuation(clip, max_attempts)
        recovery = None
        if allowed and not dry_run:
            assert clip.current_attempt is not None
            recovery = recover(
                controller,
                RecoveryInput(
                    plan=controller.runtime.plan,
                    reason=f"{reason.strip()} [{detail}]",
                    operation=ContinueReview(
                        request=clip.request,
                        after_attempt=clip.current_attempt,
                        max_attempts=max_attempts,
                    ),
                ),
            )
        decisions.append(
            ContinuationDecision(
                clip_id=clip.clip_id,
                request=clip.request,
                state=clip.state,
                after_attempt=clip.current_attempt,
                action="continue" if allowed else "hold",
                reason=detail,
                recovery=recovery,
            )
        )
    return tuple(decisions)
