"""Task semantics, PTS interpolation and observed-only supervision."""

from __future__ import annotations

import math
from collections.abc import Iterable
from fractions import Fraction

from src.tasks.base.annotation.artifacts import read_document, verify_reference
from src.tasks.base.annotation.contracts import ArtifactRef, ClipManifest, FrameRange
from src.tasks.base.annotation.editing import ReviewContext
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import load_plan, references
from src.tasks.base.annotation.predictions import PredictionReceipt

from ..contracts import PredictionDocument
from ..validation import validate_predictions
from .contracts import (
    BallFrame,
    BallLabels,
    BallObservation,
    BallReviewPolicy,
    TeacherRow,
)


def inputs(
    layout: AnnotationLayout, context: ReviewContext, manifest: ClipManifest
) -> tuple[ArtifactRef, BallReviewPolicy, PredictionDocument]:
    plan = load_plan(verify_reference(layout, context.plan))
    task = next((t for t in plan.tasks if t.task == "ball"), None)
    if (
        context.task != "ball"
        or task is None
        or task.review is None
        or not plan.allows_adoption
        or plan.annotation_root != str(layout.root)
    ):
        raise ValueError(
            "ball review requires a production policy fixed after the pilot"
        )
    for ref in references(task.review):
        verify_reference(layout, ref)
    if read_document(layout, context.manifest, ClipManifest) != manifest:
        raise ValueError("review manifest differs from its context")
    receipt = read_document(layout, context.prediction, PredictionReceipt)
    if (
        receipt.context.task,
        receipt.context.manifest,
        receipt.context.clip_id,
        receipt.context.model,
        receipt.context.configuration,
    ) != (
        "ball",
        context.manifest,
        manifest.clip_id,
        task.model,
        task.inference,
    ) or receipt.prediction.request_id != receipt.context.request_id:
        raise ValueError("ball prediction differs from its frozen campaign")
    policy: BallReviewPolicy = read_document(
        layout, task.review.policy, BallReviewPolicy
    )
    predictions = validate_predictions(
        layout, receipt.context, manifest, receipt.prediction.artifact
    )
    return task.review.policy, policy, predictions


def ranges(indices: Iterable[int]) -> tuple[FrameRange, ...]:
    result: list[FrameRange] = []
    for index in sorted(set(indices)):
        if result and result[-1].stop == index:
            result[-1] = FrameRange(start=result[-1].start, stop=index + 1)
        else:
            result.append(FrameRange(start=index, stop=index + 1))
    return tuple(result)


def resolved(frame: BallFrame) -> bool:
    decision = frame.decision
    return bool(
        decision.reviewed
        and decision.presence != "unknown"
        and not any(n.blocking for n in decision.notes)
        and not any(
            o.state == "unresolved" or (o.state == "occluded" and o.center is None)
            for o in decision.observations
        )
    )


def observation(frame: BallFrame, track_id: str) -> BallObservation | None:
    return next(
        (o for o in frame.decision.observations if o.track_id == track_id), None
    )


def interpolation_centers(
    frames: tuple[BallFrame, ...] | list[BallFrame],
    manifest: ClipManifest,
    policy: BallReviewPolicy,
    track_id: str,
    start: int,
    end: int,
) -> dict[int, tuple[float, float]]:
    if not 0 <= start < end - 1 < len(frames) - 1:
        raise ValueError(
            "interpolation endpoints lie outside the clip or have no interior"
        )
    if policy.maximum_interpolation_seconds is None:
        raise ValueError("interpolation is disabled by the review policy")
    selected = frames[start : end + 1]
    if any(
        not f.decision.reviewed
        or f.decision.boundaries
        or any(n.blocking for n in f.decision.notes)
        for f in selected
    ):
        raise ValueError(
            "interpolation crosses an unreviewed frame, blocking note or event/track boundary"
        )
    left, right = (
        observation(frames[start], track_id),
        observation(frames[end], track_id),
    )
    if (
        left is None
        or right is None
        or left.state != "visible"
        or right.state != "visible"
        or left.center is None
        or right.center is None
    ):
        raise ValueError(
            "interpolation endpoints must be visible observations of the same track"
        )
    first, last = frames[start].frame.source_pts, frames[end].frame.source_pts
    seconds = (last - first) * Fraction(manifest.time_base)
    if seconds > Fraction(str(policy.maximum_interpolation_seconds)):
        raise ValueError("interpolation exceeds the explicit policy time gap")
    result: dict[int, tuple[float, float]] = {}
    for index in range(start + 1, end):
        ratio = Fraction(frames[index].frame.source_pts - first, last - first)
        x = float(left.center[0] + ratio * (right.center[0] - left.center[0]))
        y = float(left.center[1] + ratio * (right.center[1] - left.center[1]))
        result[index] = (x, y)
    return result


def validate_labels(
    labels: BallLabels,
    context: ReviewContext,
    manifest: ClipManifest,
    policy_ref: ArtifactRef,
    policy: BallReviewPolicy,
    predictions: PredictionDocument,
) -> None:
    if (labels.clip_id, labels.prediction, labels.policy, len(labels.frames)) != (
        manifest.clip_id,
        context.prediction,
        policy_ref,
        len(manifest.frames),
    ):
        raise ValueError("ball labels changed clip, prediction, policy or frame count")
    seen_spans: set[tuple[str, int, int]] = set()
    for frame, native, proposal in zip(
        labels.frames, manifest.frames, predictions.rows, strict=True
    ):
        if frame.frame != native or frame.target != (
            manifest.target_range.start
            <= native.frame_index
            < manifest.target_range.stop
        ):
            raise ValueError(
                "ball labels must preserve every native frame/PTS and target flag"
            )
        for item in frame.decision.observations:
            if item.center is not None and not (
                0 <= item.center[0] <= manifest.width - 1
                and 0 <= item.center[1] <= manifest.height - 1
            ):
                raise ValueError("ball center lies outside native pixels")
            if item.origin == "model_confirmed" and (
                not policy.allow_prediction_confirmation
                or proposal.center is None
                or item.center != proposal.center
            ):
                raise ValueError(
                    "model confirmation must preserve the actual proposal under an allowing policy"
                )
            if item.interpolation is not None:
                bounds = item.interpolation
                if not bounds.start < native.frame_index < bounds.end:
                    raise ValueError("interpolated frame lies outside its endpoints")
                span = (item.track_id, bounds.start, bounds.end)
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                centers = interpolation_centers(labels.frames, manifest, policy, *span)
                for i, expected in centers.items():
                    derived = observation(labels.frames[i], item.track_id)
                    if (
                        derived is None
                        or derived.state != "interpolated"
                        or derived.interpolation != bounds
                        or derived.center is None
                        or not all(
                            math.isclose(a, b, rel_tol=0, abs_tol=1e-6)
                            for a, b in zip(derived.center, expected, strict=True)
                        )
                    ):
                        raise ValueError(
                            "interpolation has a gap, stale endpoint or inconsistent PTS coordinate"
                        )


def teacher_rows(labels: BallLabels) -> tuple[TeacherRow, ...]:
    """Confirmed empty is heatmap-negative; MDD never receives a negative coordinate."""
    result: list[TeacherRow] = []
    for frame in labels.frames:
        decision = frame.decision
        reasons = []
        if not frame.target:
            reasons.append("context_only")
        if not decision.reviewed:
            reasons.append("unreviewed")
        if not resolved(frame):
            reasons.append("unresolved")
        if any(
            o.state in {"interpolated", "occluded", "unresolved"}
            for o in decision.observations
        ):
            reasons.append("estimated_or_unknown_position")
        positive = tuple(
            o.track_id for o in decision.observations if o.state == "visible"
        )
        eligible = not reasons
        heatmap = "ignore" if not eligible else "positive" if positive else "negative"
        coordinate = (
            positive[0]
            if eligible and len(decision.observations) == 1 and len(positive) == 1
            else None
        )
        if eligible and positive and coordinate is None:
            reasons.append("multiple_instances_coordinate_ignored")
        result.append(
            TeacherRow(
                frame_index=frame.frame.frame_index,
                target=frame.target,
                reviewed=decision.reviewed,
                resolved=resolved(frame),
                heatmap=heatmap,
                coordinate_track=coordinate,
                positive_tracks=positive if eligible else (),
                ignore_reasons=tuple(reasons),
            )
        )
    return tuple(result)


def check_acceptance(labels: BallLabels, policy: BallReviewPolicy) -> None:
    targets = [f for f in labels.frames if f.target]
    if (
        sum(f.decision.reviewed for f in targets) / len(targets)
        < policy.minimum_reviewed_target_fraction
    ):
        raise ValueError("reviewed target coverage is below the explicit policy")
    if (
        sum(resolved(f) for f in targets) / len(targets)
        < policy.minimum_resolved_target_fraction
    ):
        raise ValueError("resolved target coverage is below the explicit policy")
    if not any(row.heatmap != "ignore" for row in teacher_rows(labels)):
        raise ValueError("no eligible ball teachers; keep the clip deferred")
