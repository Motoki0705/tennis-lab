"""Preserve native rows and candidate lineage; derive only complete observed teachers."""

from __future__ import annotations

from collections.abc import Iterable

from src.tasks.base.annotation.artifacts import read_document, verify_reference
from src.tasks.base.annotation.contracts import ArtifactRef, ClipManifest, FrameRange
from src.tasks.base.annotation.editing import ReviewContext
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import load_plan, references
from src.tasks.base.annotation.predictions import PredictionReceipt

from ..contracts import PlayerPredictions
from ..inference import PlayerPredictionAdapter
from .contracts import (
    PlayerFrame,
    PlayerLabels,
    PlayerReviewPolicy,
    TeacherRow,
    manual_observation_id,
)


def inputs(
    layout: AnnotationLayout, context: ReviewContext, manifest: ClipManifest
) -> tuple[ArtifactRef, PlayerReviewPolicy, PlayerPredictions]:
    plan = load_plan(verify_reference(layout, context.plan))
    task = next((t for t in plan.tasks if t.task == "player"), None)
    if (
        context.task != "player"
        or task is None
        or task.review is None
        or not plan.allows_adoption
        or plan.annotation_root != str(layout.root)
    ):
        raise ValueError(
            "player review requires a production policy fixed after the pilot"
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
        "player",
        context.manifest,
        manifest.clip_id,
        task.model,
        task.inference,
    ) or receipt.prediction.request_id != receipt.context.request_id:
        raise ValueError("player prediction differs from its frozen campaign")
    PlayerPredictionAdapter().validate(
        layout, receipt.context, manifest, receipt.prediction.artifact
    )
    return (
        task.review.policy,
        read_document(layout, task.review.policy, PlayerReviewPolicy),
        read_document(layout, receipt.prediction.artifact, PlayerPredictions),
    )


def ranges(indices: Iterable[int]) -> tuple[FrameRange, ...]:
    result: list[FrameRange] = []
    for index in sorted(set(indices)):
        if result and result[-1].stop == index:
            result[-1] = FrameRange(start=result[-1].start, stop=index + 1)
        else:
            result.append(FrameRange(start=index, stop=index + 1))
    return tuple(result)


def resolved(frame: PlayerFrame) -> bool:
    d = frame.decision
    return bool(
        d.reviewed
        and d.coverage == "complete"
        and not any(n.blocking for n in d.notes)
        and not any(
            o.classification == "unresolved" or o.box_state == "unresolved"
            for o in d.observations
        )
    )


def validate_labels(
    labels: PlayerLabels,
    context: ReviewContext,
    manifest: ClipManifest,
    policy_ref: ArtifactRef,
    policy: PlayerReviewPolicy,
    predictions: PlayerPredictions,
) -> None:
    if (labels.clip_id, labels.prediction, labels.policy, len(labels.frames)) != (
        manifest.clip_id,
        context.prediction,
        policy_ref,
        len(manifest.frames),
    ):
        raise ValueError(
            "player labels changed clip, prediction, policy or frame count"
        )
    seen: set[str] = set()
    for frame, native, proposals in zip(
        labels.frames, manifest.frames, predictions.rows, strict=True
    ):
        if frame.frame != native or frame.target != (
            manifest.target_range.start
            <= native.frame_index
            < manifest.target_range.stop
        ):
            raise ValueError(
                "player labels must preserve every native frame/PTS and target flag"
            )
        raw = {p.observation_id: p for p in proposals.observations}
        referenced = [
            o.proposal_id
            for o in frame.decision.observations
            if o.proposal_id is not None
        ]
        if frame.decision.reviewed and (
            len(set(referenced)) != len(referenced) or set(referenced) != set(raw)
        ):
            raise ValueError(
                "reviewed frames must explicitly account for every original candidate"
            )
        for item in frame.decision.observations:
            if item.observation_id in seen:
                raise ValueError(
                    "player observation identity must be unique across the clip"
                )
            seen.add(item.observation_id)
            if item.proposal_id is not None:
                if (
                    item.proposal_id not in raw
                    or item.observation_id != item.proposal_id
                ):
                    raise ValueError(
                        "observation does not retain its original frame candidate identity"
                    )
            elif item.observation_id != manual_observation_id(
                labels.clip_id, native.frame_index, str(item.manual_id)
            ):
                raise ValueError(
                    "manual observation identity differs from its clip/frame/local key"
                )
            if item.box is not None and (
                item.box.x2 > manifest.width or item.box.y2 > manifest.height
            ):
                raise ValueError("player bbox lies outside native image edges")
            if item.origin == "model_confirmed":
                proposal = raw[item.proposal_id]  # identity was checked above
                if (
                    not policy.allow_prediction_confirmation
                    or item.box != proposal.box
                    or item.confidence != proposal.confidence
                ):
                    raise ValueError(
                        "model confirmation must preserve the exact proposal bbox and score under an allowing policy"
                    )


def teacher_rows(labels: PlayerLabels) -> tuple[TeacherRow, ...]:
    rows = []
    for frame in labels.frames:
        d = frame.decision
        reasons = []
        if not frame.target:
            reasons.append("context_only")
        if not d.reviewed:
            reasons.append("unreviewed")
        if not resolved(frame):
            reasons.append("incomplete_or_unresolved")
        if any(o.box_state == "estimated" for o in d.observations):
            reasons.append("estimated_bbox")
        ids = tuple(
            o.observation_id for o in d.observations if o.classification == "player"
        )
        rows.append(
            TeacherRow(
                frame_index=frame.frame.frame_index,
                target=frame.target,
                reviewed=d.reviewed,
                resolved=resolved(frame),
                supervision="ignore" if reasons else "positive" if ids else "negative",
                observation_ids=() if reasons else ids,
                ignore_reasons=tuple(reasons),
            )
        )
    return tuple(rows)


def check_acceptance(labels: PlayerLabels, policy: PlayerReviewPolicy) -> None:
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
    if not any(row.supervision != "ignore" for row in teacher_rows(labels)):
        raise ValueError("no eligible player teachers; keep the clip deferred")
