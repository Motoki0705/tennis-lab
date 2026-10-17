"""Batch operations return a complete candidate; the shared store commits it atomically."""

from __future__ import annotations

from src.tasks.base.annotation.contracts import ArtifactRef, ClipManifest
from src.tasks.base.annotation.editing import ChangeResult, ChangeSummary
from src.tasks.base.annotation.editing.contracts import DeltaItem

from ..contracts import BallPredictions
from .contracts import (
    BallEdits,
    BallFrame,
    BallLabels,
    BallObservation,
    BallReviewPolicy,
    ConfirmAbsence,
    ConfirmPredictions,
    FrameDecision,
    Interpolate,
    ReplaceFrame,
)
from .evidence import Catalog
from .validation import interpolation_centers, observation, ranges, resolved


def brief(frame: BallFrame) -> str:
    d = frame.decision
    states = ",".join(f"{o.track_id}:{o.state}" for o in d.observations)
    return f"reviewed={d.reviewed} presence={d.presence} notes={len(d.notes)} {states}"[
        :200
    ]


def apply_operations(
    labels: BallLabels,
    edits: BallEdits,
    manifest: ClipManifest,
    policy: BallReviewPolicy,
    predictions: BallPredictions,
    saved: Catalog,
    actor: ArtifactRef,
) -> ChangeResult:
    frames = list(labels.frames)
    for operation in edits.operations:
        if isinstance(operation, ReplaceFrame):
            index = operation.frame_index
            if (
                index >= len(frames)
                or frames[index].sha256 != operation.expected_frame_digest
            ):
                raise ValueError(
                    "explicit replacement must acknowledge the exact current frame"
                )
            frames[index] = frames[index].model_copy(
                update={
                    "decision": operation.decision,
                    "review_actor": actor if operation.decision.reviewed else None,
                }
            )
        elif isinstance(operation, (ConfirmPredictions, ConfirmAbsence)):
            if operation.scope.stop > len(frames):
                raise ValueError("confirmation range lies outside the clip")
            if (
                isinstance(operation, ConfirmPredictions)
                and not policy.allow_prediction_confirmation
            ):
                raise ValueError("bulk prediction confirmation is disabled by policy")
            for index in range(operation.scope.start, operation.scope.stop):
                old = frames[index]
                if old.decision.reviewed:
                    raise ValueError(
                        "bulk confirmation cannot overwrite a reviewed frame; use an explicit replacement"
                    )
                if isinstance(operation, ConfirmPredictions):
                    proposal = predictions.rows[index]
                    observations: tuple[BallObservation, ...] = (
                        BallObservation(
                            track_id=operation.track_id,
                            state="visible",
                            center=(proposal.x, proposal.y),
                            origin="model_confirmed",
                        ),
                    )
                    presence = "present"
                else:
                    observations = tuple(
                        BallObservation(
                            track_id=track,
                            state="out_of_frame",
                            center=None,
                            origin="manual",
                        )
                        for track in operation.out_of_frame_tracks
                    )
                    presence = "absent"
                decision = FrameDecision(
                    reviewed=True,
                    presence=presence,
                    observations=observations,
                    evidence=saved.frames(operation.evidence, index),
                    notes=old.decision.notes,
                    boundaries=old.decision.boundaries,
                )
                frames[index] = old.model_copy(
                    update={"decision": decision, "review_actor": actor}
                )
        elif isinstance(operation, Interpolate):
            start, end = operation.endpoints.start, operation.endpoints.end
            centers = interpolation_centers(
                frames, manifest, policy, operation.track_id, start, end
            )
            for index, center in centers.items():
                old = frames[index]
                existing = observation(old, operation.track_id)
                if old.decision.presence == "absent" or (
                    existing is not None
                    and existing.state not in {"unresolved", "interpolated"}
                ):
                    raise ValueError(
                        "interpolation cannot overwrite confirmed observations or absence"
                    )
                derived = BallObservation(
                    track_id=operation.track_id,
                    state="interpolated",
                    center=center,
                    origin="interpolation",
                    interpolation=operation.endpoints,
                )
                observations = tuple(
                    sorted(
                        (
                            o
                            for o in (*old.decision.observations, derived)
                            if o is derived or o.track_id != operation.track_id
                        ),
                        key=lambda o: o.track_id,
                    )
                )
                frames[index] = old.model_copy(
                    update={
                        "decision": old.decision.model_copy(
                            update={"presence": "present", "observations": observations}
                        )
                    }
                )
        else:
            raise ValueError("unsupported ball edit operation")
    changed = [
        i
        for i, (before, after) in enumerate(zip(labels.frames, frames, strict=True))
        if before != after
    ]
    if not changed:
        raise ValueError("ball edit does not change any frame decision")
    candidate = labels.model_copy(
        update={"frames": tuple(frames), "images": tuple(saved.images.values())}
    )
    return ChangeResult(
        payload=candidate,
        summary=ChangeSummary(
            changed_ranges=ranges(changed),
            changed_rows=len(changed),
            examples=tuple(
                DeltaItem(
                    frame_index=i,
                    row_id=str(i),
                    before=brief(labels.frames[i]),
                    after=brief(frames[i]),
                )
                for i in changed[:32]
            ),
            unresolved_rows=sum(not resolved(f) for f in frames if f.target),
        ),
    )
