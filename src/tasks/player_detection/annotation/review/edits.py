"""Atomic candidate decisions: explicit replacement, proposal confirmation, or empty review."""

from src.tasks.base.annotation.contracts import ArtifactRef, ClipManifest
from src.tasks.base.annotation.editing import ChangeResult, ChangeSummary
from src.tasks.base.annotation.editing.contracts import DeltaItem

from ..contracts import PlayerPredictions
from .contracts import (
    ConfirmEmpty,
    ConfirmPredictions,
    FrameDecision,
    PlayerEdits,
    PlayerFrame,
    PlayerLabels,
    PlayerObservation,
    PlayerReviewPolicy,
    ReplaceFrame,
)
from .evidence import Catalog
from .validation import ranges, resolved


def brief(frame: PlayerFrame) -> str:
    d = frame.decision
    return (
        f"reviewed={d.reviewed} coverage={d.coverage} "
        f"players={sum(o.classification == 'player' for o in d.observations)} "
        f"candidates={len(d.observations)} notes={len(d.notes)}"
    )


def apply_operations(
    labels: PlayerLabels,
    edits: PlayerEdits,
    manifest: ClipManifest,
    policy: PlayerReviewPolicy,
    predictions: PlayerPredictions,
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
            old_ids = {o.observation_id for o in frames[index].decision.observations}
            if not old_ids.issubset(
                o.observation_id for o in operation.decision.observations
            ):
                raise ValueError(
                    "retain observation IDs when rejecting/removing boxes; do not drop their history"
                )
            frames[index] = frames[index].model_copy(
                update={
                    "decision": operation.decision,
                    "review_actor": actor if operation.decision.reviewed else None,
                }
            )
        elif isinstance(operation, (ConfirmPredictions, ConfirmEmpty)):
            if operation.scope.stop > len(frames):
                raise ValueError("confirmation range lies outside the clip")
            if (
                isinstance(operation, ConfirmPredictions)
                and not policy.allow_prediction_confirmation
            ):
                raise ValueError("bulk prediction confirmation is disabled by policy")
            for index in range(operation.scope.start, operation.scope.stop):
                old, proposals = frames[index], predictions.rows[index].observations
                if old.decision.reviewed:
                    raise ValueError(
                        "bulk confirmation cannot overwrite a reviewed frame; use explicit replacement"
                    )
                if isinstance(operation, ConfirmEmpty) and proposals:
                    raise ValueError(
                        "nonempty proposals require explicit candidate rejection, not empty confirmation"
                    )
                if isinstance(operation, ConfirmPredictions) and not proposals:
                    raise ValueError(
                        "empty predictions require explicit full-frame empty confirmation"
                    )
                observations = tuple(
                    sorted(
                        (
                            PlayerObservation(
                                observation_id=p.observation_id,
                                proposal_id=p.observation_id,
                                manual_id=None,
                                classification="player",
                                box_state="observed",
                                box=p.box,
                                origin="model_confirmed",
                                confidence=p.confidence,
                                score_origin="model",
                                duplicate_of=None,
                            )
                            for p in proposals
                        ),
                        key=lambda o: o.observation_id,
                    )
                )
                decision = FrameDecision(
                    reviewed=True,
                    coverage="complete",
                    observations=observations,
                    evidence=saved.frames(operation.evidence, index),
                    notes=old.decision.notes,
                )
                frames[index] = old.model_copy(
                    update={"decision": decision, "review_actor": actor}
                )
        else:
            raise ValueError("unsupported player edit operation")
    changed = [
        i
        for i, (before, after) in enumerate(zip(labels.frames, frames, strict=True))
        if before != after
    ]
    if not changed:
        raise ValueError("player edit does not change any frame decision")
    return ChangeResult(
        payload=labels.model_copy(
            update={"frames": tuple(frames), "images": tuple(saved.images.values())}
        ),
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
