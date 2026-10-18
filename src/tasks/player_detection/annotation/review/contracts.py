"""Bbox decisions retain candidate identity, coverage and score provenance separately."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal, Protocol, Self, TypeVar, cast

from pydantic import Field

from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    Digest,
    Document,
    FrameMap,
    FrameRange,
    Identifier,
    after_validator,
    content_digest,
)
from src.tasks.base.annotation.telemetry.contracts import RenderedImage

from ..contracts import Box

WORKER_ADAPTER = "player_bbox_review"
WORKER_VERSION = "v2"

_M = TypeVar("_M")


class _Validator(Protocol):
    def __call__(self, function: Callable[[_M], _M]) -> Callable[[_M], _M]: ...


_after = cast(_Validator, after_validator)


class PlayerReviewPolicy(Document):
    """All semantic and numerical choices require an explicit caller decision."""

    schema_version: Literal["player.review-policy.v1"] = "player.review-policy.v1"
    target_definition: str = Field(min_length=1, max_length=2000)
    bbox_definition: str = Field(min_length=1, max_length=2000)
    allow_prediction_confirmation: bool
    minimum_reviewed_target_fraction: float = Field(ge=0, le=1)
    minimum_resolved_target_fraction: float = Field(ge=0, le=1)
    minimum_image_scale: float = Field(gt=0, le=1)
    minimum_full_frame_image_scale: float = Field(gt=0, le=1)
    unreviewed_teacher: Literal["exclude"]
    supervision: Literal["observed_only"]

    @_after
    def definitions(self) -> Self:
        if not self.target_definition.strip() or not self.bbox_definition.strip():
            raise ValueError("player target and bbox definitions must be explicit")
        return self


def manual_observation_id(clip_id: str, frame_index: int, manual_id: str) -> str:
    result: str = content_digest(["player-manual", clip_id, frame_index, manual_id])
    return result


class PlayerObservation(Document):
    observation_id: Digest
    proposal_id: Digest | None
    manual_id: Identifier | None
    classification: Literal["player", "non_player", "duplicate", "unresolved"]
    box_state: Literal["observed", "estimated", "unresolved", "not_applicable"]
    box: Box | None
    origin: Literal["manual", "model_confirmed"]
    confidence: float = Field(ge=0, le=1)
    score_origin: Literal["model", "missing"]
    duplicate_of: Digest | None

    @_after
    def meaning(self) -> Self:
        if (self.proposal_id is None) == (self.manual_id is None):
            raise ValueError(
                "an observation needs one original proposal or manual identity"
            )
        if self.classification == "player":
            if self.box_state == "not_applicable":
                raise ValueError(
                    "a player needs an observed, estimated or unresolved bbox"
                )
        elif self.classification == "unresolved":
            if self.box_state != "unresolved":
                raise ValueError("unresolved classification cannot supply a bbox")
        elif self.box_state != "not_applicable":
            raise ValueError("rejected candidates cannot carry a teacher bbox")
        if (self.box is not None) != (self.box_state in {"observed", "estimated"}):
            raise ValueError("only observed/estimated bboxes carry coordinates")
        if (self.duplicate_of is not None) != (self.classification == "duplicate"):
            raise ValueError("a duplicate must name its retained player observation")
        if self.origin == "model_confirmed":
            if (
                self.proposal_id is None
                or self.classification != "player"
                or self.box_state != "observed"
                or self.score_origin != "model"
            ):
                raise ValueError(
                    "model confirmation requires an observed original player proposal"
                )
        elif self.score_origin != "missing" or self.confidence != 0:
            raise ValueError(
                "manual corrections/additions require a masked-zero missing score"
            )
        return self


class Note(Document):
    note_id: Identifier = Field(max_length=64)
    text: str = Field(min_length=1, max_length=500)
    blocking: bool


class FrameDecision(Document):
    reviewed: bool
    coverage: Literal["unknown", "partial", "complete"]
    observations: tuple[PlayerObservation, ...]
    evidence: tuple[Digest, ...] = Field(max_length=32)
    notes: tuple[Note, ...] = Field(max_length=32)

    @_after
    def meaning(self) -> Self:
        ids = [o.observation_id for o in self.observations]
        if ids != sorted(set(ids)):
            raise ValueError("player observations must have unique sorted identities")
        if len({n.note_id for n in self.notes}) != len(self.notes):
            raise ValueError("note IDs must be unique within a frame")
        if len(set(self.evidence)) != len(self.evidence):
            raise ValueError("image evidence must be unique")
        if self.reviewed != bool(self.evidence):
            raise ValueError("a visual review claim requires image references")
        if not self.reviewed and (self.coverage != "unknown" or self.observations):
            raise ValueError("unreviewed proposals cannot become label decisions")
        for item in self.observations:
            if item.duplicate_of is not None and not any(
                o.observation_id == item.duplicate_of and o.classification == "player"
                for o in self.observations
            ):
                raise ValueError(
                    "duplicate target must be a retained player in this frame"
                )
        return self


class PlayerFrame(Document):
    frame: FrameMap
    target: bool
    decision: FrameDecision
    review_actor: ArtifactRef | None

    @_after
    def actor(self) -> Self:
        if self.decision.reviewed != (self.review_actor is not None):
            raise ValueError("review claims must retain their worker launch identity")
        return self


class PlayerLabels(Document):
    schema_version: Literal["player.review-labels.v1"] = "player.review-labels.v1"
    clip_id: Digest
    prediction: ArtifactRef
    policy: ArtifactRef
    frames: tuple[PlayerFrame, ...] = Field(min_length=1)
    images: tuple[RenderedImage, ...]


class ReplaceFrame(Document):
    operation: Literal["replace"] = "replace"
    frame_index: int = Field(ge=0)
    expected_frame_digest: Digest
    decision: FrameDecision


class ConfirmPredictions(Document):
    operation: Literal["confirm_predictions"] = "confirm_predictions"
    scope: FrameRange
    evidence: tuple[Digest, ...] = Field(min_length=1, max_length=128)


class ConfirmEmpty(Document):
    """Only empty raw proposals; rejecting existing candidates requires explicit edits."""

    operation: Literal["confirm_empty"] = "confirm_empty"
    scope: FrameRange
    evidence: tuple[Digest, ...] = Field(min_length=1, max_length=128)


Operation = Annotated[
    ReplaceFrame | ConfirmPredictions | ConfirmEmpty, Field(discriminator="operation")
]


class PlayerEdits(Document):
    schema_version: Literal["player.review-edits.v1"] = "player.review-edits.v1"
    operations: tuple[Operation, ...] = Field(min_length=1, max_length=128)
    images: tuple[RenderedImage, ...] = Field(default=(), max_length=128)
    id_review: ArtifactRef | None = None


class PlayerReviewProof(Document):
    schema_version: Literal["player.review-proof.v1"] = "player.review-proof.v1"
    telemetry: tuple[ArtifactRef, ...] = Field(min_length=1)


class TeacherRow(Document):
    frame_index: int = Field(ge=0)
    target: bool
    reviewed: bool
    resolved: bool
    supervision: Literal["positive", "negative", "ignore"]
    observation_ids: tuple[Digest, ...]
    ignore_reasons: tuple[Identifier, ...]


class PlayerTeacherMask(Document):
    schema_version: Literal["player.teacher-mask.v1"] = "player.teacher-mask.v1"
    clip_id: Digest
    revision: ArtifactRef
    policy: ArtifactRef
    rows: tuple[TeacherRow, ...]
