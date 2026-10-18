"""Dense ball labels keep image review, existence, location and provenance distinct."""

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
)
from src.tasks.base.annotation.telemetry.contracts import RenderedImage

_M = TypeVar("_M")


class _Validator(Protocol):
    def __call__(self, function: Callable[[_M], _M]) -> Callable[[_M], _M]: ...


_after = cast(_Validator, after_validator)
TrackId = Annotated[Identifier, Field(max_length=64)]
Event = Literal["cut", "hit", "bounce", "track_change", "unknown_break"]


class BallReviewPolicy(Document):
    """No production defaults: values are fixed only after a qualitative pilot."""

    schema_version: Literal["ball.review-policy.v1"] = "ball.review-policy.v1"
    allow_prediction_confirmation: bool
    minimum_reviewed_target_fraction: float = Field(ge=0, le=1)
    minimum_resolved_target_fraction: float = Field(ge=0, le=1)
    minimum_image_scale: float = Field(gt=0, le=1)
    minimum_absence_image_scale: float = Field(gt=0, le=1)
    maximum_interpolation_seconds: float | None = Field(gt=0)
    unreviewed_teacher: Literal["exclude"]
    supervision: Literal["observed_only"]


class Interpolation(Document):
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @_after
    def endpoints(self) -> Self:
        if self.end <= self.start + 1:
            raise ValueError(
                "interpolation needs two distinct endpoints and an interior"
            )
        return self


class BallObservation(Document):
    track_id: TrackId
    state: Literal["visible", "occluded", "interpolated", "out_of_frame", "unresolved"]
    center: tuple[float, float] | None
    origin: Literal["manual", "model_confirmed", "interpolation"]
    interpolation: Interpolation | None = None

    @_after
    def meaning(self) -> Self:
        if self.state in {"visible", "interpolated"} and self.center is None:
            raise ValueError("visible/interpolated ball requires a position")
        if self.state in {"out_of_frame", "unresolved"} and self.center is not None:
            raise ValueError("out-of-frame/unresolved ball cannot carry a position")
        if (self.state == "interpolated") != (self.origin == "interpolation") or (
            self.state == "interpolated"
        ) != (self.interpolation is not None):
            raise ValueError("interpolation state, origin and endpoints must agree")
        if self.origin == "model_confirmed" and self.state != "visible":
            raise ValueError("coordinate proposals do not predict absence or occlusion")
        return self


class Note(Document):
    note_id: Identifier = Field(max_length=64)
    text: str = Field(min_length=1, max_length=500)
    blocking: bool


class FrameDecision(Document):
    reviewed: bool
    presence: Literal["unknown", "present", "absent"]
    observations: tuple[BallObservation, ...] = Field(max_length=32)
    evidence: tuple[Digest, ...] = Field(max_length=32)
    notes: tuple[Note, ...] = Field(max_length=32)
    boundaries: tuple[Event, ...] = Field(max_length=5)

    @_after
    def meaning(self) -> Self:
        ids = [observation.track_id for observation in self.observations]
        if ids != sorted(set(ids)):
            raise ValueError("ball tracks must be unique and sorted within a frame")
        if len({n.note_id for n in self.notes}) != len(self.notes):
            raise ValueError("note IDs must be unique within a frame")
        if len(set(self.evidence)) != len(self.evidence) or len(
            set(self.boundaries)
        ) != len(self.boundaries):
            raise ValueError("image evidence and boundary kinds must be unique")
        if self.reviewed != bool(self.evidence):
            raise ValueError("a visual review claim requires image references")
        if not self.reviewed and (
            self.presence != "unknown" or self.observations or self.boundaries
        ):
            raise ValueError("unreviewed predictions cannot become label decisions")
        if self.presence == "unknown" and self.observations:
            raise ValueError("unknown existence cannot carry confirmed ball instances")
        if self.presence == "present" and not any(
            o.state != "out_of_frame" for o in self.observations
        ):
            raise ValueError(
                "present ball requires a non-absent instance, possibly unresolved"
            )
        if self.presence == "absent" and any(
            o.state != "out_of_frame" for o in self.observations
        ):
            raise ValueError(
                "confirmed absence cannot contain a visible or uncertain instance"
            )
        return self


class BallFrame(Document):
    frame: FrameMap
    target: bool
    decision: FrameDecision
    review_actor: ArtifactRef | None

    @_after
    def actor(self) -> Self:
        if self.decision.reviewed != (self.review_actor is not None):
            raise ValueError("review claims must retain their worker launch identity")
        return self


class BallLabels(Document):
    schema_version: Literal["ball.review-labels.v1"] = "ball.review-labels.v1"
    clip_id: Digest
    prediction: ArtifactRef
    policy: ArtifactRef
    frames: tuple[BallFrame, ...] = Field(min_length=32)
    images: tuple[RenderedImage, ...]


class ReplaceFrame(Document):
    operation: Literal["replace"] = "replace"
    frame_index: int = Field(ge=0)
    expected_frame_digest: Digest
    decision: FrameDecision


class ConfirmPredictions(Document):
    operation: Literal["confirm_predictions"] = "confirm_predictions"
    scope: FrameRange
    track_id: TrackId
    evidence: tuple[Digest, ...] = Field(min_length=1, max_length=128)


class ConfirmAbsence(Document):
    operation: Literal["confirm_absence"] = "confirm_absence"
    scope: FrameRange
    out_of_frame_tracks: tuple[TrackId, ...]
    evidence: tuple[Digest, ...] = Field(min_length=1, max_length=128)


class Interpolate(Document):
    operation: Literal["interpolate"] = "interpolate"
    track_id: TrackId
    endpoints: Interpolation


Operation = Annotated[
    ReplaceFrame | ConfirmPredictions | ConfirmAbsence | Interpolate,
    Field(discriminator="operation"),
]


class BallEdits(Document):
    schema_version: Literal["ball.review-edits.v1"] = "ball.review-edits.v1"
    operations: tuple[Operation, ...] = Field(min_length=1, max_length=128)
    images: tuple[RenderedImage, ...] = Field(default=(), max_length=128)


class BallReviewProof(Document):
    schema_version: Literal["ball.review-proof.v1"] = "ball.review-proof.v1"
    telemetry: tuple[ArtifactRef, ...] = Field(min_length=1)


class TeacherRow(Document):
    frame_index: int = Field(ge=0)
    target: bool
    reviewed: bool
    resolved: bool
    heatmap: Literal["positive", "negative", "ignore"]
    coordinate_track: TrackId | None
    positive_tracks: tuple[TrackId, ...]
    ignore_reasons: tuple[Identifier, ...]


class BallTeacherMask(Document):
    schema_version: Literal["ball.teacher-mask.v1"] = "ball.teacher-mask.v1"
    clip_id: Digest
    revision: ArtifactRef
    policy: ArtifactRef
    rows: tuple[TeacherRow, ...]
