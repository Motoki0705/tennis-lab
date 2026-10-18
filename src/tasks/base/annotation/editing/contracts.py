"""Immutable revision envelopes and bounded edit summaries, without task labels."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, Self, TypeVar, cast

from pydantic import Field

from ..contracts import (
    AdoptionReceipt,
    ArtifactRef,
    ClipManifest,
    Document,
    FrameRange,
    Identifier,
    after_validator,
)
from ..paths import AnnotationLayout

_M = TypeVar("_M")


class _Validator(Protocol):
    def __call__(self, function: Callable[[_M], _M]) -> Callable[[_M], _M]: ...


_after = cast(_Validator, after_validator)


class ReviewContext(Document):
    plan: ArtifactRef
    task: Identifier
    manifest: ArtifactRef
    prediction: ArtifactRef


class Revision(Document):
    context: ArtifactRef
    sequence: int = Field(ge=0)
    previous: ArtifactRef | None
    payload: ArtifactRef
    command: ArtifactRef | None

    @_after
    def ancestry(self) -> Self:
        if (self.sequence == 0) != (self.previous is None) or (self.sequence == 0) != (
            self.command is None
        ):
            raise ValueError(
                "only the initial revision omits a previous revision and edit command"
            )
        return self


class EditCommand(Document):
    context: ArtifactRef
    expected_revision: ArtifactRef
    changes: ArtifactRef
    actor: ArtifactRef
    reason: str = Field(min_length=1, max_length=500)


class DeltaItem(Document):
    frame_index: int = Field(ge=0)
    row_id: str = Field(min_length=1, max_length=128)
    before: str = Field(max_length=200)
    after: str = Field(max_length=200)


class ChangeSummary(Document):
    changed_ranges: tuple[FrameRange, ...] = Field(max_length=128)
    changed_rows: int = Field(ge=0)
    examples: tuple[DeltaItem, ...] = Field(max_length=32)
    unresolved_rows: int = Field(ge=0)


class EditReceipt(Document):
    command: ArtifactRef
    previous: ArtifactRef
    revision: ArtifactRef
    summary: ChangeSummary


@dataclass(frozen=True)
class ChangeResult:
    payload: Document
    summary: ChangeSummary


class ReviewAdapter(Protocol):
    task: str

    def retained_references(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        revision: ArtifactRef,
        receipt: AdoptionReceipt,
    ) -> tuple[ArtifactRef, ...]:
        """Explicitly enumerate task-owned label and evidence dependencies."""
        ...

    def initialize(
        self, layout: AnnotationLayout, context: ReviewContext, manifest: ClipManifest
    ) -> Document: ...

    def validate(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        payload: Document,
    ) -> None: ...

    def apply(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        current_payload: ArtifactRef,
        command: EditCommand,
    ) -> ChangeResult:
        """Validate the entire batch, protected observations and task boundaries first."""
        ...

    def validate_adoption(
        self,
        layout: AnnotationLayout,
        context: ReviewContext,
        manifest: ClipManifest,
        revision: ArtifactRef,
        receipt: AdoptionReceipt,
    ) -> None:
        """Verify actual review evidence, policy and teacher masks; no independent QA."""
        ...
