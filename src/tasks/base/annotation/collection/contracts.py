"""Collection inputs and receipts; no annotation-kit or task-label dependency."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, Self, TypeVar, cast

from pydantic import Field

from ..contracts import (
    ArtifactRef,
    Digest,
    Document,
    Identifier,
    PositiveInt,
    SourceIdentity,
    after_validator,
    content_digest,
)

_Model = TypeVar("_Model")


class _Validator(Protocol):
    def __call__(
        self, function: Callable[[_Model], _Model]
    ) -> Callable[[_Model], _Model]: ...


# Keep the decorator typed even when staged-only mypy skips its defining module.
_after = cast(_Validator, after_validator)


class SourceCandidate(Document):
    provider: Literal["youtube", "local"]
    channel_id: Identifier
    video_id: Identifier
    title: str = Field(min_length=1)
    origin_url: str = Field(min_length=1)
    selection_reason: str = Field(min_length=1)
    evidence: ArtifactRef
    local_media: ArtifactRef | None = None

    @_after
    def local_input(self) -> Self:
        if (self.provider == "local") != (self.local_media is not None):
            raise ValueError("only local candidates require a pinned local media input")
        return self

    @property
    def source_id(self) -> str:
        identity: str = content_digest([self.provider, self.video_id])
        return identity


class PreparationConfig(Document):
    schema_version: Literal["annotation.preparation.v1"] = "annotation.preparation.v1"
    format_selector: str = Field(min_length=1)
    js_runtimes: str | None = None
    remote_components: str | None = None
    crf: int = Field(ge=0, le=51)
    preset: Literal[
        "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"
    ]
    minimum_clip_frames: PositiveInt
    maximum_unprepared_sources: PositiveInt


class AcquisitionReceipt(Document):
    schema_version: Literal["annotation.acquisition.v1"] = "annotation.acquisition.v1"
    candidate_digest: Digest
    configuration: ArtifactRef
    source: SourceIdentity


class TimeWindow(Document):
    """Seconds from the first displayed source frame, with a separate target."""

    start: float = Field(ge=0)
    stop: float = Field(gt=0)
    target_start: float = Field(ge=0)
    target_stop: float = Field(gt=0)

    @_after
    def ordered(self) -> Self:
        if not self.start <= self.target_start < self.target_stop <= self.stop:
            raise ValueError("target must be nonempty and inside its media window")
        return self


class SelectionReceipt(Document):
    schema_version: Literal["annotation.selection.v1"] = "annotation.selection.v1"
    source_id: Digest
    source_digest: Digest
    policy: ArtifactRef
    evidence: ArtifactRef
    requested_maximum: PositiveInt
    windows: tuple[TimeWindow, ...]
    shortage_reason: str | None

    @_after
    def selected(self) -> Self:
        if len(self.windows) > self.requested_maximum:
            raise ValueError("selected windows exceed the requested maximum")
        if (
            len(self.windows) < self.requested_maximum
            and not (self.shortage_reason or "").strip()
        ):
            raise ValueError("insufficient selected windows require an explicit reason")
        if (
            len(self.windows) == self.requested_maximum
            and self.shortage_reason is not None
        ):
            raise ValueError("a full selection cannot report a shortage")
        if any(
            a.stop > b.start
            for a, b in zip(self.windows, self.windows[1:], strict=False)
        ):
            raise ValueError("selected windows must be ordered and nonoverlapping")
        return self


class PreparationReceipt(Document):
    schema_version: Literal["annotation.prepared.v1"] = "annotation.prepared.v1"
    source_id: Digest
    source_digest: Digest
    selection: ArtifactRef
    configuration: ArtifactRef
    clips: tuple[ArtifactRef, ...]


class RegistrationResult(Document):
    source_id: Digest
    disposition: Literal[
        "registered", "existing", "duplicate_source", "duplicate_channel", "limit"
    ]
    owner_campaign: Identifier | None
    reason: str | None
