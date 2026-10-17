"""Selection policy, source-bound image evidence, and worker decisions."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Literal, Protocol, Self, TypeVar, cast

from pydantic import Field

from ..contracts import (
    ArtifactRef,
    Digest,
    Document,
    PositiveInt,
    SourceIdentity,
    after_validator,
)

_M = TypeVar("_M")


class _Validator(Protocol):
    def __call__(self, function: Callable[[_M], _M]) -> Callable[[_M], _M]: ...


_after = cast(_Validator, after_validator)


class ScreeningPolicy(Document):
    schema_version: Literal["annotation.screening-policy.v1"] = (
        "annotation.screening-policy.v1"
    )
    window_seconds: float = Field(default=15.0, ge=15, le=15)
    context_seconds: float = Field(default=1.0, ge=1, le=1)
    minimum_play_fraction_exclusive: float = Field(default=0.5, ge=0.5, le=0.5)
    play_definition: Literal["normal_speed_toss_release_to_point_end"] = (
        "normal_speed_toss_release_to_point_end"
    )
    overview_step_seconds: float = Field(default=2.0, gt=0)
    detail_step_seconds: float = Field(default=0.5, gt=0, le=0.5)
    boundary_step_seconds: float = Field(default=0.1, gt=0, le=0.1)
    maximum_source_seconds: float = Field(default=1500.0, ge=15)


class Interval(Document):
    start: float = Field(ge=0)
    stop: float = Field(gt=0)

    @_after
    def ordered(self) -> Self:
        if self.stop <= self.start:
            raise ValueError("interval stop must follow its start")
        return self


class ClipDecision(Interval):
    normal_speed: bool
    in_play: tuple[Interval, ...] = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence_images: tuple[str, ...] = Field(min_length=1)

    @_after
    def play_intervals(self) -> Self:
        previous = self.start
        for interval in self.in_play:
            if not previous <= interval.start < interval.stop <= self.stop:
                raise ValueError(
                    "in-play intervals must be ordered, disjoint and inside the window"
                )
            previous = interval.stop
        return self

    def validate_policy(self, policy: ScreeningPolicy) -> None:
        if abs(self.stop - self.start - policy.window_seconds) > 1e-6:
            raise ValueError("screened media window must last 15 seconds")
        if not self.normal_speed:
            raise ValueError(
                "slow-motion/replay cannot be selected as normal-speed play"
            )
        seconds = sum(
            Fraction(str(interval.stop)) - Fraction(str(interval.start))
            for interval in self.in_play
        )
        minimum_seconds = Fraction(str(policy.window_seconds)) * Fraction(
            str(policy.minimum_play_fraction_exclusive)
        )
        if seconds <= minimum_seconds:
            raise ValueError(
                "strictly more than 7.5 seconds of normal-speed play is required"
            )


class RejectedInterval(Interval):
    reason: str = Field(min_length=1)


class SourceDecision(Document):
    schema_version: Literal["annotation.screening-decision.v1"] = (
        "annotation.screening-decision.v1"
    )
    source_sha256: Digest
    requested_maximum: PositiveInt
    status: Literal["accepted", "insufficient", "deferred"]
    match_type: Literal["singles", "doubles", "mixed_content", "unknown"]
    surface: str = Field(min_length=1)
    setting: str = Field(min_length=1)
    camera: str = Field(min_length=1)
    clips: tuple[ClipDecision, ...]
    rejected: tuple[RejectedInterval, ...]
    notes: str = Field(min_length=1)

    @_after
    def count_and_overlap(self) -> Self:
        if len(self.clips) > self.requested_maximum:
            raise ValueError("selected clips exceed the requested maximum")
        if self.status != "deferred" and (
            (self.status == "accepted") != (len(self.clips) == self.requested_maximum)
        ):
            raise ValueError(
                "accepted requires the requested count; a shortage is insufficient"
            )
        if any(
            a.stop > b.start for a, b in zip(self.clips, self.clips[1:], strict=False)
        ):
            raise ValueError(
                "screened media windows must be ordered and nonoverlapping"
            )
        return self


class ImageSample(Document):
    source_pts: int
    seconds: float = Field(ge=0)


class ImagePage(Document):
    image: ArtifactRef
    samples: tuple[ImageSample, ...] = Field(min_length=1, max_length=24)


class PreviewReceipt(Document):
    schema_version: Literal["annotation.screening-preview.v1"] = (
        "annotation.screening-preview.v1"
    )
    source: ArtifactRef
    origin_pts: int
    time_base: str
    start: float = Field(ge=0)
    stop: float = Field(gt=0)
    step: float = Field(gt=0)
    decoded_until: float = Field(gt=0)
    pages: tuple[ImagePage, ...] = Field(min_length=1)

    @_after
    def timeline(self) -> Self:
        clock = Fraction(self.time_base)
        if (
            clock <= 0
            or self.stop <= self.start
            or self.decoded_until + 1e-6 < self.stop
        ):
            raise ValueError("preview must cover a positive source interval")
        if len({page.image for page in self.pages}) != len(self.pages):
            raise ValueError("preview pages must be distinct")
        previous: int | None = None
        for page in self.pages:
            for sample in page.samples:
                timestamp = float((sample.source_pts - self.origin_pts) * clock)
                if (
                    abs(timestamp - sample.seconds) > 1e-6
                    or not self.start - 1e-6 <= timestamp < self.stop
                ):
                    raise ValueError(
                        "preview sample differs from its source clock/interval"
                    )
                if previous is not None and sample.source_pts <= previous:
                    raise ValueError(
                        "preview samples must increase in source presentation order"
                    )
                previous = sample.source_pts
        return self


class ScreeningInput(Document):
    plan_digest: Digest
    source: SourceIdentity
    policy: ArtifactRef
    source_duration: float = Field(gt=0)
    source_origin_pts: int
    time_base: str
    overview: ArtifactRef


class ScreeningRequest(ScreeningInput):
    schema_version: Literal["annotation.screening-request.v1"] = (
        "annotation.screening-request.v1"
    )
    prepared_input: ArtifactRef
    requested_maximum: PositiveInt
    attempt_path: str


class PresentedImages(Document):
    """Produced by the launcher from successful image-tool events, never by the worker."""

    request_digest: Digest
    events: ArtifactRef
    images: tuple[ArtifactRef, ...]


class ScreeningEvidence(Document):
    request: ArtifactRef
    decision: ArtifactRef
    presentation: ArtifactRef
    previews: tuple[ArtifactRef, ...]


def check_decision(
    request: ScreeningRequest, policy: ScreeningPolicy, decision: SourceDecision
) -> None:
    if decision.source_sha256 != request.source.media.digest:
        raise ValueError("decision refers to a different source")
    if decision.requested_maximum != request.requested_maximum:
        raise ValueError("decision changes the requested clip count")
    for clip in decision.clips:
        clip.validate_policy(policy)
        if clip.stop > request.source_duration:
            raise ValueError("selected window extends beyond the source")
    if any(interval.stop > request.source_duration for interval in decision.rejected):
        raise ValueError("rejected interval extends beyond the source")
