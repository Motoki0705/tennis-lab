"""The task-owned ``video_ball_annotation.v3`` contract for one active ball."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Annotated, Literal, Protocol, Self, TypeVar, cast

from pydantic import ConfigDict, Field

from src.tasks.base.annotation.contracts import (
    Digest,
    Document,
    FrameRange,
    Identifier,
    after_validator,
)

TargetSemantics = Literal["active_ball"]
BallState = Literal[
    "in_frame", "out_of_frame", "no_active_ball", "unknown", "unprocessed"
]
Occlusion = Literal["not_occluded", "occluded", "unknown", "not_applicable"]
PointKind = Literal["observed", "interpolated", "estimated", "occlusion_estimated"]
FrameIndex = Annotated[int, Field(ge=0)]
_M = TypeVar("_M")


class _Validator(Protocol):
    def __call__(self, function: Callable[[_M], _M]) -> Callable[[_M], _M]: ...


_after = cast(_Validator, after_validator)


class _VideoDocument(Document):
    # Editing code can use model_copy(), which deliberately bypasses validation.
    # Revalidate nested instances as well as JSON at this public contract boundary.
    model_config = ConfigDict(revalidate_instances="always")


class VideoIdentity(_VideoDocument):
    """Original bytes and native decoding metadata, independent of a live path."""

    recording_id: str = Field(min_length=1)
    camera_id: Identifier
    file_name: str = Field(min_length=1)
    media_sha256: Digest
    size_bytes: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_count: int = Field(gt=0)
    time_base: str
    nominal_fps: str
    rotation: Literal[0] = 0

    @_after
    def timing(self) -> Self:
        try:
            time_base = Fraction(self.time_base)
            fps = Fraction(self.nominal_fps)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(
                "time_base and nominal_fps must be positive rationals"
            ) from error
        if time_base <= 0 or fps <= 0:
            raise ValueError("time_base and nominal_fps must be positive rationals")
        if not self.recording_id.strip() or not self.file_name.strip():
            raise ValueError("recording_id and file_name must not be blank")
        return self


class BallTarget(_VideoDocument):
    # Required even on JSON input: never infer active-ball semantics from v2.
    target_semantics: TargetSemantics
    label: Literal["tennis_ball"] = "tennis_ball"
    track_id: Identifier


class FrameIdentity(_VideoDocument):
    """Zero-based presentation index and exact integer PTS in the original video."""

    frame_index: FrameIndex
    pts: int


class PointProvenance(_VideoDocument):
    kind: PointKind
    # All indices refer to this same source/camera, including context outside scope.
    support_frames: tuple[FrameIndex, ...] = ()

    @_after
    def support(self) -> Self:
        if tuple(sorted(set(self.support_frames))) != self.support_frames:
            raise ValueError("support_frames must be unique and increasing")
        if self.kind == "observed" and self.support_frames:
            raise ValueError("observed coordinates do not have inferred support_frames")
        if self.kind == "interpolated" and len(self.support_frames) != 2:
            raise ValueError("interpolated coordinates require two support_frames")
        return self


class PixelPosition(_VideoDocument):
    """One native pixel centre; image bounds are checked against VideoIdentity."""

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    provenance: PointProvenance


class VideoBallFrame(FrameIdentity):
    state: BallState
    occlusion: Occlusion
    position: PixelPosition | None
    break_before: bool = False
    notes: str = ""

    @_after
    def meaning(self) -> Self:
        if self.state in {"unknown", "unprocessed"}:
            if self.position is not None or self.occlusion != "unknown":
                raise ValueError(
                    "unknown/unprocessed requires no position and unknown occlusion"
                )
            if self.state == "unprocessed" and self.break_before:
                raise ValueError("unprocessed frames cannot assert a continuity break")
        elif self.state in {"out_of_frame", "no_active_ball"}:
            if self.position is not None or self.occlusion != "not_applicable":
                raise ValueError(
                    "out_of_frame/no_active_ball requires no position and not_applicable occlusion"
                )
        else:
            if self.occlusion == "not_applicable":
                raise ValueError("in_frame requires an applicable occlusion state")
            if self.position is None:
                if self.occlusion != "occluded":
                    raise ValueError(
                        "in_frame without a position requires known occlusion; otherwise use unknown"
                    )
            else:
                origin = self.position.provenance
                if origin.kind == "observed" and self.occlusion != "not_occluded":
                    raise ValueError("observed coordinates require not_occluded")
                if (
                    origin.kind == "occlusion_estimated"
                    and self.occlusion != "occluded"
                ):
                    raise ValueError("occlusion_estimated coordinates require occluded")
                if origin.kind == "interpolated":
                    start, end = origin.support_frames
                    if not start < self.frame_index < end:
                        raise ValueError(
                            "interpolation support_frames must enclose the source frame"
                        )
        return self


class _VideoScope(_VideoDocument):
    source: VideoIdentity
    target: BallTarget
    # Half-open source-frame interval. Every row is retained, even if unprocessed.
    frame_range: FrameRange


def _validate_timeline(
    source: VideoIdentity, scope: FrameRange, frames: Sequence[FrameIdentity]
) -> None:
    if scope.stop > source.frame_count:
        raise ValueError("frame_range exceeds the original video's frame_count")
    if len(frames) != scope.stop - scope.start:
        raise ValueError("every source frame in frame_range must appear exactly once")
    previous: FrameIdentity | None = None
    for index, frame in enumerate(frames, start=scope.start):
        if frame.frame_index != index:
            raise ValueError(
                "source frame indices must be dense and in presentation order"
            )
        if previous is not None and frame.pts <= previous.pts:
            raise ValueError(
                "source PTS must be strictly increasing; never derive them from FPS"
            )
        previous = frame


class VideoAnnotationInput(_VideoScope):
    """Expected input from a pinned request/decoder, used to bind reader and writer."""

    frames: tuple[FrameIdentity, ...] = Field(min_length=1)

    @_after
    def timeline(self) -> Self:
        _validate_timeline(self.source, self.frame_range, self.frames)
        return self


class VideoBallAnnotation(_VideoScope):
    # These literals are required on disk, rather than silently supplied by readers.
    schema_version: Literal["video_ball_annotation.v3"]
    coordinate_system: Literal["native_pixels_top_left_endpoint"]
    frames: tuple[VideoBallFrame, ...] = Field(min_length=1)

    @_after
    def timeline_and_positions(self) -> Self:
        _validate_timeline(self.source, self.frame_range, self.frames)
        for frame in self.frames:
            point = frame.position
            if point is None:
                continue
            if point.x > self.source.width - 1 or point.y > self.source.height - 1:
                raise ValueError(
                    "position lies outside native pixel endpoints; use out_of_frame without coordinates"
                )
            if any(
                index >= self.source.frame_count
                for index in point.provenance.support_frames
            ):
                raise ValueError("support_frames exceed the original source timeline")
        return self

    @property
    def input_identity(self) -> VideoAnnotationInput:
        return VideoAnnotationInput(
            source=self.source,
            target=self.target,
            frame_range=self.frame_range,
            frames=tuple(
                FrameIdentity(frame_index=f.frame_index, pts=f.pts) for f in self.frames
            ),
        )
