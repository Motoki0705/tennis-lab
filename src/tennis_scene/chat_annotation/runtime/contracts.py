"""Single authority for the exchange format and its generated JSON Schemas."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Self, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

_ModelT = TypeVar("_ModelT")


class _AfterModelValidator(Protocol):
    def __call__(
        self, function: Callable[[_ModelT], _ModelT]
    ) -> Callable[[_ModelT], _ModelT]: ...


_after_model_validator = cast(_AfterModelValidator, model_validator(mode="after"))

KIT_VERSION = "4.2.0"
SCHEMA_VERSION: Literal["tennis_chat_annotation.v2"] = "tennis_chat_annotation.v2"
BALL_SCHEMA_VERSION: Literal["tennis_chat_ball_annotation.v1"] = (
    "tennis_chat_ball_annotation.v1"
)
PLAYER_SCHEMA_VERSION: Literal["tennis_chat_player_annotation.v1"] = (
    "tennis_chat_player_annotation.v1"
)
Point = Annotated[list[float], Field(min_length=2, max_length=2)]
Box = Annotated[list[float], Field(min_length=4, max_length=4)]
Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")]
FileName = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class FrameRange(StrictModel):
    """Half-open interval in decoded presentation-frame order."""

    start: int = Field(ge=0)
    stop: int = Field(gt=0)

    @_after_model_validator
    def ordered(self) -> Self:
        if self.stop <= self.start:
            raise ValueError("frame range must be nonempty and increasing")
        return self


class SourceInfo(StrictModel):
    source_id: Identifier
    youtube_id: str | None
    url: str | None
    title: str
    filename: FileName
    sha256: Digest
    bytes: int = Field(gt=0)
    acquired_at: str


class FrameMap(StrictModel):
    frame_index: int = Field(ge=0)
    source_frame_index: int = Field(ge=0)
    source_pts: int
    clip_pts: int = Field(ge=0)
    duration_pts: int = Field(gt=0)
    is_target: bool


class Policies(StrictModel):
    ball_max_gap_seconds: float = Field(gt=0)


class ClipManifest(StrictModel):
    schema_version: Literal["tennis_chat_clip.v1"]
    kit_version: str
    kit_id: Digest
    clip_id: Identifier
    source: SourceInfo
    filename: FileName
    sha256: Digest
    bytes: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    time_base: str
    nominal_fps: str
    source_start_pts: int
    media_range: FrameRange
    target_range: FrameRange
    frames: list[FrameMap] = Field(min_length=1)
    policies: Policies

    @_after_model_validator
    def consistent(self) -> Self:
        if Fraction(self.time_base) <= 0 or Fraction(self.nominal_fps) <= 0:
            raise ValueError("positive time base and nominal fps required")
        if not (
            self.media_range.start
            <= self.target_range.start
            < self.target_range.stop
            <= self.media_range.stop
        ):
            raise ValueError("target must be inside media range")
        if len(self.frames) != self.media_range.stop - self.media_range.start:
            raise ValueError("frame map length does not cover media range")
        previous: FrameMap | None = None
        for index, frame in enumerate(self.frames):
            if frame.frame_index != index:
                raise ValueError("clip frame indices must be dense from zero")
            if frame.source_frame_index != self.media_range.start + index:
                raise ValueError("source frame indices must be dense")
            expected_target = (
                self.target_range.start
                <= frame.source_frame_index
                < self.target_range.stop
            )
            if frame.is_target != expected_target:
                raise ValueError("target flag differs from target range")
            if frame.clip_pts != frame.source_pts - self.frames[0].source_pts:
                raise ValueError("clip/source PTS offset mismatch")
            if previous and frame.clip_pts != previous.clip_pts + previous.duration_pts:
                raise ValueError("noncontiguous frame presentation times")
            previous = frame
        return self


class Player(StrictModel):
    """Only participants on the target court; full-body, amodal boxes."""

    track_id: Identifier
    bbox_xyxy: Box | None = Field(
        description="Full-body [x_min,y_min,x_max,y_max] box in source-image pixels."
    )
    bbox_source: Literal["observed", "inferred", "unresolved"] = Field(
        description="Whether the full-body box was seen, inferred, or cannot be located."
    )
    occluded: bool = Field(description="True when another object hides part of the player.")
    truncated: bool = Field(description="True when the player extends beyond the image.")

    @_after_model_validator
    def consistent(self) -> Self:
        if (self.bbox_xyxy is None) != (self.bbox_source == "unresolved"):
            raise ValueError("null bbox requires unresolved source")
        if self.bbox_xyxy is not None:
            x1, y1, x2, y2 = self.bbox_xyxy
            if x2 <= x1 or y2 <= y1:
                raise ValueError("bbox must have positive area")
        if self.bbox_source == "observed" and (self.occluded or self.truncated):
            raise ValueError("occluded/truncated full-body bbox must be inferred")
        return self


class Ball(StrictModel):
    """A ball active in play on the target court."""

    track_id: Identifier
    center_px: Point | None = Field(
        description="Ball center [x,y] in source-image pixels, or null when unavailable."
    )
    status: Literal[
        "visible", "occluded", "interpolated", "out_of_frame", "unresolved"
    ] = Field(description="Observation or inference state for this ball in this frame.")
    interpolation_frames: Annotated[
        list[int], Field(min_length=2, max_length=2)
    ] | None = Field(
        description="Visible endpoint frame indices for interpolation; otherwise null."
    )

    @_after_model_validator
    def consistent(self) -> Self:
        if self.status in ("visible", "interpolated") and self.center_px is None:
            raise ValueError("visible/interpolated ball needs a position")
        if self.status in ("out_of_frame", "unresolved") and self.center_px is not None:
            raise ValueError("out_of_frame/unresolved ball position must be null")
        if (self.status == "interpolated") != (self.interpolation_frames is not None):
            raise ValueError("only interpolated balls require two endpoint frames")
        if self.interpolation_frames is not None:
            start, stop = self.interpolation_frames
            if start < 0 or stop <= start + 1:
                raise ValueError("interpolation endpoints must enclose a gap")
        return self


class FrameAnnotation(StrictModel):
    frame_index: int = Field(ge=0)
    reviewed: bool
    players: list[Player]
    balls: list[Ball]
    interpolation_break: bool
    notes: str


class Annotation(StrictModel):
    schema_version: Literal["tennis_chat_annotation.v2"]
    clip_id: FileName
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_count: int = Field(gt=0)
    status: Literal["completed", "partial"]
    issues: list[str]
    frames: list[FrameAnnotation]


class BallFrameAnnotation(StrictModel):
    """One frame in a ball-only annotation; reviewed applies to balls."""

    frame_index: int = Field(ge=0)
    reviewed: bool = Field(description="Whether the ball target was reviewed in this frame.")
    balls: list[Ball]
    interpolation_break: bool = Field(
        description="True on a hit, bounce, cut, or play boundary that interpolation cannot cross."
    )
    notes: str


class BallAnnotation(StrictModel):
    """Clip-level ball-only annotation payload."""

    schema_version: Literal["tennis_chat_ball_annotation.v1"]
    clip_id: FileName
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_count: int = Field(gt=0)
    status: Literal["completed", "partial"] = Field(
        description="Completed only when every frame was reviewed and no issue remains."
    )
    issues: list[str] = Field(description="Clip-wide unresolved issues, or an empty list.")
    frames: list[BallFrameAnnotation] = Field(
        description="One entry per video frame in display order."
    )


class PlayerFrameAnnotation(StrictModel):
    """One frame in a player-only annotation; reviewed applies to players."""

    frame_index: int = Field(ge=0)
    reviewed: bool = Field(
        description="Whether the player target was reviewed in this frame."
    )
    players: list[Player]
    notes: str


class PlayerAnnotation(StrictModel):
    """Clip-level player-only annotation payload."""

    schema_version: Literal["tennis_chat_player_annotation.v1"]
    clip_id: FileName
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_count: int = Field(gt=0)
    status: Literal["completed", "partial"] = Field(
        description="Completed only when every frame was reviewed and no issue remains."
    )
    issues: list[str] = Field(description="Clip-wide unresolved issues, or an empty list.")
    frames: list[PlayerFrameAnnotation] = Field(
        description="One entry per video frame in display order."
    )


SupportedAnnotation = Annotation | BallAnnotation | PlayerAnnotation


def parse_annotation(value: Any) -> SupportedAnnotation:
    if not isinstance(value, dict):
        raise ValueError("annotation must be a JSON object")
    schema_version = value.get("schema_version")
    if schema_version == BALL_SCHEMA_VERSION:
        ball_annotation: BallAnnotation = BallAnnotation.model_validate(value)
        return ball_annotation
    if schema_version == PLAYER_SCHEMA_VERSION:
        player_annotation: PlayerAnnotation = PlayerAnnotation.model_validate(value)
        return player_annotation
    if schema_version == SCHEMA_VERSION:
        combined_annotation: Annotation = Annotation.model_validate(value)
        return combined_annotation
    raise ValueError(f"unsupported annotation schema version: {schema_version!r}")


class ValidationReport(StrictModel):
    status: Literal["completed", "partial", "failed"]
    reviewed_frames: int
    target_frames: int
    errors: list[str]
    issues: list[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def annotation_clip_id(manifest: ClipManifest) -> str:
    """Use the globally unique input stem, not its source-local sampling range."""
    return Path(manifest.filename).stem


def make_template(
    manifest: ClipManifest, target: str = "combined"
) -> SupportedAnnotation:
    common: dict[str, Any] = {
        "clip_id": annotation_clip_id(manifest),
        "width": manifest.width,
        "height": manifest.height,
        "frame_count": len(manifest.frames),
        "status": "partial",
        "issues": [],
    }
    if target == "ball":
        return BallAnnotation(
            schema_version=BALL_SCHEMA_VERSION,
            **common,
            frames=[
                BallFrameAnnotation(
                    frame_index=frame.frame_index,
                    reviewed=False,
                    balls=[],
                    interpolation_break=False,
                    notes="",
                )
                for frame in manifest.frames
            ],
        )
    if target == "player":
        return PlayerAnnotation(
            schema_version=PLAYER_SCHEMA_VERSION,
            **common,
            frames=[
                PlayerFrameAnnotation(
                    frame_index=frame.frame_index,
                    reviewed=False,
                    players=[],
                    notes="",
                )
                for frame in manifest.frames
            ],
        )
    if target != "combined":
        raise ValueError(f"unsupported annotation target: {target}")
    return Annotation(
        schema_version=SCHEMA_VERSION,
        **common,
        frames=[
            FrameAnnotation(
                frame_index=frame.frame_index,
                reviewed=False,
                players=[],
                balls=[],
                interpolation_break=False,
                notes="",
            )
            for frame in manifest.frames
        ],
    )
