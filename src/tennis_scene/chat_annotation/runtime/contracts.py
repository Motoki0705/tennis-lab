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

KIT_VERSION = "1.0.0"
SCHEMA_VERSION: Literal["tennis_chat_annotation.v1"] = "tennis_chat_annotation.v1"
Point = Annotated[list[float], Field(min_length=2, max_length=2)]
Box = Annotated[list[float], Field(min_length=4, max_length=4)]
Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")]
FileName = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Review = Literal["complete", "partial", "unreviewed", "unusable"]
NonPlayerRole = Literal[
    "spectator",
    "chair_umpire",
    "line_umpire",
    "ball_person",
    "coach",
    "staff",
    "other",
    "unknown",
]


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
    static_tolerance_px_at_1080p: float = Field(gt=0)
    homography_max_error_px_at_1080p: float = Field(gt=0)
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


class Person(StrictModel):
    track_id: Identifier
    kind: Literal["player", "non_player", "unknown"]
    non_player_role: NonPlayerRole | None
    court_relation: Literal["target", "other", "unknown", "not_applicable"]
    bbox_xyxy: Box | None
    bbox_source: Literal["observed", "inferred", "unresolved"]
    occluded: bool
    truncated: bool
    source_frames: list[int]

    @_after_model_validator
    def consistent(self) -> Self:
        if (self.kind == "non_player") != (self.non_player_role is not None):
            raise ValueError("only non_player must have a non_player_role")
        if (self.bbox_xyxy is None) != (self.bbox_source == "unresolved"):
            raise ValueError("null bbox requires unresolved source")
        if self.bbox_xyxy is not None:
            x1, y1, x2, y2 = self.bbox_xyxy
            if x2 <= x1 or y2 <= y1:
                raise ValueError("bbox must have positive area")
        if self.bbox_source == "inferred" and not self.source_frames:
            raise ValueError("inferred bbox requires evidence frame indices")
        return self


class Ball(StrictModel):
    track_id: Identifier
    center_px: Point | None
    status: Literal["visible", "occluded", "interpolated"] | None
    missing_reason: Literal["out_of_frame", "unresolved"] | None
    source_frames: list[int]

    @_after_model_validator
    def consistent(self) -> Self:
        if self.center_px is None:
            if self.missing_reason is None or self.status not in (None, "occluded"):
                raise ValueError(
                    "unlocalized ball needs a reason and no observed position"
                )
        elif self.status is None or self.missing_reason is not None:
            raise ValueError(
                "localized ball needs one of the three statuses and no missing reason"
            )
        if self.status in ("occluded", "interpolated") and not self.source_frames:
            raise ValueError("estimated ball needs evidence frames")
        if self.status == "interpolated" and len(self.source_frames) != 2:
            raise ValueError("interpolation requires exactly two endpoint frames")
        return self


class CourtPoint(StrictModel):
    index: int = Field(ge=0, le=19)
    name: Identifier
    point_px: Point | None
    visibility: Literal[
        "visible", "occluded", "out_of_frame", "unassessed", "unresolved"
    ]
    source: Literal["observed", "inferred", "homography", "unresolved"]
    source_frames: list[int]
    anchor_indices: list[int]

    @_after_model_validator
    def consistent(self) -> Self:
        if (self.point_px is None) != (self.source == "unresolved"):
            raise ValueError("null court point requires unresolved source")
        if self.point_px is not None and self.visibility == "unresolved":
            raise ValueError(
                "localized court point cannot have unresolved visibility; use unassessed for derived geometry"
            )
        if self.visibility == "visible" and self.source != "observed":
            raise ValueError("visible court point must be directly observed")
        if self.source in ("inferred", "homography") and not self.source_frames:
            raise ValueError("derived court point needs source frames")
        if self.source == "homography" and (
            self.index > 14 or len(self.anchor_indices) < 4
        ):
            raise ValueError(
                "homography only completes 0..14 from at least four anchors"
            )
        if self.source != "homography" and self.anchor_indices:
            raise ValueError("only homography points have anchor indices")
        return self


class CourtSample(StrictModel):
    frame_index: int = Field(ge=0)
    orientation: Literal["known", "ambiguous"]
    orientation_note: str
    points: list[CourtPoint] = Field(min_length=20, max_length=20)


class IgnoreRegion(StrictModel):
    bbox_xyxy: Box
    reason: Literal["inseparable_crowd"]


class FrameAnnotation(StrictModel):
    frame_index: int = Field(ge=0)
    source_frame_index: int = Field(ge=0)
    people_review: Review
    balls_review: Review
    court_review: Review
    people: list[Person]
    balls: list[Ball]
    ignore_regions: list[IgnoreRegion]
    court_reference_frame: int | None
    events: list[Literal["hit", "bounce", "cut", "play_start", "play_end"]]
    shot_id: Identifier
    notes: str


class Annotation(StrictModel):
    schema_version: Literal["tennis_chat_annotation.v1"]
    clip_id: Identifier
    kit_id: Digest
    manifest_sha256: Digest
    teacher: str
    inspection_ranges: list[FrameRange]
    camera_review_ranges: list[FrameRange]
    camera_motion: Literal["none", "moving", "unknown"]
    court_mode: Literal["static", "dynamic", "unavailable", "unreviewed"]
    court_samples: list[CourtSample]
    frames: list[FrameAnnotation]


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


def make_template(manifest: ClipManifest, manifest_sha256: str) -> Annotation:
    return Annotation(
        schema_version=SCHEMA_VERSION,
        clip_id=manifest.clip_id,
        kit_id=manifest.kit_id,
        manifest_sha256=manifest_sha256,
        teacher="GPT-6 Astra Pro (Chat)",
        inspection_ranges=[],
        camera_review_ranges=[],
        camera_motion="unknown",
        court_mode="unreviewed",
        court_samples=[],
        frames=[
            FrameAnnotation(
                frame_index=frame.frame_index,
                source_frame_index=frame.source_frame_index,
                people_review="unreviewed",
                balls_review="unreviewed",
                court_review="unreviewed",
                people=[],
                balls=[],
                ignore_regions=[],
                court_reference_frame=None,
                events=[],
                shot_id="shot_000",
                notes="",
            )
            for frame in manifest.frames
            if frame.is_target
        ],
    )


def covered_indices(ranges: list[FrameRange], count: int) -> set[int]:
    result: set[int] = set()
    for interval in ranges:
        if interval.stop > count:
            raise ValueError("review range extends beyond clip")
        result.update(range(interval.start, interval.stop))
    return result
