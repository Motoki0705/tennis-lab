"""Explicit v2 conversion; target and processing evidence come from the caller."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field

from src.tasks.base.annotation.contracts import Digest, Document

from .contracts import (
    PixelPosition,
    PointProvenance,
    VideoAnnotationInput,
    VideoBallAnnotation,
    VideoBallFrame,
)


class MigrationEvidence(Document):
    """Decisions bound to exact old bytes, never inferred from a track ID."""

    original_sha256: Digest
    target_semantics: Literal["active_ball"]
    target_evidence: str = Field(min_length=1)
    processing_complete: Literal[True]
    processing_evidence: str = Field(min_length=1)


class MigrationRecord(Document):
    schema_version: Literal["ball.v2-migration.v1"] = "ball.v2-migration.v1"
    evidence: MigrationEvidence
    expected_input_sha256: Digest
    annotation_sha256: Digest
    original_scope: str
    counts: dict[str, int]
    # These fields stay in the retained, hash-pinned original JSON.
    legacy_metadata: tuple[str, ...] = (
        "annotation_notes", "review", "summary", "coordinate_system",
        "frames.timestamp_seconds", "frames.coordinate_source",
        "frames.occlusion_estimation", "frames.segment_id", "frames.image_score",
        "frames.review_decision",
    )


class _LegacyDocument(Document):
    # Historical diagnostic fields vary between exporters. They are retained in
    # the original artifact, not silently promoted into v3 teaching fields.
    model_config = ConfigDict(extra="allow")


class _Point(Document):
    x: float
    y: float


class _Source(_LegacyDocument):
    file_name: str
    media_sha256: Digest = Field(alias="sha256")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_count: int = Field(gt=0)
    rotation: Literal[0]
    time_base: str
    fps_numerator: int = Field(gt=0)
    fps_denominator: int = Field(gt=0)


class _Target(_LegacyDocument):
    track_id: int | str
    label: Literal["tennis_ball"]
    scope: str = Field(min_length=1)


class _Frame(_LegacyDocument):
    frame_index: int = Field(ge=0)
    pts: int
    track_id: int | str
    label: Literal["tennis_ball"]
    status: Literal["observed", "interpolated", "occlusion_estimated", "unresolved"]
    visibility: str
    center_px: _Point | None
    center_normalized: _Point | None
    support_frames: tuple[int, ...]
    break_before: bool
    notes: str = ""


class _LegacyAnnotation(_LegacyDocument):
    schema_version: Literal["video_ball_annotation.v2"]
    source: _Source
    target: _Target
    coordinate_system: dict[str, Any]
    summary: dict[str, Any]
    frames: tuple[_Frame, ...]


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def convert_v2_annotation(
    path: Path, *, expected_input: VideoAnnotationInput, evidence: MigrationEvidence
) -> tuple[VideoBallAnnotation, MigrationRecord]:
    """Convert a complete, independently identified active-ball input without IO writes.

    The caller validates original media bytes/decoding and preserves the old JSON.
    Evidence may cite the old processing report; it must not claim new recognition
    QA. Absence, unprocessed status and occlusion of interpolation are unrecoverable.
    Missing completion/target evidence must be resolved before calling this API.
    """
    expected = VideoAnnotationInput.model_validate_json(expected_input.canonical_bytes())
    evidence = MigrationEvidence.model_validate_json(evidence.canonical_bytes())
    if not evidence.target_evidence.strip() or not evidence.processing_evidence.strip():
        raise ValueError("migration requires nonblank target and processing evidence")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != evidence.original_sha256:
        raise ValueError("v2 bytes differ from migration evidence")
    json.loads(payload, object_pairs_hook=_unique_keys)
    legacy = _LegacyAnnotation.model_validate_json(payload)
    source = legacy.source
    identity = expected.source
    if (
        (source.file_name, source.media_sha256, source.width, source.height, source.frame_count)
        != (identity.file_name, identity.media_sha256, identity.width, identity.height, identity.frame_count)
        or Fraction(source.time_base) != Fraction(identity.time_base)
        or Fraction(source.fps_numerator, source.fps_denominator) != Fraction(identity.nominal_fps)
        or expected.frame_range.start != 0
        or expected.frame_range.stop != source.frame_count
        or str(legacy.target.track_id) != expected.target.track_id
    ):
        raise ValueError("v2 source/target differs from pinned input")
    coordinates = legacy.coordinate_system
    if (
        coordinates.get("origin") != "top_left"
        or coordinates.get("frame_index") != "zero_based"
        or coordinates.get("x_axis") != "right"
        or coordinates.get("y_axis") != "down"
        or coordinates.get("normalization") != {"x": "x_px / width", "y": "y_px / height"}
    ):
        raise ValueError("unsupported v2 coordinate contract")
    if len(legacy.frames) != len(expected.frames):
        raise ValueError("v2 frame count differs from pinned input")
    visibility = {
        "observed": "visible", "interpolated": "not_independently_visible",
        "occlusion_estimated": "occluded", "unresolved": "unknown",
    }
    frames = []
    for row, pinned in zip(legacy.frames, expected.frames, strict=True):
        if (row.frame_index, row.pts) != (pinned.frame_index, pinned.pts):
            raise ValueError("v2 frame/PTS differs from pinned input")
        if row.track_id != legacy.target.track_id or row.visibility != visibility[row.status]:
            raise ValueError("v2 row target/visibility contradicts its status")
        point, normalized = row.center_px, row.center_normalized
        position = None
        if row.status == "unresolved":
            if point is not None or normalized is not None or row.support_frames:
                raise ValueError("unresolved v2 row must have null coordinates and no support")
        else:
            if point is None or normalized is None:
                raise ValueError("located v2 row requires both coordinate fields")
            if max(abs(point.x - normalized.x * source.width), abs(point.y - normalized.y * source.height)) > 1e-3:
                raise ValueError("v2 pixel and normalized coordinates disagree")
            position = PixelPosition(x=point.x, y=point.y, provenance=PointProvenance(
                kind=row.status, support_frames=row.support_frames,
            ))
        frames.append(VideoBallFrame(
            frame_index=row.frame_index, pts=row.pts,
            state="unknown" if row.status == "unresolved" else "in_frame",
            occlusion=("not_occluded" if row.status == "observed" else
                       "occluded" if row.status == "occlusion_estimated" else "unknown"),
            position=position, break_before=row.break_before, notes=row.notes,
        ))
    counts = {kind: sum(row.status == kind for row in legacy.frames) for kind in visibility}
    if legacy.summary.get("counts") != counts:
        raise ValueError("v2 summary counts contradict its rows")
    if legacy.summary.get("frame_count") != len(frames):
        raise ValueError("v2 summary frame count contradicts its rows")
    annotation = VideoBallAnnotation(
        schema_version="video_ball_annotation.v3", source=identity, target=expected.target,
        frame_range=expected.frame_range, coordinate_system="native_pixels_top_left_endpoint",
        frames=tuple(frames),
    )
    return annotation, MigrationRecord(
        evidence=evidence, expected_input_sha256=expected.sha256,
        annotation_sha256=annotation.sha256, original_scope=legacy.target.scope,
        counts=dict(Counter(row.status for row in legacy.frames)),
    )
