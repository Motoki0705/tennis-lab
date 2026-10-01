"""Coverage and uncertainty indicators, never an inferred accuracy score."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..runtime.contracts import (
    Ball,
    BallAnnotation,
    BallFrameAnnotation,
    ClipManifest,
    Player,
    PlayerAnnotation,
    SupportedAnnotation,
)
from ..runtime.validation import validate_annotation


def frame_ranges(indices: list[int]) -> list[list[int]]:
    result: list[list[int]] = []
    for index in sorted(set(indices)):
        if result and result[-1][1] == index:
            result[-1][1] = index + 1
        else:
            result.append([index, index + 1])
    return result


def missing_summary(count: int) -> dict[str, Any]:
    return {
        "state": "missing",
        "declared_status": None,
        "validation_status": None,
        "frames": count,
        "reviewed": 0,
        "unreviewed": count,
        "reviewed_percent": 0.0,
        "uncertain_frames": 0,
        "localized_frames": 0,
        "absent_frames": 0,
        "object_count": 0,
        "null_objects": 0,
        "interpolated_objects": 0,
        "out_of_frame_objects": 0,
        "inferred_boxes": 0,
        "occluded_boxes": 0,
        "truncated_boxes": 0,
        "notes_frames": 0,
        "event_boundaries": 0,
        "unreviewed_ranges": [[0, count]],
        "uncertain_ranges": [],
        "errors": [],
        "issues": [],
    }


def inspect_annotation(
    annotation: SupportedAnnotation, manifest: ClipManifest, target: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if (target == "ball" and not isinstance(annotation, BallAnnotation)) or (
        target == "player" and not isinstance(annotation, PlayerAnnotation)
    ):
        raise ValueError(f"{target} 専用JSONではありません")
    if not isinstance(annotation, (BallAnnotation, PlayerAnnotation)):
        raise ValueError("対象別JSONが必要です")
    report = validate_annotation(annotation, manifest)
    count = len(manifest.frames)
    summary = missing_summary(count)
    summary.update(
        declared_status=annotation.status,
        validation_status=report.status,
        errors=report.errors,
        issues=annotation.issues,
    )
    reviewed: set[int] = set()
    uncertain: set[int] = set()
    localized: set[int] = set()
    absent: set[int] = set()
    timeline: list[dict[str, Any]] = []
    for row in annotation.frames:
        index = row.frame_index
        if not 0 <= index < count:
            continue
        objects: Sequence[Ball | Player] = (
            row.balls if isinstance(row, BallFrameAnnotation) else row.players
        )
        is_uncertain = bool(row.notes)
        has_position = False
        null_count = 0
        for obj in objects:
            if isinstance(obj, Ball):
                position = obj.center_px
                missing = position is None and obj.status != "out_of_frame"
                if row.reviewed:
                    summary["interpolated_objects"] += obj.status == "interpolated"
                    summary["out_of_frame_objects"] += obj.status == "out_of_frame"
            else:
                position = obj.bbox_xyxy
                missing = position is None
                if row.reviewed:
                    summary["inferred_boxes"] += obj.bbox_source == "inferred"
                    summary["occluded_boxes"] += obj.occluded
                    summary["truncated_boxes"] += obj.truncated
            has_position |= position is not None
            is_uncertain |= missing
            null_count += missing
        if row.reviewed:
            reviewed.add(index)
            if is_uncertain:
                uncertain.add(index)
            if has_position:
                localized.add(index)
            if not objects and not row.notes:
                absent.add(index)
            summary["object_count"] += len(objects)
            summary["null_objects"] += null_count
            summary["notes_frames"] += bool(row.notes)
            if isinstance(row, BallFrameAnnotation):
                summary["event_boundaries"] += row.interpolation_break
        timeline.append(
            {
                "frame": index,
                "reviewed": row.reviewed,
                "uncertain": is_uncertain if row.reviewed else False,
                "localized": has_position if row.reviewed else False,
                "objects": len(objects),
                "notes": row.notes,
            }
        )
    pending = sorted(set(range(count)) - reviewed)
    state = (
        "invalid"
        if report.errors
        else "unreviewed"
        if not reviewed
        else "in_progress"
        if pending
        else "completed"
        if report.status == "completed"
        else "reviewed_partial"
    )
    summary.update(
        state=state,
        reviewed=len(reviewed),
        unreviewed=len(pending),
        reviewed_percent=round(100 * len(reviewed) / count, 2),
        uncertain_frames=len(uncertain),
        localized_frames=len(localized),
        absent_frames=len(absent),
        unreviewed_ranges=frame_ranges(pending),
        uncertain_ranges=frame_ranges(list(uncertain)),
    )
    return summary, timeline
