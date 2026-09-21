"""Semantic validation supplements the generated structural JSON Schema."""

from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import Annotation, ClipManifest, ValidationReport, covered_indices
from .geometry import complete_court, court_mode, interpolation_values


def validate_annotation(
    annotation: Annotation,
    manifest: ClipManifest,
    manifest_sha256: str,
    definition: dict[str, Any],
) -> ValidationReport:
    errors: list[str] = []
    issues: list[str] = []
    expected = [frame.frame_index for frame in manifest.frames if frame.is_target]
    if (annotation.clip_id, annotation.kit_id, annotation.manifest_sha256) != (
        manifest.clip_id,
        manifest.kit_id,
        manifest_sha256,
    ):
        errors.append("annotation clip/kit/manifest identity mismatch")
    if [frame.frame_index for frame in annotation.frames] != expected:
        errors.append(
            "annotation frames must exactly match ordered target frames (no context or duplicates)"
        )
    count = len(manifest.frames)
    try:
        inspected = covered_indices(annotation.inspection_ranges, count)
        covered_indices(annotation.camera_review_ranges, count)
    except ValueError as error:
        errors.append(str(error))
        inspected = set()
    samples = {sample.frame_index: sample for sample in annotation.court_samples}
    if len(samples) != len(annotation.court_samples):
        errors.append("duplicate court sample frame")

    def evidence(indices: list[int], label: str) -> None:
        if any(index < 0 or index >= count for index in indices) or len(indices) != len(
            set(indices)
        ):
            errors.append(f"{label}: invalid/duplicate evidence frame indices")

    def inside(point: list[float]) -> bool:
        return 0 <= point[0] < manifest.width and 0 <= point[1] < manifest.height

    for sample in annotation.court_samples:
        label = f"court frame {sample.frame_index}"
        if not 0 <= sample.frame_index < count:
            errors.append(f"{label}: frame outside clip")
        if [p.index for p in sample.points] != list(range(20)) or [
            p.name for p in sample.points
        ] != definition["names"]:
            errors.append(f"{label}: CourtKP20 index/name/order mismatch")
            continue
        if sample.orientation == "ambiguous":
            issues.append(f"{label}: ambiguous orientation")
        for point in sample.points:
            evidence(point.source_frames, label)
            if point.point_px is None:
                issues.append(f"{label} point {point.index}: unresolved")
            elif point.visibility == "unresolved":
                errors.append(
                    f"{label} point {point.index}: localized point has unresolved visibility"
                )
            elif point.visibility == "visible" and not inside(point.point_px):
                errors.append(f"{label}: visible point outside image")
            elif point.visibility == "out_of_frame" and inside(point.point_px):
                errors.append(f"{label}: out_of_frame point inside image")
            if point.source == "observed" and point.source_frames != [
                sample.frame_index
            ]:
                errors.append(f"{label}: observed point must cite its own frame")
        derived = [point for point in sample.points if point.source == "homography"]
        if derived:
            erased = sample.model_copy(deep=True)
            for point in erased.points:
                if point.source == "homography":
                    point.point_px = None
                    point.source = "unresolved"
                    point.visibility = "unresolved"
                    point.source_frames = []
                    point.anchor_indices = []
            try:
                reproduced = complete_court(erased, definition, manifest)
                for point in derived:
                    computed = reproduced.points[point.index]
                    if (
                        point.anchor_indices != computed.anchor_indices
                        or point.source_frames != computed.source_frames
                        or point.point_px is None
                        or computed.point_px is None
                        or not np.allclose(
                            point.point_px, computed.point_px, atol=1e-5, rtol=0
                        )
                    ):
                        errors.append(
                            f"{label}: homography point/provenance cannot be reproduced"
                        )
            except ValueError as error:
                errors.append(f"{label}: {error}")
    if annotation.court_mode == "static":
        try:
            if court_mode(annotation, manifest) != "static":
                errors.append(
                    "static court lacks matching anchors or complete motion review"
                )
        except ValueError as error:
            errors.append(str(error))
    if annotation.court_mode == "unreviewed":
        issues.append("court mode has not been reviewed")
    reviewed = 0
    for frame in annotation.frames:
        index = frame.frame_index
        if not 0 <= index < count:
            errors.append(f"frame {index}: outside clip")
            continue
        label = f"frame {index}"
        if frame.source_frame_index != manifest.frames[index].source_frame_index:
            errors.append(f"{label}: source frame mapping mismatch")
        complete = all(
            state == "complete"
            for state in (frame.people_review, frame.balls_review, frame.court_review)
        )
        if complete and index in inspected:
            reviewed += 1
        else:
            issues.append(f"{label}: not completely inspected/annotated")
        for kind, objects in (("person", frame.people), ("ball", frame.balls)):
            ids = [obj.track_id for obj in objects]
            if len(ids) != len(set(ids)):
                errors.append(f"{label}: duplicate {kind} track ID")
        for person in frame.people:
            evidence(person.source_frames, label)
            if (
                person.kind == "unknown"
                or person.non_player_role == "unknown"
                or person.court_relation == "unknown"
                or person.bbox_xyxy is None
            ):
                issues.append(f"{label} person {person.track_id}: unresolved")
            if person.bbox_xyxy is not None:
                x1, y1, x2, y2 = person.bbox_xyxy
                if (
                    not (
                        0 <= x1 < x2 <= manifest.width
                        and 0 <= y1 < y2 <= manifest.height
                    )
                    and not person.truncated
                ):
                    errors.append(
                        f"{label}: amodal bbox outside image must be marked truncated"
                    )
            if person.bbox_source == "observed" and (
                person.occluded or person.truncated
            ):
                errors.append(
                    f"{label}: occluded/truncated full-body bbox must be inferred"
                )
        for region in frame.ignore_regions:
            x1, y1, x2, y2 = region.bbox_xyxy
            if not (0 <= x1 < x2 <= manifest.width and 0 <= y1 < y2 <= manifest.height):
                errors.append(
                    f"{label}: ignore region must be inside image with positive area"
                )
        for ball in frame.balls:
            evidence(ball.source_frames, label)
            if ball.center_px is None and ball.missing_reason == "unresolved":
                issues.append(f"{label} ball {ball.track_id}: unresolved location")
            if ball.center_px is not None and not inside(ball.center_px):
                errors.append(
                    f"{label}: ball center outside image must be null/out_of_frame"
                )
            if ball.status == "visible" and ball.source_frames != [index]:
                errors.append(f"{label}: visible ball must cite its own frame")
            if ball.status == "interpolated":
                try:
                    values = interpolation_values(
                        annotation, manifest, ball.track_id, *ball.source_frames
                    )
                    if (
                        index not in values
                        or ball.center_px is None
                        or not np.allclose(
                            ball.center_px, values[index], atol=1e-5, rtol=0
                        )
                    ):
                        errors.append(
                            f"{label}: interpolated position cannot be reproduced"
                        )
                except (ValueError, IndexError) as error:
                    errors.append(f"{label}: {error}")
        reference = frame.court_reference_frame
        if reference is not None and reference not in samples:
            errors.append(f"{label}: missing referenced court sample")
        if annotation.court_mode == "static" and reference != 0:
            errors.append(f"{label}: static court must reference anchor frame 0")
        if (
            annotation.court_mode == "dynamic"
            and frame.court_review == "complete"
            and reference != index
        ):
            errors.append(f"{label}: dynamic court needs its own frame's sample")
        if annotation.court_mode == "unavailable" and reference is not None:
            errors.append(f"{label}: unavailable court must not reference geometry")
    return ValidationReport(
        status="failed" if errors else "partial" if issues else "completed",
        reviewed_frames=reviewed,
        target_frames=len(expected),
        errors=errors,
        issues=issues,
    )
