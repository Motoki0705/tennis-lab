"""Validate complete clip coverage, identity, coordinates and interpolation."""

from __future__ import annotations

import numpy as np

from .contracts import Annotation, ClipManifest, ValidationReport, annotation_clip_id
from .geometry import interpolation_values


def validate_annotation(
    annotation: Annotation, manifest: ClipManifest
) -> ValidationReport:
    errors: list[str] = []
    issues = list(annotation.issues)
    count = len(manifest.frames)
    if (
        annotation.clip_id != annotation_clip_id(manifest)
        or annotation.width != manifest.width
        or annotation.height != manifest.height
        or annotation.frame_count != count
    ):
        errors.append("annotation clip identity/dimensions/frame count mismatch")
    if [frame.frame_index for frame in annotation.frames] != list(range(count)):
        errors.append(
            "annotation frames must exactly match all ordered clip frames, including context (no gaps or duplicates)"
        )
    reviewed = 0
    for frame in annotation.frames:
        index = frame.frame_index
        label = f"frame {index}"
        if not 0 <= index < count:
            errors.append(f"{label}: outside clip")
            continue
        if frame.reviewed:
            reviewed += 1
        else:
            issues.append(f"{label}: not fully reviewed")
        if frame.notes:
            issues.append(f"{label}: {frame.notes}")
        for kind, objects in (("player", frame.players), ("ball", frame.balls)):
            ids = [obj.track_id for obj in objects]
            if len(ids) != len(set(ids)):
                errors.append(f"{label}: duplicate {kind} track ID")
        for player in frame.players:
            if player.bbox_xyxy is None:
                issues.append(
                    f"{label} player {player.track_id}: unresolved full-body bbox"
                )
                continue
            x1, y1, x2, y2 = player.bbox_xyxy
            if (
                not (0 <= x1 < x2 <= manifest.width and 0 <= y1 < y2 <= manifest.height)
                and not player.truncated
            ):
                errors.append(
                    f"{label}: amodal bbox outside image must be marked truncated"
                )
        for ball in frame.balls:
            if ball.center_px is None and ball.status != "out_of_frame":
                issues.append(f"{label} ball {ball.track_id}: unresolved location")
            if ball.center_px is not None and not (
                0 <= ball.center_px[0] < manifest.width
                and 0 <= ball.center_px[1] < manifest.height
            ):
                errors.append(f"{label}: ball outside image must be null/out_of_frame")
            if ball.status == "interpolated":
                try:
                    if ball.interpolation_frames is None:
                        raise ValueError("missing interpolation endpoints")
                    values = interpolation_values(
                        annotation, manifest, ball.track_id, *ball.interpolation_frames
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
    if annotation.status == "completed" and issues:
        errors.append("completed annotation contains unreviewed or unresolved work")
    if annotation.status == "partial" and not issues:
        errors.append(
            "partial annotation requires an explanation in issues or frame notes"
        )
    return ValidationReport(
        status="failed" if errors else "partial" if issues else "completed",
        reviewed_frames=reviewed,
        target_frames=count,
        errors=errors,
        issues=issues,
    )
