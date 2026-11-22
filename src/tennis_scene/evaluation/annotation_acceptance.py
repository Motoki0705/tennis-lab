"""Post-run coverage reports; clip000 recognition acceptance is explicitly scoped."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from src.tasks.ball_detection.annotation.video import (
    VideoAnnotationInput,
    VideoBallAnnotation,
    summarize_annotation,
)
from src.tennis_scene.ground_truth import UserReference


def _intervals(frames: Sequence[int]) -> list[list[int]]:
    result: list[list[int]] = []
    for frame in frames:
        if result and result[-1][1] == frame:
            result[-1][1] = frame + 1
        else:
            result.append([frame, frame + 1])
    return result


def summarize_video_annotations(
    annotations: Sequence[VideoBallAnnotation],
    *,
    expected_inputs: Sequence[VideoAnnotationInput],
) -> dict[str, Any]:
    expected = {a.source.camera_id: a for a in expected_inputs}
    if (
        not expected
        or len(expected) != len(expected_inputs)
        or len(annotations) != len(expected)
        or {a.source.camera_id for a in annotations} != set(expected)
    ):
        raise ValueError("Report requires exactly the pinned camera/input set")
    cameras = {}
    totals: Counter[str] = Counter()
    for annotation in annotations:
        camera = annotation.source.camera_id
        if annotation.input_identity != expected[camera]:
            raise ValueError("Report annotation differs from its pinned input")
        counts = asdict(summarize_annotation(annotation))
        totals.update(counts)
        cameras[camera] = {
            **counts,
            "states": dict(Counter(f.state for f in annotation.frames)),
            "unknown_intervals": _intervals(
                [f.frame_index for f in annotation.frames if f.state == "unknown"]
            ),
            "unprocessed_intervals": _intervals(
                [f.frame_index for f in annotation.frames if f.state == "unprocessed"]
            ),
        }
    scope = {c: [i.frame_range.start, i.frame_range.stop] for c, i in expected.items()}
    return {
        "schema": "scene_annotation_summary.v1",
        "totals": dict(totals),
        "cameras": cameras,
        "scope": scope,
    }


def clip000_acceptance(report: dict[str, Any]) -> dict[str, Any]:
    """Never let a tiny fixture pass an absolute-count production criterion."""
    if (
        report["scope"] != {c: [0, 1010] for c in ("cam0", "cam1", "cam2")}
        or report["totals"]["total"] != 3030
    ):
        return {
            "status": "not_applicable",
            "reason": "requires all 3030 camera/frames of clip000",
        }
    counts = report["totals"]
    return {
        "status": "passed"
        if counts["position_unavailable"] <= 81
        and counts["final_unknown"] <= 25
        and counts["unprocessed"] == 0
        else "failed",
        "limits": {"position_unavailable": 81, "final_unknown": 25, "unprocessed": 0},
    }


def summarize_reference(reference: UserReference) -> dict[str, Any]:
    """Score finalized reference_kind; source_status describes the original CSV."""
    cameras = {}
    totals: Counter[str] = Counter()
    for camera in reference.identity.camera_ids:
        rows = [
            reference.position(camera, f) for f in range(reference.identity.num_frames)
        ]
        counts = {
            "total": len(rows),
            "position_usable": sum(
                r.valid_for_scoring and r.uv_px is not None for r in rows
            ),
            "final_unknown": sum(r.reference_kind == "unknown" for r in rows),
            "unprocessed": 0,
        }
        counts["position_unavailable"] = counts["total"] - counts["position_usable"]
        totals.update(counts)
        cameras[camera] = {
            **counts,
            "states": dict(Counter(r.reference_kind for r in rows)),
        }
    return {"totals": dict(totals), "cameras": cameras}
