"""Read-only review summaries of saved truth, including explicit unknown states."""

from __future__ import annotations

from collections import Counter
from typing import Any


def target_court_id(sample: dict[str, Any]) -> str | None:
    """V2/V3 bind the target per sample; V1 records it in sparse metadata."""
    target = sample.get("target_court")
    if isinstance(target, dict):
        binding = target.get("binding", target)
        return str(binding["court_instance_id"])
    value = sample.get("metadata", {}).get("target_court")
    return str(value) if value is not None else None


def visibility_summary(projection: dict[str, Any] | None) -> dict[str, Any]:
    """Geometry and renderer visibility are separate authorities, never inferred."""
    courts = []
    total: Counter[str] = Counter()
    for court in projection["courts"] if projection is not None else []:
        counts: Counter[str] = Counter()
        rows = []
        for semantic_class in court["classes"]:
            for point in semantic_class["points"]:
                front = point.get("in_front")
                inside = point.get("in_frame")
                visible = point.get("renderer_visible")
                for value in (front, inside, visible):
                    if value is not None and not isinstance(value, bool):
                        raise ValueError("Visibility must be boolean or unrecorded.")
                counts["total"] += 1
                if inside is True:
                    counts["in_frame"] += 1
                    if visible is False:
                        counts["in_frame_hidden"] += 1
                elif inside is False:
                    counts["out_of_frame"] += 1
                else:
                    counts["unknown_geometry"] += 1
                if front is False:
                    counts["behind_camera"] += 1
                if visible is True:
                    counts["visible"] += 1
                elif visible is None:
                    counts["unknown_renderer"] += 1
                geometry = (
                    "behind_camera"
                    if front is False
                    else "in_frame"
                    if inside is True
                    else "out_of_frame"
                    if inside is False
                    else "unrecorded"
                )
                rows.append(
                    {
                        "class": semantic_class["class_name"],
                        "class_id": semantic_class["class_id"],
                        "physical_index": point["physical_index"],
                        "uv": point["uv"],
                        "geometry": geometry,
                        "renderer": "visible"
                        if visible is True
                        else "not_visible"
                        if visible is False
                        else "unrecorded",
                    }
                )
        keys = (
            "total",
            "in_frame",
            "visible",
            "in_frame_hidden",
            "out_of_frame",
            "behind_camera",
            "unknown_geometry",
            "unknown_renderer",
        )
        result = {key: counts[key] for key in keys}
        total.update(result)
        courts.append(
            {
                "id": court["court_instance_id"],
                "coverage": court["coverage_mode"],
                "counts": result,
                "points": rows,
            }
        )
    return {
        "projection_recorded": projection is not None,
        "counts": dict(total) if projection is not None else None,
        "courts": courts,
    }


def target_projection(sample: dict[str, Any]) -> dict[str, Any]:
    """Select the stored target KP without changing the all-court sparse record."""
    target = target_court_id(sample)
    projection = sample.get("projection")
    if target is None or projection is None:
        raise ValueError("Target court projection is not recorded.")
    courts = [c for c in projection["courts"] if c["court_instance_id"] == target]
    if len(courts) != 1:
        raise ValueError("Target court projection must bind exactly one court.")
    return {**projection, "courts": courts}


def sample_summary(sample: dict[str, Any]) -> dict[str, Any]:
    visibility = visibility_summary(sample["projection"])
    target = target_court_id(sample)
    target_projection = next(
        (c for c in visibility["courts"] if c["id"] == target), None
    )
    return {
        "id": sample["sample_id"],
        "frame": sample["trajectory_frame_index"],
        "view": sample["view_id"],
        "camera": sample["camera"]["camera_id"],
        "target_court": target,
        "resolution": [sample["width"], sample["height"]],
        "counts": visibility["counts"],
        "target_counts": target_projection["counts"] if target_projection else None,
        "target_coverage": target_projection["coverage"] if target_projection else None,
    }


def rejection_summary(dataset: dict[str, Any]) -> dict[str, Any]:
    """Missing candidate details do not mean zero rejections or rejected images."""
    records = dataset.get("rejected_samples")
    reasons: Counter[str] = Counter()
    rows = []
    for sample in records if records is not None else []:
        reasons.update(sample["reasons"])
        projection = sample.get("projection")
        rows.append(
            {
                "id": sample["sample_id"],
                "group": sample["trajectory_group_id"],
                "frame": sample["trajectory_frame_index"],
                "view": sample["view_id"],
                "split": sample["split"],
                "target_court": target_court_id(sample),
                "reasons": sample["reasons"],
                "projection_recorded": projection is not None,
                "renderer_recorded": any(
                    isinstance(p.get("renderer_visible"), bool)
                    for c in projection["courts"]
                    for cls in c["classes"]
                    for p in cls["points"]
                )
                if projection is not None
                else False,
            }
        )
    return {
        "count": dataset["metrics"]["rejected_frame_count"],
        "records_recorded": records is not None,
        "record_count": len(records) if records is not None else None,
        "reason_counts": dict(reasons),
        "samples": rows,
        "images": "not_retained",
    }


def split_target_counts(
    samples: list[dict[str, Any]], courts: list[str]
) -> dict[str, dict[str, int]]:
    """Count accepted sample bindings, keeping empty split/court cells visible."""
    result = {
        split: dict.fromkeys(courts, 0) for split in ("train", "validation", "test")
    }
    for sample in samples:
        target = target_court_id(sample)
        if target is not None:
            if target not in courts:
                raise ValueError("Sample target does not belong to the aligned courts.")
            result[sample["split"]][target] += 1
    return result
