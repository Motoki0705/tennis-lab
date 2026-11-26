"""Compute clip summaries and write an offline page for one or more stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.multiview_pose.schema import RowRejection
from src.tennis_scene.visualization.pose_comparison.inputs import (
    STOP_LABELS,
    ClipPlan,
    ComparisonInputs,
)
from src.tennis_scene.visualization.pose_comparison.rendering import (
    AMBER,
    BALL,
    ROW_STYLES,
    Color,
    in_core,
    torso_xy,
)


def _summary(
    counts: NDArray[np.int64], core_counts: NDArray[np.int64]
) -> dict[str, Any]:
    def histogram(values: NDArray[np.int64]) -> dict[int, int]:
        unique, frames = np.unique(values, return_counts=True)
        return {int(n): int(count) for n, count in zip(unique, frames, strict=True)}

    return {
        "populated_frames": int(np.count_nonzero(counts)),
        "count_histogram": histogram(counts),
        "core_count_histogram": histogram(core_counts),
    }


def _reprojection(
    errors: NDArray[np.floating] | None, *, empty_reason: str
) -> dict[str, int | float | str | None]:
    if errors is None or not errors.size:
        return {"points": 0, "median_px": None, "p95_px": None, "reason": empty_reason}
    return {
        "points": int(errors.size),
        "median_px": float(np.median(errors)),
        "p95_px": float(np.quantile(errors, 0.95)),
        "reason": None,
    }


def clip_metrics(
    data: ComparisonInputs, plan: ClipPlan, pose_seconds: float
) -> dict[str, Any]:
    scene, candidates = data.legacy, data.candidates
    n = data.source.num_frames
    old_counts = np.zeros(n, dtype=np.int64)
    old_core = np.zeros_like(old_counts)
    if scene is not None:
        old_counts = scene.valid.any(axis=-1).sum(axis=0)
        xy = torso_xy(scene.xyz.reshape(-1, 17, 3), scene.valid.reshape(-1, 17))
        old_core = in_core(xy).reshape(len(scene.player_ids), n).sum(axis=0)
    new_counts = np.diff(candidates.frame_offsets)
    xy = torso_xy(candidates.positions_m, candidates.joint_valid)
    frames: NDArray[np.int64] = np.repeat(np.arange(n), new_counts)
    new_core = np.bincount(frames[in_core(xy)], minlength=n)
    rejected = np.bincount(
        np.concatenate(candidates.row_rejection_codes), minlength=len(RowRejection)
    )
    stopped = plan.baseline_reason is not None and not old_counts.any()
    old_errors = scene.unused_view_errors if scene is not None else None
    if stopped:
        old_reason = "pipeline_stopped"
    elif not old_counts.any():
        old_reason = "no_valid_3d"
    elif old_errors is None:
        old_reason = "inliers_unavailable"
    else:
        old_reason = "no_unused_observations"
    new_errors = candidates.reprojection_px[
        candidates.reprojection_valid & ~candidates.inlier_mask
    ]
    new_reason = "no_unused_observations" if new_counts.any() else "no_valid_3d"
    return {
        "source_frames": n,
        "ball_valid_frames": 0 if scene is None else int(scene.ball_valid.sum()),
        "legacy": {
            **_summary(old_counts, old_core),
            "unused_view_reprojection": _reprojection(
                old_errors, empty_reason=old_reason
            ),
            "status": "stopped" if stopped else "complete",
            "baseline_stop_code": plan.baseline_reason,
            "baseline_stop_label": STOP_LABELS.get(
                plan.baseline_reason or "", (plan.baseline_reason, "")
            )[0],
            "baseline_error": plan.baseline_error,
            "node_seconds": plan.node_seconds,
        },
        "new": {
            **_summary(new_counts, new_core),
            "unused_view_reprojection": _reprojection(
                new_errors, empty_reason=new_reason
            ),
            "status": "complete",
            "reconstruction_seconds": pose_seconds,
            "ambiguous_candidates": int(candidates.ambiguous.sum()),
            "ambiguous_frames": int(len(np.unique(frames[candidates.ambiguous]))),
            "detection_rows": {code.name: int(rejected[code]) for code in RowRejection},
        },
    }


def _css_color(bgr: Color) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def write_page(output: Path, manifest: dict[str, Any]) -> None:
    """Embed the manifest so file:// playback has no JSON fetch or server dependency."""
    document = {
        **manifest,
        "legend": {
            "unassigned": [
                {
                    "code": code.name,
                    "letter": letter,
                    "label": label,
                    "color": _css_color(color),
                }
                for code, (letter, label, color) in ROW_STYLES.items()
            ],
            "ambiguous": _css_color(AMBER),
            "ball": _css_color(BALL),
        },
    }
    metadata = json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2)
    (output / "manifest.json").write_text(metadata + "\n", encoding="utf-8")
    template = Path(__file__).with_name("index.html").read_text(encoding="utf-8")
    html = template.replace("<!-- MANIFEST -->", metadata.replace("<", "\\u003c"))
    (output / "index.html").write_text(html, encoding="utf-8")
