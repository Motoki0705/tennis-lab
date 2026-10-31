"""Frame identities and location metrics shared by heatmap and coordinate evaluation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from src.tasks.ball_detection.data.windows.all_frames import LABEL_KINDS


def prediction_identities(records: list[dict[str, Any]]) -> dict[str, Any]:
    digests = {name: hashlib.sha256() for name in ("target_identity", "teacher_identity", "scored_frame")}
    for r in records:
        key = [r["clip_id"], r["frame_index"], 1, r["native_fps"]]
        # This is the manifest's compact JSONL identity, not the report's JSON encoding.
        line = (json.dumps(key, separators=(",", ":")) + "\n").encode()
        digests["target_identity"].update(line)
        code = LABEL_KINDS.index(r["kind"])
        digests["teacher_identity"].update((json.dumps([*key, code], separators=(",", ":")) + "\n").encode())
        if code:
            digests["scored_frame"].update(line)
    return {f"{name}_sha256": digest.hexdigest() for name, digest in digests.items()}


def location_metrics(
    records: list[dict[str, Any]], *, distance_px: float
) -> dict[str, Any]:
    errors = [r["error_px"] for r in records if r["kind"] == "observed"]
    counts = {kind: sum(r["kind"] == kind for r in records) for kind in LABEL_KINDS}
    return dict(
        support=dict(
            frames=len(records), scored_frames=len(records) - counts["masked"], **counts
        ),
        mean_error_px=float(np.mean(errors)) if errors else None,
        median_error_px=float(np.median(errors)) if errors else None,
        p95_error_px=float(np.quantile(errors, 0.95, method="linear"))
        if errors
        else None,
        within_distance_rate=sum(e <= distance_px for e in errors) / len(errors)
        if errors
        else None,
        distance_threshold_source_px=distance_px,
    )
