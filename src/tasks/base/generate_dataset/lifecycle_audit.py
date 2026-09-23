"""Audit completed full-source datasets without loading dense observation arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def audit_full_source_dataset(
    root: Path, *, max_concurrent: int, require_uniform: bool, tolerance: float = 0.05
) -> dict[str, Any]:
    seconds: NDArray[np.float64] = np.zeros(max_concurrent + 1, dtype=np.float64)
    scene_count = 0
    tracks = 0
    for path in sorted((root / "scenes").glob("*/meta.json")):
        meta = json.loads(path.read_text())
        scalars_path = path.parent / "scalars.json"
        scalars = json.loads(scalars_path.read_text()) if scalars_path.exists() else {}
        instances = meta.get("track_instances", scalars.get("track_instances"))
        if not isinstance(instances, list) or not instances:
            raise ValueError(f"{path}: missing lifecycle records")
        t = int(meta["num_frames"])
        fps = float(meta["fps"] if "fps" in meta else meta["fps_out"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"{path}: invalid fps")
        changes: NDArray[np.int64] = np.zeros(t + 1, dtype=np.int64)
        for track in instances:
            birth, death = int(track["birth_frame"]), int(track["death_frame"])
            if (
                not 0 <= birth < death <= t
                or track["source_start"] != 0
                or track["source_end"] != death - birth
            ):
                raise ValueError(f"{path}: truncated or invalid full-source lifecycle")
            changes[birth] += 1
            changes[death] -= 1
        count = np.cumsum(changes)[:-1]
        if (
            count[0] == 0
            or int(count.max()) > max_concurrent
            or max(x["death_frame"] for x in instances) != t
        ):
            raise ValueError(f"{path}: invalid first birth, concurrency, or scene end")
        seconds += np.bincount(count, minlength=max_concurrent + 1) / fps
        scene_count += 1
        tracks += len(instances)
    if not scene_count:
        raise ValueError(f"{root}: no scenes to audit")
    fractions = seconds[1:] / seconds[1:].sum()
    report = {
        "lifecycle_contract": "full_source_birth_balanced_v1",
        "scene_count": scene_count,
        "source_count": tracks,
        "occupancy_seconds": seconds.tolist(),
        "positive_occupancy_fractions": fractions.tolist(),
        "uniform_tolerance": tolerance,
        "uniform_required": require_uniform,
        "max_uniform_deviation": float(np.max(np.abs(fractions - 1 / max_concurrent))),
    }
    (root / "lifecycle_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if require_uniform and report["max_uniform_deviation"] > tolerance:
        raise ValueError(
            f"Birth occupancy is outside the dataset quota: {fractions.tolist()}; see lifecycle_audit.json"
        )
    return report
