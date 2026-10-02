"""CPU metadata-only inventory and explicit extrapolation; never opens media."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["status"] != "complete":
        raise ValueError("The inventory requires a complete evidence manifest")
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"clips": 0, "frames": 0})
    for record in manifest["clips"]:
        clip = record["clip"]
        if clip["split"] not in {"train", "val"} or "video_001" in clip["clip_id"]:
            raise ValueError("Forbidden video or split in the requested context inventory")
        row = counts[clip["source"]]
        row["clips"] += 1
        row["frames"] += clip["frame_count"]
    reference = args.repository / "knowledge/runs/run-i964-coco-fullframe-r5-20260929/inference.json"
    timing = json.loads(reference.read_text())
    # Use only the already recorded video_002 timing, without any frames or labels.
    rows = [v for k, v in timing["archives"]["coco_fullframe_0.01"].items() if k.startswith("video_002/")]
    reference_frames = sum(r["frames"] for r in rows)
    reference_seconds = sum(r["frames"] * r["ms_per_frame"] / 1000 for r in rows)
    seconds_per_frame = reference_seconds / reference_frames
    total_frames = sum(r["frames"] for r in counts.values())
    total_clips = sum(r["clips"] for r in counts.values())
    if (total_clips, total_frames) != (329, 145767):
        raise ValueError("The campaign inventory changed; revise the budget explicitly")
    result: dict[str, Any] = {
        "status": "metadata_audit_complete_not_enqueued",
        "manifest": {"path": str(args.manifest), "sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        "counts": dict(counts), "total_clips": total_clips, "total_frames": total_frames,
        "reference": {"path": str(reference.relative_to(args.repository)),
                      "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                      "subset": "video_002/clip_013, three cameras; timing metadata only",
                      "frames": reference_frames, "seconds": reference_seconds,
                      "seconds_per_frame": seconds_per_frame,
                      "scope": timing["runtime_scope"], "floor": timing["floor"]},
        "estimates_not_measurements": {
            "detector_hours_by_source": {k: r["frames"] * seconds_per_frame / 3600 for k, r in counts.items()},
            "detector_hours_total": total_frames * seconds_per_frame / 3600,
            "complete_job_hours_range": [22, 33], "peak_vram_gib_range": [6, 9],
            "selected_pose_raw_bytes_at_cap6": total_frames * (6 * (17 * 3 * 4 + 3 * 4 + 1) + 8 * 2 + 4),
            "context_cache_budget_gib": 0.5, "additional_output_budget_gib": 1,
        },
        "current_grant": {"wall_hours_total": 4, "peak_vram_gb": 10, "new_jobs": 0},
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
