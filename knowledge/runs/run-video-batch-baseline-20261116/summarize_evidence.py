"""Promote bounded recorder evidence and derive the reported prefix metrics."""

from __future__ import annotations

import hashlib
import json
import shutil
import statistics
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
SOURCE = REPOSITORY / "outputs/tennis_scene/video_batch_baseline/20261116"
DESTINATION = Path(__file__).resolve().parent


def main() -> None:
    rows = []
    files = []
    for repeat in range(1, 4):
        for condition in ("cold_component_cache", "cache_hit"):
            name = f"repeat-{repeat}-{condition}.json"
            path = SOURCE / "recordings" / name
            record = json.loads(path.read_text())
            receipts = json.loads((SOURCE / "receipts" / name).read_text())
            outcomes = json.loads((SOURCE / "outcomes" / name).read_text())
            wall = record["total_wall_seconds"]
            row = {key: record[key] for key in ("metadata", "total_wall_seconds", "stage_wall_seconds", "stage_inclusive_seconds", "models", "resources", "batch_shapes")}
            completed = sum(item["status"] == "generated" for item in outcomes)
            row.update(
                recording_status=record["status"],
                reconstruction_completed_clips=completed,
                effective_cache_condition="cold_component_cache" if condition == "cold_component_cache" else "completed_prefix_cache_only",
                all_node_cache_hit=False,
                stage_wall_fractions={stage: seconds / wall for stage, seconds in record["stage_wall_seconds"].items()},
                clip_outcomes=[{"clip_id": item["clip_id"], "status": item["status"], "stop_node": receipt.get("active_stage"),
                                "node_statuses": receipt.get("stage_status", {}), "error": item["error"]}
                               for item, receipt in zip(outcomes, receipts, strict=True)],
            )
            if completed or any(item["stop_node"] is None or not item["stop_node"].startswith("court_detection/") for item in row["clip_outcomes"]):
                raise ValueError("This summary is only for the observed Court failure prefix; review other outcomes separately")
            rows.append(row)
            for folder in ("recordings", "receipts", "outcomes"):
                original = SOURCE / folder / name
                size = original.stat().st_size
                digest = hashlib.sha256(original.read_bytes()).hexdigest()
                destination = DESTINATION / folder / name
                if size > 256 * 1024:
                    raise ValueError(f"Evidence exceeds knowledge storage policy: {original}")
                destination.parent.mkdir(exist_ok=True)
                shutil.copyfile(original, destination)
                files.append({"path": f"{folder}/{name}", "sha256": digest, "bytes": size})
    cold_warm = [row["total_wall_seconds"] for row in rows if row["metadata"]["condition"] == "cold_component_cache" and row["metadata"]["repeat"] > 1]
    prefix = [row["total_wall_seconds"] for row in rows if row["metadata"]["condition"] == "cache_hit"]
    metrics = {
        "scope": "Court-failure-prefix-only; full-model baseline and full-cache-hit are not established",
        "cold_initial_seconds": rows[0]["total_wall_seconds"],
        "cold_process_warm_seconds": {"values": cold_warm, "mean": statistics.mean(cold_warm), "sample_stdev": statistics.stdev(cold_warm)},
        "completed_prefix_cache_seconds": {"values": prefix, "mean": statistics.mean(prefix), "sample_stdev": statistics.stdev(prefix)},
        "rows": rows,
    }
    (DESTINATION / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    (DESTINATION / "evidence-files.json").write_text(json.dumps(files, indent=2) + "\n")
    total = sum(path.stat().st_size for path in DESTINATION.rglob("*") if path.is_file())
    if total > 2 * 1024**2:
        raise ValueError(f"Run exceeds knowledge total storage policy: {total}")
    print(json.dumps({"run_bytes": total, "cold_initial_seconds": metrics["cold_initial_seconds"],
                      "cold_warm": metrics["cold_process_warm_seconds"], "prefix": metrics["completed_prefix_cache_seconds"]}))


if __name__ == "__main__":
    main()
