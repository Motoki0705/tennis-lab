"""Derive storage-only costs from the replay; does not choose experiment sizes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    measurement = json.loads(args.measurement.read_text())
    by_name = {step["name"]: step for step in measurement["steps"]}
    caches = [step for name, step in by_name.items() if name.startswith("new-cache-")]
    steady = [step for step in caches if step["name"] != "new-cache-00"]
    values = sorted(step["seconds"] for step in steady)
    cache_seconds = statistics.median(values)
    result: dict[str, Any] = {
        "schema": "court_scale_transport_analysis_v1",
        "new_cache_count": len(caches),
        "steady_cache_seconds": {
            "median": cache_seconds,
            "min": min(values),
            "max": max(values),
            "p95_nearest_rank": values[math.ceil(0.95 * len(values)) - 1],
            "mean": statistics.mean(values),
        },
        "steady_cache_mean_listing_busy_seconds": statistics.mean(
            s["listing_busy_seconds"] for s in steady
        ),
        "steady_cache_mean_copy_busy_seconds": statistics.mean(
            s["copy_busy_seconds"] for s in steady
        ),
        "steady_cache_mean_local_residual_seconds": statistics.mean(
            s["seconds"] - s["rc_busy_seconds"] for s in steady
        ),
        "first_cache_seconds": by_name["new-cache-00"]["seconds"],
        "same_host_cache_save_speedup": by_name["baseline-new-cache"]["seconds"]
        / cache_seconds,
        "identical_cache_save_speedup": by_name["baseline-new-cache"]["seconds"]
        / by_name["new-cache-00"]["seconds"],
        "same_host_cache_recovery_speedup": by_name["baseline-cache-recovery"][
            "seconds"
        ]
        / by_name["resume-cache-restore"]["seconds"],
        "cache_save_seconds_sum": sum(step["seconds"] for step in caches),
        "replayed_initial_attempt_seconds": by_name["initial-session"]["seconds"],
        "replayed_resume_attempt_seconds": by_name["resume-session"]["seconds"],
        "full_fetch_seconds": by_name["completed-fetch"]["seconds"],
        "full_fetch_vs_prior_colab_ratio": 3663.088
        / by_name["completed-fetch"]["seconds"],
        "cross_host_ratio_is_descriptive_only": True,
        "cost_model": "T = Tinputs + sum(render_seconds_i) + cache_seconds*Knew + Trestore + Tpublication + Tattempt + Tlocal_validation; replay did not render",
        "assumptions": {
            "accepted_in_smoke": 282,
            "candidate_in_smoke": 303,
            "caches_in_smoke": 24,
            "smoke_cache_archive_bytes": 19537920,
            "same_fill_K": "ceil(24*N/282)",
            "full_chunk_K": "ceil(N/(64*(282/303))); optimistic, ignores trajectory/birth-level ends",
        },
        "illustrative_storage_overhead_only": [],
    }
    for images in (6000, 12000, 24000, 48000):
        same_fill = math.ceil(24 * images / 282)
        full_chunk = math.ceil(images / (64 * (282 / 303)))
        result["illustrative_storage_overhead_only"].append(
            {
                "accepted_images": images,
                "same_fill_caches": same_fill,
                "same_fill_save_hours": same_fill * cache_seconds / 3600,
                "full_chunk_caches": full_chunk,
                "full_chunk_save_hours": full_chunk * cache_seconds / 3600,
                "cache_archive_GB": 19537920 * images / 282 / 1e9,
            }
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
