"""Derive timing/inventory and explicit rclone-call estimates from smoke evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new analysis output file.")
    evidence = Path("knowledge/runs/run-court-scale-colab-resume-l4-20261023")
    request = read(evidence / "drive-receipts/request.json")
    final = read(evidence / "drive-receipts/completed.json")
    resumed = read(evidence / "resumed-attempt.json")
    caches = [
        read(evidence / f"drive-receipts/caches/{event['cache_key']}.json")
        for event in resumed["caches"]
    ]
    metrics = [
        read(evidence / f"cache-metrics/{cache['cache_key']}.json") for cache in caches
    ]
    assert len(caches) == 24
    assert len({cache["archive_path"] for cache in caches}) == len(caches)
    assert [event["reused"] for event in resumed["caches"]] == [True] + [False] * 23
    elapsed = {}
    for stage in ("interrupt", "resume", "fetch"):
        log = evidence / f"colab-operations/{stage}/stdout.log"
        found = re.findall(r"^elapsed_seconds=([0-9.]+)$", log.read_text(), re.M)
        assert len(found) == 1
        elapsed[stage] = float(found[0])
    new_times = [datetime.fromisoformat(cache["verified_at"]) for cache in caches[1:]]
    assert new_times == sorted(new_times)
    intervals = [
        (later - earlier).total_seconds()
        for earlier, later in zip(new_times, new_times[1:], strict=False)
    ]
    new_seconds = sum(metric["generation_seconds"] for metric in metrics[1:])
    generating_between = sum(metric["generation_seconds"] for metric in metrics[2:])
    version = read(evidence / final["dataset_version_path"])
    accepted = sum(scene["counts"]["qa_accepted"] for scene in version["scenes"])
    candidates = sum(scene["counts"]["candidate_unique"] for scene in version["scenes"])
    cache_bytes = sum(cache["archive_bytes"] for cache in caches)
    # d includes tennis_lab itself. These count explicit RcloneBackend.run calls,
    # not HTTP requests hidden inside rclone. All ancestor folders already exist.
    object_depth = len(request["archive_drive"]["path"].split("/")) + 1
    receipt_depth = object_depth + 1
    get_object = 2 * object_depth + 3  # pre/post resolve, copyto, two stat calls
    get_receipt = 2 * receipt_depth + 3
    fetch_cache = receipt_depth + get_receipt + get_object
    fetch_fixed_bundle = object_depth + 2 * get_object
    new_put_object = 4 * object_depth + 3  # exists, upload pre/post, inspect
    new_put_receipt = 4 * receipt_depth + 3
    new_cache = (
        2 * receipt_depth  # prepare exists, publish receipt check
        + new_put_object
        + new_put_receipt
        + receipt_depth
        + get_receipt  # receipt readback
    )
    new_fixed_bundle = 2 * object_depth + 2 * new_put_object + get_object
    restored_cache = receipt_depth + fetch_cache
    resumed_calls = (
        2 * fetch_fixed_bundle
        + object_depth
        + restored_cache
        + 23 * new_cache
        + new_fixed_bundle
        + new_put_object
    )
    fetch_calls = 2 * fetch_fixed_bundle + len(caches) * fetch_cache
    # Proposal budget: verified put returns the ID from its verified readback;
    # optional get returns absence from its own preflight, retaining pre/post checks.
    proposed_new_cache = 2 * receipt_depth + get_object + 2 * get_receipt
    proposed_fixed_fetch = 2 * get_object
    proposed_fixed_publish = object_depth + 3 * get_object
    proposed_fetch_calls = 2 * proposed_fixed_fetch + len(caches) * (
        get_receipt + get_object
    )
    proposed_resume_calls = (
        2 * proposed_fixed_fetch
        + object_depth
        + get_receipt
        + get_object
        + 23 * proposed_new_cache
        + proposed_fixed_publish
        + get_object
    )
    assert (object_depth, receipt_depth, fetch_cache, new_cache) == (7, 8, 44, 109)
    steady_seconds = (sum(intervals) - generating_between) / len(intervals)
    inputs = (
        [
            evidence / "colab-operations" / stage / "stdout.log"
            for stage in ("interrupt", "resume", "fetch")
        ]
        + [evidence / "resumed-attempt.json"]
        + sorted((evidence / "drive-receipts").rglob("*.json"))
        + sorted((evidence / "cache-metrics").glob("*.json"))
    )
    result: dict[str, Any] = {
        "schema": "court_scale_durability_cost_analysis_v1",
        "source_run": evidence.name,
        "command_seconds": elapsed,
        "generation_seconds": {
            "total": sum(metric["generation_seconds"] for metric in metrics),
            "resume_new_only": new_seconds,
            "resume_other": elapsed["resume"] - new_seconds,
        },
        "inventory": {
            "caches": len(caches),
            "candidates": candidates,
            "accepted": accepted,
            "input_archive_bytes": request["archive_bytes"],
            "cache_archive_bytes": cache_bytes,
            "publication_archive_bytes": final["archive_bytes"],
            "fetch_archive_bytes": request["archive_bytes"]
            + cache_bytes
            + final["archive_bytes"],
            "archive_count": len(caches) + 2,
            "receipt_count": len(caches) + 2,
            "attempt_file_count": 2,
            "archive_member_count": sum(len(cache["files"]) for cache in caches)
            + len(request["files"])
            + len(final["files"]),
            "resume_input_archive_downloads": 2,
        },
        "cache_intervals": {
            "count": len(intervals),
            "seconds": intervals,
            "median_seconds": statistics.median(intervals),
            "min_seconds": min(intervals),
            "max_seconds": max(intervals),
            "non_generation_mean_seconds": steady_seconds,
            "note": "Between archive verified_at points; includes previous receipt completion and next cache work, not isolated upload timings.",
        },
        "code_derived_rclone_calls": {
            "scope": "success path; existing ancestor folders; no retries; includes local stat; not HTTP counts",
            "fetch_cache": fetch_cache,
            "new_cache": new_cache,
            "resume": resumed_calls,
            "fetch": fetch_calls,
            "fetch_resolve_listings": 2 * (5 * object_depth)
            + len(caches) * (3 * receipt_depth + 2 * object_depth),
            "fetch_local_stat": 2 * (len(caches) + 2),
            "fetch_remote_stat": 2 * (len(caches) + 2),
            "fetch_copyto": 2 * (len(caches) + 2),
            "proposed_new_cache": proposed_new_cache,
            "proposed_resume": proposed_resume_calls,
            "proposed_fetch": proposed_fetch_calls,
            "proposal_status": "design budget only; not implemented or benchmarked",
        },
        "scenarios_not_main_experiment_decisions": [
            {
                "accepted_images": number,
                "caches_at_observed_occupancy": math.ceil(
                    number * len(caches) / accepted
                ),
                "caches_if_full_64_and_observed_acceptance": math.ceil(
                    number / (64 * accepted / candidates)
                ),
                "overhead_hours_at_observed_occupancy": math.ceil(
                    number * len(caches) / accepted
                )
                * steady_seconds
                / 3600,
                "overhead_hours_if_full_64": math.ceil(
                    number / (64 * accepted / candidates)
                )
                * steady_seconds
                / 3600,
                "cache_archive_gb_at_observed_ratio": number
                * cache_bytes
                / accepted
                / 1e9,
            }
            for number in (6000, 24000, 96000)
        ],
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
        },
    }
    calls = result["code_derived_rclone_calls"]
    assert fetch_calls == sum(
        calls[key]
        for key in (
            "fetch_resolve_listings",
            "fetch_local_stat",
            "fetch_remote_stat",
            "fetch_copyto",
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "source_sha256"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
