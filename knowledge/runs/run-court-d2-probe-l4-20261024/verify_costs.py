"""Require the revised estimator to reproduce the old budgets under old inputs."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

from analyze_probe import FROZEN, HERE, read, write
from update_costs import PILOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", required=True, type=Path)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=False)
    evidence = PILOT / "evidence"
    counts = read(evidence / "dataset-version.json")["scenes"]
    metrics = read(evidence / "cache-metrics.json")
    previous = read(FROZEN / "cost-scenarios.json")
    selected = read(FROZEN / "render-selection.json")["recipes"]
    probes = []
    for variant in selected:
        for scene in counts:
            name = scene["scene_id"]
            jpeg = sum(
                r["storage"]["jpeg_shard_bytes"]
                for r in metrics
                if r["cache"]["scene_id"] == name
            )
            probes.append(
                {
                    "variant": variant,
                    "scene": name,
                    "weighted_acceptance_given_geometry": scene["counts"]["qa_accepted"]
                    / scene["counts"]["geometry_pass"],
                    "weighted_bytes_per_accepted_jpeg": jpeg
                    / scene["counts"]["qa_accepted"],
                }
            )
    fixture = args.scratch / "old-input-fixture.json"
    result_path = args.scratch / "old-input-costs.json"
    write(
        fixture,
        {
            "fixture_only": True,
            "scenarios": probes,
            "variants": dict.fromkeys(selected),
        },
    )
    subprocess.run(
        [
            sys.executable,
            str(HERE / "update_costs.py"),
            "--analysis",
            str(fixture),
            "--output",
            str(result_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = read(result_path)
    comparisons = 0
    for variant in selected:
        old_variant = previous["variants"][variant]
        new_variant = result["variants"][variant]
        for old, new in zip(
            old_variant["conditions"], new_variant["conditions"], strict=True
        ):
            assert old["name"] == new["name"]
            for left, right in (
                ("accepted_if_old_scene_qa_rates", "accepted_estimate"),
                ("durable_bytes_if_old_jpeg_sizes", "durable_bytes_estimate"),
                (
                    "generation_with_persistence_seconds",
                    "generation_with_persistence_seconds",
                ),
                ("fetch_seconds", "fetch_seconds"),
                ("planning_seconds", "planning_seconds"),
            ):
                assert math.isclose(
                    old[left], new[right], rel_tol=1e-12, abs_tol=1e-5
                ), (variant, old["name"], left)
                comparisons += 1
        for old, new in zip(
            old_variant["budgets"], new_variant["budgets"], strict=True
        ):
            for key in ("main_only_l4_hold_hours", "with_additions_l4_hold_hours"):
                for left, right in zip(old[key], new[key], strict=True):
                    assert math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-8)
                    comparisons += 1
            assert math.isclose(
                old["drive_increment_bytes"],
                new["drive_increment_bytes"],
                rel_tol=1e-12,
            )
            comparisons += 1
    print(
        f"Old-input cost identity passed: {comparisons} scalar comparisons across {len(selected)} variants"
    )


if __name__ == "__main__":
    main()
