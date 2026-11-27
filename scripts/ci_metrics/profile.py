"""Rebuild scheduling hints from a run's latest complete shards, including failures."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .lanes import parse_attempt
from .shards import inspect_shards


@dataclass(frozen=True)
class PhaseMeasurements:
    by_file: dict[str, float]
    by_case: dict[str, float]
    by_shard: dict[int, float]
    outcomes: dict[str, str]
    split_files: set[str]


def measured_phases(root: Path) -> PhaseMeasurements:
    """Read phase totals rebuilt by the shard audit from the raw observations."""
    file_seconds: dict[str, float] = defaultdict(float)
    test_seconds: dict[str, float] = {}
    shard_seconds: dict[int, float] = {}
    observed_splits: set[str] = set()
    outcomes: dict[str, str] = {}
    for directory in sorted(root.glob("*/tests")):
        manifest = json.loads(next(directory.glob("shard-*.json")).read_text())
        observed_splits.update(manifest["split_files"])
        total = 0.0
        for result in json.loads((directory / "results.json").read_text()):
            nodeid = result["nodeid"]
            phases = result["phase_s"].values()
            seconds = sum(phases)
            if not math.isfinite(seconds) or any(
                not math.isfinite(value) or value < 0 for value in phases
            ):
                raise ValueError(f"invalid measured phase cost: {nodeid}")
            test_seconds[nodeid] = seconds
            file_seconds[nodeid.split("::", 1)[0]] += seconds
            outcomes[nodeid] = result["status"]
            total += seconds
        shard_seconds[manifest["index"]] = total
    return PhaseMeasurements(
        file_seconds, test_seconds, shard_seconds, outcomes, observed_splits
    )


def build_profile(root: Path, split_files: list[str] | None = None) -> dict[str, Any]:
    directories = sorted(root.glob("*/tests"))
    if not directories:
        raise ValueError("no shard observations found under root")
    observations = [
        json.loads((directory / "environment.json").read_text())
        for directory in directories
    ]
    metadata = observations[0]
    context = metadata["environment"]
    count = metadata["shard"]["count"]
    if (
        type(count) is not int
        or not 1 <= count <= 32
        or any(
            not isinstance(context.get(key), str) or not context[key]
            for key in ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")
        )
    ):
        raise ValueError(
            "profile requires a valid Actions source/run/attempt and shard count"
        )
    audit = inspect_shards(
        root,
        count,
        sha=context["GITHUB_SHA"],
        run_id=context["GITHUB_RUN_ID"],
        attempt=str(
            max(
                parse_attempt(row["environment"].get("GITHUB_RUN_ATTEMPT"))
                for row in observations
            )
        ),
    )
    if not audit["observations_complete"]:
        raise ValueError(f"cannot profile incomplete observations: {audit['issues']}")
    if not audit["full_cpu_coverage"]:
        raise ValueError("cannot profile a partial CPU scope with deferred tests")
    return profile_from_audit(audit, measured_phases(root), split_files)


def profile_from_audit(
    audit: dict[str, Any], phases: PhaseMeasurements, split_files: list[str] | None
) -> dict[str, Any]:
    selected_splits = sorted(phases.split_files if split_files is None else split_files)
    if len(set(selected_splits)) != len(selected_splits):
        raise ValueError("split files must be unique measured files")
    if unmeasured := sorted(set(selected_splits) - phases.by_file.keys()):
        raise ValueError(
            f"split files must be unique measured files: {', '.join(unmeasured)}"
        )
    return {
        "schema_version": 1,
        "source": {
            "run_id": audit["run_id"],
            "attempt": audit["attempt"],
            "plan_attempt": audit["plan_attempt"],
            "shard_attempts": audit["shard_attempts"],
            "commit": audit["source_sha"],
            "shard_count": audit["shard_count"],
            "selected_count": audit["selected_count"],
            "inventory_sha256": hashlib.sha256(
                "\n".join(sorted(phases.by_case)).encode()
            ).hexdigest(),
            "counts": audit["counts"],
            "successful_test_observations": audit["successful_test_observations"],
            "unsuccessful_cases": {
                name: status
                for name, status in sorted(phases.outcomes.items())
                if status not in {"passed", "skipped", "xfailed"}
            },
        },
        "file_seconds": {
            name: max(round(seconds, 6), 0.000001)
            for name, seconds in sorted(phases.by_file.items())
        },
        "split_files": selected_splits,
        "test_seconds": {
            name: max(round(seconds, 6), 0.000001)
            for name, seconds in sorted(phases.by_case.items())
            if name.split("::", 1)[0] in selected_splits
        },
    }
