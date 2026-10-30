"""Rebuild scheduling hints from one complete Actions attempt, including failures."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .shards import inspect_shards


def build_profile(root: Path, split_files: list[str] | None = None) -> dict[str, Any]:
    directories = sorted(root.glob("*/tests"))
    if not directories:
        raise ValueError("no shard observations found under root")
    metadata = json.loads((directories[0] / "environment.json").read_text())
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
        attempt=context["GITHUB_RUN_ATTEMPT"],
    )
    if not audit["observations_complete"]:
        raise ValueError(f"cannot profile incomplete observations: {audit['issues']}")
    if not audit["full_cpu_coverage"]:
        raise ValueError("cannot profile a partial CPU scope with deferred tests")
    file_seconds: dict[str, float] = defaultdict(float)
    test_seconds: dict[str, float] = {}
    observed_splits: set[str] = set()
    outcomes: dict[str, str] = {}
    for directory in directories:
        plan = json.loads(next(directory.glob("shard-*.json")).read_text())
        observed_splits.update(plan["split_files"])
        for result in json.loads((directory / "results.json").read_text()):
            nodeid = result["nodeid"]
            seconds = sum(result["phase_s"].values())
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError(f"invalid measured phase cost: {nodeid}")
            test_seconds[nodeid] = max(round(seconds, 6), 0.000001)
            file_seconds[nodeid.split("::", 1)[0]] += seconds
            outcomes[nodeid] = result["status"]
    selected_splits = sorted(observed_splits if split_files is None else split_files)
    if len(set(selected_splits)) != len(selected_splits) or any(
        name not in file_seconds for name in selected_splits
    ):
        raise ValueError("split files must be unique measured files")
    return {
        "schema_version": 1,
        "source": {
            "run_id": audit["run_id"],
            "attempt": audit["attempt"],
            "commit": audit["source_sha"],
            "shard_count": count,
            "selected_count": audit["selected_count"],
            "inventory_sha256": hashlib.sha256(
                "\n".join(sorted(test_seconds)).encode()
            ).hexdigest(),
            "counts": audit["counts"],
            "successful_test_observations": audit["successful_test_observations"],
            "unsuccessful_cases": {
                name: status
                for name, status in sorted(outcomes.items())
                if status not in {"passed", "skipped", "xfailed"}
            },
        },
        "file_seconds": {
            name: max(round(seconds, 6), 0.000001)
            for name, seconds in sorted(file_seconds.items())
        },
        "split_files": selected_splits,
        "test_seconds": {
            name: seconds
            for name, seconds in sorted(test_seconds.items())
            if name.split("::", 1)[0] in selected_splits
        },
    }
