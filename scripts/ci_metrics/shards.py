"""Assign the complete collected CPU suite and verify all shard observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .lanes import read_plan, validate_plan
from .report import summarize


def read_timing_profile(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    data = json.loads(raw)
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or not isinstance(data.get("file_seconds"), dict)
    ):
        raise ValueError("invalid CPU file timing profile")
    weights = data["file_seconds"]
    if not weights or any(
        not isinstance(name, str)
        or not name.startswith("tests/")
        or type(value) not in (int, float)
        or not math.isfinite(value)
        or value <= 0
        for name, value in weights.items()
    ):
        raise ValueError("CPU file timings must be finite positive seconds")
    test_weights = data.get("test_seconds", {})
    split_files = data.get("split_files", [])
    if not isinstance(test_weights, dict) or any(
        not isinstance(name, str)
        or "::" not in name
        or type(value) not in (int, float)
        or not math.isfinite(value)
        or value <= 0
        for name, value in test_weights.items()
    ):
        raise ValueError("CPU test timings must be finite positive seconds")
    if (
        not isinstance(split_files, list)
        or any(not isinstance(name, str) or name not in weights for name in split_files)
        or len(set(split_files)) != len(split_files)
    ):
        raise ValueError("split files must be unique measured files")
    return data, hashlib.sha256(raw).hexdigest()


def group_key(nodeid: str, split_files: list[str]) -> str:
    filename = nodeid.split("::", 1)[0]
    return nodeid if filename in split_files else filename


def assign_groups(
    nodeids: list[str],
    count: int,
    weights: dict[str, float],
    *,
    split_files: list[str] | None = None,
    test_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    if count < 1 or count > 32:
        raise ValueError("shard count must be between 1 and 32")
    if not nodeids or len(set(nodeids)) != len(nodeids):
        raise ValueError("CPU inventory must be nonempty and unique")
    files = sorted({nodeid.split("::", 1)[0] for nodeid in nodeids})
    split_files = split_files or []
    test_weights = test_weights or {}
    groups = sorted({group_key(nodeid, split_files) for nodeid in nodeids})
    if len(groups) < count:
        raise ValueError("CPU inventory has fewer files/groups than shards")
    # New files always run. Their explicit estimate uses the measured median;
    # the manifest names them so this is not mistaken for a measured cost.
    unknown_seconds = statistics.median(weights.values())
    counts = Counter(nodeid.split("::", 1)[0] for nodeid in nodeids)
    costs = {
        name: test_weights.get(
            name, weights[name.split("::", 1)[0]] / counts[name.split("::", 1)[0]]
        )
        if "::" in name
        else weights.get(name, unknown_seconds)
        for name in groups
    }
    loads = [0.0] * count
    assignments = {}
    for name in sorted(groups, key=lambda name: (-costs[name], name)):
        index = min(range(count), key=lambda i: (loads[i], i))
        assignments[name] = index
        loads[index] += costs[name]
    return {
        "assignments": assignments,
        "estimated_phase_seconds": loads,
        "unmeasured_files": [name for name in files if name not in weights],
        "unmeasured_file_estimate_seconds": unknown_seconds,
        "split_files": sorted(split_files),
        "unmeasured_cases_in_split_files": [
            name for name in groups if "::" in name and name not in test_weights
        ],
    }


def inspect_shards(
    root: Path,
    count: int,
    *,
    sha: str,
    run_id: str,
    attempt: str,
    lane_plan: dict[str, Any] | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Audit complete observations, retaining failed tests as measured evidence."""
    issues: list[str] = []
    successful = True
    directories = sorted(root.glob("*/tests"))
    if len(directories) != count:
        issues.append(f"expected {count} shard artifacts, found {len(directories)}")
    full_inventory: list[str] | None = None
    requested_inventory: list[str] | None = None
    deferred: dict[str, str] = {}
    reference_lane = lane_plan
    reference_assignments = None
    schema = None
    profile: str | None = None
    indices: list[int] = []
    executed: list[str] = []
    outcomes: Counter[str] = Counter()
    case_outcomes: dict[str, str] = {}
    if lane_plan is not None:
        try:
            validate_plan(lane_plan, root=source_root)
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(f"invalid fixed lane plan: {error}")
    for directory in directories:
        try:
            metadata = json.loads((directory / "environment.json").read_text())
            context = metadata["environment"]
            if (
                metadata["source"]["commit"] != sha
                or metadata["source"]["status"]
                or context["GITHUB_SHA"] != sha
                or context["GITHUB_RUN_ID"] != run_id
                or context["GITHUB_RUN_ATTEMPT"] != attempt
            ):
                raise ValueError("source revision/run/attempt mismatch or dirty source")
            shard = metadata["shard"]
            if shard["count"] != count or not 0 <= shard["index"] < count:
                raise ValueError("unexpected shard index/count")
            if metadata.get("cpu_target_before_sharding") is not True:
                raise ValueError("shard did not request the full CPU target")
            indices.append(shard["index"])
            manifests = sorted(directory.glob("shard-*.json"))
            workers = metadata["workers"]
            expected_names = (
                {f"shard-gw{i}.json" for i in range(workers)}
                if workers
                else {"shard-controller.json"}
            )
            if {path.name for path in manifests} != expected_names:
                raise ValueError("missing/extra worker shard inventories")
            plans = [json.loads(path.read_text()) for path in manifests]
            plan = plans[0]
            if any(other != plan for other in plans):
                raise ValueError(
                    "workers disagree on the full CPU inventory/assignment"
                )
            if (
                plan["schema_version"] not in {1, 2}
                or plan["index"] != shard["index"]
                or plan["count"] != count
                or plan["profile_sha256"] != shard["profile_sha256"]
            ):
                raise ValueError("shard metadata and worker plan disagree")
            if schema is None:
                schema = plan["schema_version"]
            if plan["schema_version"] != schema:
                raise ValueError("mixed legacy and lane-aware shard schemas")
            if schema == 2:
                observed_lane = read_plan(directory / "lane-plan.json")
                if (
                    observed_lane["source_sha"] != sha
                    or observed_lane["run_id"] != run_id
                    or observed_lane["attempt"] != attempt
                    or observed_lane["plan_id"] != metadata.get("lane_plan_id")
                ):
                    raise ValueError("lane plan source/run/attempt mismatch")
                if reference_lane is None:
                    reference_lane = observed_lane
                if observed_lane != reference_lane:
                    raise ValueError(
                        "shards disagree with the fixed lane plan/policy/scope"
                    )
                all_ids = plan["universe_nodeids"]
                requested_ids = plan["requested_nodeids"]
                if reference_assignments is None:
                    reference_assignments = plan["assignments"]
                    deferred = plan["deferred"]
                if (
                    reference_assignments != plan["assignments"]
                    or deferred != plan["deferred"]
                ):
                    raise ValueError(
                        "shards disagree on requested assignment/deferred cases"
                    )
            else:
                if lane_plan is not None or metadata.get("lane_plan_id") is not None:
                    raise ValueError(
                        "legacy shard cannot establish lane-aware coverage"
                    )
                all_ids = requested_ids = plan["full_nodeids"]
            selected = plan["selected_nodeids"]
            if not all_ids or len(set(all_ids)) != len(all_ids):
                raise ValueError("invalid full CPU inventory")
            expected_selection = [
                nodeid
                for nodeid in requested_ids
                if plan["assignments"][group_key(nodeid, plan["split_files"])]
                == shard["index"]
            ]
            if selected != expected_selection:
                raise ValueError(
                    "selected inventory does not match the file assignment"
                )
            if full_inventory is None:
                full_inventory = all_ids
                requested_inventory = requested_ids
                profile = plan["profile_sha256"]
            if (
                all_ids != full_inventory
                or requested_ids != requested_inventory
                or plan["profile_sha256"] != profile
            ):
                raise ValueError("shards disagree on full inventory or timing profile")
            summary = summarize(directory)
            inventory = json.loads((directory / "inventory.json").read_text())
            if (
                summary["issues"]
                or summary["run"]["status"] != "completed"
                or summary["run"]["exitcode"] not in {0, 1}
                or any(
                    row["outcome"] == "failed" for row in summary["collection_results"]
                )
            ):
                raise ValueError(f"incomplete test observations: {summary['issues']}")
            successful &= summary["successful_test_baseline"] is True
            if inventory != selected:
                raise ValueError("executed inventory differs from the shard plan")
            executed.extend(inventory)
            outcomes.update(summary["counts"])
            case_outcomes.update(
                {
                    row["nodeid"]: row["status"]
                    for row in json.loads((directory / "results.json").read_text())
                }
            )
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            issues.append(f"{directory.parent.name}: {error}")
    if sorted(indices) != list(range(count)):
        issues.append("missing/duplicate shard indices")
    if requested_inventory is None or Counter(executed) != Counter(requested_inventory):
        issues.append(
            "shard union differs from the requested CPU target (missing/duplicate tests)"
        )
    suite_results = {}
    if reference_lane is not None:
        for suite in reference_lane["registry"]["suites"]:
            nodes = [
                node
                for node in (full_inventory or [])
                if node.split("::", 1)[0] in suite["files"]
            ]
            measured = Counter(
                case_outcomes[node] for node in nodes if node in case_outcomes
            )
            suite_results[suite["id"]] = {
                "universe_count": len(nodes),
                "requested_count": sum(node in case_outcomes for node in nodes),
                "deferred_count": sum(node in deferred for node in nodes),
                "counts": dict(measured),
                "verified_success": not issues
                and successful
                and bool(nodes)
                and measured["passed"] > 0
                and sum(measured.values()) == len(nodes),
            }
    return {
        "schema_version": 2 if schema == 2 else 1,
        "observations_complete": not issues,
        "successful_test_observations": not issues and successful,
        "issues": issues,
        "source_sha": sha,
        "run_id": run_id,
        "attempt": attempt,
        "shard_count": count,
        "universe_count": len(full_inventory or []),
        "requested_count": len(requested_inventory or []),
        "deferred_count": len(deferred),
        "full_cpu_coverage": not issues and not deferred,
        "full_cpu_success": not issues and successful and not deferred,
        "scoped_cpu_success": not issues and successful,
        "plan_id": reference_lane["plan_id"] if reference_lane else None,
        "policy_sha256": reference_lane["policy_sha256"] if reference_lane else None,
        "event": reference_lane["event"] if reference_lane else None,
        "ref": reference_lane["ref"] if reference_lane else None,
        "scope": reference_lane["scope"] if reference_lane else "legacy_full",
        "suite_results": suite_results,
        "selected_count": len(executed),
        "counts": dict(outcomes),
    }


def verify_shards(
    root: Path,
    count: int,
    *,
    sha: str,
    run_id: str,
    attempt: str,
    needs: dict[str, Any],
    lane_plan: dict[str, Any] | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    result = inspect_shards(
        root,
        count,
        sha=sha,
        run_id=run_id,
        attempt=attempt,
        lane_plan=lane_plan,
        source_root=source_root,
    )
    if not result["successful_test_observations"]:
        result["issues"].append("incomplete/failed test observations")
    if set(needs) != {"plan", "lint", "tests"} or any(
        row.get("result") != "success" for row in needs.values()
    ):
        result["issues"].append("plan, lint and every CPU test job must succeed")
    result["success"] = not result["issues"]
    result["full_cpu_success"] &= result["success"]
    result["scoped_cpu_success"] &= result["success"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--lane-plan", required=True, type=Path)
    args = parser.parse_args()
    plan_error: str | None
    try:
        lane_plan = read_plan(args.lane_plan, root=Path.cwd())
    except (OSError, ValueError, KeyError, TypeError) as error:
        lane_plan = None
        plan_error = f"fixed lane plan unavailable/invalid: {error}"
    else:
        plan_error = None
    result = verify_shards(
        args.root,
        args.count,
        sha=os.environ["GITHUB_SHA"],
        run_id=os.environ["GITHUB_RUN_ID"],
        attempt=os.environ["GITHUB_RUN_ATTEMPT"],
        needs=json.loads(os.environ["CI_JOB_RESULTS"]),
        lane_plan=lane_plan,
        source_root=Path.cwd(),
    )
    if plan_error:
        result["issues"].append(plan_error)
        result["success"] = False
    result["full_cpu_success"] &= result["success"]
    result["scoped_cpu_success"] &= result["success"]
    print(json.dumps(result, indent=2))
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "coverage.json").write_text(json.dumps(result, indent=2) + "\n")
    if summary_file := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary_file).open("a") as stream:
            stream.write(
                "## CPU shard coverage\n\n"
                + (
                    "Full CPU success.\n\n"
                    if result["full_cpu_success"]
                    else "Requested scope success; deferred tests are unverified.\n\n"
                    if result["scoped_cpu_success"]
                    else "CPU coverage failed.\n\n"
                )
                + "```json\n"
                + json.dumps(result, indent=2)
                + "\n```\n"
            )
            if lane_plan is not None:
                from .plan import markdown

                stream.write(markdown(lane_plan))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
