"""Informational test growth and cost reports, independent of the coverage gate."""

from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

# The policy and units are owned by tests/README.md, not by the CI gate.
CASE_SECONDS_LIMIT = 5
EXPANSION_LIMIT = 50
SETUP_SECONDS_LIMIT = 1
SCHEMA_VERSION = 1
DISPLAY_LIMIT = 20


def inventory_details(
    copies: list[list[dict[str, Any]] | None], nodeids: list[str]
) -> dict[str, Any]:
    """Keep metadata separate from the existing nodeid-only inventory contract."""
    try:
        if not copies or any(copy is None for copy in copies):
            raise ValueError("missing collection details (originalname/callspec)")
        reference = copies[0]
        if any(copy != reference for copy in copies):
            raise ValueError("workers disagree on collection details")
        if reference is None or [item["nodeid"] for item in reference] != nodeids:
            raise ValueError("collection details do not match the inventory")
        for item in reference:
            if (
                item["file"] != item["nodeid"].split("::", 1)[0]
                or not isinstance(item["family"], str)
                or not item["family"]
                or not isinstance(item["originalname"], str)
                or not item["originalname"]
                or type(item["parametrized"]) is not bool
                or (
                    item["callspec_id"] is not None
                    and not isinstance(item["callspec_id"], str)
                )
                or not isinstance(item["fixtures"], list)
                or any(not isinstance(name, str) for name in item["fixtures"])
            ):
                raise ValueError("invalid collection details")
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        return {"schema_version": SCHEMA_VERSION, "items": [], "issues": [str(error)]}
    return {"schema_version": SCHEMA_VERSION, "items": reference, "issues": []}


def validate_snapshot(data: dict[str, Any]) -> dict[str, Any]:
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"incompatible growth schema: {data['schema_version']}")
    if data["complete"] is not True:
        raise ValueError(
            "incomplete growth observations: " + str(data.get("reason", "unspecified"))
        )
    nodeids = data["nodeids"]
    if (
        not isinstance(nodeids, list)
        or not nodeids
        or any(not isinstance(node, str) for node in nodeids)
        or len(set(nodeids)) != len(nodeids)
    ):
        raise ValueError("invalid growth inventory")
    details = inventory_details([data["details"]], nodeids)
    if details["issues"]:
        raise ValueError("; ".join(details["issues"]))
    if [row["nodeid"] for row in data["results"]] != nodeids:
        raise ValueError("growth results do not match the inventory")
    for row in data["results"]:
        if row["status"] not in {"passed", "failed", "skipped", "xfailed", "xpassed"}:
            raise ValueError("incomplete growth case")
        phases = row["phase_s"]
        if not {"setup", "teardown"} <= phases.keys() <= {"setup", "call", "teardown"}:
            raise ValueError("invalid growth phases")
        for seconds in phases.values():
            if (
                type(seconds) not in (float, int)
                or not math.isfinite(seconds)
                or seconds < 0
            ):
                raise ValueError("invalid growth phase seconds")
    for fixture in data["fixtures"]:
        if (
            not isinstance(fixture["fixture"], str)
            or not isinstance(fixture["name"], str)
            or not isinstance(fixture["scope"], str)
            or type(fixture["count"]) is not int
            or fixture["count"] < 1
            or any(
                type(fixture[key]) not in (int, float)
                or not math.isfinite(fixture[key])
                or fixture[key] < 0
                for key in ("setup_s", "max_setup_s")
            )
            or fixture["max_setup_s"] > fixture["setup_s"]
        ):
            raise ValueError("invalid growth fixture observation")
    return data


def aggregate_snapshot(root: Path, audit: dict[str, Any]) -> dict[str, Any]:
    """Read audited shards once; fixture events are never added to test phases."""
    if not audit["observations_complete"]:
        raise ValueError("current shard observations are incomplete")
    results: dict[str, Any] = {}
    details: dict[str, Any] = {}
    fixtures: dict[str, Any] = {}
    for directory in sorted(root.glob("*/tests")):
        selected = json.loads((directory / "inventory.json").read_text())
        collected = json.loads((directory / "inventory-details.json").read_text())
        if collected["schema_version"] != SCHEMA_VERSION or collected["issues"]:
            raise ValueError(
                f"{directory.parent.name}: unavailable collection details: {collected['issues']}"
            )
        selected_set = set(selected)
        details.update(
            {
                item["nodeid"]: item
                for item in collected["items"]
                if item["nodeid"] in selected_set
            }
        )
        for row in json.loads((directory / "results.json").read_text()):
            if row["nodeid"] in results:
                raise ValueError("duplicate growth case across shards")
            results[row["nodeid"]] = row
        summary = json.loads((directory / "summary.json").read_text())
        for row in summary["fixtures"]:
            key = row["fixture"]
            if key not in fixtures:
                fixtures[key] = dict(row)
            else:
                for name in ("count", "setup_s", "errors", "skipped", "xfailed"):
                    fixtures[key][name] += row[name]
                fixtures[key]["max_setup_s"] = max(
                    fixtures[key]["max_setup_s"], row["max_setup_s"]
                )
    if len(results) != audit["requested_count"]:
        raise ValueError("growth inventory differs from audited R")
    nodeids = sorted(results)
    return validate_snapshot(
        {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            **{
                key: audit[key]
                for key in (
                    "source_sha",
                    "run_id",
                    "attempt",
                    "universe_count",
                    "requested_count",
                    "deferred_count",
                )
            },
            "nodeids": nodeids,
            "details": [details[node] for node in nodeids],
            "results": [results[node] for node in nodeids],
            "fixtures": sorted(fixtures.values(), key=lambda row: row["fixture"]),
        }
    )


def analyze(
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
    *,
    baseline_reason: str | None = None,
    profile: dict[str, Any] | None = None,
    profile_reason: str | None = None,
) -> dict[str, Any]:
    """Compare file scopes so deferred tests never appear as removals or 0 seconds."""
    requested_files = {node.split("::", 1)[0] for node in current["nodeids"]}
    partial = current["deferred_count"] > 0
    old_nodes = (
        {
            node
            for node in baseline["nodeids"]
            if not partial or node.split("::", 1)[0] in requested_files
        }
        if baseline is not None
        else set()
    )
    new_nodes = set(current["nodeids"])
    file_changes = []
    if baseline is not None:
        before_files: dict[str, set[str]] = defaultdict(set)
        after_files: dict[str, set[str]] = defaultdict(set)
        for node in old_nodes:
            before_files[node.split("::", 1)[0]].add(node)
        for node in new_nodes:
            after_files[node.split("::", 1)[0]].add(node)
        for filename in sorted(before_files.keys() | after_files.keys()):
            before = before_files[filename]
            after = after_files[filename]
            if before != after:
                file_changes.append(
                    {
                        "file": filename,
                        "before": len(before),
                        "after": len(after),
                        "added": len(after - before),
                        "removed": len(before - after),
                        "net": len(after) - len(before),
                    }
                )
    old_results = (
        {row["nodeid"]: row for row in baseline["results"]}
        if baseline is not None
        else {}
    )
    details = {row["nodeid"]: row for row in current["details"]}
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    slow_cases = []
    setup_cases = []
    measured: dict[str, float] = defaultdict(float)
    for row in current["results"]:
        node = row["nodeid"]
        total = sum(row["phase_s"].values())
        measured[node.split("::", 1)[0]] += total
        families[details[node]["family"]].append(row)
        if total > CASE_SECONDS_LIMIT:
            previous = old_results.get(node)
            change = (
                "unknown (no baseline)"
                if baseline is None
                else "new case"
                if previous is None
                else "crossed threshold"
                if sum(previous["phase_s"].values()) <= CASE_SECONDS_LIMIT
                else "existing"
            )
            slow_cases.append(
                {
                    "nodeid": node,
                    "total_s": total,
                    "phase_s": row["phase_s"],
                    "status": row["status"],
                    "change": change,
                }
            )
        if row["phase_s"]["setup"] > SETUP_SECONDS_LIMIT:
            setup_cases.append(
                {
                    "nodeid": node,
                    "setup_s": row["phase_s"]["setup"],
                    "status": row["status"],
                    "fixtures": details[node]["fixtures"],
                }
            )
    expansions = [
        {
            "family": family,
            "count": len(rows),
            "total_s": sum(sum(row["phase_s"].values()) for row in rows),
        }
        for family, rows in families.items()
        if len(rows) > EXPANSION_LIMIT
        and any(details[row["nodeid"]]["parametrized"] for row in rows)
    ]
    drift = []
    if profile is not None:
        for filename, seconds in measured.items():
            estimate = profile["file_seconds"].get(filename)
            drift.append(
                {
                    "file": filename,
                    "profile_s": estimate,
                    "measured_s": seconds,
                    "delta_s": seconds - estimate if estimate is not None else None,
                    "delta_percent": 100 * (seconds / estimate - 1)
                    if estimate is not None
                    else None,
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "scope": "R only (requested files); D is unmeasured and excluded from comparisons"
        if partial
        else "Full CPU target",
        **{
            key: current[key]
            for key in ("universe_count", "requested_count", "deferred_count")
        },
        "baseline": None
        if baseline is None
        else {key: baseline[key] for key in ("source_sha", "run_id", "attempt")},
        "baseline_reason": (baseline_reason or "no main baseline was supplied")
        if baseline is None
        else None,
        "file_changes": file_changes if baseline is not None else None,
        "slow_cases": sorted(
            slow_cases,
            key=lambda row: (
                row["change"] == "existing",
                -row["total_s"],
                row["nodeid"],
            ),
        ),
        "expanded_functions": sorted(
            expansions, key=lambda row: (-row["count"], row["family"])
        ),
        "setup_cases": sorted(
            setup_cases, key=lambda row: (-row["setup_s"], row["nodeid"])
        ),
        "slow_fixtures": sorted(
            (
                row
                for row in current["fixtures"]
                if row["max_setup_s"] > SETUP_SECONDS_LIMIT
            ),
            key=lambda row: (-row["max_setup_s"], row["fixture"]),
        ),
        "profile_drift": sorted(
            drift,
            key=lambda row: (
                -abs(row["delta_s"]) if row["delta_s"] is not None else float("-inf"),
                row["file"],
            ),
        )
        if profile is not None
        else None,
        "profile_reason": (profile_reason or "no timing profile was supplied")
        if profile is None
        else None,
    }


def markdown(report: dict[str, Any]) -> str:
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return html.escape(str(value)).replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "## Test growth and cost warnings",
        "",
        "Informational only; thresholds never fail CI. Full rows are saved in growth-summary.json.",
        "",
    ]
    if not report["available"]:
        return "\n".join([*lines, "Report unavailable: " + cell(report["reason"]), ""])
    lines += [
        report["scope"],
        f"U={report['universe_count']}, R={report['requested_count']}, D={report['deferred_count']}.",
        "",
    ]
    if report["baseline"] is None:
        lines.append(
            "Baseline comparison unavailable: " + cell(report["baseline_reason"])
        )
    else:
        base = report["baseline"]
        lines.append(
            f"Baseline: main ancestor {cell(base['source_sha'])}, run {cell(base['run_id'])}, attempt {cell(base['attempt'])}."
        )

    def table(title: str, headers: list[str], rows: list[list[Any]]) -> None:
        lines.extend(["", f"### {title} ({len(rows)})", ""])
        if not rows:
            lines.append("None.")
            return
        lines.extend(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)])
        for row in rows[:DISPLAY_LIMIT]:
            lines.append("| " + " | ".join(cell(value) for value in row) + " |")
        if len(rows) > DISPLAY_LIMIT:
            lines.append(
                f"Showing {DISPLAY_LIMIT} of {len(rows)}; see growth-summary.json for all rows."
            )

    if report["file_changes"] is not None:
        table(
            "Case count changes by file",
            ["File", "Before", "After", "Added", "Removed", "Net"],
            [
                [
                    row[key]
                    for key in ("file", "before", "after", "added", "removed", "net")
                ]
                for row in report["file_changes"]
            ],
        )
    table(
        f"Cases > {CASE_SECONDS_LIMIT} s (setup + call + teardown)",
        ["Case", "Change", "Status", "Total s", "Setup s", "Call s", "Teardown s"],
        [
            [
                row["nodeid"],
                row["change"],
                row["status"],
                row["total_s"],
                *(
                    row["phase_s"].get(phase, "unmeasured")
                    for phase in ("setup", "call", "teardown")
                ),
            ]
            for row in report["slow_cases"]
        ],
    )
    table(
        f"Parametrized functions > {EXPANSION_LIMIT} cases",
        ["Function", "Cases", "Phase s"],
        [
            [row["family"], row["count"], row["total_s"]]
            for row in report["expanded_functions"]
        ],
    )
    table(
        f"Case setups > {SETUP_SECONDS_LIMIT} s",
        ["Case", "Status", "Setup s", "Related fixtures"],
        [
            [row["nodeid"], row["status"], row["setup_s"], ", ".join(row["fixtures"])]
            for row in report["setup_cases"]
        ],
    )
    table(
        f"Individual fixture setups > {SETUP_SECONDS_LIMIT} s",
        [
            "Fixture (scope)",
            "Actual executions",
            "Total setup s",
            "Max single setup s",
            "Errors / skips / xfails",
        ],
        [
            [
                row["fixture"],
                row["count"],
                row["setup_s"],
                row["max_setup_s"],
                f"{row['errors']} / {row['skipped']} / {row['xfailed']}",
            ]
            for row in report["slow_fixtures"]
        ],
    )
    lines.extend(
        [
            "",
            "Fixture setup is contained in case setup; it is never added again. Phase sums are worker elapsed time, not job wall time.",
        ]
    )
    if report["profile_drift"] is None:
        lines.extend(
            ["", "Profile comparison unavailable: " + cell(report["profile_reason"])]
        )
    else:
        table(
            "Profile versus measured phase time",
            ["File", "Profile s", "Measured s", "Delta s", "Delta %"],
            [
                [
                    row["file"],
                    row["profile_s"]
                    if row["profile_s"] is not None
                    else "unmeasured in profile",
                    row["measured_s"],
                    row["delta_s"],
                    row["delta_percent"],
                ]
                for row in report["profile_drift"]
            ],
        )
    return "\n".join(lines) + "\n"
