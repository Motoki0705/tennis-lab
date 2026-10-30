"""Validate inventory/report completeness before calling an observation a baseline."""

from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .lanes import audit_partition, read_plan
from .runner import write_json


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _junit_address(nodeid: str) -> tuple[str, str]:
    address, bracket, parameters = nodeid.partition("[")
    parts = address.split("::")
    parts[0] = parts[0].replace("/", ".").removesuffix(".py")
    parts[-1] += bracket + parameters
    return ".".join(parts[:-1]), parts[-1]


def _elapsed(stop: float | None, start: float | None) -> float | None:
    return stop - start if stop is not None and start is not None else None


def summarize(root: Path) -> dict[str, Any]:
    report_started = time.perf_counter()
    issues: list[str] = []
    processes = []
    controller: list[dict[str, Any]] = []
    inventories: list[list[str]] = []
    universes: dict[str, list[list[str]]] = defaultdict(list)
    fixtures: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("events-*.jsonl")):
        events = []
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                issues.append(f"invalid event JSON: {path.name}:{number}")
        configured = [e for e in events if e.get("event") == "configured"]
        if any(e.get("monotonic_s") is None for e in events):
            issues.append(f"missing monotonic event timing: {path.name}")
        if any(
            e["event"] == "phase" and e.get("reported_monotonic_s") is None
            for e in events
        ):
            issues.append(f"missing monotonic phase report timing: {path.name}")
        if len(configured) != 1:
            issues.append(f"missing/duplicate process configuration: {path.name}")
            continue
        role = configured[0]["role"]
        universes[role].extend(
            e["nodeids"] for e in events if e["event"] == "cpu_universe"
        )
        finished = [e for e in events if e["event"] == "session_finish"]
        if len(finished) != 1:
            issues.append(f"missing/duplicate session finish: {path.name}")
        collections = [e for e in events if e["event"] == "collection"]
        # The xdist controller has no items; inventories arrive from each worker.
        if role != "controller":
            if len(collections) != 1:
                issues.append(f"missing/duplicate worker collection: {path.name}")
            inventories.extend(e["nodeids"] for e in collections)
        if role == "controller":
            if controller:
                issues.append("multiple pytest controllers")
            controller = events
        first_phase = next((e["start"] for e in events if e["event"] == "phase"), None)
        processes.append(
            {
                "role": role,
                "pid": configured[0]["pid"],
                "plugin_import_at": configured[0]["imported_at"],
                "configured_at": configured[0]["at"],
                "plugin_import_monotonic_s": configured[0].get("imported_monotonic_s"),
                "configured_monotonic_s": configured[0].get("monotonic_s"),
                "session_start_at": next(
                    (e["at"] for e in events if e["event"] == "session_start"), None
                ),
                "collection_finished_at": collections[0]["at"] if collections else None,
                "session_start_monotonic_s": next(
                    (
                        e.get("monotonic_s")
                        for e in events
                        if e["event"] == "session_start"
                    ),
                    None,
                ),
                "collection_finished_monotonic_s": (
                    collections[0].get("monotonic_s") if collections else None
                ),
                "first_phase_at": first_phase,
                "finish": finished[0] if len(finished) == 1 else None,
                "deselected_count": sum(
                    len(e["nodeids"]) for e in events if e["event"] == "deselected"
                ),
            }
        )
        for event in events:
            if event["event"] != "fixture":
                continue
            key = f"{event['source']} [{event['scope']}]"
            item = fixtures.setdefault(
                key,
                {
                    "fixture": key,
                    "count": 0,
                    "setup_s": 0.0,
                    "errors": 0,
                    "skipped": 0,
                    "xfailed": 0,
                },
            )
            item["count"] += 1
            item["setup_s"] += event["setup_s"]
            item["errors"] += event["outcome"] == "error"
            item["skipped"] += event["outcome"] == "skipped"
            item["xfailed"] += event["outcome"] == "xfailed"
    metadata = _json(root / "environment.json")
    run = _json(root / "run.json")
    if not metadata:
        issues.append("missing environment metadata")
    if not run or run.get("finished_at") is None:
        issues.append("missing runner completion")
    if run.get("source_unchanged") is False:
        issues.append("source changed after measurement launch")
    if run.get("started_at") is not None and (
        run.get("started_monotonic_s") is None
        or run.get("command_finished_monotonic_s") is None
    ):
        issues.append("missing monotonic command timing")
    if (
        run.get("started_monotonic_s") is not None
        and run.get("command_finished_monotonic_s") is not None
        and run["command_finished_monotonic_s"] < run["started_monotonic_s"]
    ):
        issues.append("negative monotonic command duration")
    for process_row in processes:
        if any(
            process_row[key] is None
            for key in (
                "plugin_import_monotonic_s",
                "configured_monotonic_s",
                "session_start_monotonic_s",
            )
        ):
            issues.append(f"missing monotonic process timing: {process_row['role']}")
    workers = metadata.get("workers")
    worker_collections = [e for e in controller if e["event"] == "worker_collection"]
    if workers is not None and workers > 0:
        if (
            len(worker_collections) != workers
            or len({e["worker"] for e in worker_collections}) != workers
        ):
            issues.append("missing/duplicate controller worker inventory")
        if len(inventories) != workers:
            issues.append("missing/duplicate worker process inventory")
        inventories.extend(e["nodeids"] for e in worker_collections)
    else:
        inventories.extend(
            e["nodeids"] for e in controller if e["event"] == "collection"
        )
    nodeids = inventories[0] if inventories else []
    if not inventories or not nodeids:
        issues.append("empty/missing selected inventory")
    if any(ids != nodeids for ids in inventories):
        issues.append("worker collection mismatch")
    if len(set(nodeids)) != len(nodeids):
        issues.append("duplicate selected nodeids")
    lane_scope = None
    if metadata.get("lane_plan_id") is not None:
        try:
            plan = read_plan(root / "lane-plan.json")
            if plan["plan_id"] != metadata["lane_plan_id"]:
                raise ValueError("lane plan differs from runner metadata")
            roles = {f"gw{i}" for i in range(workers)} if workers else {"controller"}
            manifests = {
                p.stem.removeprefix("shard-"): json.loads(p.read_text())
                for p in root.glob("shard-*.json")
            }
            if (
                set(manifests) != roles
                or {role for role, ids in universes.items() if ids} != roles
            ):
                raise ValueError(
                    "missing/extra pre-selection CPU universe or worker manifest"
                )
            reference = None
            for role in sorted(roles):
                if len(universes[role]) != 1:
                    raise ValueError("missing/duplicate pre-selection CPU universe")
                universe = universes[role][0]
                if reference is not None and reference != universe:
                    raise ValueError("workers disagree on pre-selection CPU universe")
                reference = universe
                audit_partition(manifests[role], universe, plan)
                if manifests[role]["selected_nodeids"] != nodeids:
                    raise ValueError(
                        "executed inventory differs from lane/shard selection"
                    )
            manifest = manifests[sorted(roles)[0]]
            lane_scope = {
                "plan_id": plan["plan_id"],
                "scope": plan["scope"],
                "universe_count": len(manifest["universe_nodeids"]),
                "requested_count": len(manifest["requested_nodeids"]),
                "deferred_count": len(manifest["deferred"]),
                "deferred_suites": plan["deferred_suites"],
            }
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            issues.append(f"invalid lane evidence: {error}")
    elif (root / "lane-plan.json").exists() or any(universes.values()):
        issues.append("lane evidence has no runner plan identity")
    phases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in controller:
        if event["event"] == "phase":
            phases[event["nodeid"]].append(event)
    if set(phases) - set(nodeids):
        issues.append("reports outside selected inventory")
    results = []
    for nodeid in nodeids:
        reports = phases.get(nodeid, [])
        names = Counter(e["phase"] for e in reports)
        setup = next((e for e in reports if e["phase"] == "setup"), None)
        complete = names["setup"] == names["teardown"] == 1 and all(
            n == 1 for n in names.values()
        )
        if setup is None or (setup["outcome"] == "passed" and names["call"] != 1):
            complete = False
        outcomes = {e["outcome"] for e in reports}
        if not complete:
            status = "incomplete"
        elif "failed" in outcomes:
            status = "failed"
        elif any(e.get("wasxfail") is not None for e in reports):
            status = "xfailed" if "skipped" in outcomes else "xpassed"
        elif "skipped" in outcomes:
            status = "skipped"
        else:
            status = "passed"
        results.append(
            {
                "nodeid": nodeid,
                "status": status,
                "phase_s": {e["phase"]: e["duration_s"] for e in reports},
                "phase_outcomes": {e["phase"]: e["outcome"] for e in reports},
                "failure_details": {
                    e["phase"]: e["failure_detail"]
                    for e in reports
                    if e.get("failure_detail") is not None
                },
                "worker": reports[0]["worker"] if reports else None,
                "torch_threads": [e.get("torch_threads") for e in reports],
            }
        )
    counts = dict(Counter(item["status"] for item in results))
    if counts.get("incomplete"):
        issues.append("selected tests have missing/duplicate phases")
    collection_results = [e for e in controller if e["event"] == "collection_result"]
    junit: dict[str, Any] = {}
    try:
        cases = list(ET.parse(root / "junit.xml").iter("testcase"))
        junit = {
            "testcases": len(cases),
            "time_s": sum(float(case.get("time", "0")) for case in cases),
            "failed": sum(
                case.find("failure") is not None or case.find("error") is not None
                for case in cases
            ),
            "skipped": sum(case.find("skipped") is not None for case in cases),
        }
        double_failures = {
            nodeid
            for nodeid, events in phases.items()
            if {e["phase"] for e in events if e["outcome"] == "failed"}
            >= {"call", "teardown"}
        }
        if len(cases) != len(phases) + len(collection_results) + len(double_failures):
            issues.append(
                "JUnit testcase count differs from execution/collection reports"
            )
        failed_cases = sum(
            any(e["outcome"] == "failed" for e in events) for events in phases.values()
        )
        failed_cases += len(double_failures) + sum(
            e["outcome"] == "failed" for e in collection_results
        )
        skipped_cases = sum(
            any(e["outcome"] == "skipped" for e in events) for events in phases.values()
        )
        skipped_cases += sum(e["outcome"] == "skipped" for e in collection_results)
        if junit["failed"] != failed_cases:
            issues.append("JUnit failure count differs from phase reports")
        if junit["skipped"] != skipped_cases:
            issues.append("JUnit skip count differs from phase reports")
        addresses = [_junit_address(nodeid) for nodeid in [*phases, *double_failures]]
        addresses.extend(_junit_address(e["nodeid"]) for e in collection_results)
        if Counter(addresses) != Counter(
            (case.get("classname"), case.get("name")) for case in cases
        ):
            issues.append("JUnit test identities differ from phase reports")
        phase_total = sum(e["duration_s"] for events in phases.values() for e in events)
        if abs(junit["time_s"] - phase_total) > 0.001 * len(cases) + 0.01:
            issues.append(
                "JUnit total time differs from phase reports (beyond XML rounding)"
            )
    except (FileNotFoundError, ET.ParseError, ValueError) as error:
        issues.append(f"missing/invalid JUnit: {type(error).__name__}")
    phase_s = {
        phase: sum(
            e["duration_s"]
            for events in phases.values()
            for e in events
            if e["phase"] == phase
        )
        for phase in ("setup", "call", "teardown")
    }
    for process_row in processes:
        process_row["first_phase_at"] = min(
            (
                e["start"]
                for events in phases.values()
                for e in events
                if e["worker"] == process_row["role"]
            ),
            default=None,
        )
    finish_codes = [p["finish"]["exitstatus"] for p in processes if p["finish"]]
    successful = (
        not issues
        and run.get("status") == "completed"
        and run.get("exitcode") == 0
        and bool(finish_codes)
        and all(code == 0 for code in finish_codes)
        and not counts.get("failed")
        and not any(
            e["event"] == "collection_result" and e["outcome"] == "failed"
            for e in controller
        )
    )
    files: dict[str, dict[str, Any]] = {}
    for item in results:
        filename = item["nodeid"].split("::", 1)[0]
        row = files.setdefault(
            filename,
            {"file": filename, "count": 0, "phase_s": dict.fromkeys(phase_s, 0.0)},
        )
        row["count"] += 1
        for phase, seconds in item["phase_s"].items():
            row["phase_s"][phase] += seconds
    started = run.get("started_monotonic_s")
    command_finished = run.get("command_finished_monotonic_s")
    controller_process = next((p for p in processes if p["role"] == "controller"), None)
    last_phase_report = max(
        (
            e["reported_monotonic_s"]
            for events in phases.values()
            for e in events
            if e.get("reported_monotonic_s") is not None
        ),
        default=None,
    )
    timing = {
        "command_to_plugin_import_s": _elapsed(
            controller_process["plugin_import_monotonic_s"], started
        )
        if controller_process
        else None,
        "plugin_import_to_configure_s": _elapsed(
            controller_process["configured_monotonic_s"],
            controller_process["plugin_import_monotonic_s"],
        )
        if controller_process
        else None,
        "worker_session_to_collection_s": {
            p["role"]: _elapsed(
                p["collection_finished_monotonic_s"], p["session_start_monotonic_s"]
            )
            for p in processes
            if p["collection_finished_at"] is not None
        },
        "command_to_all_workers_collected_s": _elapsed(
            max(
                (e["monotonic_s"] for e in worker_collections if "monotonic_s" in e),
                default=None,
            ),
            started,
        ),
        "last_phase_report_to_command_exit_s": _elapsed(
            command_finished, last_phase_report
        ),
    }
    summary = {
        "schema_version": 1,
        "successful_test_baseline": successful,
        "scoped_cpu_success": successful and lane_scope is not None,
        "lane_scope": lane_scope,
        "full_cpu_baseline": (
            successful
            and metadata.get("full_cpu_target") is True
            and metadata.get("shard") is None
            and (lane_scope is None or lane_scope["deferred_count"] == 0)
            and bool(metadata.get("source", {}).get("commit"))
            and run.get("source_unchanged") is True
        ),
        "issues": issues,
        "selected_count": len(nodeids),
        "counts": counts,
        "collection_results": collection_results,
        "inventory_sha256": hashlib.sha256(
            "\n".join(sorted(nodeids)).encode()
        ).hexdigest(),
        "phase_elapsed_s": phase_s,
        "torch_intraop_threads_observed": sorted(
            {
                e["torch_threads"]["intraop"]
                for events in phases.values()
                for e in events
                if e.get("torch_threads")
            }
        ),
        "junit": junit,
        "run": run,
        "processes": processes,
        "timing_breakdown": timing,
        "report_generation_s": time.perf_counter() - report_started,
        "test_command_wall_s": _elapsed(command_finished, started),
        "wall_clock_adjustment_s": (
            run["command_finished_at"]
            - run["started_at"]
            - (command_finished - started)
            if started is not None
            and command_finished is not None
            and run.get("command_finished_at") is not None
            and run.get("started_at") is not None
            else None
        ),
        "event_write_s": sum(
            p["finish"]["event_write_s"] for p in processes if p["finish"]
        ),
        "largest_process_max_rss_kib": max(
            (p["finish"]["max_rss_kib"] for p in processes if p["finish"]), default=None
        ),
        "fixtures": sorted(
            fixtures.values(), key=lambda item: item["setup_s"], reverse=True
        ),
        "files": sorted(
            files.values(), key=lambda item: sum(item["phase_s"].values()), reverse=True
        ),
    }
    write_json(root / "inventory.json", nodeids)
    write_json(root / "results.json", results)
    lines = [
        "# CPU test measurements",
        "",
        f"Successful test baseline: **{successful}**; successful full CPU baseline: **{summary['full_cpu_baseline']}**.",
        f"Full CPU target requested: **{metadata.get('full_cpu_target')}**.",
        f"Selected: {len(nodeids)}; outcomes: `{json.dumps(counts, sort_keys=True)}`.",
        f"Lane coverage (deferred cases are not skips or passes): `{json.dumps(lane_scope, sort_keys=True)}`.",
        f"Command wall seconds: {summary['test_command_wall_s']}; lock wait seconds: {run.get('lock_wait_s')}.",
        f"Wall-clock adjustment relative to monotonic elapsed seconds: {summary['wall_clock_adjustment_s']}.",
        f"Summed worker phase seconds: `{json.dumps(phase_s, sort_keys=True)}`.",
        f"Largest individual pytest process RSS (KiB, not tree peak): {summary['largest_process_max_rss_kib']}.",
        f"Measured event serialization/write seconds (lower bound on overhead): {summary['event_write_s']:.4f}.",
        "",
        "## Completeness issues",
        "",
        *(f"- {issue}" for issue in issues),
    ]
    if not issues:
        lines.append("None.")
    lines += [
        "",
        "## Largest test files",
        "",
        "| File | Cases | Setup s | Call s | Teardown s |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary["files"][:20]:
        times = item["phase_s"]
        lines.append(
            f"| {item['file']} | {item['count']} | {times['setup']:.3f} | {times['call']:.3f} | {times['teardown']:.3f} |"
        )
    lines += [
        "",
        "## Largest fixture setups",
        "",
        "| Fixture | Executions | Setup s | Errors |",
        "|---|---:|---:|---:|",
    ]
    for fixture in summary["fixtures"][:20]:
        lines.append(
            f"| {fixture['fixture']} | {fixture['count']} | {fixture['setup_s']:.3f} | {fixture['errors']} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    summary["report_generation_s"] = time.perf_counter() - report_started
    write_json(root / "summary.json", summary)
    return summary
