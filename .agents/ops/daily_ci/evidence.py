"""Version 1 artifact contract shared by workflows, collectors and recovery."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .model import instant, positive, request_id, sha, window
from .store import load

# Job IDs and shard IDs, not display names. Changes require a contract migration.
PLANS = {
    ("ci.yml", "lint-and-test", "all"): (
        "environment",
        "cpu-backend",
        "system-dependencies",
        "lint",
        "test",
    ),
    ("knowledge.yml", "knowledge", "all"): (
        "environment",
        "graph",
        "repro-paths",
        "test",
    ),
    ("knowledge.yml", "knowledge-webui", "all"): (
        "environment",
        "lint",
        "test",
        "build",
        "browser-install",
        "test-e2e",
    ),
}
IDENTITY_KEYS = {
    "repository",
    "workflow",
    "job",
    "shard",
    "cycle_id",
    "generation",
    "request_id",
    "target_sha",
    "checkout_sha",
    "baseline_sha",
    "deadline_at",
    "run_id",
    "attempt",
}


def validate_identity(identity: dict[str, Any]) -> tuple[str, ...]:
    if set(identity) != IDENTITY_KEYS:
        raise ValueError("incomplete evidence identity")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", identity["repository"], flags=re.ASCII):
        raise ValueError("invalid repository")
    for key in ("generation", "run_id", "attempt"):
        positive(identity[key])
    for key in ("target_sha", "checkout_sha", "baseline_sha"):
        sha(identity[key])
    if identity["target_sha"] != identity["checkout_sha"]:
        raise ValueError("checkout SHA mismatch")
    plan_key = identity["workflow"], identity["job"], identity["shard"]
    if plan_key not in PLANS:
        raise ValueError("unknown workflow/job/shard")
    cycle, _, deadline = window(instant(identity["deadline_at"]))
    if cycle != identity["cycle_id"] or instant(identity["deadline_at"]) != deadline:
        raise ValueError("invalid cycle/deadline")
    if identity["request_id"] != request_id(
        identity["repository"], cycle, identity["generation"], identity["workflow"]
    ):
        raise ValueError("request identity mismatch")
    return PLANS[plan_key]


def artifact_name(identity: dict[str, Any]) -> str:
    validate_identity(identity)
    return "daily-ci-{}-{}-{}-{}-{}-{}".format(
        identity["workflow"],
        identity["job"],
        identity["shard"],
        identity["run_id"],
        identity["attempt"],
        identity["request_id"],
    )


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def pytest_outcomes(report: dict[str, Any]) -> dict[str, str]:
    """Exact, unmodified node IDs. A rename/absence never implies recovery."""
    if (
        set(report)
        != {
            "schema_version",
            "identity",
            "collected",
            "reports",
            "collection_errors",
            "collection_consistent",
            "finished",
            "exit_code",
        }
        or type(report["schema_version"]) is not int
        or report["schema_version"] != 1
    ):
        raise ValueError("invalid pytest manifest schema")
    nodes = report["collected"]
    if (
        not isinstance(nodes, list)
        or not nodes
        or any(not isinstance(n, str) or not n for n in nodes)
        or len(set(nodes)) != len(nodes)
        or not isinstance(report["reports"], dict)
        or not set(report["reports"]) <= set(nodes)
        or type(report["finished"]) is not bool
        or type(report["collection_consistent"]) is not bool
        or not isinstance(report["collection_errors"], list)
        or type(report["exit_code"]) is not int
    ):
        raise ValueError("invalid pytest collection")
    result = {}
    for node in nodes:
        phases = report["reports"].get(node, [])
        if not isinstance(phases, list):
            raise ValueError("invalid pytest reports")
        seen = set()
        for phase in phases:
            if (
                set(phase) != {"when", "outcome", "wasxfail"}
                or phase["when"] not in {"setup", "call", "teardown"}
                or phase["outcome"] not in {"passed", "failed", "skipped"}
                or type(phase["wasxfail"]) is not bool
                or phase["when"] in seen
            ):
                raise ValueError("invalid/duplicate pytest phase")
            seen.add(phase["when"])
        outcomes = {p["outcome"] for p in phases}
        if "failed" in outcomes:
            result[node] = "failed"
        elif "skipped" in outcomes or any(p["wasxfail"] for p in phases):
            result[node] = "skipped"
        elif (
            seen == {"setup", "call", "teardown"}
            and report["finished"]
            and report["collection_consistent"]
            and not report["collection_errors"]
            and report["exit_code"] in (0, 1)
        ):
            result[node] = "passed"
        else:
            result[node] = "incomplete" if phases else "not_run"
    return result


def validate_bundle(
    directory: Path, expected: dict[str, Any], *, remote_conclusion: str
) -> dict[str, Any]:
    """Read an extracted single-job artifact; reject mismatches/missing evidence.

    The caller must derive expected from the cycle and GitHub run/attempt/job,
    not from the artifact. Missing/cancelled jobs remain incomplete at collection.
    Returns result, identity, checks and exact nodeid -> outcome mapping.
    """
    plan = validate_identity(expected)
    if directory.is_symlink() or (directory / "manifest.json").is_symlink():
        raise ValueError("symlink is not an evidence artifact")
    if remote_conclusion not in {"success", "failure"}:
        raise ValueError("remote job is not a completed success/failure")
    manifest = read_json(directory / "manifest.json")
    if (
        set(manifest)
        != {
            "schema_version",
            "identity",
            "started_at",
            "finished_at",
            "checks",
            "files",
            "validation_error",
        }
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["identity"] != expected
    ):
        raise ValueError("manifest schema/identity mismatch")
    if manifest["validation_error"] is not None or not manifest["finished_at"]:
        raise ValueError("unvalidated or unfinished job")
    _, start, deadline = window(instant(expected["deadline_at"]))
    started, finished = (
        instant(manifest["started_at"]),
        instant(manifest["finished_at"]),
    )
    if not start <= started <= finished < deadline:
        raise ValueError("job outside maintenance window")
    checks = manifest["checks"]
    if not isinstance(checks, dict) or tuple(checks) != plan:
        raise ValueError("required checks missing/changed")
    for check in checks.values():
        if set(check) != {"outcome", "started_at", "finished_at", "exit_code", "argv"}:
            raise ValueError("invalid check schema")
        outcome = check["outcome"]
        if outcome not in {"passed", "failed", "not_run", "incomplete"}:
            raise ValueError("invalid check outcome")
        if outcome in {"passed", "failed"} and (
            type(check["exit_code"]) is not int
            or (check["exit_code"] == 0) != (outcome == "passed")
            or not started
            <= instant(check["started_at"])
            <= instant(check["finished_at"])
            <= finished
            or not isinstance(check["argv"], list)
            or not check["argv"]
        ):
            raise ValueError("check completion mismatch")
    files = manifest["files"]
    if not isinstance(files, dict) or "checks.junit.xml" not in files:
        raise ValueError("JUnit evidence missing")
    for name, digest in files.items():
        if not re.fullmatch(r"[a-z][a-z0-9_.-]*", name):
            raise ValueError("unsafe evidence filename")
        path = directory / name
        if path.is_symlink() or not path.is_file() or digest_file(path) != digest:
            raise ValueError(f"evidence missing/modified: {name}")
        if name.endswith(".xml"):
            ET.parse(path)
    cases = ET.parse(directory / "checks.junit.xml").findall(".//testcase")
    if [c.attrib["name"] for c in cases] != list(plan):
        raise ValueError("check JUnit coverage mismatch")
    for case in cases:
        outcome = checks[case.attrib["name"]]["outcome"]
        tags = [child.tag for child in case]
        if tags != (
            []
            if outcome == "passed"
            else ["failure"]
            if outcome == "failed"
            else ["skipped"]
        ):
            raise ValueError("check JUnit outcome mismatch")
    tests: dict[str, str] = {}
    if expected["job"] in {"lint-and-test", "knowledge"} and checks["test"][
        "outcome"
    ] in {"passed", "failed"}:
        if (
            not {"pytest.json", "pytest.junit.xml", "pytest.events.jsonl"}
            <= files.keys()
        ):
            raise ValueError("pytest evidence missing")
        report = read_json(directory / "pytest.json")
        if report.get("identity") != expected:
            raise ValueError("pytest identity mismatch")
        tests = pytest_outcomes(report)
        if report["exit_code"] != checks["test"]["exit_code"]:
            raise ValueError("pytest exit code mismatch")
        if checks["test"]["outcome"] == "passed" and (
            report["collection_errors"]
            or not report["collection_consistent"]
            or not report["finished"]
            or any(s in {"failed", "incomplete", "not_run"} for s in tests.values())
        ):
            raise ValueError("pytest success is not supported by reports")
    outcomes = {c["outcome"] for c in checks.values()}
    result = (
        "failure"
        if "failed" in outcomes
        else "incomplete"
        if outcomes != {"passed"}
        else "success"
    )
    if result == "success" and remote_conclusion != "success":
        raise ValueError("remote failure contradicts successful artifact")
    return {"result": result, "identity": expected, "checks": checks, "tests": tests}


def passed_nodeids(evidence: dict[str, Any]) -> frozenset[str]:
    """Only call on validate_bundle's result (after remote identity verification)."""
    return frozenset(
        n for n, outcome in evidence["tests"].items() if outcome == "passed"
    )


def validate_workflow(
    directories: Mapping[tuple[str, str], Path],
    expected: dict[str, Any],
    conclusions: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    """Require every job/shard in one workflow and one run attempt.

    expected is the full trusted identity except job/shard. Map keys are
    (job, shard). GitHub job conclusions must come from that specific attempt.
    """
    if set(expected) != IDENTITY_KEYS - {"job", "shard"}:
        raise ValueError("incomplete workflow identity")
    required = {
        (job, shard)
        for workflow, job, shard in PLANS
        if workflow == expected["workflow"]
    }
    if not required or set(directories) != required or set(conclusions) != required:
        raise ValueError("missing/unknown/duplicate job or shard")
    jobs = {
        key: validate_bundle(
            directories[key],
            {**expected, "job": key[0], "shard": key[1]},
            remote_conclusion=conclusions[key],
        )
        for key in sorted(required)
    }
    results = {job["result"] for job in jobs.values()}
    result = (
        "failure"
        if "failure" in results
        else "incomplete"
        if "incomplete" in results
        else "success"
    )
    return {"result": result, "jobs": jobs}
