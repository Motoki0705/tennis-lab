"""Refresh CPU scheduling weights from a verified successful main push."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shlex
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .artifacts import fetch_artifacts, inspect_artifacts, read_manifest
from .growth import phase_statistics
from .lanes import parse_attempt, read_plan
from .plan import _api
from .profile import PhaseMeasurements, measured_phases, profile_from_audit
from .runner import write_json
from .shards import assign_groups, group_key, inspect_shards, read_timing_profile

REPOSITORY = "Motoki0705/tennis-lab"
TIMING_PROFILE = Path("scripts/ci_metrics/cpu-file-times.json")


class IneligibleRun(ValueError):
    """The run lacks eligible evidence. Transport and local I/O errors propagate."""


@dataclass(frozen=True)
class RunMeasurements:
    audit: dict[str, Any]
    phases: PhaseMeasurements
    coverage_artifact_id: int
    plan_artifact_id: int
    shard_artifact_ids: dict[str, int]
    scheduling_profile_sha256: str


def _validate_run(run: dict[str, Any], repo: str, run_id: int) -> None:
    if (
        run["id"] != run_id
        or run["path"] != ".github/workflows/ci.yml"
        or run["event"] != "push"
        or run["head_branch"] != "main"
        or run["status"] != "completed"
        or run["conclusion"] != "success"
        or run["repository"]["full_name"].lower() != repo.lower()
        or run["head_repository"]["full_name"].lower() != repo.lower()
        or not re.fullmatch(r"[0-9a-f]{40}", run["head_sha"])
    ):
        raise IneligibleRun("expected a completed successful CI push on main")
    if type(run["run_attempt"]) is not int or run["run_attempt"] < 1:
        raise IneligibleRun("invalid Actions run attempt")


def _latest_runs(repo: str) -> Iterator[dict[str, Any]]:
    page = 1
    while True:
        runs = json.loads(
            _api(
                f"repos/{repo}/actions/workflows/ci.yml/runs"
                f"?branch=main&event=push&status=success&per_page=30&page={page}"
            )
        )["workflow_runs"]
        yield from runs
        if len(runs) < 30:
            return
        page += 1


def _download(
    repo: str,
    run: dict[str, Any],
    artifact: dict[str, Any],
    destination: Path,
    *,
    shard: bool = False,
) -> None:
    artifact_id = artifact["id"]
    identity = artifact["workflow_run"]
    if (
        type(artifact_id) is not int
        or artifact_id <= 0
        or artifact["expired"] is not False
        or identity["id"] != run["id"]
        or identity["head_sha"] != run["head_sha"]
        or identity["head_branch"] != "main"
        or identity["repository_id"] != run["repository"]["id"]
        or identity["head_repository_id"] != run["head_repository"]["id"]
    ):
        raise IneligibleRun(f"artifact {artifact_id} has mismatched run/HEAD metadata")
    archive_bytes = _api(f"repos/{repo}/actions/artifacts/{artifact_id}/zip")
    digest = artifact.get("digest")
    if (
        digest is not None
        and digest != "sha256:" + hashlib.sha256(archive_bytes).hexdigest()
    ):
        raise IneligibleRun(f"artifact {artifact_id} failed the SHA-256 check")
    destination.mkdir(parents=True)
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            members = archive.namelist()
            if len(set(members)) != len(members):
                raise IneligibleRun(
                    f"artifact {artifact_id} has duplicate archive members"
                )
            if shard:
                names = [
                    "tests/" + name
                    for name in (
                        "environment.json",
                        "lane-plan.json",
                        "run.json",
                        "junit.xml",
                    )
                ]
                names += [
                    name
                    for name in members
                    if re.fullmatch(
                        r"tests/(?:events-[^/]+\.jsonl|shard-[^/]+\.json)", name
                    )
                ]
            else:
                names = (
                    ["plan.json"]
                    if artifact["name"].startswith("cpu-lane-plan-")
                    else ["coverage.json", "artifacts.json"]
                )
            for name in names:
                with (
                    archive.open(name) as source,
                    (destination / Path(name).name).open("xb") as stream,
                ):
                    shutil.copyfileobj(source, stream)
    except (KeyError, zipfile.BadZipFile) as error:
        raise IneligibleRun(
            f"artifact {artifact_id} is missing or malformed: {error}"
        ) from error


def _validate_coverage(coverage: dict[str, Any], run: dict[str, Any]) -> None:
    expected = {
        "schema_version": 3,
        "success": True,
        "full_cpu_success": True,
        "observations_complete": True,
        "deferred_count": 0,
        "event": "push",
        "ref": "refs/heads/main",
        "scope": "full",
        "source_sha": run["head_sha"],
        "run_id": str(run["id"]),
        "attempt": str(run["run_attempt"]),
    }
    for key, value in expected.items():
        if type(coverage.get(key)) is not type(value) or coverage[key] != value:
            raise IneligibleRun(f"coverage.json {key} must equal {value!r}")
    counts = coverage["counts"]
    if (
        not isinstance(counts, dict)
        or set(counts) - {"passed", "skipped", "xfailed"}
        or any(type(value) is not int or value < 0 for value in counts.values())
        or counts.get("passed", 0) <= 0
    ):
        raise IneligibleRun(
            "coverage.json must record successful test outcomes with zero failures"
        )


def _measure_run(repo: str, run: dict[str, Any], root: Path) -> RunMeasurements:
    run_id = str(run["id"])
    sha = run["head_sha"]
    attempt = str(run["run_attempt"])
    artifacts = fetch_artifacts(repo, run_id)
    coverage_name = f"cpu-test-coverage-{run_id}-{attempt}"
    coverage_rows = [
        a for a in artifacts if a["name"] == coverage_name and a["expired"] is not True
    ]
    if len(coverage_rows) != 1:
        raise IneligibleRun(
            f"expected exactly one live {coverage_name} artifact, found {len(coverage_rows)}"
        )
    coverage_row = coverage_rows[0]
    aggregate_dir = root / "aggregate"
    _download(repo, run, coverage_row, aggregate_dir)
    coverage = json.loads((aggregate_dir / "coverage.json").read_text())
    _validate_coverage(coverage, run)
    count = coverage["shard_count"]
    if type(count) is not int or not 1 <= count <= 32:
        raise IneligibleRun("invalid coverage shard count")
    manifest = inspect_artifacts(artifacts, run_id, count)
    if not manifest["success"]:
        raise IneligibleRun("; ".join(manifest["issues"]))
    pinned = manifest["artifacts"]
    if (
        read_manifest(aggregate_dir / "artifacts.json", run_id, count) != manifest
        or coverage["artifacts"] != pinned
        or coverage_row["id"] in {row["id"] for row in pinned}
    ):
        raise IneligibleRun(
            "live artifact IDs differ from the successful aggregate's download set"
        )

    pinned_names = {row["name"] for row in pinned}
    by_id = {
        row["id"]: row
        for row in artifacts
        if row["name"] in pinned_names and row["expired"] is False
    }
    plan_row, *shard_rows = pinned
    _download(repo, run, by_id[plan_row["id"]], root / "plan")
    lane_plan = read_plan(root / "plan/plan.json")
    if any(
        lane_plan[key] != coverage[field]
        for key, field in (
            ("plan_id", "plan_id"),
            ("policy_sha256", "policy_sha256"),
            ("source_sha", "source_sha"),
            ("run_id", "run_id"),
            ("event", "event"),
            ("ref", "ref"),
            ("attempt", "plan_attempt"),
        )
    ):
        raise IneligibleRun("fixed lane plan differs from coverage.json")

    def download_shard(item: tuple[int, dict[str, Any]]) -> None:
        index, row = item
        directory = root / row["name"] / "tests"
        _download(repo, run, by_id[row["id"]], directory, shard=True)
        metadata = json.loads((directory / "environment.json").read_text())
        if metadata["shard"]["index"] != index:
            raise IneligibleRun(
                f"artifact {row['id']} does not contain its named shard {index}"
            )

    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(download_shard, enumerate(shard_rows)):
            pass

    audit = inspect_shards(
        root, count, sha=sha, run_id=run_id, attempt=attempt, lane_plan=lane_plan
    )
    if not audit["full_cpu_success"]:
        raise IneligibleRun(
            f"incomplete/failed shard observations: {audit['issues']} (outcomes {audit['counts']})"
        )
    if mismatches := [key for key in audit if audit[key] != coverage.get(key)]:
        raise IneligibleRun(
            f"recomputed shard audit differs from coverage.json: {', '.join(mismatches)}"
        )
    metadata = json.loads(
        (root / shard_rows[0]["name"] / "tests/environment.json").read_text()
    )
    return RunMeasurements(
        {**audit, "success": coverage["success"], "artifacts": pinned},
        measured_phases(root),
        coverage_row["id"],
        plan_row["id"],
        {str(index): row["id"] for index, row in enumerate(shard_rows)},
        metadata["shard"]["profile_sha256"],
    )


def _select_run(
    repo: str, run_id: int | None, attempt: str | None
) -> tuple[RunMeasurements, list[dict[str, Any]]]:
    endpoint = f"repos/{repo}/actions/runs"
    runs = (
        _latest_runs(repo)
        if run_id is None
        else [json.loads(_api(f"{endpoint}/{run_id}"))]
    )
    rejected: list[dict[str, Any]] = []
    for run in runs:
        ident = run["id"] if run_id is None else run_id
        try:
            _validate_run(run, repo, ident)
            if attempt is not None and str(run["run_attempt"]) != attempt:
                raise IneligibleRun(
                    "requested attempt is no longer the live run attempt"
                )
            with TemporaryDirectory(prefix="cpu-profile-evidence-") as stage:
                measurements = _measure_run(repo, run, Path(stage))
        except IneligibleRun as error:
            if run_id is not None:
                raise ValueError(f"run {ident}: {error}") from error
            rejected.append({"run_id": ident, "reason": str(error)})
            print(f"Rejected main run {ident}: {error}", file=sys.stderr)
            continue
        current = json.loads(_api(f"{endpoint}/{ident}"))
        _validate_run(current, repo, ident)
        if any(current[key] != run[key] for key in ("head_sha", "run_attempt")):
            raise ValueError(f"run {ident}: HEAD or attempt changed during retrieval")
        return measurements, rejected
    raise ValueError(
        "no eligible successful main CI run with complete retained artifacts"
    )


def _compare(
    phases: PhaseMeasurements,
    audit: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    system = set(audit["system_deps"]["nodeids"])
    regular = sorted(set(phases.by_case) - system)
    count = audit["shard_count"]
    comparisons = {}
    file_nodes: dict[str, list[str]] = defaultdict(list)
    for node in regular:
        file_nodes[node.split("::", 1)[0]].append(node)
    file_costs = {
        file: sum(phases.by_case[node] for node in nodes)
        for file, nodes in file_nodes.items()
    }
    heavy_files = sorted(file_costs, key=lambda file: (-file_costs[file], file))[:10]
    placements: dict[str, dict[str, dict[int, float]]] = {
        file: {} for file in heavy_files
    }
    for label, profile in profiles.items():
        plan = assign_groups(
            regular,
            count,
            profile["file_seconds"],
            split_files=profile.get("split_files", []),
            test_weights=profile.get("test_seconds", {}),
        )
        assignments = plan["assignments"]
        splits = plan["split_files"]
        measured = [0.0] * count
        for node in regular:
            measured[assignments[group_key(node, splits)]] += phases.by_case[node]
        for filename in heavy_files:
            costs: dict[int, float] = defaultdict(float)
            for node in file_nodes[filename]:
                costs[assignments[group_key(node, splits)]] += phases.by_case[node]
            placements[filename][label] = dict(sorted(costs.items()))
        comparisons[label] = {
            **plan,
            "estimated": phase_statistics(plan["estimated_phase_seconds"]),
            "measured_phase_seconds_on_assignment": measured,
            "measured_on_assignment": phase_statistics(measured),
            "unmeasured_file_count": len(plan["unmeasured_files"]),
            "unmeasured_split_case_count": len(plan["unmeasured_cases_in_split_files"]),
        }
    return comparisons, [
        {
            "file": file,
            "cases": len(file_nodes[file]),
            "phase_seconds": file_costs[file],
            "by_shard": placements[file],
        }
        for file in heavy_files
    ]


def _write_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".cpu-profile-") as stage:
        candidate = Path(stage) / path.name
        write_json(candidate, document)
        candidate.replace(path)


def update_profile(
    repo: str,
    output: Path,
    baseline: Path,
    report_path: Path | None,
    *,
    run_id: int | None = None,
    attempt: str | None = None,
    split_files: list[str] | None = None,
) -> dict[str, Any]:
    if run_id is not None and run_id < 1:
        raise ValueError("--run-id must be positive")
    if attempt is not None:
        parse_attempt(attempt)
        if run_id is None:
            raise ValueError("--attempt requires --run-id")
    if report_path is not None and report_path.resolve() in {
        output.resolve(),
        baseline.resolve(),
    }:
        raise ValueError("--report must be separate from the timing profiles")
    old, old_sha = read_timing_profile(baseline)
    evidence, rejected = _select_run(repo, run_id, attempt)
    audit, phases = evidence.audit, evidence.phases
    previous_splits = set(old.get("split_files", []))
    unobserved_splits = sorted(previous_splits - phases.by_file.keys())
    weights_only = profile_from_audit(
        audit, phases, sorted(previous_splits & phases.by_file.keys())
    )
    candidate = profile_from_audit(audit, phases, split_files)
    source = candidate["source"]
    source.update(
        repository=repo,
        plan_id=audit["plan_id"],
        policy_sha256=audit["policy_sha256"],
        coverage_artifact_id=evidence.coverage_artifact_id,
        plan_artifact_id=evidence.plan_artifact_id,
        shard_artifact_ids=evidence.shard_artifact_ids,
        scheduling_profile_sha256=evidence.scheduling_profile_sha256,
    )
    comparisons, heavy_files = _compare(
        phases, audit, {"A": old, "B": weights_only, "C": candidate}
    )
    old_files, new_files = old["file_seconds"], candidate["file_seconds"]
    common = old_files.keys() & new_files.keys()
    command = [
        ".venv/bin/python",
        "-m",
        "scripts.ci_metrics",
        "update-profile",
        "--repo",
        repo,
        "--run-id",
        source["run_id"],
        "--attempt",
        source["attempt"],
        "--baseline",
        str(baseline),
        "--output",
        str(output),
    ]
    for name in candidate["split_files"]:
        command.extend(("--split-file", name))
    if report_path is not None:
        command.extend(("--report", str(report_path)))
    report = {
        "source": source,
        "reproduce": shlex.join(command),
        "rejected_runs": rejected,
        "baseline_sha256": old_sha,
        "baseline_source": old.get("source"),
        "baseline_matches_observed_profile": old_sha
        == evidence.scheduling_profile_sha256,
        "unobserved_baseline_split_files": unobserved_splits,
        "candidate_file_count": len(new_files),
        "candidate_split_case_count": len(candidate["test_seconds"]),
        "file_differences": {
            "added": sorted(new_files.keys() - old_files.keys()),
            "removed": sorted(old_files.keys() - new_files.keys()),
            "changed": {
                name: [old_files[name], new_files[name]]
                for name in sorted(common)
                if old_files[name] != new_files[name]
            },
        },
        "ordinary_shard_phase_seconds_observed": [
            phases.by_shard[index] for index in range(audit["shard_count"])
        ],
        "system_deps_phase_seconds_observed": phases.by_shard[audit["shard_count"]],
        "coverage": audit,
        "assignment_comparison": comparisons,
        "heaviest_ordinary_files": heavy_files,
    }
    if report_path is not None:
        _write_atomic(report_path, report)
    _write_atomic(output, candidate)
    return report


def add_parser(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--run-id", type=int, help="Use this exact successful main run"
    )
    selection.add_argument(
        "--latest-main", action="store_true", help="Find the newest eligible main run"
    )
    parser.add_argument("--attempt", help="Require this live attempt with --run-id")
    parser.add_argument("--repo", default=REPOSITORY)
    parser.add_argument(
        "--output", type=Path, required=True, help="Write the verified candidate here"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=TIMING_PROFILE,
        help="Existing profile used for the comparison",
    )
    parser.add_argument(
        "--report", type=Path, help="Write the full comparison and provenance as JSON"
    )
    parser.add_argument(
        "--split-file",
        action="append",
        help="Replace the observed split list with these reviewed files (repeatable)",
    )


def run_update(args: argparse.Namespace) -> int:
    try:
        report = update_profile(
            args.repo,
            args.output,
            args.baseline,
            args.report,
            run_id=args.run_id,
            attempt=args.attempt,
            split_files=args.split_file,
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        subprocess.SubprocessError,
    ) as error:
        message = (
            error.stderr.decode(errors="replace").strip()
            if isinstance(error, subprocess.CalledProcessError) and error.stderr
            else str(error)
        )
        print(f"CPU profile update failed: {message}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "source": report["source"],
                "reproduce": report["reproduce"],
                "candidate_file_count": report["candidate_file_count"],
                "unobserved_baseline_split_files": report[
                    "unobserved_baseline_split_files"
                ],
                "file_differences": {
                    key: len(value) for key, value in report["file_differences"].items()
                },
                "ordinary_shard_comparison": {
                    label: {
                        key: comparison[key]
                        for key in (
                            "estimated_phase_seconds",
                            "estimated",
                            "measured_on_assignment",
                            "unmeasured_file_count",
                            "unmeasured_split_case_count",
                        )
                    }
                    for label, comparison in report["assignment_comparison"].items()
                },
            },
            indent=2,
        )
    )
    return 0
