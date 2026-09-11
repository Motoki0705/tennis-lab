"""Collect pytest JUnit timings without a custom pytest/xdist scheduler."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from .sharding import (
    CI_EXCLUDED_FILES,
    Shard,
    ci_test_files,
    read_durations,
    validate_test_path,
)


def junit_timings(path: Path, files: Sequence[str]) -> dict[str, float]:
    """Include setup/call/teardown; deselected files legitimately contribute zero."""
    result = dict.fromkeys(files, 0.0)
    for case in ET.parse(path).getroot().iter("testcase"):
        name = case.get("file")
        if name is None:
            # Collection errors have no testcase file; they must never become a
            # successful timing baseline (run_tests preserves pytest's exit code).
            if case.find("error") is not None:
                continue
            raise ValueError(
                f"JUnit testcase lacks file; use junit_family=legacy: {path}"
            )
        validate_test_path(name)
        if name not in result:
            raise ValueError(f"JUnit contains an unassigned test file: {name}")
        seconds = float(case.attrib["time"])
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"Invalid JUnit duration for {name}: {seconds}")
        result[name] += seconds
    return {name: round(seconds, 6) for name, seconds in result.items()}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_tests(
    command: Sequence[str], *, repo_root: Path, shard: Shard, output: Path
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    junit = output / "junit.xml"
    metrics = output / "metrics.json"
    junit.unlink(missing_ok=True)
    metrics.unlink(missing_ok=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    plan = {**asdict(shard), "revision": revision, "excluded_files": CI_EXCLUDED_FILES}
    write_json(output / "plan.json", plan)
    started = time.monotonic()
    result = subprocess.run(
        [*command, f"--junitxml={junit}", "-o", "junit_family=legacy"],
        cwd=repo_root,
        check=False,
    )
    elapsed = time.monotonic() - started
    durations = junit_timings(junit, shard.files) if junit.is_file() else None
    write_json(
        metrics,
        {
            **plan,
            "schema_version": 1,
            "exit_code": result.returncode,
            "elapsed_seconds": round(elapsed, 3),
            "seconds_by_file": durations,
        },
    )
    summary = (
        f"### Python shard {shard.index}/{shard.count}\n\n"
        f"- pytest wall time: {elapsed:.1f}s\n"
        f"- Exit code: {result.returncode}\n"
        f"- Test files: {len(shard.files)}\n"
        f"- Files without historical timings: {len(shard.unmeasured_files)}\n"
        f"- Explicitly excluded from CI: {', '.join(CI_EXCLUDED_FILES)}\n"
    )
    if durations:
        summary += "\n| Slowest files | Total test seconds |\n|---|---:|\n"
        for name in sorted(durations, key=lambda item: (-durations[item], item))[:10]:
            summary += f"| `{name}` | {durations[name]:.2f} |\n"
    (output / "summary.md").write_text(summary, encoding="utf-8")
    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if github_summary:
        with Path(github_summary).open("a", encoding="utf-8") as stream:
            stream.write(summary)
    if durations is None and result.returncode == 0:
        raise ValueError("Successful pytest invocation did not produce JUnit output")
    return result.returncode


def update_durations(reports: Path, *, repo_root: Path, output: Path) -> None:
    """Only accept a complete, successful run from one revision and shard count."""
    paths = sorted(reports.rglob("metrics.json"))
    if not paths:
        raise ValueError(f"No metrics.json reports under {reports}")
    seen: set[int] = set()
    identity: tuple[str, int] | None = None
    merged: dict[str, float] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("exit_code") != 0:
            raise ValueError(
                f"Only successful shard reports can update timings: {path}"
            )
        current = (payload["revision"], payload["count"])
        if identity is not None and current != identity:
            raise ValueError("Cannot merge different revisions or shard counts")
        identity = current
        index = payload["index"]
        if index in seen or not 1 <= index <= current[1]:
            raise ValueError(f"Duplicate or invalid shard index: {index}")
        seen.add(index)
        durations = read_durations(path)
        if len(payload["files"]) != len(set(payload["files"])) or set(durations) != set(
            payload["files"]
        ):
            raise ValueError(f"Timing coverage differs from the shard plan: {path}")
        if merged.keys() & durations.keys():
            raise ValueError("Test files were executed by more than one shard")
        merged.update(durations)
    if identity is None or seen != set(range(1, identity[1] + 1)):
        raise ValueError("Missing shard reports")
    expected = set(ci_test_files(repo_root))
    if set(merged) != expected:
        raise ValueError(
            f"Timing coverage differs from current CI files: missing={sorted(expected - merged.keys())}, "
            f"extra={sorted(merged.keys() - expected)}"
        )
    write_json(
        output,
        {
            "schema_version": 1,
            "source_revision": identity[0],
            "seconds_by_file": merged,
        },
    )
