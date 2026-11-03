"""Bind aggregate downloads to a paginated, unambiguous run artifact inventory."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.ci_metrics.artifacts import read_manifest

REPO = Path(__file__).resolve().parents[3]


def _artifact(identifier: int, name: str, *, expired: bool = False) -> dict[str, Any]:
    return {
        "id": identifier,
        "name": name,
        "created_at": "2026-10-30T01:02:03Z",
        "size_in_bytes": 100 + identifier,
        "expired": expired,
    }


@pytest.mark.parametrize(
    "scenario,reason",
    [
        ("complete", None),
        ("expired_old_duplicate", None),
        ("duplicate_plan", "expected exactly one"),
        ("duplicate_shard", "expected exactly one"),
        ("missing_plan", "found 0"),
        ("missing_shard", "found 0"),
        ("extra_shard", "unexpected shard artifact"),
        ("expired_shard", "found 0"),
        ("api_failure", "API unavailable"),
    ],
)
def test_artifact_selection_pages_checks_and_pins_exact_download_ids(
    tmp_path: Path, scenario: str, reason: str | None
) -> None:
    plan = "cpu-lane-plan-123"
    prefix = "cpu-test-measurements-123-shard-"
    rows = [
        _artifact(10, plan),
        _artifact(20, prefix + "0"),
        _artifact(30, prefix + "1"),
    ]
    if scenario.startswith("duplicate_"):
        rows.append(
            _artifact(40, plan if scenario == "duplicate_plan" else prefix + "1")
        )
    elif scenario == "expired_old_duplicate":
        rows.append(_artifact(40, prefix + "1", expired=True))
    elif scenario.startswith("missing_"):
        rows.pop(0 if scenario == "missing_plan" else 2)
    elif scenario == "extra_shard":
        rows.append(_artifact(40, prefix + "2"))
    elif scenario == "expired_shard":
        rows[2]["expired"] = True
    # Keep the expected inventory (and duplicate errors) on a later page. Other
    # runs/coverage artifacts must not become inputs to the aggregate.
    pages = [
        {
            "artifacts": [_artifact(i + 100, f"unrelated-{i}") for i in range(98)]
            + [
                _artifact(300, "cpu-lane-plan-456"),
                _artifact(301, "cpu-test-coverage-123-1"),
            ]
        },
        {"artifacts": rows},
    ]
    binary = tmp_path / "bin"
    binary.mkdir()
    gh = binary / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "assert sys.argv[1:3] == ['api', 'repos/owner/repo/actions/runs/123/artifacts?per_page=100']\n"
        + ("sys.exit('API unavailable')\n" if scenario == "api_failure" else "")
        + f"pages = {pages!r}\n"
        "print(json.dumps(pages if '--paginate' in sys.argv and '--slurp' in sys.argv else pages[:1]))\n"
    )
    gh.chmod(0o755)
    output = tmp_path / "artifacts.json"
    github_output = tmp_path / "github-output"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ci_metrics.artifacts",
            "--repo",
            "owner/repo",
            "--run-id",
            "123",
            "--count",
            "2",
            "--output",
            str(output),
        ],
        cwd=REPO,
        env={
            **os.environ,
            "PATH": f"{binary}{os.pathsep}{os.environ['PATH']}",
            "GITHUB_OUTPUT": str(github_output),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    manifest = json.loads(output.read_text())
    assert manifest["run_id"] == "123"
    assert manifest["shard_count"] == 2
    assert manifest["success"] == (reason is None)
    assert completed.returncode == (0 if reason is None else 1)
    if reason is None:
        assert manifest["issues"] == []
        assert manifest["artifacts"] == [
            _artifact(10, plan),
            _artifact(20, prefix + "0"),
            _artifact(30, prefix + "1"),
        ]
        assert github_output.read_text().splitlines() == [
            "plan_artifact_id=10",
            "shard_artifact_ids=20,30",
        ]
        assert read_manifest(output, "123", 2) == manifest
    else:
        assert reason in completed.stdout + completed.stderr
        assert any(reason in issue for issue in manifest["issues"])
        assert not github_output.exists()


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "duplicate_plan",
        "missing_manifest",
        "wrong_run",
        "wrong_count",
        "wrong_download_set",
    ],
)
def test_artifact_selection_controls_the_coverage_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    from scripts.ci_metrics import shards

    manifest = tmp_path / "artifacts.json"
    data = {
        "schema_version": 1,
        "run_id": "123",
        "shard_count": 2,
        "success": True,
        "issues": [],
        "artifacts": [
            _artifact(10, "cpu-lane-plan-123"),
            _artifact(20, "cpu-test-measurements-123-shard-0"),
            _artifact(30, "cpu-test-measurements-123-shard-1"),
        ],
    }
    if fault == "duplicate_plan":
        data.update(
            success=False,
            issues=[
                "expected exactly one non-expired artifact for cpu-lane-plan-123, found 2"
            ],
        )
    elif fault == "wrong_run":
        data["run_id"] = "456"
    elif fault == "wrong_count":
        data["shard_count"] = 3
    elif fault == "wrong_download_set":
        data["artifacts"] = []
    if fault != "missing_manifest":
        manifest.write_text(json.dumps(data))
    # Isolate the CLI's failure propagation from the existing shard/content audit.
    monkeypatch.setattr(shards, "read_plan", lambda *a, **k: {})
    monkeypatch.setattr(
        shards,
        "verify_shards",
        lambda *a, **k: {
            "issues": [],
            "success": True,
            "observations_complete": True,
            "successful_test_observations": True,
            "full_cpu_coverage": True,
            "full_cpu_success": True,
            "scoped_cpu_success": True,
            "source_sha": "head",
        },
    )
    monkeypatch.setattr(shards, "write_growth_report", lambda *a, **k: "")
    for key, value in {
        "GITHUB_SHA": "head",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "CI_JOB_RESULTS": "{}",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shards",
            "--root",
            str(tmp_path),
            "--count",
            "2",
            "--lane-plan",
            "plan.json",
            "--artifact-manifest",
            str(manifest),
        ],
    )
    assert shards.main() == (0 if fault is None else 1)
    result = json.loads((tmp_path / "coverage.json").read_text())
    assert result["success"] == (fault is None)
    assert result["full_cpu_success"] == (fault is None)
    assert result["observations_complete"] == (fault is None)
    assert result["full_cpu_coverage"] == (fault is None)
    if fault is None:
        assert result["artifacts"] == data["artifacts"]
    else:
        assert any("artifact inventory" in issue for issue in result["issues"])
    if fault == "duplicate_plan":
        assert any("cpu-lane-plan-123, found 2" in issue for issue in result["issues"])
