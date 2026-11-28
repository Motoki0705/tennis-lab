"""Exercise real CPU selection and reject incomplete or mismatched shard evidence."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from scripts.ci_metrics import update_profile
from scripts.ci_metrics.__main__ import main as metrics_main
from scripts.ci_metrics.artifacts import inspect_artifacts
from scripts.ci_metrics.lanes import digest, load_registry, make_plan
from scripts.ci_metrics.profile import build_profile
from scripts.ci_metrics.report import summarize
from scripts.ci_metrics.runner import test_environment as headless_environment
from scripts.ci_metrics.shards import (
    assign_groups,
    inspect_shards,
    read_timing_profile,
    verify_shards,
    write_growth_report,
)

REPO = Path(__file__).resolve().parents[3]
CONTEXT: dict[str, Any] = {"sha": "a" * 40, "run_id": "123", "attempt": "1"}
SUCCESS = {
    name: {"result": "success"} for name in ("plan", "lint", "tests", "system-deps")
}


def test_assignment_uses_cost_and_preserves_files_including_new_tests() -> None:
    weights = {
        "tests/test_heavy.py": 90.0,
        "tests/test_medium.py": 40.0,
        "tests/test_light.py": 10.0,
    }
    nodeids = [f"{name}::test_{i}" for name in weights for i in range(3)]
    nodeids.append("tests/test_new.py::test_new")
    plan = assign_groups(nodeids, 2, weights)
    assert (
        plan["assignments"]["tests/test_heavy.py"]
        != plan["assignments"]["tests/test_medium.py"]
    )
    assert plan["estimated_phase_seconds"] == [90.0, 90.0]
    assert plan["unmeasured_files"] == ["tests/test_new.py"]
    assert plan == assign_groups(list(reversed(nodeids)), 2, weights)
    with pytest.raises(ValueError, match="unique"):
        assign_groups(nodeids + nodeids[:1], 2, weights)
    with pytest.raises(ValueError, match="fewer files"):
        assign_groups(nodeids, 5, weights)


@pytest.mark.parametrize("count", [1, 2, 4, 6, 8, 10])
def test_every_dispatch_partition_covers_new_and_split_cases_once(count: int) -> None:
    weights = {f"tests/test_{i}.py": float(20 - i) for i in range(10)}
    nodeids = [f"{name}::test_{i}" for name in weights for i in range(3)]
    nodeids += ["tests/new.py::test_new", "tests/test_0.py::test_new"]
    plan = assign_groups(nodeids, count, weights, split_files=["tests/test_0.py"])
    selected = [
        nodeid
        for index in range(count)
        for nodeid in nodeids
        if plan["assignments"].get(
            nodeid, plan["assignments"].get(nodeid.split("::")[0])
        )
        == index
    ]
    assert sorted(selected) == sorted(nodeids)
    assert set(plan["assignments"].values()) == set(range(count))


def test_only_explicit_function_fixture_files_split_by_measured_case_cost() -> None:
    nodeids = [f"tests/test_gate.py::test_{i}" for i in range(4)]
    weights = {"tests/test_gate.py": 100.0, "tests/test_shared.py": 10.0}
    nodeids += ["tests/test_shared.py::test_one", "tests/test_shared.py::test_two"]
    cases = {nodeids[0]: 50.0, nodeids[1]: 20.0, nodeids[2]: 20.0, nodeids[3]: 10.0}
    plan = assign_groups(
        nodeids, 4, weights, split_files=["tests/test_gate.py"], test_weights=cases
    )
    assert plan["assignments"][nodeids[0]] != plan["assignments"][nodeids[1]]
    assert "tests/test_shared.py" in plan["assignments"]
    assert "tests/test_shared.py::test_one" not in plan["assignments"]
    assert plan["unmeasured_cases_in_split_files"] == []
    new = "tests/test_gate.py::test_new"
    updated = assign_groups(
        [*nodeids, new],
        4,
        weights,
        split_files=["tests/test_gate.py"],
        test_weights=cases,
    )
    assert new in updated["assignments"]
    assert updated["unmeasured_cases_in_split_files"] == [new]


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_timing_profiles_fail_explicitly(tmp_path: Path, bad: Any) -> None:
    path = tmp_path / "weights.json"
    path.write_text(
        json.dumps({"schema_version": 1, "file_seconds": {"tests/a.py": bad}})
    )
    with pytest.raises(ValueError, match="positive seconds"):
        read_timing_profile(path)


def _run_observed_shard(
    suite: Path,
    output: Path,
    index: int,
    count: int,
    *,
    workers: int = 2,
    lane_plan: Path | None = None,
    attempt: str = "1",
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ci_metrics",
            "run",
            "--output",
            str(output),
            "--workers",
            str(workers),
            "--shard-index",
            str(index),
            "--shard-count",
            str(count),
            "--shard-weights",
            str(suite / "weights.json"),
            *(["--lane-plan", str(lane_plan)] if lane_plan else []),
        ],
        cwd=suite,
        env={
            **headless_environment(dict(os.environ)),
            "GITHUB_SHA": CONTEXT["sha"],
            "GITHUB_RUN_ID": CONTEXT["run_id"],
            "GITHUB_RUN_ATTEMPT": attempt,
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    path = output / "environment.json"
    metadata = json.loads(path.read_text())
    # The isolated suite is not a repository; fix the expected CI source context.
    metadata["source"] = {"commit": CONTEXT["sha"], "status": ""}
    path.write_text(json.dumps(metadata))


@pytest.fixture(scope="module")
def observed_shards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root: Path = tmp_path_factory.mktemp("observed-shards")
    suite = root / "suite"
    suite.mkdir()
    (suite / ".spin").mkdir()
    shutil.copy2(REPO / ".spin/cmds.py", suite / ".spin/cmds.py")
    shutil.copy2(REPO / "pyproject.toml", suite / "pyproject.toml")
    shutil.copytree(
        REPO / "scripts/ci_metrics",
        suite / "scripts/ci_metrics",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (suite / "scripts/ci_metrics/lanes.json").write_text(
        json.dumps({"schema_version": 1, "suites": []})
    )
    (suite / "src").mkdir()
    (suite / "src/__init__.py").touch()
    (suite / "tests").mkdir()
    for name in ("first", "second", "third", "new"):
        (suite / f"tests/test_{name}.py").write_text(
            "import pytest\n"
            + (
                "@pytest.mark.parametrize('case', [0, 1])\ndef test_cpu(case): assert True\n"
                if name == "first"
                else "def test_cpu(): assert True\n"
            )
            + (
                "@pytest.mark.system_deps\ndef test_system(): pass\n"
                if name == "new"
                else ""
            )
            + "@pytest.mark.cuda\n"
            "def test_gpu(): assert False\n"
            "@pytest.mark.local_data\n"
            "def test_local(): assert False\n"
        )
    restored = (
        suite / "tests/integration/synthetic_data_generation/test_court_dataset.py"
    )
    restored.parent.mkdir(parents=True)
    restored.write_text("def test_court_contract(): pass\n")
    weights = suite / "weights.json"
    weights.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split_files": ["tests/test_first.py"],
                "file_seconds": {
                    "tests/test_first.py": 20,
                    "tests/test_second.py": 10,
                    "tests/test_third.py": 5,
                },
            }
        )
    )
    artifacts = root / "artifacts"
    for index in range(3):
        _run_observed_shard(suite, artifacts / f"shard-{index}/tests", index, 2)
    return artifacts


def test_real_shards_cover_exact_cpu_inventory(observed_shards: Path) -> None:
    result = verify_shards(observed_shards, 2, needs=SUCCESS, **CONTEXT)
    assert result["success"], result
    assert result["selected_count"] == 7
    assert result["counts"] == {"passed": 7}
    assert result["plan_attempt"] == "1"
    assert result["shard_attempts"] == {"0": "1", "1": "1", "2": "1"}
    for directory in observed_shards.glob("*/tests"):
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["full_cpu_baseline"] is False
        plan = json.loads((directory / "shard-gw0.json").read_text())
        assert plan["unmeasured_files"] == [
            "tests/integration/synthetic_data_generation/test_court_dataset.py",
            "tests/test_new.py",
        ]
        assert len(plan["universe_nodeids"]) == 7
        assert plan["system_deps_nodeids"] == ["tests/test_new.py::test_system"]
        if plan["index"] == 2:
            assert plan["selected_nodeids"] == plan["system_deps_nodeids"]
        else:
            assert not set(plan["selected_nodeids"]) & set(plan["system_deps_nodeids"])
        assert plan["requested_nodeids"] == plan["universe_nodeids"]
        assert plan["deferred"] == {}
        for path in directory.glob("events-gw*.jsonl"):
            events = [json.loads(line) for line in path.read_text().splitlines()]
            universe = next(
                e["nodeids"] for e in events if e["event"] == "cpu_universe"
            )
            assert universe == plan["universe_nodeids"]
            assert all(
                "test_gpu" not in node and "test_local" not in node for node in universe
            )
        assert (
            plan["assignments"]["tests/test_first.py::test_cpu[0]"]
            != plan["assignments"]["tests/test_first.py::test_cpu[1]"]
        )


def _replace_lane_plan(directory: Path, **changes: Any) -> dict[str, Any]:
    path = directory / "lane-plan.json"
    plan: dict[str, Any] = json.loads(path.read_text())
    plan.update(changes)
    plan["plan_id"] = digest({k: v for k, v in plan.items() if k != "plan_id"})
    path.write_text(json.dumps(plan))
    metadata_path = directory / "environment.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["lane_plan_id"] = plan["plan_id"]
    metadata_path.write_text(json.dumps(metadata))
    for manifest_path in directory.glob("shard-*.json"):
        manifest = json.loads(manifest_path.read_text())
        manifest["plan_id"] = plan["plan_id"]
        manifest_path.write_text(json.dumps(manifest))
    return plan


@pytest.mark.parametrize(
    "plan_attempt,shard_attempts,aggregate_attempt",
    [
        ("1", ["1", "1", "1"], "1"),
        ("1", ["1", "2", "1"], "2"),
        ("2", ["2", "2", "2"], "3"),
        ("1", ["2", "10", "2"], "10"),
        ("1", ["1", "1", "2"], "2"),
    ],
    ids=[
        "single-attempt",
        "failed-shard-rerun",
        "python-only-rerun",
        "numeric-maximum",
        "system-shard-rerun",
    ],
)
def test_aggregation_and_profile_accept_latest_shards_from_the_same_fixed_plan(
    observed_shards: Path,
    tmp_path: Path,
    plan_attempt: str,
    shard_attempts: list[str],
    aggregate_attempt: str,
) -> None:
    root = tmp_path / "artifacts"
    shutil.copytree(observed_shards, root)
    for index, attempt in enumerate(shard_attempts):
        directory = root / f"shard-{index}/tests"
        plan = _replace_lane_plan(directory, attempt=plan_attempt)
        metadata_path = directory / "environment.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["environment"]["GITHUB_RUN_ATTEMPT"] = attempt
        metadata_path.write_text(json.dumps(metadata))
    result = verify_shards(
        root,
        2,
        needs=SUCCESS,
        lane_plan=plan,
        **{**CONTEXT, "attempt": aggregate_attempt},
    )
    expected_attempts = {str(i): value for i, value in enumerate(shard_attempts)}
    assert result["success"], result
    assert result["counts"] == {"passed": 7}
    assert result["attempt"] == aggregate_attempt
    assert result["plan_attempt"] == plan_attempt
    assert result["shard_attempts"] == expected_attempts
    profile = build_profile(root)
    assert profile["source"]["attempt"] == max(shard_attempts, key=int)
    assert profile["source"]["plan_attempt"] == plan_attempt
    assert profile["source"]["shard_attempts"] == expected_attempts
    assert profile["source"]["selected_count"] == 7


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("shard_before_plan", "attempt"),
        ("shard_after_aggregate", "attempt"),
        ("plan_after_aggregate", "attempt"),
        ("different_plan", "fixed lane plan"),
        ("plan_wrong_run", "source/run"),
        ("plan_wrong_head", "source/run"),
    ],
)
def test_aggregation_rejects_attempt_bounds_and_a_valid_but_different_plan(
    observed_shards: Path, tmp_path: Path, fault: str, reason: str
) -> None:
    root = tmp_path / "artifacts"
    shutil.copytree(observed_shards, root)
    plan = json.loads((root / "shard-0/tests/lane-plan.json").read_text())
    second = root / "shard-1/tests"
    if fault == "different_plan":
        _replace_lane_plan(second, main_successes={"unrelated": {"status": "unknown"}})
    elif fault == "plan_wrong_run":
        _replace_lane_plan(second, run_id="456")
    elif fault == "plan_wrong_head":
        _replace_lane_plan(second, source_sha="b" * 40)
    elif fault in {"shard_before_plan", "plan_after_aggregate"}:
        for directory in root.glob("*/tests"):
            plan = _replace_lane_plan(directory, attempt="2")
    else:
        path = second / "environment.json"
        metadata = json.loads(path.read_text())
        metadata["environment"]["GITHUB_RUN_ATTEMPT"] = "3"
        path.write_text(json.dumps(metadata))
    aggregate_attempt = "1" if fault == "plan_after_aggregate" else "2"
    result = verify_shards(
        root,
        2,
        needs=SUCCESS,
        lane_plan=plan,
        **{**CONTEXT, "attempt": aggregate_attempt},
    )
    assert not result["success"]
    assert any(reason in issue for issue in result["issues"]), result
    if fault != "shard_after_aggregate":
        with pytest.raises(ValueError, match="incomplete"):
            build_profile(root)


@pytest.mark.parametrize("field", ["plan", "shard", "aggregate"])
@pytest.mark.parametrize("bad", [None, 1, True, "0", "-1", "1.5", " 1", "1\n", "one"])
def test_attempts_must_be_positive_integer_strings(
    observed_shards: Path, tmp_path: Path, field: str, bad: Any
) -> None:
    root = tmp_path / "artifacts"
    shutil.copytree(observed_shards, root)
    directory = root / "shard-0/tests"
    aggregate_attempt = "2"
    if field == "plan":
        _replace_lane_plan(directory, attempt=bad)
    elif field == "shard":
        path = directory / "environment.json"
        metadata = json.loads(path.read_text())
        metadata["environment"]["GITHUB_RUN_ATTEMPT"] = bad
        path.write_text(json.dumps(metadata))
    else:
        aggregate_attempt = bad
    result = verify_shards(
        root, 2, needs=SUCCESS, **{**CONTEXT, "attempt": aggregate_attempt}
    )
    assert not result["success"]
    assert any("attempt" in issue for issue in result["issues"]), result
    if field != "aggregate":
        with pytest.raises(ValueError, match="attempt"):
            build_profile(root)


def test_profile_recomputes_measured_file_and_split_costs(
    observed_shards: Path,
    tmp_path: Path,
) -> None:
    profile = build_profile(observed_shards)
    results = [
        row
        for path in observed_shards.glob("*/tests/results.json")
        for row in json.loads(path.read_text())
    ]
    assert profile["source"]["selected_count"] == len(results) == 7
    assert profile["source"]["counts"] == {"passed": 7}
    assert profile["source"]["successful_test_observations"] is True
    assert profile["split_files"] == ["tests/test_first.py"]
    assert profile["file_seconds"]["tests/test_first.py"] == pytest.approx(
        sum(
            sum(row["phase_s"].values())
            for row in results
            if "test_first.py::" in row["nodeid"]
        ),
        abs=1e-6,
    )
    assert profile["test_seconds"] == {
        row["nodeid"]: max(round(sum(row["phase_s"].values()), 6), 0.000001)
        for row in results
        if "test_first.py::" in row["nodeid"]
    }
    assert profile == build_profile(observed_shards)
    assert build_profile(observed_shards, ["tests/test_second.py"])["split_files"] == [
        "tests/test_second.py"
    ]
    with pytest.raises(ValueError, match="unique measured"):
        build_profile(observed_shards, ["tests/unmeasured.py"])
    destination = tmp_path / "profile.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ci_metrics",
            "profile",
            "--root",
            str(observed_shards),
            "--output",
            str(destination),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert read_timing_profile(destination)[0] == profile


def _mock_update_actions(
    monkeypatch: pytest.MonkeyPatch,
    coverage: dict[str, Any],
    root: Path | None = None,
    *,
    newer_files: dict[str, bytes] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    repo = update_profile.REPOSITORY
    run = {
        "id": 123,
        "path": ".github/workflows/ci.yml",
        "event": "push",
        "head_branch": "main",
        "head_sha": CONTEXT["sha"],
        "run_attempt": 3,
        "status": "completed",
        "conclusion": "success",
        "repository": {"id": 1, "full_name": repo},
        "head_repository": {"id": 1, "full_name": repo},
    }
    names = [
        "cpu-lane-plan-123",
        *(f"cpu-test-measurements-123-shard-{i}" for i in range(3)),
    ]
    artifacts: list[dict[str, Any]] = [
        {
            "name": name,
            "id": 10 + index,
            "expired": False,
            "created_at": "2026-11-27T00:00:00Z",
            "size_in_bytes": 100,
            "workflow_run": {
                "id": 123,
                "head_sha": CONTEXT["sha"],
                "head_branch": "main",
                "repository_id": 1,
                "head_repository_id": 1,
            },
        }
        for index, name in enumerate([*names, "cpu-test-coverage-123-3"])
    ]
    manifest = inspect_artifacts(artifacts, "123", 2)
    coverage["artifacts"] = manifest["artifacts"]
    newer_run = {**run, "id": 124}
    newer_artifacts = (
        [
            {
                **item,
                "id": item["id"] + 100,
                "name": item["name"].replace("-123", "-124"),
                "workflow_run": {**item["workflow_run"], "id": 124},
            }
            for item in artifacts
        ]
        if newer_files is not None
        else []
    )
    newer_manifest = inspect_artifacts(newer_artifacts, "124", 2)

    def list_artifacts(repository: str, run_id: str) -> list[dict[str, Any]]:
        assert repository == repo
        return (
            artifacts if run_id == "123" else newer_artifacts if run_id == "124" else []
        )

    def api(endpoint: str) -> bytes:
        if endpoint == f"repos/{repo}/actions/runs/123":
            return json.dumps(run).encode()
        if endpoint == f"repos/{repo}/actions/runs/124":
            return json.dumps(newer_run).encode()
        if endpoint.startswith(f"repos/{repo}/actions/workflows/ci.yml/runs?"):
            return json.dumps({"workflow_runs": [newer_run, run]}).encode()
        identifier = int(endpoint.split("/")[-2])
        assert endpoint == f"repos/{repo}/actions/artifacts/{identifier}/zip"
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            if identifier == 114:
                assert newer_files is not None
                newer_coverage = {
                    **coverage,
                    "run_id": "124",
                    "artifacts": newer_manifest["artifacts"],
                }
                archive.writestr(
                    "coverage.json",
                    newer_files.get(
                        "coverage.json", json.dumps(newer_coverage).encode()
                    ),
                )
                archive.writestr("artifacts.json", json.dumps(newer_manifest))
            elif identifier == 110:
                assert newer_files is not None
                archive.writestr("plan.json", newer_files["plan.json"])
            elif identifier == 14:
                archive.writestr("coverage.json", json.dumps(coverage))
                archive.writestr("artifacts.json", json.dumps(manifest))
            else:
                assert root is not None, f"unexpected download: {identifier}"
                if identifier == 10:
                    archive.write(root / "shard-0/tests/lane-plan.json", "plan.json")
                else:
                    for path in (root / f"shard-{identifier - 11}/tests").iterdir():
                        archive.write(path, f"tests/{path.name}")
        return output.getvalue()

    monkeypatch.setattr(update_profile, "fetch_artifacts", list_artifacts)
    monkeypatch.setattr(update_profile, "_api", api)
    return run, artifacts


@pytest.mark.parametrize(
    "location, key, value, reason",
    [
        ("run", "event", "pull_request", "successful CI push on main"),
        ("run", "head_branch", "topic", "successful CI push on main"),
        ("run", "conclusion", "cancelled", "successful CI push on main"),
        ("coverage", "success", False, "coverage.json success"),
        ("coverage", "full_cpu_success", False, "coverage.json full_cpu_success"),
        (
            "coverage",
            "observations_complete",
            False,
            "coverage.json observations_complete",
        ),
        ("coverage", "deferred_count", 1, "coverage.json deferred_count"),
        ("coverage", "counts", {"passed": 6, "failed": 1}, "zero failures"),
        ("coverage", "source_sha", "b" * 40, "coverage.json source_sha"),
        ("coverage", "run_id", "124", "coverage.json run_id"),
        ("coverage", "attempt", "2", "coverage.json attempt"),
        ("coverage", "artifacts", [], "download set"),
        ("artifacts", "missing", None, "found 0"),
        ("artifacts", "expired", None, "found 0"),
        ("artifacts", "duplicate", None, "found 2"),
        ("artifacts", "wrong_run", None, "mismatched run/HEAD"),
    ],
)
def test_update_profile_rejects_ineligible_evidence_before_replacing_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    location: str,
    key: str,
    value: Any,
    reason: str,
) -> None:
    coverage = {
        "schema_version": 3,
        "success": True,
        "full_cpu_success": True,
        "observations_complete": True,
        "deferred_count": 0,
        "event": "push",
        "ref": "refs/heads/main",
        "scope": "full",
        "source_sha": CONTEXT["sha"],
        "run_id": "123",
        "attempt": "3",
        "counts": {"passed": 7},
        "shard_count": 2,
    }
    run, artifacts = _mock_update_actions(monkeypatch, coverage)
    if location == "run":
        run[key] = value
    elif location == "coverage":
        coverage[key] = value
    elif key == "missing":
        artifacts.pop(1)
    elif key == "expired":
        artifacts[1]["expired"] = True
    elif key == "duplicate":
        artifacts.append({**artifacts[1], "id": 42})
    else:
        artifacts[0]["workflow_run"]["id"] = 99

    profile = tmp_path / "cpu-file-times.json"
    before = b'{"schema_version":1,"file_seconds":{"tests/test_first.py":1}}\n'
    profile.write_bytes(before)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ci_metrics",
            "update-profile",
            "--run-id",
            "123",
            "--output",
            str(profile),
            "--baseline",
            str(profile),
        ],
    )
    assert metrics_main() == 1
    assert reason in capsys.readouterr().err
    assert profile.read_bytes() == before


def _main_push_observations(
    observed_shards: Path, root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    shutil.copytree(observed_shards, root)
    for index in range(3):
        directory = root / f"shard-{index}/tests"
        plan = _replace_lane_plan(
            directory, event="push", ref="refs/heads/main", reasons=["main_push"]
        )
        path = directory / "environment.json"
        metadata = json.loads(path.read_text())
        metadata["environment"]["GITHUB_RUN_ATTEMPT"] = "2" if index == 1 else "1"
        path.write_text(json.dumps(metadata))
    coverage = verify_shards(
        root, 2, needs=SUCCESS, lane_plan=plan, **{**CONTEXT, "attempt": "3"}
    )
    assert coverage["full_cpu_success"], coverage["issues"]
    return plan, coverage


def test_invalid_setup_and_missing_call_cannot_publish_a_profile(
    observed_shards: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "actions"
    plan, coverage = _main_push_observations(observed_shards, root)
    directory = root / "shard-0/tests"
    journal = next(directory.glob("events-controller-*.jsonl"))
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    setup = next(
        row for row in rows if row["event"] == "phase" and row["phase"] == "setup"
    )
    assert setup["outcome"] == "passed"
    setup["outcome"] = "unknown-setup"
    nodeid = setup["nodeid"]
    call = next(
        row
        for row in rows
        if row["event"] == "phase"
        and row["phase"] == "call"
        and row["nodeid"] == nodeid
    )
    rows.remove(call)
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = summarize(directory)
    assert result["successful_test_baseline"] is False
    assert result["counts"]["incomplete"] == 1
    assert any(
        "unknown-setup" in issue and nodeid in issue for issue in result["issues"]
    )
    checked = inspect_shards(root, 2, lane_plan=plan, **{**CONTEXT, "attempt": "3"})
    assert checked["observations_complete"] is False
    assert any("unknown-setup" in issue for issue in checked["issues"])
    with pytest.raises(ValueError, match="unknown-setup"):
        build_profile(root)

    _mock_update_actions(monkeypatch, coverage, root)
    output = tmp_path / "cpu-file-times.json"
    before = (observed_shards.parent / "suite/weights.json").read_bytes()
    output.write_bytes(before)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ci_metrics",
            "update-profile",
            "--run-id",
            "123",
            "--baseline",
            str(output),
            "--output",
            str(output),
        ],
    )
    assert metrics_main() == 1
    assert "unknown-setup" in capsys.readouterr().err
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "file, data, reason",
    [
        (
            "plan.json",
            b'{"schema_version":1,"plan_id":"not-the-plan"}',
            "lane plan identity mismatch",
        ),
        ("coverage.json", b'{"success": true,', "JSONDecodeError"),
        ("coverage.json", b"[]", "AttributeError"),
    ],
)
def test_profile_search_rejects_corrupt_evidence_and_uses_the_next_run(
    observed_shards: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    file: str,
    data: bytes,
    reason: str,
) -> None:
    root = tmp_path / "actions"
    _, coverage = _main_push_observations(observed_shards, root)
    _mock_update_actions(monkeypatch, coverage, root, newer_files={file: data})
    profile = tmp_path / "cpu-file-times.json"
    before = (observed_shards.parent / "suite/weights.json").read_bytes()
    profile.write_bytes(before)
    report = tmp_path / "report.json"
    args = [
        "ci_metrics",
        "update-profile",
        "--baseline",
        str(profile),
        "--output",
        str(profile),
        "--report",
        str(report),
    ]

    monkeypatch.setattr(sys, "argv", [*args, "--run-id", "124"])
    assert metrics_main() == 1
    diagnostic = capsys.readouterr().err
    assert "run 124" in diagnostic and file in diagnostic and reason in diagnostic
    assert profile.read_bytes() == before
    assert not report.exists()

    monkeypatch.setattr(sys, "argv", [*args, "--latest-main"])
    assert metrics_main() == 0
    assert "Rejected main run 124" in capsys.readouterr().err
    assert read_timing_profile(profile)[0]["source"]["run_id"] == "123"
    (rejected,) = json.loads(report.read_text())["rejected_runs"]
    assert (
        rejected["run_id"] == 124
        and file in rejected["reason"]
        and reason in rejected["reason"]
    )


@pytest.mark.parametrize("fault", ["transport", "local-io"])
def test_profile_search_stops_on_transport_or_local_io_failure(
    observed_shards: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fault: str,
) -> None:
    root = tmp_path / "actions"
    _, coverage = _main_push_observations(observed_shards, root)
    run, _ = _mock_update_actions(monkeypatch, coverage, root)
    looked_up = []
    list_artifacts: Callable[[str, str], list[dict[str, Any]]] = (
        update_profile.fetch_artifacts
    )

    def track_lookup(repo: str, run_id: str) -> list[dict[str, Any]]:
        looked_up.append(run_id)
        return list_artifacts(repo, run_id)

    monkeypatch.setattr(
        update_profile, "_latest_runs", lambda _: iter((run, {**run, "id": 122}))
    )
    monkeypatch.setattr(update_profile, "fetch_artifacts", track_lookup)
    if fault == "transport":
        api: Callable[[str], bytes] = update_profile._api

        def unavailable(endpoint: str) -> bytes:
            if endpoint.endswith("/artifacts/11/zip"):
                return b"invalid shard ZIP"
            if endpoint.endswith("/artifacts/12/zip"):
                raise subprocess.CalledProcessError(
                    1, ["gh", "api"], stderr=b"network unavailable"
                )
            return api(endpoint)

        monkeypatch.setattr(update_profile, "_api", unavailable)
        reason = "network unavailable"
    else:
        read_text = Path.read_text

        def unreadable(path: Path, *args: Any, **kwargs: Any) -> str:
            if path.name.startswith("events-controller-") and path.parents[
                2
            ].name.startswith("cpu-profile-evidence-"):
                raise PermissionError("cannot read staged observations")
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", unreadable)
        reason = "cannot read staged observations"

    profile = tmp_path / "cpu-file-times.json"
    before = (observed_shards.parent / "suite/weights.json").read_bytes()
    profile.write_bytes(before)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ci_metrics",
            "update-profile",
            "--latest-main",
            "--baseline",
            str(profile),
            "--output",
            str(profile),
        ],
    )
    assert metrics_main() == 1
    diagnostic = capsys.readouterr().err
    assert reason in diagnostic and "Rejected main run" not in diagnostic
    assert looked_up == ["123"]
    assert profile.read_bytes() == before


def test_verified_update_profiles_a_partial_rerun_and_preserves_existing_weights_on_error(
    observed_shards: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "actions"
    plan, coverage = _main_push_observations(observed_shards, root)
    _mock_update_actions(monkeypatch, coverage, root)
    baseline = observed_shards.parent / "suite/weights.json"
    output = tmp_path / "cpu-file-times.json"
    report = tmp_path / "comparison.json"
    common = [
        "ci_metrics",
        "update-profile",
        "--baseline",
        str(baseline),
        "--output",
        str(output),
        "--report",
        str(report),
    ]
    monkeypatch.setattr(sys, "argv", [*common, "--latest-main"])
    assert metrics_main() == 0
    assert "Rejected main run 124" in capsys.readouterr().err
    profile, fingerprint = read_timing_profile(output)
    checked = json.loads(report.read_text())
    assert profile["source"]["attempt"] == "3"
    assert profile["source"]["plan_attempt"] == "1"
    assert profile["source"]["shard_attempts"] == {"0": "1", "1": "2", "2": "1"}
    assert profile["source"]["coverage_artifact_id"] == 14
    assert profile["source"]["plan_artifact_id"] == 10
    assert profile["source"]["shard_artifact_ids"] == {"0": 11, "1": 12, "2": 13}
    assert checked["assignment_comparison"]["A"]["unmeasured_file_count"] == 2
    assert checked["assignment_comparison"]["C"]["unmeasured_split_case_count"] == 0
    assert checked["coverage"]["selected_count"] == 7
    assert checked["baseline_matches_observed_profile"] is True
    assert checked["ordinary_shard_phase_seconds_observed"] == pytest.approx(
        checked["assignment_comparison"]["A"]["measured_phase_seconds_on_assignment"]
    )

    monkeypatch.setattr(sys, "argv", [*common, "--run-id", "123", "--attempt", "3"])
    assert metrics_main() == 0
    assert read_timing_profile(output)[1] == fingerprint
    before = output.read_bytes()
    second = root / "shard-1/tests"
    _replace_lane_plan(second, main_successes={"different-plan": {}})
    assert metrics_main() == 1
    assert "fixed lane plan" in capsys.readouterr().err
    assert output.read_bytes() == before
    _replace_lane_plan(second, main_successes=plan["main_successes"])

    path = second / "run.json"
    successful_run = path.read_text()
    path.write_text(json.dumps({**json.loads(successful_run), "exitcode": 1}))
    assert metrics_main() == 1
    assert "failed shard observations" in capsys.readouterr().err
    assert output.read_bytes() == before
    path.write_text(successful_run)

    replace = Path.replace

    def fail_to_publish(candidate: Path, target: Path) -> Path:
        if target == output:
            raise OSError("cannot publish candidate")
        return replace(candidate, target)

    with monkeypatch.context() as publish:
        publish.setattr(Path, "replace", fail_to_publish)
        assert metrics_main() == 1
    assert "cannot publish candidate" in capsys.readouterr().err
    assert output.read_bytes() == before
    assert not list(tmp_path.glob(".cpu-profile-*"))

    suite = tmp_path / "suite"
    shutil.copytree(observed_shards.parent / "suite", suite)
    shutil.copy2(output, suite / "weights.json")
    with (suite / "tests/test_first.py").open("a") as stream:
        stream.write(
            "\n@pytest.mark.skip(reason='optional dependency unavailable')\ndef test_new_case(): pass\n"
        )
    (suite / "tests/test_unmeasured.py").write_text("def test_new_file(): pass\n")
    new_observations = tmp_path / "after-update"
    for index in range(3):
        _run_observed_shard(
            suite, new_observations / f"shard-{index}/tests", index, 2, workers=1
        )
    result = verify_shards(
        new_observations, 2, needs=SUCCESS, source_root=suite, **CONTEXT
    )
    assert result["full_cpu_success"], result
    assert result["selected_count"] == 9
    assert result["counts"] == {"passed": 8, "skipped": 1}
    assert result["system_deps"]["counts"] == {"passed": 1}
    manifest = json.loads(
        (new_observations / "shard-0/tests/shard-gw0.json").read_text()
    )
    assert manifest["profile_sha256"] == fingerprint
    assert manifest["unmeasured_files"] == ["tests/test_unmeasured.py"]
    assert manifest["unmeasured_cases_in_split_files"] == [
        "tests/test_first.py::test_new_case"
    ]


def test_real_lane_selection_records_the_universe_before_deferring_a_file(
    tmp_path: Path,
    observed_shards: Path,
) -> None:
    suite = tmp_path / "suite"
    shutil.copytree(observed_shards.parent / "suite", suite)
    timing_profile = suite / "weights.json"
    timing_profile.write_text(json.dumps(build_profile(observed_shards)))
    registry_path = suite / "scripts/ci_metrics/lanes.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suites": [
                    {
                        "id": "new",
                        "files": ["tests/test_new.py"],
                        "paths": ["tools/new/**"],
                        "reason": "Independent broad regression",
                        "owner": "repository-owner",
                    }
                ],
            }
        )
    )
    registry, files = load_registry(suite)
    plan = make_plan(
        registry,
        files,
        source_sha=CONTEXT["sha"],
        event="pull_request",
        run_id=CONTEXT["run_id"],
        attempt=CONTEXT["attempt"],
        base_sha="base",
        head_sha="head",
        changes=[{"status": "M", "paths": ["docs/readme.md"]}],
    )
    plan_path = suite / "plan.json"
    plan_path.write_text(json.dumps(plan))
    artifacts = tmp_path / "artifacts"
    output = artifacts / "shard-0/tests"
    for index in range(2):
        _run_observed_shard(
            suite,
            artifacts / f"shard-{index}/tests",
            index,
            1,
            workers=1,
            lane_plan=plan_path,
            attempt="2",
        )
    audit = verify_shards(
        artifacts,
        1,
        needs=SUCCESS,
        lane_plan=plan,
        source_root=suite,
        **{**CONTEXT, "attempt": "2"},
    )
    assert audit["success"], audit
    assert (
        audit["universe_count"],
        audit["requested_count"],
        audit["deferred_count"],
    ) == (7, 6, 1)
    assert audit["counts"] == {"passed": 6}
    assert audit["full_cpu_success"] is False
    assert json.loads((output / "shard-gw0.json").read_text())["deferred"] == {
        "tests/test_new.py::test_cpu": "new"
    }
    with pytest.raises(ValueError, match="partial CPU scope"):
        build_profile(artifacts)
    write_growth_report(
        artifacts,
        audit,
        timing_profile,
        source_root=suite,
        repository=None,
        ancestor=CONTEXT["sha"],
    )
    report = json.loads((artifacts / "growth-summary.json").read_text())
    assert report["profile_freshness"]["observed_case_count"] == 6
    assert report["profile_freshness"]["full_selected_count_delta"] is None
    assert report["ordinary_shard_balance"]["measured"]["max_over_mean"] == 1.0
    assert len(report["ordinary_shard_balance"]["shards"]) == 1

    write_growth_report(
        artifacts,
        audit,
        suite / "missing.json",
        source_root=suite,
        repository=None,
        ancestor=CONTEXT["sha"],
    )
    report = json.loads((artifacts / "growth-summary.json").read_text())
    assert report["profile_freshness"] is None
    assert "FileNotFoundError" in report["profile_reason"]
    assert report["ordinary_shard_balance"]["shards"][0]["index"] == 0
    assert audit["success"] is True


def test_split_file_rejects_new_shared_fixture(
    tmp_path: Path,
    observed_shards: Path,
) -> None:
    suite = tmp_path / "suite"
    shutil.copytree(observed_shards.parent / "suite", suite)
    first = suite / "tests/test_first.py"
    first.write_text(
        first.read_text()
        + "\n@pytest.fixture(scope='module')\ndef shared(): return 1\ndef test_shared(shared): assert shared == 1\n"
    )
    output = tmp_path / "observed"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ci_metrics",
            "run",
            "--output",
            str(output),
            "--workers",
            "1",
            "--shard-index",
            "0",
            "--shard-count",
            "2",
            "--shard-weights",
            str(suite / "weights.json"),
        ],
        cwd=suite,
        env=headless_environment(dict(os.environ)),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "shared fixtures" in (output / "pytest.log").read_text()
    assert not json.loads((output / "summary.json").read_text())[
        "successful_test_baseline"
    ]


def test_storage_boundary_files_pass_the_case_split_collection_guard(
    tmp_path: Path,
) -> None:
    files = [
        f"tests/integration/agents_ops/test_storage_tier_{kind}_boundaries.py"
        for kind in ("crash", "recovery")
    ]
    profile, _ = read_timing_profile(REPO / "scripts/ci_metrics/cpu-file-times.json")
    # Exercise the same fixture guard when the baseline profile is checked out
    # for a timing comparison and has no measurement for these files yet.
    weights = tmp_path / "weights.json"
    weights.write_text(
        json.dumps(
            {
                **profile,
                "split_files": files,
                "file_seconds": {
                    name: profile["file_seconds"].get(name, 1.0) for name in files
                },
            }
        )
    )
    _, fingerprint = read_timing_profile(weights)
    registry, source_files = load_registry(REPO)
    lane = tmp_path / "plan.json"
    lane.write_text(
        json.dumps(make_plan(registry, source_files, source_sha=None, event="local"))
    )
    output = tmp_path / "observations"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "--collect-only",
            "-q",
            "-m",
            "not local_data and not cuda and not postgres",
            "-p",
            "scripts.ci_metrics.pytest_plugin",
            "-p",
            "scripts.ci_metrics.shard_plugin",
            "--ci-shard-index",
            "0",
            "--ci-shard-count",
            "10",
            "--ci-shard-weights",
            str(weights),
            "--ci-lane-plan",
            str(lane),
            "--ci-metrics-dir",
            str(output),
            *files,
        ],
        cwd=REPO,
        env=headless_environment(dict(os.environ)),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "shard-controller.json").read_text())
    assert manifest["profile_sha256"] == fingerprint
    assert {node.split("::", 1)[0] for node in manifest["universe_nodeids"]} == set(
        files
    )
    assert set(manifest["assignments"]) == set(manifest["universe_nodeids"])
    assert set(manifest["assignments"].values()) == set(range(10))


@pytest.mark.parametrize(
    "fault",
    [
        "missing_artifact",
        "duplicate_index",
        "old_head",
        "wrong_run",
        "wrong_source_commit",
        "future_attempt",
        "subset",
        "worker_inventory",
        "missing_worker",
        "lost_phase",
        "duplicate_phase",
        "missing_test",
        "duplicate_test",
        "profile",
        "all_manifests_omit_case",
        "missing_universe_event",
        "mixed_plan",
    ],
)
def test_aggregation_rejects_incomplete_or_mixed_evidence(
    tmp_path: Path,
    observed_shards: Path,
    fault: str,
) -> None:
    root = tmp_path / "artifacts"
    shutil.copytree(observed_shards, root)
    first = root / "shard-0/tests"
    second = root / "shard-1/tests"
    metadata_path = second / "environment.json"
    metadata = json.loads(metadata_path.read_text())
    if fault == "missing_artifact":
        shutil.rmtree(second.parent)
    elif fault in {
        "duplicate_index",
        "old_head",
        "wrong_run",
        "wrong_source_commit",
        "future_attempt",
        "subset",
    }:
        if fault == "duplicate_index":
            metadata["shard"]["index"] = 0
        elif fault == "old_head":
            metadata["environment"]["GITHUB_SHA"] = "b" * 40
        elif fault == "wrong_run":
            metadata["environment"]["GITHUB_RUN_ID"] = "456"
        elif fault == "wrong_source_commit":
            metadata["source"]["commit"] = "b" * 40
        elif fault == "future_attempt":
            metadata["environment"]["GITHUB_RUN_ATTEMPT"] = "2"
        else:
            metadata["cpu_target_before_sharding"] = False
        metadata_path.write_text(json.dumps(metadata))
    elif fault == "missing_worker":
        (second / "shard-gw1.json").unlink()
    elif fault in {"lost_phase", "duplicate_phase"}:
        events = next(second.glob("events-controller-*.jsonl"))
        rows = [json.loads(line) for line in events.read_text().splitlines()]
        phase = next(row for row in rows if row["event"] == "phase")
        if fault == "lost_phase":
            rows.remove(phase)
        else:
            rows.append(phase)
        events.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif fault == "all_manifests_omit_case":
        omitted = json.loads((first / "shard-gw0.json").read_text())[
            "universe_nodeids"
        ][0]
        for path in root.glob("*/tests/shard-*.json"):
            plan = json.loads(path.read_text())
            for key in ("universe_nodeids", "requested_nodeids", "selected_nodeids"):
                plan[key] = [node for node in plan[key] if node != omitted]
            path.write_text(json.dumps(plan))
    elif fault == "missing_universe_event":
        for path in second.glob("events-gw*.jsonl"):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            path.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in rows
                    if row["event"] != "cpu_universe"
                )
            )
    elif fault == "mixed_plan":
        path = second / "lane-plan.json"
        plan = json.loads(path.read_text())
        plan["scope"] = "pr"
        path.write_text(json.dumps(plan))
    else:
        paths = (
            [second / "shard-gw0.json"]
            if fault == "worker_inventory"
            else list(second.glob("shard-*.json"))
        )
        for path in paths:
            plan = json.loads(path.read_text())
            if fault in {"worker_inventory", "missing_test"}:
                plan["universe_nodeids"].pop()
            elif fault == "duplicate_test":
                other = json.loads((first / "shard-gw0.json").read_text())
                plan["selected_nodeids"].append(other["selected_nodeids"][0])
            else:
                plan["profile_sha256"] = "different"
            path.write_text(json.dumps(plan))
    result = verify_shards(root, 2, needs=SUCCESS, **CONTEXT)
    assert result["success"] is False, result
    assert result["issues"]
    if fault == "all_manifests_omit_case":
        assert any("pre-selection CPU universe" in issue for issue in result["issues"])
    # Profiling derives its upper attempt bound from the shards themselves.
    if fault == "future_attempt":
        assert build_profile(root)["source"]["attempt"] == "2"
        return
    with pytest.raises(ValueError, match="incomplete"):
        build_profile(root)


@pytest.mark.parametrize("job", ["plan", "lint", "tests", "system-deps"])
@pytest.mark.parametrize("outcome", ["failure", "cancelled", "skipped", None])
def test_aggregation_requires_all_upstream_jobs(
    observed_shards: Path,
    job: str,
    outcome: str | None,
) -> None:
    needs = {**SUCCESS, job: {"result": outcome}}
    assert not verify_shards(observed_shards, 2, needs=needs, **CONTEXT)["success"]


@pytest.mark.parametrize("result", ["success", "failure", "skipped", None])
def test_postgres_job_is_required_by_the_current_workflow(
    observed_shards: Path, result: str | None
) -> None:
    needs = dict(SUCCESS)
    if result is not None:
        needs["postgres"] = {"result": result}
    audit = verify_shards(
        observed_shards, 2, needs=needs, require_postgres=True, **CONTEXT
    )
    assert audit["success"] is (result == "success")
