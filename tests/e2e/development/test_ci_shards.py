"""Exercise real CPU selection and reject incomplete or mismatched shard evidence."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.ci_metrics.profile import build_profile
from scripts.ci_metrics.runner import test_environment as headless_environment
from scripts.ci_metrics.shards import assign_groups, read_timing_profile, verify_shards

REPO = Path(__file__).resolve().parents[3]
CONTEXT = {"sha": "a" * 40, "run_id": "123", "attempt": "1"}
SUCCESS = {"lint": {"result": "success"}, "tests": {"result": "success"}}


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
            + "@pytest.mark.cuda\n"
            "def test_gpu(): assert False\n"
            "@pytest.mark.local_data\n"
            "def test_local(): assert False\n"
        )
    excluded = (
        suite / "tests/integration/synthetic_data_generation/test_court_dataset.py"
    )
    excluded.parent.mkdir(parents=True)
    excluded.write_text(
        "raise AssertionError('manual integration must not be collected')\n"
    )
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
    for index in range(2):
        output = artifacts / f"shard-{index}" / "tests"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.ci_metrics",
                "run",
                "--output",
                str(output),
                "--workers",
                "2",
                "--shard-index",
                str(index),
                "--shard-count",
                "2",
                "--shard-weights",
                str(weights),
            ],
            cwd=suite,
            env={
                **headless_environment(dict(os.environ)),
                "GITHUB_SHA": CONTEXT["sha"],
                "GITHUB_RUN_ID": CONTEXT["run_id"],
                "GITHUB_RUN_ATTEMPT": CONTEXT["attempt"],
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
    return artifacts


def test_real_shards_cover_exact_cpu_inventory(observed_shards: Path) -> None:
    result = verify_shards(observed_shards, 2, needs=SUCCESS, **CONTEXT)
    assert result["success"], result
    assert result["selected_count"] == 5
    assert result["counts"] == {"passed": 5}
    for directory in observed_shards.glob("*/tests"):
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["full_cpu_baseline"] is False
        plan = json.loads((directory / "shard-gw0.json").read_text())
        assert plan["unmeasured_files"] == ["tests/test_new.py"]
        assert len(plan["full_nodeids"]) == 5
        assert (
            plan["assignments"]["tests/test_first.py::test_cpu[0]"]
            != plan["assignments"]["tests/test_first.py::test_cpu[1]"]
        )


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
    assert profile["source"]["selected_count"] == len(results) == 5
    assert profile["source"]["counts"] == {"passed": 5}
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


@pytest.mark.parametrize(
    "fault",
    [
        "missing_artifact",
        "duplicate_index",
        "old_head",
        "old_attempt",
        "subset",
        "worker_inventory",
        "missing_worker",
        "lost_phase",
        "duplicate_phase",
        "missing_test",
        "duplicate_test",
        "profile",
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
    elif fault in {"duplicate_index", "old_head", "old_attempt", "subset"}:
        if fault == "duplicate_index":
            metadata["shard"]["index"] = 0
        elif fault == "old_head":
            metadata["environment"]["GITHUB_SHA"] = "b" * 40
        elif fault == "old_attempt":
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
    else:
        paths = (
            [second / "shard-gw0.json"]
            if fault == "worker_inventory"
            else list(second.glob("shard-*.json"))
        )
        for path in paths:
            plan = json.loads(path.read_text())
            if fault in {"worker_inventory", "missing_test"}:
                plan["full_nodeids"].pop()
            elif fault == "duplicate_test":
                other = json.loads((first / "shard-gw0.json").read_text())
                plan["selected_nodeids"].append(other["selected_nodeids"][0])
            else:
                plan["profile_sha256"] = "different"
            path.write_text(json.dumps(plan))
    result = verify_shards(root, 2, needs=SUCCESS, **CONTEXT)
    assert result["success"] is False, result
    assert result["issues"]
    with pytest.raises(ValueError, match="incomplete"):
        build_profile(root)


@pytest.mark.parametrize("job", ["lint", "tests"])
@pytest.mark.parametrize("outcome", ["failure", "cancelled", "skipped", None])
def test_aggregation_requires_all_upstream_jobs(
    observed_shards: Path,
    job: str,
    outcome: str | None,
) -> None:
    needs = {**SUCCESS, job: {"result": outcome}}
    assert not verify_shards(observed_shards, 2, needs=needs, **CONTEXT)["success"]
