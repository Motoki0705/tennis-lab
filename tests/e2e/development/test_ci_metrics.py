"""Measurement failures must not become successful performance baselines."""

from __future__ import annotations

import copy
import io
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.ci_metrics import plan as lane_planner
from scripts.ci_metrics import runner as measurement_runner
from scripts.ci_metrics.actions import summarize_attempts
from scripts.ci_metrics.growth import aggregate_snapshot, analyze
from scripts.ci_metrics.lanes import digest, make_plan, partition_nodeids
from scripts.ci_metrics.profile import build_profile
from scripts.ci_metrics.report import summarize
from scripts.ci_metrics.runner import test_environment as headless_environment
from scripts.ci_metrics.runner import write_json
from scripts.ci_metrics.shards import assign_groups, verify_shards, write_growth_report

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "field,value",
    [
        ("attempt", "3"),
        ("run_id", "456"),
        ("source_sha", "other-head"),
        ("attempt", None),
        ("attempt", 1),
        ("attempt", "invalid"),
        ("attempt", "0"),
        ("current_attempt", None),
        ("current_attempt", 2),
        ("current_attempt", "invalid"),
        ("current_attempt", "0"),
    ],
)
def test_runner_rejects_invalid_plan_context_before_launching_pytest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_one.py").touch()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\npython_files = ["test_*.py"]\n'
    )
    registry = tmp_path / "scripts/ci_metrics/lanes.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"schema_version": 1, "suites": []}))
    context: dict[str, Any] = {"source_sha": "head", "run_id": "123", "attempt": "1"}
    if field != "current_attempt":
        context[field] = value
    plan = make_plan(
        {"schema_version": 1, "suites": []},
        ["tests/test_one.py"],
        event="push",
        ref="refs/heads/main",
        **context,
    )
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    weights = tmp_path / "weights.json"
    write_json(weights, {"schema_version": 1, "file_seconds": {"tests/test_one.py": 1}})
    metadata = {
        "source": {"commit": "head", "status": ""},
        "environment": {
            "GITHUB_SHA": "head",
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": value if field == "current_attempt" else "2",
        },
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(measurement_runner, "environment_metadata", lambda *a: metadata)
    monkeypatch.setattr(
        measurement_runner.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("invalid plan launched pytest"),
    )
    with pytest.raises(ValueError, match="attempt|revision|run"):
        measurement_runner.run(
            tmp_path / "output", 1, [], None, (0, 1, weights), plan_path
        )


def _collection_details(nodeids: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "nodeid": node,
            "file": node.split("::", 1)[0],
            "family": node,
            "originalname": node.rsplit("::", 1)[-1],
            "parametrized": False,
            "callspec_id": None,
            "fixtures": ["shared"],
        }
        for node in nodeids
    ]


def _observation(root: Path, outcome: str = "passed") -> list[dict[str, Any]]:
    root.mkdir(exist_ok=True)
    write_json(root / "environment.json", {"workers": 0, "full_cpu_target": True})
    exitcode = int(outcome in {"setup_failed", "failed", "teardown_failed"})
    write_json(
        root / "run.json",
        {
            "status": "completed",
            "exitcode": exitcode,
            "started_at": 1,
            "started_monotonic_s": 1,
            "command_finished_monotonic_s": 5,
            "finished_at": 5,
        },
    )
    nodeid = "tests/test_sample.py::test_one"
    reports = [
        {
            "phase": "setup",
            "outcome": outcome.removeprefix("setup_")
            if outcome.startswith("setup_")
            else "passed",
        },
        *(
            []
            if outcome.startswith("setup_")
            else [
                {
                    "phase": "call",
                    "outcome": outcome
                    if outcome in {"passed", "failed", "skipped"}
                    else "passed",
                }
            ]
        ),
        {
            "phase": "teardown",
            "outcome": "failed" if outcome == "teardown_failed" else "passed",
        },
    ]
    if outcome in {"xfailed", "xpassed"}:
        reports[1]["outcome"] = "skipped" if outcome == "xfailed" else "passed"
        reports[1]["wasxfail"] = ""  # An empty reason still identifies xfail/xpass.
    events = [
        {
            "event": "configured",
            "role": "controller",
            "pid": 10,
            "at": 1,
            "imported_at": 0.5,
            "imported_monotonic_s": 1.01,
            "monotonic_s": 1.02,
        },
        {"event": "session_start", "at": 1.1, "monotonic_s": 1.1},
        {
            "event": "collection",
            "at": 1.2,
            "monotonic_s": 1.2,
            "nodeids": [nodeid],
            "details": _collection_details([nodeid]),
        },
        *[
            {
                "event": "phase",
                "nodeid": nodeid,
                "duration_s": 0.5,
                "worker": "controller",
                "start": 2,
                "stop": 2.5,
                "monotonic_s": 2.6,
                "reported_monotonic_s": 2.6,
                **report,
            }
            for report in reports
        ],
        {
            "event": "session_finish",
            "at": 5,
            "monotonic_s": 5,
            "exitstatus": exitcode,
            "max_rss_kib": 100,
            "event_write_s": 0.01,
        },
    ]
    tag = (
        '<error message="fixture failure"/>'
        if exitcode
        else "<skipped/>"
        if any(report["outcome"] == "skipped" for report in reports)
        else ""
    )
    (root / "junit.xml").write_text(
        f'<testsuites><testsuite><testcase classname="tests.test_sample" name="test_one" time="{0.5 * len(reports)}">{tag}</testcase></testsuite></testsuites>'
    )
    _events(root, events)
    return events


def _events(root: Path, events: list[dict[str, Any]], name: str = "controller") -> None:
    (root / f"events-{name}.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
    )


@pytest.mark.parametrize(
    "outcome,expected",
    [
        ("passed", "passed"),
        ("setup_skipped", "skipped"),
        ("skipped", "skipped"),
        ("setup_failed", "failed"),
        ("failed", "failed"),
        ("teardown_failed", "failed"),
        ("xfailed", "xfailed"),
        ("xpassed", "xpassed"),
    ],
)
def test_normal_failure_and_setup_skip_are_complete_observations(
    tmp_path: Path, outcome: str, expected: str
) -> None:
    _observation(tmp_path, outcome)
    result = summarize(tmp_path)
    assert result["issues"] == []
    assert result["counts"] == {expected: 1}
    assert result["successful_test_baseline"] == (expected != "failed")


@pytest.mark.parametrize("outcome", ["passed", "failed", "setup_failed"])
def test_profile_retains_failures_without_accepting_them_in_python_job(
    tmp_path: Path, outcome: str
) -> None:
    directory = tmp_path / "shard-0/tests"
    directory.mkdir(parents=True)
    _observation(directory, outcome)
    nodeids = ["tests/test_sample.py::test_one"]
    context: dict[str, Any] = {"sha": "a" * 40, "run_id": "123", "attempt": "1"}
    write_json(
        directory / "environment.json",
        {
            "workers": 0,
            "cpu_target_before_sharding": True,
            "source": {"commit": context["sha"], "status": ""},
            "environment": dict(
                zip(
                    ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"),
                    context.values(),
                    strict=True,
                )
            ),
            "shard": {"index": 0, "count": 1, "profile_sha256": "profile"},
        },
    )
    write_json(
        directory / "shard-controller.json",
        {
            "schema_version": 1,
            "index": 0,
            "count": 1,
            "profile_sha256": "profile",
            "full_nodeids": nodeids,
            "selected_nodeids": nodeids,
            **assign_groups(nodeids, 1, {"tests/test_sample.py": 1}),
        },
    )
    profile = build_profile(tmp_path)
    assert profile["file_seconds"] == {
        "tests/test_sample.py": 1.0 if outcome == "setup_failed" else 1.5
    }
    assert profile["source"]["successful_test_observations"] == (outcome == "passed")
    expected = {} if outcome == "passed" else {nodeids[0]: "failed"}
    assert profile["source"]["unsuccessful_cases"] == expected
    audit = verify_shards(
        tmp_path,
        1,
        needs={name: {"result": "success"} for name in ("plan", "lint", "tests")},
        **context,
    )
    assert audit["selected_count"] == 1
    assert audit["observations_complete"] is True
    assert audit["success"] == (outcome == "passed")


@pytest.fixture
def scoped_observation(tmp_path: Path) -> Path:
    directory = tmp_path / "shard-0/tests"
    directory.mkdir(parents=True)
    events = _observation(directory)
    universe = ["tests/test_sample.py::test_one", "tests/test_wide.py::test_deferred"]
    registry = {
        "schema_version": 1,
        "suites": [
            {
                "id": "wide",
                "files": ["tests/test_wide.py"],
                "paths": ["tools/wide/**"],
                "reason": "Broad independent regression",
                "owner": "repository-owner",
            }
        ],
    }
    plan = make_plan(
        registry,
        ["tests/test_sample.py", "tests/test_wide.py"],
        source_sha="a" * 40,
        event="pull_request",
        run_id="123",
        attempt="1",
        base_sha="base",
        head_sha="head",
    )
    requested, deferred = partition_nodeids(universe, plan)
    events.insert(
        2,
        {
            "event": "cpu_universe",
            "nodeids": universe,
            "details": _collection_details(universe),
            "at": 1.15,
            "monotonic_s": 1.15,
        },
    )
    _events(directory, events)
    write_json(directory / "lane-plan.json", plan)
    write_json(
        directory / "environment.json",
        {
            "workers": 0,
            "full_cpu_target": True,
            "cpu_target_before_sharding": True,
            "lane_plan_id": plan["plan_id"],
            "source": {"commit": "a" * 40, "status": ""},
            "environment": {
                "GITHUB_SHA": "a" * 40,
                "GITHUB_RUN_ID": "123",
                "GITHUB_RUN_ATTEMPT": "1",
            },
            "shard": {"index": 0, "count": 1, "profile_sha256": "profile"},
        },
    )
    write_json(
        directory / "shard-controller.json",
        {
            "schema_version": 2,
            "index": 0,
            "count": 1,
            "profile_sha256": "profile",
            "plan_id": plan["plan_id"],
            "policy_sha256": plan["policy_sha256"],
            "scope": plan["scope"],
            "universe_nodeids": universe,
            "requested_nodeids": requested,
            "deferred": deferred,
            "selected_nodeids": requested,
            **assign_groups(requested, 1, {"tests/test_sample.py": 1}),
        },
    )
    return tmp_path


def test_scoped_success_does_not_count_deferred_tests_as_skips_or_full_success(
    scoped_observation: Path,
) -> None:
    summary = summarize(scoped_observation / "shard-0/tests")
    assert summary["issues"] == []
    assert summary["scoped_cpu_success"] is True
    assert summary["full_cpu_baseline"] is False
    assert summary["counts"] == {"passed": 1}
    audit = verify_shards(
        scoped_observation,
        1,
        sha="a" * 40,
        run_id="123",
        attempt="1",
        needs={name: {"result": "success"} for name in ("plan", "lint", "tests")},
    )
    assert audit["success"] and audit["scoped_cpu_success"]
    assert not audit["full_cpu_success"] and not audit["full_cpu_coverage"]
    assert (
        audit["universe_count"],
        audit["requested_count"],
        audit["deferred_count"],
    ) == (2, 1, 1)
    assert audit["suite_results"]["wide"]["verified_success"] is False
    with pytest.raises(ValueError, match="partial CPU scope"):
        build_profile(scoped_observation)
    snapshot = aggregate_snapshot(scoped_observation, audit)
    assert snapshot["nodeids"] == ["tests/test_sample.py::test_one"]
    assert (
        snapshot["universe_count"],
        snapshot["requested_count"],
        snapshot["deferred_count"],
    ) == (2, 1, 1)


def test_growth_combines_shards_without_recounting_fixture_time_or_inventory(
    scoped_observation: Path,
) -> None:
    first = scoped_observation / "shard-0/tests"
    scoped_plan = json.loads((first / "lane-plan.json").read_text())
    plan = make_plan(
        scoped_plan["registry"],
        scoped_plan["source_files"],
        source_sha="a" * 40,
        event="push",
        ref="refs/heads/main",
        run_id="123",
        attempt="1",
    )
    universe = ["tests/test_sample.py::test_one", "tests/test_wide.py::test_deferred"]
    assignment = assign_groups(
        universe, 2, {"tests/test_sample.py": 1, "tests/test_wide.py": 1}
    )
    base_metadata = json.loads((first / "environment.json").read_text())
    for index, node in enumerate(universe):
        directory = scoped_observation / f"shard-{index}/tests"
        directory.mkdir(parents=True, exist_ok=True)
        events = _observation(directory)
        for event in events:
            if event["event"] == "collection":
                event.update(nodeids=[node], details=_collection_details([node]))
            elif event["event"] == "phase":
                event["nodeid"] = node
                event["duration_s"] = {"setup": 1.2, "call": 2, "teardown": 2}[
                    event["phase"]
                ]
        events.insert(
            2,
            {
                "event": "cpu_universe",
                "at": 1.15,
                "monotonic_s": 1.15,
                "nodeids": universe,
                "details": _collection_details(universe),
            },
        )
        events.insert(
            4,
            {
                "event": "fixture",
                "at": 1.5,
                "monotonic_s": 1.5,
                "name": "shared",
                "scope": "session",
                "source": "support.shared",
                "nodeid": "",
                "outcome": "passed",
                "setup_s": 0.6 if index == 0 else 1.1,
            },
        )
        _events(directory, events)
        (directory / "junit.xml").write_text(
            f'<testsuites><testsuite><testcase classname="{node.split("::")[0].removesuffix(".py").replace("/", ".")}" name="{node.split("::")[-1]}" time="5.2"/></testsuite></testsuites>'
        )
        metadata = copy.deepcopy(base_metadata)
        metadata.update(
            lane_plan_id=plan["plan_id"],
            shard={"index": index, "count": 2, "profile_sha256": "profile"},
        )
        metadata["environment"]["GITHUB_RUN_ATTEMPT"] = str(index + 1)
        write_json(directory / "environment.json", metadata)
        write_json(directory / "lane-plan.json", plan)
        write_json(
            directory / "shard-controller.json",
            {
                "schema_version": 2,
                "index": index,
                "count": 2,
                "profile_sha256": "profile",
                "plan_id": plan["plan_id"],
                "policy_sha256": plan["policy_sha256"],
                "scope": plan["scope"],
                "universe_nodeids": universe,
                "requested_nodeids": universe,
                "deferred": {},
                "selected_nodeids": [node],
                **assignment,
            },
        )
    audit = verify_shards(
        scoped_observation,
        2,
        sha="a" * 40,
        run_id="123",
        attempt="2",
        needs={name: {"result": "success"} for name in ("plan", "lint", "tests")},
        lane_plan=plan,
    )
    assert audit["success"], audit
    assert audit["plan_attempt"] == "1"
    assert audit["shard_attempts"] == {"0": "1", "1": "2"}
    snapshot = aggregate_snapshot(scoped_observation, audit)
    assert snapshot["attempt"] == "2"
    report = analyze(snapshot, None, baseline_reason="first full run")
    assert len(snapshot["nodeids"]) == 2
    assert sum(row["total_s"] for row in report["slow_cases"]) == pytest.approx(10.4)
    assert report["slow_fixtures"][0]["count"] == 2
    assert report["slow_fixtures"][0]["setup_s"] == pytest.approx(1.7)
    assert report["slow_fixtures"][0]["max_setup_s"] == 1.1
    # Missing comparison evidence and heavy tests leave the verified gate intact.
    document = write_growth_report(
        scoped_observation,
        audit,
        scoped_observation / "missing-profile.json",
        source_root=scoped_observation,
        repository=None,
        ancestor="a" * 40,
    )
    assert "GITHUB_REPOSITORY is unavailable" in document
    assert "Profile comparison unavailable" in document
    assert audit["success"] is True
    (first / "inventory-details.json").unlink()
    unavailable = write_growth_report(
        scoped_observation,
        audit,
        scoped_observation / "missing-profile.json",
        source_root=scoped_observation,
        repository=None,
        ancestor="a" * 40,
    )
    assert (
        "Report unavailable:" in unavailable and "inventory-details.json" in unavailable
    )
    assert (
        json.loads((scoped_observation / "growth.json").read_text())["complete"]
        is False
    )
    assert audit["success"] is True


@pytest.mark.parametrize(
    "fault",
    [
        "unjustified_defer",
        "missing_from_r_and_d",
        "r_d_overlap",
        "missing_from_u_r_d",
        "wrong_policy",
        "extra_assignment",
        "mixed_schema",
    ],
)
def test_scope_audit_rejects_unexplained_or_missing_cases(
    scoped_observation: Path, fault: str
) -> None:
    path = scoped_observation / "shard-0/tests/shard-controller.json"
    manifest = json.loads(path.read_text())
    if fault == "unjustified_defer":
        manifest["requested_nodeids"] = []
        manifest["deferred"]["tests/test_sample.py::test_one"] = "wide"
    elif fault == "missing_from_r_and_d":
        manifest["deferred"] = {}
    elif fault == "r_d_overlap":
        manifest["deferred"]["tests/test_sample.py::test_one"] = "wide"
    elif fault == "missing_from_u_r_d":
        manifest["universe_nodeids"].pop()
        manifest["deferred"] = {}
    elif fault == "wrong_policy":
        manifest["policy_sha256"] = "old"
    elif fault == "extra_assignment":
        manifest["assignments"]["tests/test_outside.py"] = 0
    else:
        manifest["schema_version"] = 1
    write_json(path, manifest)
    summary = summarize(path.parent)
    assert not summary["successful_test_baseline"]
    assert any("invalid lane evidence" in issue for issue in summary["issues"])


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "all_skipped",
        "partial",
        "old_attempt",
        "old_policy",
        "expired",
        "api_error",
    ],
)
def test_last_main_success_requires_same_attempt_full_audit_and_executed_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str | None,
) -> None:
    registry = {
        "schema_version": 1,
        "suites": [
            {
                "id": "wide",
                "files": ["tests/test_wide.py"],
                "paths": ["tools/wide/**"],
                "reason": "Broad regression",
                "owner": "repository-owner",
            }
        ],
    }
    coverage = {
        "success": True,
        "full_cpu_success": fault != "partial",
        "source_sha": "main-sha",
        "run_id": "42",
        "attempt": "1" if fault != "old_attempt" else "2",
        "event": "push",
        "ref": "refs/heads/main",
        "policy_sha256": digest(registry) if fault != "old_policy" else "old",
        "suite_results": {
            "wide": {
                "verified_success": fault != "all_skipped",
                "counts": {"passed": 1} if fault != "all_skipped" else {"skipped": 1},
            }
        },
    }
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("coverage.json", json.dumps(coverage))

    def api(endpoint: str) -> bytes:
        if fault == "api_error":
            raise subprocess.CalledProcessError(1, "gh api")
        if "/workflows/" in endpoint:
            return json.dumps(
                {
                    "workflow_runs": [
                        {
                            "id": 42,
                            "run_attempt": 1,
                            "head_sha": "main-sha",
                            "event": "push",
                            "head_branch": "main",
                            "conclusion": "success",
                            "html_url": "https://github.com/owner/repo/actions/runs/42",
                        }
                    ]
                }
            ).encode()
        if endpoint.endswith("/zip"):
            return archive.getvalue()
        return json.dumps(
            {
                "artifacts": [
                    {
                        "name": "cpu-test-coverage-42-1",
                        "id": 7,
                        "expired": fault == "expired",
                    }
                ]
            }
        ).encode()

    monkeypatch.setattr(lane_planner, "_api", api)
    monkeypatch.setattr(
        lane_planner.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0)
    )
    result = lane_planner.main_successes(tmp_path, "owner/repo", "head", registry)[
        "wide"
    ]
    assert result["commit"] == ("main-sha" if fault is None else None)
    plan = make_plan(
        registry,
        ["tests/test_wide.py"],
        source_sha="head",
        event="pull_request",
        base_sha="base",
        head_sha="head",
        main_successes={"wide": result},
    )
    document = lane_planner.markdown(plan)
    assert ("main-sha" in document) == (fault is None)
    assert ("unknown:" in document) == (fault is not None)


@pytest.mark.parametrize(
    "fault",
    [
        "junit",
        "finish",
        "phase",
        "identity",
        "partial_json",
        "runner",
        "cancel",
        "worker_inventory",
        "monotonic",
    ],
)
def test_missing_interrupted_and_mismatched_measurements_are_not_baselines(
    tmp_path: Path, fault: str
) -> None:
    events = _observation(tmp_path)
    if fault == "junit":
        (tmp_path / "junit.xml").unlink()
    elif fault == "finish":
        _events(tmp_path, events[:-1])
    elif fault == "phase":
        _events(tmp_path, [e for e in events if e.get("phase") != "teardown"])
    elif fault == "identity":
        xml = tmp_path / "junit.xml"
        xml.write_text(xml.read_text().replace('name="test_one"', 'name="test_other"'))
    elif fault == "partial_json":
        with (tmp_path / "events-controller.jsonl").open("a") as stream:
            stream.write('{"event":')
    elif fault == "runner":
        (tmp_path / "run.json").unlink()
    elif fault == "cancel":
        write_json(
            tmp_path / "run.json",
            {"status": "cancelled", "exitcode": 0, "finished_at": 5},
        )
    elif fault == "worker_inventory":
        write_json(tmp_path / "environment.json", {"workers": 2})
        _events(
            tmp_path,
            [
                *events,
                {
                    "event": "worker_collection",
                    "at": 1.3,
                    "worker": "gw0",
                    "nodeids": ["other"],
                },
            ],
        )
    elif fault == "monotonic":
        run_path = tmp_path / "run.json"
        state = json.loads(run_path.read_text())
        del state["started_monotonic_s"]
        write_json(run_path, state)
    result = summarize(tmp_path)
    assert result["successful_test_baseline"] is False
    assert result["full_cpu_baseline"] is False
    assert result["issues"] or fault == "cancel"
    if fault == "phase":
        assert result["counts"] == {"incomplete": 1}


def _attempt(attempt: int = 1, conclusion: str = "success") -> dict[str, Any]:
    return {
        "run": {
            "id": 42,
            "run_attempt": attempt,
            "status": "completed",
            "conclusion": conclusion,
            "created_at": "2026-01-01T00:00:00Z",
            "run_started_at": "2026-01-01T00:10:00Z",
            "head_sha": "same-revision",
        },
        "requested_at": "2026-01-01T00:00:00Z",
        "jobs": [
            {
                "id": 100 + attempt,
                "run_id": 42,
                "head_sha": "same-revision",
                "run_attempt": attempt,
                "name": "Python",
                "status": "completed",
                "conclusion": conclusion,
                "created_at": "2026-01-01T00:00:05Z",
                "started_at": "2026-01-01T00:00:10Z",
                "completed_at": "2026-01-01T00:02:00Z",
                "steps": [
                    {
                        "name": "Test",
                        "status": "completed",
                        "conclusion": conclusion,
                        "started_at": "2026-01-01T00:00:20Z",
                        "completed_at": "2026-01-01T00:01:50Z",
                    }
                ],
            },
            {
                "id": 200 + attempt,
                "run_id": 42,
                "head_sha": "same-revision",
                "run_attempt": attempt,
                "name": "Other",
                "status": "completed",
                "conclusion": "success",
                "created_at": "2026-01-01T00:00:00Z",
                "started_at": "2026-01-01T00:00:05Z",
                "completed_at": "2026-01-01T00:00:35Z",
                "steps": [],
            },
        ],
    }


@pytest.mark.parametrize("changed_identity", [False, True])
def test_collection_skip_does_not_disable_junit_identity_or_time_checks(
    tmp_path: Path, changed_identity: bool
) -> None:
    events = _observation(tmp_path)
    events.append(
        {
            "event": "collection_result",
            "at": 1.2,
            "monotonic_s": 1.2,
            "nodeid": "tests/test_skip.py",
            "outcome": "skipped",
        }
    )
    _events(tmp_path, events)
    xml = tmp_path / "junit.xml"
    text = xml.read_text().replace(
        "</testsuite>",
        '<testcase classname="" name="tests.test_skip" time="0"><skipped/></testcase></testsuite>',
    )
    if changed_identity:
        text = text.replace('name="test_one"', 'name="wrong_identity"')
    xml.write_text(text)
    result = summarize(tmp_path)
    assert result["successful_test_baseline"] is not changed_identity
    assert result["issues"] == (
        ["JUnit test identities differ from phase reports"] if changed_identity else []
    )


def test_attempts_separate_check_wait_from_all_runner_time_and_include_cancelled() -> (
    None
):
    result = summarize_attempts([_attempt(1, "cancelled"), _attempt(2)], ["Python"])
    assert result["conclusions"] == {"cancelled": 1, "success": 1}
    assert result["all_attempts_runner_s"] == 280
    first, second = result["attempts"]
    assert (
        first["observed_check_completion_s"]
        == second["observed_check_completion_s"]
        == 120
    )
    assert first["all_jobs_runner_s"] == 140
    assert first["dispatch_to_first_job_s"] == 5
    assert first["jobs"][0]["queue_s"] == 5
    assert first["phase_step_s"] == {"pytest": 90}
    assert first["successful_observed_checks"] is False
    assert second["successful_observed_checks"] is True
    assert result["successful_completion_distribution"] == {
        "n": 1,
        "median_s": 120,
        "p95_s": 120,
        "p95_method": "nearest rank: sorted[ceil(0.95*n)-1]",
        "p95_is_reference_only": True,
        "comparison_scope": "descriptive only; do not infer improvement across commits, runners, cache states or test sets",
    }


@pytest.mark.parametrize(
    "fault",
    [
        "timestamp",
        "check",
        "attempt",
        "steps",
        "pending",
        "backwards",
        "run",
        "head",
        "missing_head",
    ],
)
def test_actions_missing_or_failed_data_is_not_success(
    tmp_path: Path, fault: str
) -> None:
    record = _attempt()
    if fault == "timestamp":
        record["jobs"][0]["completed_at"] = None
    elif fault == "check":
        record["jobs"][0]["name"] = "Unrelated"
    elif fault == "attempt":
        record["jobs"][0]["run_attempt"] = 2
    elif fault == "steps":
        del record["jobs"][0]["steps"]
    elif fault == "pending":
        record["run"]["status"] = "in_progress"
    elif fault == "backwards":
        record["jobs"][0]["completed_at"] = "2025-01-01T00:00:00Z"
    elif fault == "run":
        record["jobs"][0]["run_id"] = 43
    elif fault == "head":
        record["jobs"][0]["head_sha"] = "other-revision"
    elif fault == "missing_head":
        del record["jobs"][0]["head_sha"]
    result = summarize_attempts([record], ["Python"])
    assert result["attempts"][0]["successful_observed_checks"] is False
    assert result["successful_completion_distribution"]["n"] == 0
    if fault in {"attempt", "run", "head", "missing_head"}:
        assert result["all_attempts_runner_s"] is None


def test_duplicate_and_missing_attempts_are_visible() -> None:
    with pytest.raises(ValueError, match="duplicate run/attempt"):
        summarize_attempts([_attempt(), copy.deepcopy(_attempt())], ["Python"])
    result = summarize_attempts([_attempt(2)], ["Python"])
    assert result["missing_attempts"] == [{"run_id": 42, "attempt": 1}]
    assert result["all_attempts_runner_s"] is None


def test_recorded_actions_retry_keeps_apt_cancellation_and_test_time_separate() -> None:
    records = json.loads(
        (REPO / "tests/support/development/ci_attempts.json").read_text()
    )
    result = summarize_attempts(records, ["Python"])
    first, second = result["attempts"]
    assert result["conclusions"] == {"cancelled": 1, "success": 1}
    assert first["phase_step_s"]["apt"] == 1480  # 24m40s
    assert first["phase_step_s"]["pytest"] == 0  # skipped, not successful execution
    assert first["successful_observed_checks"] is False
    assert second["phase_step_s"]["pytest"] == 2966  # 49m26s
    assert result["all_attempts_runner_s"] == 1518 + 3027
    assert result["successful_completion_distribution"]["n"] == 1
    assert second["queue_origin"].startswith("run_started_at proxy")


def test_test_subprocess_is_headless_and_cpu_bounded() -> None:
    original = {
        "HERDR_ENV": "1",
        "HERDR_SOCKET_PATH": "live",
        "CODEX_SESSION_ID": "parent",
        "CLAUDE_CODE_PARENT_SESSION_ID": "real-parent",
        "PYTEST_ADDOPTS": "-p outer_plugin",
        "PYTEST_XDIST_WORKER": "outer-worker",
        "PYTEST_CURRENT_TEST": "outer-case",
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "8",
        "PATH": "keep",
    }
    result = headless_environment(original)
    assert result == {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": "keep",
    }
    assert original["HERDR_ENV"] == "1"


def test_source_change_while_queued_requires_a_fresh_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = {"workers": 0, "source": {"revision": "before"}}
    after = {"workers": 0, "source": {"revision": "after"}}
    observations = iter([before, after, after])
    monkeypatch.setattr(
        measurement_runner, "environment_metadata", lambda *args: next(observations)
    )

    def must_not_start(*args: Any, **kwargs: Any) -> None:
        pytest.fail(
            "pytest must not start with already imported stale measurement code"
        )

    monkeypatch.setattr(measurement_runner.subprocess, "Popen", must_not_start)
    output = tmp_path / "measurement"
    with pytest.raises(RuntimeError, match="source changed while waiting"):
        measurement_runner.run(output, 0, [], tmp_path / "nested.lock")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["successful_test_baseline"] is False
    assert summary["run"]["status"] == "launcher_error"
    assert summary["run"]["exitcode"] is None


def test_runner_elapsed_time_survives_wall_clock_reversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wall_clock = iter([1000.0, 100.0, 90.0, 80.0])
    elapsed_clock = iter([10.0, 20.0, 24.0, 25.0])
    monkeypatch.setattr(
        measurement_runner,
        "time",
        SimpleNamespace(
            time=lambda: next(wall_clock), perf_counter=lambda: next(elapsed_clock)
        ),
    )
    metadata = {"workers": 0, "source": {"commit": "fixed-source"}}
    monkeypatch.setattr(
        measurement_runner, "environment_metadata", lambda *args: dict(metadata)
    )
    output = tmp_path / "measurement"

    class FinishedTest:
        pid = 12345
        stdout = io.BytesIO()

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            events = _observation(output)
            for event in events:
                event["monotonic_s"] += 19
                for key in ("imported_monotonic_s", "reported_monotonic_s"):
                    if key in event:
                        event[key] += 19
            _events(output, events)

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(measurement_runner.subprocess, "Popen", FinishedTest)
    assert measurement_runner.run(output, 0, [], tmp_path / "nested.lock") == 0
    result = json.loads((output / "summary.json").read_text())
    assert result["successful_test_baseline"] is True
    assert result["run"]["lock_wait_s"] == 10
    assert result["test_command_wall_s"] == 4
    assert result["wall_clock_adjustment_s"] == -14
    assert result["timing_breakdown"]["command_to_plugin_import_s"] == pytest.approx(
        0.01
    )
    assert result["timing_breakdown"]["last_phase_report_to_command_exit_s"] == (
        pytest.approx(2.4)
    )


def test_default_measurement_runs_while_legacy_pytest_lock_is_held(
    tmp_path: Path,
) -> None:
    import fcntl

    lock = tmp_path / "legacy.lock"
    output = tmp_path / "observed"
    suite = tmp_path / "test_small.py"
    suite.write_text(
        "import pytest\n@pytest.mark.parametrize('value', [4], ids=['outer[inner::id]'])\ndef test_one(value):\n    assert 2 + 2 == value\n"
    )
    environment = headless_environment(dict(os.environ))
    environment["PYTEST_LOCK"] = str(lock)
    with lock.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
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
                "--",
                str(suite),
            ],
            cwd=REPO,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["successful_test_baseline"] is True
    assert summary["counts"] == {"passed": 1}
    assert json.loads((output / "environment.json").read_text())["lock_path"] is None
    detail = json.loads((output / "inventory-details.json").read_text())
    assert detail["issues"] == []
    assert len(detail["items"]) == 1
    item = detail["items"][0]
    assert item["originalname"] == "test_one"
    assert item["family"].endswith("test_small.py::test_one")
    assert item["callspec_id"] == "outer[inner::id]"
    assert item["parametrized"] is True


def test_runner_cancellation_while_waiting_for_lock_leaves_partial_artifacts(
    tmp_path: Path,
) -> None:
    import fcntl

    lock = tmp_path / "nested.lock"
    output = tmp_path / "observed"
    with lock.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.ci_metrics",
                "run",
                "--output",
                str(output),
                "--lock",
                str(lock),
            ],
            cwd=REPO,
            env=headless_environment(dict(os.environ)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 15
            while not (output / "run.json").exists():
                assert process.poll() is None
                assert time.monotonic() < deadline
                time.sleep(0.01)
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=15)
            assert process.returncode == 143, stdout + stderr
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "cancelled"
    assert run.get("started_at") is None
    assert run["exitcode"] is None
    assert (
        json.loads((output / "summary.json").read_text())["successful_test_baseline"]
        is False
    )


def test_closed_job_log_terminates_the_test_process(tmp_path: Path) -> None:
    output = tmp_path / "observed"
    suite = tmp_path / "test_wait.py"
    suite.write_text("import time\ndef test_wait(): time.sleep(60)\n")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "scripts.ci_metrics",
            "run",
            "--output",
            str(output),
            "--workers",
            "1",
            "--",
            str(suite),
        ],
        cwd=REPO,
        env=headless_environment(dict(os.environ)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    process.stdout.close()
    process.stdout = None
    try:
        _, stderr = process.communicate(timeout=30)
        assert process.returncode != 0, stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
    state = json.loads((output / "run.json").read_text())
    assert state["status"] == "launcher_error"
    assert "BrokenPipeError" in state["launcher_error"]
    with pytest.raises(ProcessLookupError):
        os.kill(state["process_group_id"], 0)
    assert (
        json.loads((output / "summary.json").read_text())["successful_test_baseline"]
        is False
    )


def test_runner_cancellation_during_test_keeps_inventory_and_incomplete_result(
    tmp_path: Path,
) -> None:
    output = tmp_path / "observed"
    started = tmp_path / "test-started"
    suite = tmp_path / "test_wait.py"
    suite.write_text(
        f"from pathlib import Path\nimport time\ndef test_wait():\n    Path({str(started)!r}).touch()\n    time.sleep(60)\n"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "scripts.ci_metrics",
            "run",
            "--output",
            str(output),
            "--lock",
            str(tmp_path / "nested.lock"),
            "--workers",
            "1",
            "--",
            str(suite),
        ],
        cwd=REPO,
        env=headless_environment(dict(os.environ)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not started.exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 143, stdout + stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["successful_test_baseline"] is False
    assert summary["selected_count"] == 1
    assert summary["counts"] == {"incomplete": 1}
    assert summary["run"]["status"] == "cancelled"
