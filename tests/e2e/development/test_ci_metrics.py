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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.ci_metrics import runner as measurement_runner
from scripts.ci_metrics.actions import summarize_attempts
from scripts.ci_metrics.profile import build_profile
from scripts.ci_metrics.report import summarize
from scripts.ci_metrics.runner import test_environment as headless_environment
from scripts.ci_metrics.runner import write_json
from scripts.ci_metrics.shards import assign_groups, verify_shards

REPO = Path(__file__).resolve().parents[3]


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
        {"event": "collection", "at": 1.2, "monotonic_s": 1.2, "nodeids": [nodeid]},
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
    context = {"sha": "a" * 40, "run_id": "123", "attempt": "1"}
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
        needs={"lint": {"result": "success"}, "tests": {"result": "success"}},
        **context,
    )
    assert audit["selected_count"] == 1
    assert audit["observations_complete"] is True
    assert audit["success"] == (outcome == "passed")


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
    suite.write_text("def test_one():\n    assert 2 + 2 == 4\n")
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
