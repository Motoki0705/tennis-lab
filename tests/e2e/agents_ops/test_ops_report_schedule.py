"""Fixed-clock coverage for three-day slots and crash/retry recovery."""

from __future__ import annotations

import importlib
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops/ops-report"))

schedule: Any = importlib.import_module("opsreport.schedule")


@pytest.mark.parametrize(
    ("installed", "anchor"),
    [
        ("2026-10-18T07:59:59.999999+09:00", "2026-10-18T08:00:00+09:00"),
        ("2026-10-18T08:00:00+09:00", "2026-10-19T08:00:00+09:00"),
        ("2026-10-18T08:00:01+09:00", "2026-10-19T08:00:00+09:00"),
        ("2026-10-17T22:59:59+00:00", "2026-10-18T08:00:00+09:00"),
        ("2026-12-31T23:00:00+09:00", "2027-01-01T08:00:00+09:00"),
    ],
)
def test_anchor_is_first_08_after_installation(installed: str, anchor: str) -> None:
    assert schedule.first_slot(schedule.aware_time(installed)).isoformat() == anchor


@pytest.mark.parametrize(
    ("anchor", "next_date"),
    [
        ("2026-01-31", "2026-02-03"),
        ("2026-12-30", "2027-01-02"),
        ("2027-02-28", "2027-03-03"),
        ("2028-02-28", "2028-03-02"),
        ("2028-02-29", "2028-03-03"),
    ],
)
def test_slots_keep_three_days_across_calendar_boundaries(
    anchor: str, next_date: str
) -> None:
    first = schedule.aware_time(anchor + "T08:00:00+09:00")
    second = schedule.aware_time(next_date + "T08:00:00+09:00")
    assert schedule.latest_slot(first, first - timedelta(seconds=1)) is None
    assert schedule.latest_slot(first, second - timedelta(seconds=1)) == first
    assert schedule.latest_slot(first, second) == second
    for k in range(2, 40):
        expected = first + timedelta(days=3 * k)
        assert schedule.latest_slot(first, expected + timedelta(hours=23)) == expected


def _path(ops_env: Any) -> Path:
    return (
        Path(ops_env.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/ops-report/schedule.json"
    )


def _status(ops_env: Any, now: str | None = None) -> dict[str, Any]:
    args = ("--now", now) if now else ()
    result = ops_env.run("schedule", "status", *args)
    assert result.returncode == 0, result.stderr
    data: dict[str, Any] = json.loads(result.stdout)
    return data


def _no_prs(ops_env: Any) -> None:
    state = ops_env.state
    state["prs"] = []
    ops_env.write_state(state)
    payload = json.loads(Path(ops_env.env["FAKE_AGENT_OUTPUT"]).read_text())
    payload["stale_pr_notes"] = {}
    ops_env.set_agent_output(payload)


def _reports(ops_env: Any) -> list[dict[str, Any]]:
    return [i for i in ops_env.state["issues"] if "ops-report" in i["labels"]]


def test_init_status_and_reinitialization_are_explicit(ops_env: Any) -> None:
    _path(ops_env).unlink()
    result = ops_env.run("schedule", "init", "--now", "2026-10-18T07:59:00+09:00")
    assert result.returncode == 0, result.stderr
    initial = _path(ops_env).read_bytes()
    status = _status(ops_env, "2026-10-18T07:59:59+09:00")
    assert status["anchor"] == "2026-10-18T08:00:00+09:00"
    assert status["next_slot"] == status["anchor"]
    assert status["pending_slot"] is None
    assert status["last_attempt"] is None and status["last_success"] is None
    again = ops_env.run("schedule", "init", "--now", "2026-10-20T09:00:00+09:00")
    assert again.returncode == 1
    assert "init never resets" in again.stderr
    assert _path(ops_env).read_bytes() == initial
    assert ops_env.calls() == []


@pytest.mark.parametrize("content", [None, "", "{", "{}", '{"version": 99}'])
def test_missing_or_corrupt_schedule_fails_closed(
    ops_env: Any, content: str | None
) -> None:
    path = _path(ops_env)
    if content is None:
        path.unlink()
    else:
        path.write_text(content)
    for args in (("schedule", "status"), ("report", "--if-due")):
        result = ops_env.run(*args)
        assert result.returncode == 1
        assert (
            "not initialized" in result.stderr
            if content is None
            else "corrupt schedule" in result.stderr
        )
    assert not path.exists() if content is None else path.read_text() == content
    assert ops_env.calls() == []
    assert not (ops_env.tmp / "agent_used.txt").exists()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("anchor", "2026-10-12T08:00:00"),
        ("anchor", "2026-10-12T09:00:00+09:00"),
        ("anchor", "2026-10-13T08:00:00+09:00"),
        ("last_attempt", {}),
        ("last_success", {"slot": "2026-10-13T08:00:00+09:00"}),
        ("skipped", [{"from": "bad"}]),
    ],
)
def test_invalid_schedule_fields_cannot_reset_anchor(
    ops_env: Any, key: str, value: Any
) -> None:
    path = _path(ops_env)
    state = json.loads(path.read_text())
    state[key] = value
    path.write_text(json.dumps(state))
    original = path.read_bytes()
    result = ops_env.run("report", "--if-due")
    assert result.returncode == 1
    assert "corrupt schedule" in result.stderr
    assert path.read_bytes() == original
    assert ops_env.calls() == []


def test_if_due_noops_before_anchor_and_on_days_between_slots(ops_env: Any) -> None:
    initial = _path(ops_env).read_bytes()
    early = ops_env.run("report", "--if-due", "--now", "2026-10-12T07:59:59+09:00")
    assert early.returncode == 0, early.stderr
    assert json.loads(early.stdout)["status"] == "not_due"
    assert _path(ops_env).read_bytes() == initial
    assert ops_env.calls() == []
    assert ops_env.run("report", "--if-due").returncode == 0
    after_success = _path(ops_env).read_bytes()
    (ops_env.tmp / "agent_used.txt").unlink()
    for now in ("2026-10-13T08:00:00+09:00", "2026-10-15T07:59:59+09:00"):
        result = ops_env.run("report", "--if-due", "--now", now)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["status"] == "already_published"
    assert _path(ops_env).read_bytes() == after_success
    assert len(_reports(ops_env)) == 1
    assert not (ops_env.tmp / "agent_used.txt").exists()


@pytest.mark.parametrize("failure", ["agent", "github"])
def test_failed_slot_can_retry_and_success_is_not_republished(
    ops_env: Any, failure: str
) -> None:
    extra = (
        {"FAKE_AGENT_FAIL": "1"}
        if failure == "agent"
        else {"FAKE_GH_FAIL": "issue list"}
    )
    failed = ops_env.run("report", "--if-due", **extra)
    assert failed.returncode == 1
    status = _status(ops_env)
    assert status["last_attempt"]["status"] == "failed"
    assert status["last_attempt"]["error"]
    assert status["pending_slot"] == "2026-10-12T08:00:00+09:00"
    assert status["last_success"] is None
    assert len(_reports(ops_env)) == (1 if failure == "agent" else 0)
    retried = ops_env.run("report", "--if-due", "--now", "2026-10-13T12:00:00+09:00")
    assert retried.returncode == 0, retried.stderr
    status = _status(ops_env, "2026-10-13T12:00:00+09:00")
    assert status["last_attempt"]["status"] == "succeeded"
    assert status["last_success"]["slot"] == "2026-10-12T08:00:00+09:00"
    assert status["pending_slot"] is None
    assert status["next_slot"] == "2026-10-15T08:00:00+09:00"
    assert status["skipped"] == []
    assert _reports(ops_env)[0]["title"] == "運用レポート 2026-10-12"
    until = (
        "2026-10-12T08:00:00+09:00"
        if failure == "agent"
        else "2026-10-13T12:00:00+09:00"
    )
    assert f"until={until}" in _reports(ops_env)[0]["body"]
    assert ops_env.run("report", "--if-due").returncode == 0
    assert len(_reports(ops_env)) == 1


def test_restart_catches_up_only_latest_slot_and_keeps_collection_boundary(
    ops_env: Any,
) -> None:
    assert ops_env.run("report", "--if-due").returncode == 0
    _no_prs(ops_env)
    delayed_now = "2026-10-23T17:00:00+09:00"
    result = ops_env.run("report", "--if-due", "--now", delayed_now)
    assert result.returncode == 0, result.stderr
    reports = _reports(ops_env)
    assert [r["title"] for r in reports] == [
        "運用レポート 2026-10-12",
        "運用レポート 2026-10-21",
    ]
    assert (
        "slot=20261021 since=2026-10-12T08:00:00+09:00 until=2026-10-23T17:00:00+09:00"
        in reports[-1]["body"]
    )
    status = _status(ops_env, delayed_now)
    assert status["skipped"] == [
        {
            "from": "2026-10-15T08:00:00+09:00",
            "through": "2026-10-18T08:00:00+09:00",
            "count": 2,
        }
    ]
    assert status["next_slot"] == "2026-10-24T08:00:00+09:00"
    (ops_env.tmp / "agent_used.txt").unlink()
    again = ops_env.run("report", "--if-due", "--now", delayed_now)
    assert again.returncode == 0
    assert len(_reports(ops_env)) == 2
    assert not (ops_env.tmp / "agent_used.txt").exists()


@pytest.mark.parametrize("failed_first", [False, True])
def test_delayed_first_success_covers_from_anchor_and_records_skips(
    ops_env: Any, failed_first: bool
) -> None:
    if failed_first:
        assert (
            ops_env.run("report", "--if-due", FAKE_GH_FAIL="repo view").returncode == 1
        )
    _no_prs(ops_env)
    result = ops_env.run("report", "--if-due", "--now", "2026-10-21T10:00:00+09:00")
    assert result.returncode == 0, result.stderr
    (report,) = _reports(ops_env)
    assert (
        "slot=20261021 since=2026-10-09T08:00:00+09:00 until=2026-10-21T10:00:00+09:00"
        in report["body"]
    )
    status = _status(ops_env, "2026-10-21T10:00:00+09:00")
    assert status["skipped"] == [
        {
            "from": "2026-10-12T08:00:00+09:00",
            "through": "2026-10-18T08:00:00+09:00",
            "count": 3,
        }
    ]
    assert status["last_success"]["slot"] == "2026-10-21T08:00:00+09:00"


def test_missing_published_success_and_backwards_clock_do_not_republish(
    ops_env: Any,
) -> None:
    assert ops_env.run("report", "--if-due").returncode == 0
    before = _path(ops_env).read_bytes()
    data = ops_env.state
    data["issues"] = [i for i in data["issues"] if "ops-report" not in i["labels"]]
    ops_env.write_state(data)
    (ops_env.tmp / "agent_used.txt").unlink()
    result = ops_env.run("report", "--if-due")
    assert result.returncode == 1
    assert "local success has no published report" in result.stderr
    assert _path(ops_env).read_bytes() == before
    assert not (ops_env.tmp / "agent_used.txt").exists()
    # A later success must never be replaced by an older clock's pending slot.
    data = json.loads(_path(ops_env).read_text())
    data["last_attempt"]["slot"] = "2026-10-15T08:00:00+09:00"
    data["last_success"]["slot"] = "2026-10-15T08:00:00+09:00"
    _path(ops_env).write_text(json.dumps(data))
    result = ops_env.run("report", "--if-due")
    assert result.returncode == 1
    assert "clock moved behind" in result.stderr
    assert _reports(ops_env) == []


def test_running_slot_after_crash_reconciles_published_issue(ops_env: Any) -> None:
    assert ops_env.run("report", "--if-due").returncode == 0
    path = _path(ops_env)
    data = json.loads(path.read_text())
    # Model interruption between gh create and saving the successful result.
    data["last_success"] = None
    data["last_attempt"] = {
        "slot": data["anchor"],
        "started_at": data["anchor"],
        "status": "running",
    }
    path.write_text(json.dumps(data))
    (ops_env.tmp / "agent_used.txt").unlink()
    result = ops_env.run("report", "--if-due")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "already_published"
    assert _status(ops_env)["last_success"]["url"] == _reports(ops_env)[0]["url"]
    assert len(_reports(ops_env)) == 1
    assert not (ops_env.tmp / "agent_used.txt").exists()


def test_missing_previous_success_cannot_reset_collection_window(ops_env: Any) -> None:
    assert ops_env.run("report", "--if-due").returncode == 0
    data = ops_env.state
    data["issues"] = [i for i in data["issues"] if "ops-report" not in i["labels"]]
    ops_env.write_state(data)
    (ops_env.tmp / "agent_used.txt").unlink()
    result = ops_env.run("report", "--if-due", "--now", "2026-10-15T08:00:00+09:00")
    assert result.returncode == 1
    assert "local success has no published report" in result.stderr
    assert (
        _status(ops_env, "2026-10-15T08:00:00+09:00")["last_attempt"]["status"]
        == "failed"
    )
    assert _reports(ops_env) == []
    assert not (ops_env.tmp / "agent_used.txt").exists()


def test_restart_after_later_slots_does_not_mark_a_published_crash_as_skipped(
    ops_env: Any,
) -> None:
    assert ops_env.run("report", "--if-due").returncode == 0
    path = _path(ops_env)
    data = json.loads(path.read_text())
    data["last_success"] = None
    data["last_attempt"] = {
        "slot": data["anchor"],
        "started_at": data["anchor"],
        "status": "running",
    }
    path.write_text(json.dumps(data))
    _no_prs(ops_env)
    # Even an intervening network failure must not permanently misclassify it.
    args = ("report", "--if-due", "--now", "2026-10-21T10:00:00+09:00")
    assert ops_env.run(*args, FAKE_GH_FAIL="issue list").returncode == 1
    assert ops_env.run(*args, FAKE_AGENT_FAIL="1").returncode == 1
    status = _status(ops_env, "2026-10-21T10:00:00+09:00")
    assert status["last_success"]["slot"] == "2026-10-12T08:00:00+09:00"
    assert status["skipped"] == [
        {
            "from": "2026-10-15T08:00:00+09:00",
            "through": "2026-10-18T08:00:00+09:00",
            "count": 2,
        }
    ]
    assert ops_env.run(*args).returncode == 0
    assert len(_reports(ops_env)) == 2


def test_concurrent_due_runs_publish_once(ops_env: Any) -> None:
    runs = [
        ops_env.start("report", "--if-due", FAKE_GH_CREATE_DELAY="1") for _ in range(2)
    ]
    results = [(p.communicate(timeout=120), p.returncode) for p in runs]
    assert [code for _, code in results] == [0, 0], results
    assert len(_reports(ops_env)) == 1
    assert any('"status": "already_published"' in out for (out, _), _code in results)


def test_preview_leaves_schedule_unchanged(ops_env: Any) -> None:
    before = _path(ops_env).read_bytes()
    result = ops_env.run("report", "--dry-run", "--now", "2026-10-15T10:00:00+09:00")
    assert result.returncode == 0, result.stderr
    assert "slot=20261015" in result.stdout
    assert _path(ops_env).read_bytes() == before
    assert _reports(ops_env) == []


def test_naive_clock_override_and_retired_env_are_rejected(ops_env: Any) -> None:
    result = ops_env.run("report", "--now", "2026-10-12T08:00:00")
    assert result.returncode == 1
    assert "timezone is required" in result.stderr
    result = ops_env.run("report", WEEKLY_REPORT_AGENT="claude")
    assert result.returncode == 1
    assert "unsupported legacy environment" in result.stderr
    assert ops_env.calls() == []
