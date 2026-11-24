"""Persist consecutive skips and exercise the Windows notification boundary."""

from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def sync_cli(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    return importlib.import_module("git_sync.cli")


def test_notice_reports_the_last_holders_once_and_resets_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sync_cli: Any
) -> None:
    class Clock:
        current = datetime(2026, 11, 23, 0, tzinfo=UTC)
        fromisoformat = staticmethod(datetime.fromisoformat)

        @classmethod
        def now(cls, tz: object) -> datetime:
            assert tz == UTC
            return cls.current

    notices: list[tuple[str, str]] = []
    monkeypatch.setattr(sync_cli, "datetime", Clock)
    monkeypatch.setattr(
        sync_cli, "send_windows_toast", lambda *args: notices.append(args)
    )
    directory, root = tmp_path / "state", tmp_path / "repo"

    def skip(reason: str, **details: Any) -> dict[str, Any]:
        result: dict[str, Any] = sync_cli.Journal(directory, root).finish(
            "skipped", reason, **details
        )
        Clock.current += timedelta(minutes=30)
        streak = result["skip_streak"]
        assert isinstance(streak, dict)
        return streak

    old = {"entry": "python -m parent", "pid": 125, "mode": "read"}
    last = {
        "entry": "review/merge_gate.py",
        "pid": 481,
        "mode": "read",
        "parent_issue": 2278,
    }
    assert skip("checkout_jobs_running", lock_holders=[old])["count"] == 1
    assert skip("checkout_jobs_running", lock_holders=[last])["elapsed_seconds"] == 1800
    assert not notices
    streak = skip("control_operation_running")
    assert streak == {
        "count": 3,
        "since": "2026-11-23T00:00:00+00:00",
        "elapsed_seconds": 3600,
        "last_lock_holders": [last],
        "notification": "sent",
    }
    [(title, body)] = notices
    assert "git-sync" in title
    for part in (
        "3 consecutive skips",
        "1:00:00",
        "review/merge_gate.py",
        "PID 481",
        "parent #2278",
    ):
        assert part in body
    assert "PID 125" not in body
    events = [
        json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()
    ]
    [notice] = [record for record in events if record["reason"] == "consecutive_skips"]
    assert notice["last_lock_holders"] == [last]
    assert notice["count"] == 3 and notice["elapsed_seconds"] == 3600

    assert skip("checkout_jobs_running", lock_holders=[old])["count"] == 4
    assert len(notices) == 1
    result = sync_cli.Journal(directory, root).finish("unchanged", "already_current")
    assert "skip_streak" not in result
    for count in (1, 2, 3):
        assert skip("checkout_jobs_running", lock_holders=[old])["count"] == count
    assert len(notices) == 2


@pytest.mark.parametrize(
    "error_name,retried",
    [("ToastNotSent", True), ("ToastError", False)],
)
def test_notice_retries_only_confirmed_non_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sync_cli: Any,
    error_name: str,
    retried: bool,
) -> None:
    attempts = []

    def notify(title: str, body: str) -> None:
        attempts.append(body)
        if len(attempts) == 1:
            raise getattr(sync_cli, error_name)("injected notification failure")

    monkeypatch.setattr(sync_cli, "send_windows_toast", notify)
    results = [
        sync_cli.Journal(tmp_path, tmp_path / "repo").finish(
            "skipped", "checkout_jobs_running", lock_holders=[]
        )
        for _ in range(4)
    ]
    assert [result["skip_streak"]["count"] for result in results] == [1, 2, 3, 4]
    assert results[2]["skip_streak"]["notification"] == (
        "pending" if retried else "uncertain"
    )
    assert results[3]["skip_streak"]["notification"] == (
        "sent" if retried else "uncertain"
    )
    assert len(attempts) == (2 if retried else 1)
