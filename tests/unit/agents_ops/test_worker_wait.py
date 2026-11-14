"""Wait policy boundaries with durable records and an injected monotonic clock."""

from __future__ import annotations

import copy
import importlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


@dataclass
class Clock:
    now: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def read(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@dataclass
class Evidence:
    module: Any
    store: Any
    directory: Path
    row: dict[str, Any]
    record: dict[str, Any]
    live: dict[str, bool]
    timeouts: list[float | None]

    def save(self) -> None:
        self.store.write_json(self.directory / "result.json", self.record)
        self.store.write_json(
            self.store.path, {"version": 1, "parents": {"o/r#1": self.row}}
        )

    def terminal(self, *, cleanup: bool = True) -> None:
        self.record.update(
            result="completed",
            turn_end_reason="completed",
            exit_code=0,
            finished_at=100.0,
            cleanup_complete=cleanup,
        )
        self.live.update(dict.fromkeys(self.live, False))
        self.save()

    def wait(
        self, clock: Clock | None = None, **kwargs: Any
    ) -> tuple[int, dict[str, Any]]:
        timer = clock or Clock()
        value: tuple[int, dict[str, Any]] = self.module.wait_worker(
            self.store, "o/r#1", 10, clock=timer.read, sleep=timer.sleep, **kwargs
        )
        return value


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Evidence:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    module = importlib.import_module("parent.worker_wait")
    store = importlib.import_module("parent.state").Store(tmp_path / "state")
    settings = importlib.import_module("lib.agent_settings")
    selected = settings.resolve(
        settings.DEFAULT_CONFIG, "codex", None, None, None, "write"
    )
    directory = store.directory / "worker-runs/current"
    directory.mkdir(parents=True)
    task = {
        "delegated": True,
        "agent": "codex",
        "delegation_id": "a" * 32,
        "worktree": str(tmp_path / "worktree"),
        "branch": "codex/worker",
        "worker_runs": [
            {
                "delegation_id": "a" * 32,
                "run_dir": "worker-runs/current",
                "kind": "initial",
                "recorded_at": "2026-11-13T00:00:00+00:00",
            }
        ],
    }
    record = {
        "schema_version": 1,
        "run_id": "one-run",
        "result": "running",
        "turn_end_reason": None,
        "exit_code": None,
        "task_success": None,
        "cleanup_complete": False,
        "finished_at": None,
        "contract": {
            "task_id": task["delegation_id"],
            "cwd": task["worktree"],
            "branch": task["branch"],
            "mode": "write",
            "settings": selected,
        },
    }
    value = Evidence(
        module,
        store,
        directory,
        {"tasks": {"10": task}},
        record,
        {"runner_alive": True, "agent_alive": True, "agent_group_alive": True},
        [],
    )
    value.save()

    def inspect(_runtime: Any, path: Path, *, timeout: float | None) -> dict[str, Any]:
        assert path == directory
        value.timeouts.append(timeout)
        saved = json.loads((path / "result.json").read_text())
        return {"record": saved, "summary": {**saved, **value.live}}

    monkeypatch.setattr(module.Runtime, "inspect", inspect)
    return value


def test_deadline_uses_remaining_time_and_never_sleeps_past_limit(
    evidence: Evidence,
) -> None:
    clock = Clock()
    code, value = evidence.wait(clock, timeout=2.5)
    assert code == 124 and value["wait_state"] == "timeout"
    assert clock.sleeps == [1.0, 1.0, 0.5]
    assert clock.now == 2.5 and evidence.timeouts == [2.5, 1.5, 0.5]
    assert value["termination_recorded"] is False


def test_ended_run_never_calls_sleep_or_marks_registry_ended(
    evidence: Evidence,
) -> None:
    evidence.terminal()
    (evidence.directory / "final.md").write_text("worker result")
    before = evidence.store.path.read_bytes()
    clock = Clock()
    code, value = evidence.wait(clock)
    assert code == 0 and value["wait_state"] == "ended"
    assert clock.sleeps == []
    assert value["termination_recorded"] is True and value["cleanup_complete"] is True
    assert value["worker_ended_recorded"] is False
    assert value["final_file"] == str(evidence.directory / "final.md")
    assert evidence.store.path.read_bytes() == before


@pytest.mark.parametrize("live_field", [None, "agent_alive", "agent_group_alive"])
def test_end_requires_cleanup_and_no_live_cli_or_group(
    evidence: Evidence, live_field: str | None
) -> None:
    evidence.terminal(cleanup=live_field is not None)
    if live_field:
        evidence.live[live_field] = True
    code, value = evidence.wait()
    assert code == 2 and value["wait_state"] == "cleanup_unconfirmed"
    assert value["termination_recorded"] is True
    assert value["worker_ended_recorded"] is False


def test_live_cli_does_not_replace_lost_runner_proof(evidence: Evidence) -> None:
    evidence.live["runner_alive"] = False
    code, value = evidence.wait()
    assert code == 2 and value["wait_state"] == "liveness_unknown"
    assert value["runner_summary"]["agent_alive"] is True


def test_terminal_record_waits_until_runner_itself_exits(evidence: Evidence) -> None:
    evidence.terminal()
    evidence.live["runner_alive"] = True
    clock = Clock()

    def exit_runner(seconds: float) -> None:
        clock.sleep(seconds)
        evidence.live["runner_alive"] = False

    code, value = evidence.module.wait_worker(
        evidence.store, "o/r#1", 10, clock=clock.read, sleep=exit_runner
    )
    assert code == 0 and value["wait_state"] == "ended"
    assert clock.sleeps == [1.0]


@pytest.mark.parametrize(
    "failure",
    ["missing", "corrupt", "contract", "incomplete_end", "missing_end_reason"],
)
def test_unavailable_or_mismatched_records_fail_without_guessing(
    evidence: Evidence, failure: str
) -> None:
    evidence.terminal()
    if failure == "missing":
        (evidence.directory / "result.json").unlink()
    elif failure == "corrupt":
        (evidence.directory / "result.json").write_text("{incomplete")
    elif failure == "contract":
        evidence.record["contract"]["task_id"] = "b" * 32
        evidence.save()
    elif failure == "incomplete_end":
        evidence.record["exit_code"] = None
        evidence.save()
    else:
        evidence.record.pop("turn_end_reason")
        evidence.save()
    clock = Clock()
    code, value = evidence.wait(clock)
    assert code == 2 and value["wait_state"] == "liveness_unknown"
    assert value["error"] == "runner_records_unavailable_or_invalid"
    assert value["termination_recorded"] is False and clock.sleeps == []


@pytest.mark.parametrize("change", ["delegation", "latest_run"])
def test_target_is_pinned_and_never_follows_new_work(
    evidence: Evidence, change: str
) -> None:
    def retarget(_seconds: float) -> None:
        task = evidence.row["tasks"]["10"]
        if change == "delegation":
            task["delegation_id"] = "b" * 32
            task["worker_runs"][0]["delegation_id"] = "b" * 32
        else:
            task["worker_runs"].append(
                {**task["worker_runs"][0], "run_dir": "worker-runs/new"}
            )
        evidence.save()

    with pytest.raises(ValueError, match="latest run changed"):
        evidence.module.wait_worker(evidence.store, "o/r#1", 10, sleep=retarget)
    assert len(evidence.timeouts) == 1


def test_hung_status_is_bounded_by_wait_deadline(
    evidence: Evidence, monkeypatch: pytest.MonkeyPatch
) -> None:
    def hung(_runtime: Any, _path: Path, *, timeout: float | None) -> None:
        assert timeout == 3
        raise subprocess.TimeoutExpired("agent_exec.sh status", timeout)

    monkeypatch.setattr(evidence.module.Runtime, "inspect", hung)
    code, value = evidence.wait(timeout=3)
    assert code == 124 and value["wait_state"] == "timeout"


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_timeout_requires_finite_positive_seconds(
    evidence: Evidence, timeout: float
) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        evidence.wait(timeout=timeout)
    assert evidence.timeouts == []


def test_old_delegation_history_is_not_a_current_run(evidence: Evidence) -> None:
    task = evidence.row["tasks"]["10"]
    task["worker_runs"][0]["delegation_id"] = "b" * 32
    evidence.save()
    with pytest.raises(ValueError, match="no recorded run"):
        evidence.wait()


def test_invocation_contract_is_checked_in_addition_to_result(
    evidence: Evidence,
) -> None:
    evidence.terminal()
    invocation = copy.deepcopy(evidence.record)
    invocation["contract"]["branch"] = "codex/different"
    evidence.store.write_json(evidence.directory / "invocation.json", invocation)
    code, value = evidence.wait()
    assert code == 2 and value["wait_state"] == "liveness_unknown"
    assert evidence.timeouts == []
