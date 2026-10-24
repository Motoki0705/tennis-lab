"""Maintenance controller against a fake clock/API and a disposable real main."""

from __future__ import annotations

import copy
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"
sys.path.insert(0, str(OPS))
CycleAPI = importlib.import_module("daily_ci").CycleAPI
CycleStore = importlib.import_module("daily_ci").CycleStore
Controller = importlib.import_module("daily_ci.controller").Controller
GitHub = importlib.import_module("daily_ci.github").GitHub
window = importlib.import_module("daily_ci.model").window


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@dataclass
class Clock:
    value: datetime = field(
        default_factory=lambda: datetime.fromisoformat("2026-10-25T06:00:00+09:00")
    )

    def __call__(self) -> datetime:
        return self.value

    def set(self, value: str) -> None:
        self.value = datetime.fromisoformat(value)


class FakeActions:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.rows: dict[int, dict[str, Any]] = {}
        self.posts: list[tuple[str, dict[str, str]]] = []
        self.cancellations: list[int] = []
        self.lost_response: str | None = None
        self.accept = True
        self.visible = True
        self.cancel_effect = True
        self.on_main: Callable[[], None] | None = None
        self.on_dispatch: Callable[[], None] | None = None

    def main_sha(self) -> str:
        if self.on_main:
            self.on_main()
        return git(self.repo, "rev-parse", "HEAD")

    def commit(self, label: str) -> str:
        (self.repo / "main.txt").write_text(label)
        git(self.repo, "add", "main.txt")
        git(self.repo, "commit", "-m", label)
        return self.main_sha()

    def dispatch(
        self, workflow: str, inputs: dict[str, str], timeout: float
    ) -> int | None:
        assert timeout > 0
        self.posts.append((workflow, inputs))
        identifier = len(self.posts)
        if self.on_dispatch:
            self.on_dispatch()
        if self.accept:
            self.rows[identifier] = {
                "id": identifier,
                "run_attempt": 1,
                "event": "workflow_dispatch",
                "path": f".github/workflows/{workflow}",
                "head_branch": "main",
                "head_sha": inputs["target_sha"],
                "display_title": f"daily-ci:{inputs['request_id']}",
                "status": "queued",
                "conclusion": None,
            }
        if workflow == self.lost_response:
            raise subprocess.TimeoutExpired("fake dispatch", timeout)
        return identifier if self.accept else None

    def runs(self, workflow: str, requested_at: str) -> list[dict[str, Any]]:
        assert requested_at
        return (
            [
                copy.deepcopy(row)
                for row in self.rows.values()
                if row["path"] == f".github/workflows/{workflow}"
            ]
            if self.visible
            else []
        )

    def run(self, run_id: int) -> dict[str, Any]:
        return copy.deepcopy(self.rows[run_id])

    def cancel(self, run_id: int) -> None:
        self.cancellations.append(run_id)
        if self.cancel_effect:
            self.rows[run_id].update(status="completed", conclusion="cancelled")

    def complete(self, conclusion: str = "success") -> None:
        for row in self.rows.values():
            row.update(status="completed", conclusion=conclusion)


class FakeHooks:
    def __init__(self) -> None:
        self.results = False
        self.unresolved: list[dict[str, Any]] = []
        self.stop_state = "stopped"
        self.fail: tuple[str, str] | None = None
        self.calls: list[tuple[str, str]] = []

    def __call__(
        self, role: str, operation: str, cycle: dict[str, Any] | None, timeout: float
    ) -> dict[str, Any]:
        self.calls.append((role, operation))
        assert timeout > 0
        if self.fail == (role, operation):
            raise ValueError("injected adapter failure")
        if operation == "probe":
            assert cycle is None
            return {"schema_version": 1, "ready": True, "role": role}
        assert cycle is not None
        if role == "observer":
            results = []
            if self.results:
                for workflow, request in cycle["generations"][-1]["requests"].items():
                    if (
                        request["state"] == "bound"
                        and request["remote_status"] == "completed"
                    ):
                        results.append(result_for(cycle, workflow))
            return {
                "schema_version": 1,
                "results": results,
                "unresolved": self.unresolved,
            }
        if operation == "stop":
            return {
                "schema_version": 1,
                "acknowledged": True,
                "workers": {
                    worker: {
                        "state": self.stop_state,
                        "evidence": {"runtime": "checked"},
                    }
                    for worker in cycle["workers"]
                },
            }
        return {"schema_version": 1, "acknowledged": True}


def result_for(cycle: dict[str, Any], workflow: str = "ci.yml") -> dict[str, Any]:
    revision = cycle["generations"][-1]
    request = revision["requests"][workflow]
    return {
        "cycle_id": cycle["cycle_id"],
        "generation": revision["number"],
        "workflow": workflow,
        "request_id": request["request_id"],
        "run_id": request["run_id"],
        "attempt": request["attempt"],
        "target_sha": revision["target_sha"],
        "checkout_sha": revision["target_sha"],
        "baseline_sha": cycle["baseline_sha"],
        "result": "success" if request["remote_conclusion"] == "success" else "failure",
        "evidence": {"manifest": f"verified/{request['run_id']}/{request['attempt']}"},
    }


@dataclass
class Harness:
    api: Any
    clock: Clock
    actions: FakeActions
    hooks: FakeHooks

    @property
    def controller(self) -> Any:
        # A fresh controller each time exercises durable restart reconciliation.
        return Controller(self.api, self.actions, hooks=self.hooks)

    @property
    def cycle_id(self) -> str:
        value: str = window(self.clock())[0]
        return value

    def cycle(self, cycle_id: str | None = None) -> dict[str, Any]:
        value: dict[str, Any] = self.api.snapshot()["cycles"][cycle_id or self.cycle_id]
        return value

    def tick(self) -> dict[str, Any]:
        value: dict[str, Any] = self.controller.tick()
        return value


@pytest.fixture
def h(tmp_path: Path) -> Harness:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    actions = FakeActions(repo)
    baseline = actions.commit("initial")
    clock = Clock()
    api = CycleAPI(CycleStore("O/R", tmp_path / "state"), clock=clock)
    api.initialize(baseline)
    hooks = FakeHooks()
    harness = Harness(api, clock, actions, hooks)
    harness.controller.enable()
    return harness


def test_initial_once_restart_and_unchanged_main_next_day(h: Harness) -> None:
    h.clock.set("2026-10-25T05:59:59+09:00")
    h.tick()
    assert not h.actions.posts and not h.api.snapshot()["normal_merge_frozen"]
    h.clock.set("2026-10-25T06:00:00+09:00")
    for _ in range(3):
        assert not h.tick()["errors"]
    assert len(h.actions.posts) == 2
    first = h.cycle()
    assert first["cutoff_sha"] == h.actions.main_sha()
    assert h.api.snapshot()["normal_merge_frozen"]
    h.clock.set("2026-10-26T06:00:00+09:00")
    h.tick()
    assert len(h.actions.posts) == 4
    assert h.cycle()["target_sha"] == first["target_sha"]
    assert (
        h.cycle()["generations"][0]["requests"]["ci.yml"]["request_id"]
        != first["generations"][0]["requests"]["ci.yml"]["request_id"]
    )
    assert h.cycle()["carryover"]["cycle_id"] == first["cycle_id"]


@pytest.mark.parametrize("accepted", [True, False])
def test_lost_dispatch_response_is_never_resent(h: Harness, accepted: bool) -> None:
    h.actions.accept = accepted
    h.actions.lost_response = "ci.yml"
    assert h.tick()["errors"]
    h.actions.lost_response = None
    for _ in range(3):
        h.tick()
    assert len(h.actions.posts) == 2
    request = h.cycle()["generations"][0]["requests"]["ci.yml"]
    assert request["state"] == ("bound" if accepted else "dispatch_unknown")
    assert h.cycle()["status"] == "active"


def test_delayed_run_after_lost_response_binds_without_dispatch(h: Harness) -> None:
    h.actions.visible = False
    h.actions.lost_response = "ci.yml"
    h.tick()
    assert h.cycle()["generations"][0]["requests"]["ci.yml"]["run_id"] is None
    h.actions.visible = True
    h.actions.lost_response = None
    h.tick()
    assert h.cycle()["generations"][0]["requests"]["ci.yml"]["run_id"] == 1
    assert len(h.actions.posts) == 2


def test_api_registration_and_new_sha_new_run_after_repair(h: Harness) -> None:
    h.tick()
    old = h.cycle()
    h.api.register_repair(h.cycle_id, 10, "a" * 40, ["failure-1"])
    assert (
        h.api.merge_reason(11, "a" * 40, h.cycle_id) == "daily_ci_repair_not_registered"
    )
    assert (
        h.api.merge_reason(10, "b" * 40, h.cycle_id) == "daily_ci_repair_not_registered"
    )
    assert h.api.merge_reason(10, "a" * 40, None) == "daily_ci_normal_merge_frozen"
    assert h.api.merge_reason(10, "a" * 40, h.cycle_id) is None
    with h.api.store.repository_lock():
        h.api.begin_merge(h.cycle_id, 10, "a" * 40)
        merged = h.actions.commit("reviewed repair")
        h.api.finish_merge(h.cycle_id, 10, "a" * 40, merged)
    h.tick()
    assert h.cycle()["cutoff_sha"] == old["target_sha"]
    assert h.cycle()["target_sha"] == merged
    assert len(h.actions.posts) == 4
    assert h.actions.posts[-1][1]["generation"] == "2"
    assert h.actions.posts[-1][1]["target_sha"] == merged
    assert h.cycle()["generations"][-1]["kind"] == "repair"
    with pytest.raises(ValueError, match="already attempted"):
        h.api.register_repair(h.cycle_id, 10, "b" * 40, ["failure-1"])


def test_remote_success_needs_collector_evidence_then_releases_early(
    h: Harness,
) -> None:
    h.tick()
    h.api.register_worker(h.cycle_id, "worker-1", {"job_id": "dispatcher-1"})
    h.actions.complete()
    h.tick()
    assert h.cycle()["status"] == "active"
    h.hooks.results = True
    assert not h.tick()["errors"]
    assert h.cycle()["status"] == "success"
    assert not h.api.snapshot()["normal_merge_frozen"]
    assert h.cycle()["workers"]["worker-1"]["state"] == "stopped"
    assert h.cycle()["stop_complete"]
    assert h.api.snapshot()["last_success"]["target_sha"] == h.cycle()["target_sha"]
    assert h.api.merge_reason(12, "a" * 40, None) is None
    with pytest.raises(ValueError, match="does not permit"):
        h.api.permit_worker(h.cycle_id, "worker-1")
    h.tick()
    assert len(h.actions.posts) == 2


@pytest.mark.parametrize(
    "conclusion", ["failure", "cancelled", "skipped", "neutral", "timed_out"]
)
def test_non_success_is_not_recovery(h: Harness, conclusion: str) -> None:
    h.tick()
    h.actions.complete(conclusion)
    h.hooks.results = True
    h.tick()
    assert h.cycle()["status"] == "active"
    assert h.api.snapshot()["last_success"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("attempt", 2),
        ("run_id", 999),
        ("generation", 2),
        ("target_sha", "f" * 40),
        ("checkout_sha", "f" * 40),
        ("baseline_sha", "f" * 40),
        ("request_id", "other"),
        ("evidence", {}),
    ],
)
def test_result_identity_mismatch_is_rejected(
    h: Harness, field: str, value: Any
) -> None:
    h.tick()
    h.actions.complete()
    h.tick()
    result = result_for(h.cycle())
    result[field] = value
    with pytest.raises(ValueError):
        h.api.record_result(result)
    assert h.cycle()["generations"][0]["requests"]["ci.yml"]["observation"] is None


def test_attempt_change_invalidates_previous_evidence(h: Harness) -> None:
    h.tick()
    h.actions.complete()
    h.tick()
    prior = result_for(h.cycle())
    h.api.record_result(prior)
    h.actions.rows[1].update(run_attempt=2, status="in_progress", conclusion=None)
    h.tick()
    request = h.cycle()["generations"][0]["requests"]["ci.yml"]
    assert request["attempt"] == 2 and request["observation"] is None
    with pytest.raises(ValueError, match="identity"):
        h.api.record_result(prior)


def test_external_main_update_invalidates_old_green(h: Harness) -> None:
    h.tick()
    target = h.cycle()["target_sha"]
    h.actions.complete()
    h.hooks.results = True
    external = h.actions.commit("external main update")
    h.tick()
    assert h.cycle()["status"] == "blocked"
    assert h.cycle()["target_sha"] == target
    assert h.cycle()["observed_main_sha"] == external
    assert h.api.snapshot()["last_success"] is None
    assert len(h.actions.posts) == 2
    assert h.api.merge_reason(10, "a" * 40, None) == "daily_ci_normal_merge_frozen"


@pytest.mark.parametrize("stopped", [True, False])
def test_watchdog_expires_without_controller_or_merge_mutex(
    h: Harness, stopped: bool
) -> None:
    h.tick()
    h.api.register_worker(h.cycle_id, "worker", {"job_id": "existing-supervisor"})
    h.api.unresolved(h.cycle_id, [{"signature": "failed-test", "evidence": {"run": 1}}])
    h.hooks.stop_state = "stopped" if stopped else "stop_unknown"
    h.actions.cancel_effect = stopped
    h.clock.set("2026-10-25T08:59:59+09:00")
    assert h.api.permit_worker(h.cycle_id, "worker")["target_sha"]
    h.clock.set("2026-10-25T09:00:00+09:00")
    with h.api.store.repository_lock(), h.api.store.lock("controller"):
        h.controller.watchdog()
    cycle = h.cycle()
    assert cycle["status"] == "expired"
    assert cycle["stop_complete"] is stopped
    assert cycle["unresolved"][0]["signature"] == "failed-test"
    assert h.actions.cancellations == [1, 2]
    assert h.api.merge_reason(10, "a" * 40, None) is None
    assert (
        h.api.merge_reason(10, "a" * 40, h.cycle_id)
        == "daily_ci_cycle_expired_or_inactive"
    )
    with pytest.raises(ValueError, match="does not permit"):
        h.api.permit_worker(h.cycle_id, "worker")
    with pytest.raises(ValueError, match="does not permit"):
        h.api.register_repair(h.cycle_id, 10, "a" * 40, ["failed-test"])


def test_stop_failure_preserves_unknown_but_does_not_freeze_daytime(h: Harness) -> None:
    h.tick()
    h.api.register_worker(h.cycle_id, "worker", {"job_id": "existing-supervisor"})
    h.hooks.fail = ("workers", "stop")
    h.clock.set("2026-10-25T09:00:00+09:00")
    assert h.controller.watchdog()
    assert h.cycle()["workers"]["worker"]["state"] == "stop_unknown"
    assert not h.cycle()["stop_complete"]
    assert not h.api.snapshot()["normal_merge_frozen"]
    h.hooks.fail = None
    h.controller.watchdog()
    assert h.cycle()["stop_complete"]


def test_late_unknown_dispatch_is_cancelled_after_deadline(h: Harness) -> None:
    h.actions.visible = False
    h.actions.lost_response = "ci.yml"
    h.tick()
    h.clock.set("2026-10-25T09:00:00+09:00")
    h.controller.watchdog()
    assert not h.cycle()["stop_complete"]
    assert (
        h.cycle()["generations"][0]["requests"]["ci.yml"]["stop_state"]
        == "stop_unknown"
    )
    h.actions.visible = True
    h.controller.watchdog()
    assert 1 in h.actions.cancellations
    assert h.cycle()["stop_complete"]
    assert len(h.actions.posts) == 2


@pytest.mark.parametrize(
    "now", ["2026-10-25T09:00:00+09:00", "2026-10-27T15:00:00+09:00"]
)
def test_daytime_return_records_only_current_missed_cycle(h: Harness, now: str) -> None:
    h.clock.set(now)
    h.tick()
    assert not h.actions.posts
    assert h.cycle()["status"] == "missed"
    assert len(h.api.snapshot()["cycles"]) == 1


def test_window_closes_between_main_get_and_dispatch(h: Harness) -> None:
    main_reads = 0

    def boundary() -> None:
        nonlocal main_reads
        main_reads += 1
        if main_reads == 2:
            h.clock.set("2026-10-25T09:00:00+09:00")

    h.actions.on_main = boundary
    h.tick()
    assert not h.actions.posts
    assert h.cycle()["status"] == "expired"


def test_repo_mutex_serializes_preexisting_merge_before_main_pin(h: Harness) -> None:
    with h.api.store.repository_lock():
        with pytest.raises(ValueError, match="busy"):
            h.tick()
        merged_sha = h.actions.commit("ordinary merge already in flight")
    h.tick()
    assert h.cycle()["cutoff_sha"] == merged_sha


def test_duplicate_dispatch_runs_are_blocked_and_both_cancelled(h: Harness) -> None:
    h.tick()
    h.actions.rows[99] = {**h.actions.rows[1], "id": 99}
    assert h.tick()["errors"]
    assert h.cycle()["status"] == "active"
    h.clock.set("2026-10-25T09:00:00+09:00")
    h.controller.watchdog()
    assert {1, 2, 99} <= set(h.actions.cancellations)
    assert not h.cycle()["stop_complete"]


def test_missing_and_corrupt_state_never_fall_back_to_empty(h: Harness) -> None:
    store = h.api.store
    original = store.path.read_text()
    store.path.unlink()
    with pytest.raises(ValueError, match="state is missing"):
        h.api.merge_reason(10, "a" * 40, None)
    with pytest.raises(ValueError, match="state is missing"):
        h.api.initialize("a" * 40)
    store.path.write_text(
        original.replace('"schema_version": 1', '"schema_version": 99')
    )
    with pytest.raises(ValueError, match="malformed"):
        h.tick()
    store.path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate"):
        h.api.snapshot()


def test_unconnected_integrations_cannot_enable_or_report_success(
    tmp_path: Path,
) -> None:
    api = CycleAPI(CycleStore("o/r", tmp_path))
    api.initialize("a" * 40)
    controller = Controller(api, GitHub("o/r"))
    with pytest.raises(ValueError, match="observer integration is not configured"):
        controller.enable()
    assert api.snapshot()["enabled"] is False


@pytest.mark.parametrize("argument", ["--now", "--date", "--clock"])
def test_production_cli_has_no_clock_override(tmp_path: Path, argument: str) -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "daily_ci",
            "--repo",
            "o/r",
            argument,
            "2026-10-25T06:00:00+09:00",
            "tick",
        ],
        env={**os.environ, "PYTHONPATH": str(OPS), "XDG_STATE_HOME": str(tmp_path)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 2 and "usage:" in process.stderr
    assert not list(tmp_path.iterdir())


def test_cli_initializes_disabled_with_explicit_baseline(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(OPS), "XDG_STATE_HOME": str(tmp_path)}
    for args in (["init", "--baseline-sha", "a" * 40], ["status"]):
        process = subprocess.run(
            [sys.executable, "-m", "daily_ci", "--repo", "o/r", *args],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert process.returncode == 0, process.stderr
        state = json.loads(process.stdout)
        assert state["enabled"] is False and state["baseline_sha"] == "a" * 40
    assert not list(tmp_path.rglob("hook-*"))


def test_production_rest_and_adapter_commands_use_shared_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Path(__file__).with_name("fake_daily_ci.py")
    binary = tmp_path / "bin"
    binary.mkdir()
    gh = binary / "gh"
    gh.write_text(f"#!{sys.executable}\n" + fake.read_text())
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("DAILY_CI_FAKE_ROOT", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(OPS))
    (tmp_path / "remote.json").write_text(json.dumps({"main": "a" * 40, "runs": []}))
    clock = Clock()
    api = CycleAPI(CycleStore("o/r", tmp_path / "state"), clock=clock)
    api.initialize(
        "a" * 40,
        integrations={
            role: [sys.executable, str(fake), "adapter", role]
            for role in ("observer", "workers")
        },
    )
    controller = Controller(api, GitHub("o/r"))
    controller.enable()
    for _ in range(2):
        assert not controller.tick()["errors"]
    clock.set("2026-10-25T09:00:00+09:00")
    assert not controller.watchdog()
    calls = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    posts = [c for c in calls if c["method"] == "POST"]
    assert len([c for c in posts if c["endpoint"].endswith("/dispatches")]) == 2
    assert len([c for c in posts if c["endpoint"].endswith("/cancel")]) == 2
    assert api.snapshot()["cycles"][window(clock())[0]]["stop_complete"] is True
    assert not list(api.store.directory.glob("hook-*"))


def test_failed_days_keep_baseline_until_verified_repair(h: Harness) -> None:
    baseline = h.actions.main_sha()
    h.tick()
    h.api.register_repair(h.cycle_id, 10, "a" * 40, ["failed-test"])
    with h.api.store.repository_lock():
        h.api.begin_merge(h.cycle_id, 10, "a" * 40)
        repaired = h.actions.commit("fixed main")
        h.api.finish_merge(h.cycle_id, 10, "a" * 40, repaired)
    h.tick()
    assert h.cycle()["baseline_sha"] == baseline
    h.clock.set("2026-10-26T06:00:00+09:00")
    h.tick()
    assert h.cycle()["baseline_sha"] == baseline
    assert "10" in h.cycle()["carryover"]["repair_prs"]
    assert h.cycle()["repair_prs"] == {}
    h.actions.complete()
    h.hooks.results = True
    h.tick()
    assert h.cycle()["status"] == "success"
    h.clock.set("2026-10-27T06:00:00+09:00")
    h.tick()
    assert h.cycle()["baseline_sha"] == repaired
