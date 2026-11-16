"""Automatic merge contracts through the persisted controller, with fake external I/O.

GitHub, reviewer/repair agent and clock are injected. State writes and restarts
are real; no live GitHub, agent, GPU or timed waits are used.
"""

from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path
from typing import Any, TypeVar, cast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops"))
module: Any = importlib.import_module("review.progression")
Engine, RateWait, new_state, read, save = (
    module.Engine,
    module.RateWait,
    module.new_state,
    module.read,
    module.save,
)
T = TypeVar("T")


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert 0 < seconds <= 60
        self.sleeps.append(seconds)
        self.now += seconds


class Backend:
    def __init__(self, numbers: tuple[int, ...] = (11,)) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.heads = {n: str(n % 10) * 40 for n in numbers}
        self.checks: dict[int, list[str]] = {n: ["success"] for n in numbers}
        self.gates: dict[int, list[str | None]] = {n: [None] for n in numbers}
        self.stops: set[int] = set()
        self.merged: set[int] = set()
        self.reports: list[str] = ["repaired"]
        self.reviews: list[str] = ["approve"]
        self.dependencies: list[bool] = [True]
        self.main_ci = "success"
        self.closed = True
        self.local_error: str | None = None
        self.rate_once = False
        self.lose_merge = False
        self.merge_took_effect = True
        self.lose_repair = False
        self.unrelated_issues = 0
        self.review_count = 0
        self.repair_count: dict[int, int] = dict.fromkeys(numbers, 0)

    @staticmethod
    def next(values: list[T]) -> T:
        return values.pop(0) if len(values) > 1 else values[0]

    def local(self, state: dict[str, Any]) -> str | None:
        self.calls.append(("local",))
        return self.local_error

    def review(self, state: dict[str, Any], directory: Path) -> dict[str, Any]:
        self.calls.append(("review",))
        self.review_count += 1
        verdict = self.next(self.reviews)
        return {
            "verdict": verdict,
            "major": []
            if verdict == "approve"
            else [{"prs": [11], "description": "incorrect result"}],
        }

    def pr(self, number: int) -> dict[str, Any]:
        self.calls.append(("pr", number))
        return {
            "number": number,
            "head": {"sha": self.heads[number]},
            "merged": number in self.merged,
            "merge_commit_sha": "a" * 40 if number in self.merged else None,
        }

    def ci(self, number: int, head: str, *, main: bool = False) -> dict[str, Any]:
        self.calls.append(("main_ci" if main else "ci", number, head))
        if self.rate_once:
            self.rate_once = False
            raise RateWait(1180.0, "primary")
        return {
            "status": self.main_ci if main else self.next(self.checks[number]),
            "head": head,
            "checks": [],
        }

    def shape(self, target: dict[str, Any], pr: dict[str, Any]) -> str | None:
        self.calls.append(("shape", target["number"]))
        return self.next(self.gates[target["number"]])

    def human(self, state: dict[str, Any], target: dict[str, Any]) -> str | None:
        self.calls.append(("human", target["number"]))
        return "needs-human" if target["number"] in self.stops else None

    def dependency(self, target: dict[str, Any]) -> bool:
        self.calls.append(("dependencies", target["number"]))
        return self.next(self.dependencies)

    def failure(self, target: dict[str, Any], ci: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("failure_logs", target["number"]))
        return {"failed_jobs": [{"name": "tests", "log": "assertion failed"}]}

    def repair(
        self, state: dict[str, Any], target: dict[str, Any], attempt: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append(
            ("repair", target["number"], attempt["id"], attempt["details"])
        )
        self.repair_count[target["number"]] += 1
        if self.lose_repair:
            raise OSError("worker launch response lost")
        self.heads[target["number"]] = format(
            self.repair_count[target["number"]] + 10, "040x"
        )
        return {
            "outcome": self.next(self.reports),
            "head_sha": self.heads[target["number"]],
            "parent_request": "inspect requirements",
        }

    def recover_repair(
        self, state: dict[str, Any], target: dict[str, Any], attempt: dict[str, Any]
    ) -> dict[str, Any] | None:
        self.calls.append(("recover_repair", target["number"]))
        return None

    def merge(self, target: dict[str, Any], head: str) -> None:
        self.calls.append(("merge", target["number"], head))
        if self.merge_took_effect:
            self.merged.add(target["number"])
        if self.lose_merge:
            raise OSError("merge response lost")

    def issues_completed(self, target: dict[str, Any]) -> bool:
        self.calls.append(("issues_completed", target["number"]))
        return self.closed

    def changes(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            str(n): {"files": 2, "additions": 5, "deletions": 1} for n in self.heads
        }

    def notification_changes(
        self, state: dict[str, Any], pr: int | None
    ) -> dict[str, Any]:
        return self.changes(state)

    def notify(self, state: dict[str, Any], event: dict[str, Any]) -> None:
        self.calls.append(("notify", event["kind"]))


def setup(
    tmp_path: Path, numbers: tuple[int, ...] = (11,)
) -> tuple[Path, Backend, Clock]:
    path = tmp_path / "progression.json"
    request = {
        "schema_version": 4,
        "repo": "o/r",
        "parent_issue": 1,
        "review_run_id": "parent-1-" + "1" * 32,
        "prs": [
            {"number": n, "head_sha": str(n % 10) * 40, "issues": [n + 100]}
            for n in numbers
        ],
    }
    save(path, new_state(request, owner="codex:test", generation=1))
    return path, Backend(numbers), Clock()


def run(path: Path, backend: Backend, clock: Clock) -> dict[str, Any]:
    return cast(
        dict[str, Any], Engine(path, backend, clock=clock, sleep=clock.sleep).run()
    )


def test_request_runs_to_post_merge_notification_without_parent(tmp_path: Path) -> None:
    path, backend, clock = setup(tmp_path)
    result = run(path, backend, clock)
    assert result["status"] == "completed"
    assert backend.review_count == 1
    assert backend.repair_count[11] == 0
    assert ("main_ci", 11, "a" * 40) in backend.calls
    assert ("issues_completed", 11) in backend.calls
    assert backend.calls[-1] == ("notify", "completed")
    assert read(path)["events"][-1]["delivered"] is True


def test_ci_and_conflict_repairs_change_head_without_rereview_or_resolve(
    tmp_path: Path,
) -> None:
    path, backend, clock = setup(tmp_path)
    backend.checks[11] = ["failure", "success"]
    backend.gates[11] = ["conflict", None]
    assert run(path, backend, clock)["status"] == "completed"
    assert backend.review_count == 1
    assert backend.repair_count[11] == 2
    repairs = [c for c in backend.calls if c[0] == "repair"]
    assert repairs[0][3]["failed_jobs"][0]["log"] == "assertion failed"
    assert repairs[1][3]["reason"] == "conflict"
    assert [c for c in backend.calls if c[0] == "merge"] == [
        ("merge", 11, backend.heads[11])
    ]


def test_three_repairs_per_pr_other_pr_continues(tmp_path: Path) -> None:
    path, backend, clock = setup(tmp_path, (11, 12))
    backend.checks[11] = ["failure", "success", "failure", "failure"]
    backend.gates[11] = ["draft"]
    result = run(path, backend, clock)
    assert result["targets"]["11"]["reason"] == "repair_limit"
    assert backend.repair_count == {11: 3, 12: 0}
    assert backend.merged == {12}


@pytest.mark.parametrize(
    "reason", ["needs-human", "auto_merge_disabled", "owner_changed"]
)
def test_human_controls_do_not_launch_worker(tmp_path: Path, reason: str) -> None:
    path, backend, clock = setup(tmp_path)
    if reason == "needs-human":
        backend.checks[11] = ["failure"]
        backend.stops.add(11)
    else:
        backend.local_error = reason
    assert run(path, backend, clock)["status"] == "parent_wait"
    assert backend.repair_count[11] == 0
    assert not any(c[0] in {"dependencies", "merge"} for c in backend.calls)


def test_gate_short_circuits_and_dependency_wait_does_not_start_agent(
    tmp_path: Path,
) -> None:
    path, backend, clock = setup(tmp_path)
    backend.checks[11] = ["pending", "success"]
    backend.dependencies = [False, True]
    assert run(path, backend, clock)["status"] == "completed"
    first_ci = next(i for i, c in enumerate(backend.calls) if c[0] == "ci")
    assert backend.calls[first_ci + 1][0] == "local"
    ordered = [
        c[0] for c in backend.calls[first_ci + 1 :] if c[0] not in {"pr", "notify"}
    ]
    assert ordered[:5] == ["local", "ci", "shape", "human", "dependencies"]
    assert backend.repair_count[11] == 0
    assert clock.sleeps == [60, 60]


def test_worker_can_return_parent_request(tmp_path: Path) -> None:
    path, backend, clock = setup(tmp_path)
    backend.gates[11] = ["draft"]
    backend.reports = ["needs_parent"]
    result = run(path, backend, clock)
    assert result["targets"]["11"]["reason"] == "inspect requirements"
    assert not backend.merged


@pytest.mark.parametrize("rereview", [False, True])
def test_major_requires_explicit_continue_rereview_is_opt_in(
    tmp_path: Path, rereview: bool
) -> None:
    path, backend, clock = setup(tmp_path)
    backend.reviews = ["major", "approve"]
    assert run(path, backend, clock)["status"] == "parent_wait"
    assert not backend.merged
    assert run(path, backend, clock)["status"] == "parent_wait"
    engine = Engine(path, backend, clock=clock, sleep=clock.sleep)
    engine.continue_run("major repaired", rereview=rereview)
    assert engine.run()["status"] == "completed"
    assert backend.review_count == (2 if rereview else 1)
    assert read(path)["decisions"][-1]["changes"]["11"]["files"] == 2


@pytest.mark.parametrize("failure", ["issue", "main"])
def test_post_merge_failure_notifies_parent(tmp_path: Path, failure: str) -> None:
    path, backend, clock = setup(tmp_path)
    if failure == "issue":
        backend.closed = False
    else:
        backend.main_ci = "failure"
    result = run(path, backend, clock)
    assert result["status"] == "parent_wait"
    assert backend.merged == {11}
    assert backend.repair_count[11] == 0


def test_conditional_rate_limit_wait_and_ci_timeout(tmp_path: Path) -> None:
    path, backend, clock = setup(tmp_path)
    backend.rate_once = True
    assert run(path, backend, clock)["status"] == "completed"
    assert sum(clock.sleeps) == 180
    path, backend, clock = setup(tmp_path)
    backend.checks[11] = ["pending"]
    result = run(path, backend, clock)
    assert result["targets"]["11"]["reason"] == "ci_timeout"
    assert sum(clock.sleeps) == 6 * 3600
    assert backend.review_count == 1 and backend.repair_count[11] == 0


@pytest.mark.parametrize("effect", [False, True])
def test_lost_merge_response_never_resends(tmp_path: Path, effect: bool) -> None:
    path, backend, clock = setup(tmp_path)
    backend.lose_merge = True
    backend.merge_took_effect = effect
    result = run(path, backend, clock)
    assert result["status"] == ("completed" if effect else "parent_wait")
    if not effect:
        assert result["targets"]["11"]["reason"] == "recovery_required"
    run(path, backend, clock)
    assert len([c for c in backend.calls if c[0] == "merge"]) == 1


def test_interruption_at_worker_intent_requires_recovery_not_relaunch(
    tmp_path: Path,
) -> None:
    path, backend, clock = setup(tmp_path)
    backend.checks[11] = ["failure"]
    backend.lose_repair = True
    result = run(path, backend, clock)
    assert result["targets"]["11"]["reason"] == "recovery_required"
    run(path, backend, clock)
    assert backend.repair_count[11] == 1


def test_restart_saved_ci_stage_without_reviewer_and_no_registry_lock(
    tmp_path: Path,
) -> None:
    path, backend, clock = setup(tmp_path)
    backend.checks[11] = ["pending", "success"]

    def interrupted_sleep(seconds: float) -> None:
        assert read(path)["targets"]["11"]["stage"] == "ci_wait"
        assert not list(tmp_path.glob("**/registry.lock"))
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        Engine(path, backend, clock=clock, sleep=interrupted_sleep).run()
    clock.now += 60
    assert run(path, backend, clock)["status"] == "completed"
    assert backend.review_count == 1


def test_get_scope_ignores_unrelated_issues(tmp_path: Path) -> None:
    traces = []
    for count in (0, 2000):
        path, backend, clock = setup(tmp_path, (11, 12, 13))
        backend.unrelated_issues = count
        run(path, backend, clock)
        traces.append(
            copy.deepcopy(
                [
                    c
                    for c in backend.calls
                    if c[0]
                    in {
                        "pr",
                        "ci",
                        "main_ci",
                        "human",
                        "dependencies",
                        "issues_completed",
                    }
                ]
            )
        )
    assert traces[0] == traces[1]
    assert json.loads(path.read_text())["status"] == "completed"


@pytest.mark.parametrize(
    "stage", ["requested", "merge_pending", "repair_pending", "post_merge_wait"]
)
def test_human_control_pauses_saved_stages_before_further_effects(
    tmp_path: Path, stage: str
) -> None:
    path, backend, clock = setup(tmp_path)
    saved = read(path)
    if stage != "requested":
        saved["review"].update(
            stage="accepted", result={"verdict": "approve", "major": []}
        )
        saved["targets"]["11"]["stage"] = stage
    save(path, saved)
    backend.stops.add(11)
    result = run(path, backend, clock)
    assert result["status"] == "parent_wait"
    assert backend.review_count == 0 and backend.repair_count[11] == 0
    assert not any(
        c[0] in {"merge", "recover_repair", "main_ci", "issues_completed"}
        for c in backend.calls
    )


def test_dependency_wait_preserves_independent_progress_and_merge_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, backend, clock = setup(tmp_path, (11, 12, 13))

    def dependencies(target: dict[str, Any]) -> bool:
        backend.calls.append(("dependencies", target["number"]))
        return target["number"] != 11 or 12 in backend.merged

    monkeypatch.setattr(backend, "dependency", dependencies)
    assert run(path, backend, clock)["status"] == "completed"
    assert [c[1] for c in backend.calls if c[0] == "merge"] == [12, 13, 11]
    assert backend.review_count == 1 and not any(backend.repair_count.values())
    assert clock.sleeps == [60]
