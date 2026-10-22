"""Schedule, reservation crashes and managed capacity without real agents/state."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest

from tests.e2e.agents_ops.test_dispatcher_judgement import Harness, decision, issue

ROOT = Path(__file__).resolve().parents[3]


class DispatchHarness:
    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.h = Harness(tmp, monkeypatch)
        self.mod = importlib.import_module("dispatcher.supervisor")
        self.service = importlib.import_module("dispatcher.service")
        self.clock = importlib.import_module("dispatcher.schedule")
        self.schedule = self.clock.Schedule(tmp / "dispatcher")
        with self.h.store.locked():
            self.schedule.initialize(
                {
                    "repo": "o/r",
                    "cwd": str(self.h.work),
                    "runner": str(self.h.mod.RUNNER),
                    "max_parents": 1,
                    "max_children": 2,
                    "judgement_timeout": 10,
                },
                self.now("2026-10-22T07:00:00+09:00"),
            )
            state = self.schedule.read()
            state["enabled"] = True
            self.schedule.write(state)
        self.control = self.mod.Supervisor(self.h.gh, self.h.store, self.schedule)

    def now(self, value: str) -> Any:
        return self.clock.local_time(value)

    def run(
        self, value: str = "2026-10-22T08:00:00+09:00", *, explicit: bool = False
    ) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self.service.run_due(
                self.h.gh,
                self.h.store,
                self.schedule,
                self.now(value),
                explicit=explicit,
            ),
        )

    def state(self) -> dict[str, Any]:
        with self.h.store.locked():
            return copy.deepcopy(self.schedule.read())

    def row(self, n: int = 1) -> dict[str, Any]:
        with self.h.store.locked():
            return copy.deepcopy(self.h.store.parents[f"o/r#{n}"])

    def update(self, row: dict[str, Any], n: int = 1) -> None:
        with self.h.store.locked():
            self.h.store.parents[f"o/r#{n}"] = row
            self.h.store.save()

    def prepare(self, *numbers: int) -> None:
        self.h.gh.issues = {n: issue(n) for n in (numbers or (1,))}
        self.h.response(*(decision(n) for n in self.h.gh.issues))
        result = self.run()
        assert result["result"] == "recorded", result

    def job(self, row: dict[str, Any] | None = None) -> dict[str, Any]:
        dispatch = (row or self.row())["dispatch"]
        return cast(dict[str, Any], dispatch["jobs"][dispatch["active_job"]])

    def runtime(
        self, job: dict[str, Any], *, live: bool = False, state: str = "waiting"
    ) -> None:
        directory = Path(job["directory"]) / "runtime"
        directory.mkdir(exist_ok=True)
        record = {
            "schema_version": 1,
            "run_id": job["id"],
            "result": "running" if live else "completed",
            "turn_end_reason": None if live else "completed",
            "exit_code": None if live else 0,
            "cleanup_complete": not live,
            "_alive": live,
            "runtime": {"session_id": "fixture-session"},
            "contract": {
                "cwd": job["cwd"],
                "task_id": job["task_id"],
                "mode": "write",
                "settings": {
                    "role": job["role"],
                    "agent": "claude" if job["role"] == "parent" else "codex",
                    "profile": None,
                },
            },
        }
        (directory / "result.json").write_text(json.dumps(record))
        (directory / "final.md").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "state": state,
                    "reason": "CI待ち",
                    "resume_condition": "CI完了",
                }
            )
        )

    def accepted(self) -> dict[str, Any]:
        row = self.row()
        job = self.job(row)
        job["phase"] = "starting"
        self.update(row)
        self.runtime(job, live=True)
        env = {
            "CLAUDE_CODE_SESSION_ID": "fixture-session",
            "AGENT_RUNTIME_RUN_DIR": str(Path(job["directory"]) / "runtime"),
            "AGENT_RUNTIME_TASK_ID": job["task_id"],
        }
        self.control.accept(
            json.loads((Path(job["directory"]) / "input.json").read_text()), env
        )
        return env


@pytest.fixture
def d(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DispatchHarness:
    return DispatchHarness(tmp_path, monkeypatch)


class FixtureRuntime:
    def __init__(self, runner: Path) -> None:
        pass

    def inspect(self, directory: Path, *, stop: bool = False) -> dict[str, Any]:
        record = json.loads((directory / "result.json").read_text())
        if stop:
            record.update(
                _alive=False,
                result="failed",
                cleanup_complete=True,
                turn_end_reason="stopped",
            )
            (directory / "result.json").write_text(json.dumps(record))
        return {
            "record": record,
            "summary": {
                "runner_alive": record["_alive"],
                "agent_alive": record["_alive"],
                "agent_group_alive": record["_alive"],
            },
        }

    def ended(self, evidence: dict[str, Any]) -> bool:
        return (
            not evidence["record"]["_alive"] and evidence["record"]["cleanup_complete"]
        )


def fixture_runtime(d: DispatchHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(d.mod, "Runtime", FixtureRuntime)


def test_slots_always_in_tokyo_and_catch_up_only_latest(d: DispatchHarness) -> None:
    cases = {
        "2026-10-21T22:59:59+00:00": "2026-10-21T20:00:00+09:00",
        "2026-10-21T23:00:00+00:00": "2026-10-22T08:00:00+09:00",
        "2026-10-22T11:00:00+00:00": "2026-10-22T20:00:00+09:00",
    }
    for now, expected in cases.items():
        assert d.clock.latest_slot(d.now(now)).isoformat() == expected
    d.h.response(decision(1))
    assert d.run("2026-10-25T21:00:00+09:00")["result"] == "recorded"
    assert len(d.h.calls.read_text().splitlines()) == 1
    assert d.state()["cursor"] == "2026-10-25T20:00:00+09:00"
    assert d.run("2026-10-25T21:01:00+09:00")["result"] == "not_due"


def test_unchanged_slots_run_once_and_updates_never_trigger_extra_turns(
    d: DispatchHarness,
) -> None:
    d.prepare()
    for i in range(3):
        d.h.gh.issues[1]["body"] += f"\nupdate {i}"
        assert d.run()["result"] == "not_due"
    assert len(d.h.calls.read_text().splitlines()) == 1
    assert d.run("2026-10-22T20:00:00+09:00")["result"] == "recorded"
    assert len(d.h.calls.read_text().splitlines()) == 2
    assert d.run("2026-10-23T08:00:00+09:00")["result"] == "recorded"
    assert len(d.h.calls.read_text().splitlines()) == 3
    assert d.run("2026-10-23T08:01:00+09:00", explicit=True)["result"] == "recorded"
    assert len(d.h.calls.read_text().splitlines()) == 4


def test_duplicate_notification_and_previous_lifetime_lock_block_judgment(
    d: DispatchHarness,
) -> None:
    gate = importlib.import_module("dispatcher.runtime").gate
    with gate(d.schedule.directory / "judgement.lock"):
        assert d.run()["result"] == "busy"
    assert not d.h.calls.exists()
    d.prepare()
    assert d.run()["result"] == "not_due"


@pytest.mark.parametrize("outcome", ["recorded", "deferred", "preview", "failed"])
def test_v2_results_and_recovered_results_reserve_only_recorded_candidates(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    d.h.response(decision(1))
    if outcome == "preview":
        original = d.h.mod.run

        def preview(*args: Any, **kwargs: Any) -> Any:
            return original(*args, **{**kwargs, "apply": False})

        monkeypatch.setattr(d.h.mod, "run", preview)
    elif outcome == "failed":
        monkeypatch.setenv("FAKE_JUDGEMENT_SCENARIO", "cli_error")
    elif outcome == "deferred":

        def edit_after_judgement(endpoint: str) -> None:
            if d.h.calls.exists():
                d.h.gh.on_get = None
                d.h.gh.issues[1]["body"] += "\n要件変更"

        d.h.gh.on_get = edit_after_judgement
    result = d.run()
    assert result["result"] == outcome, result
    run = d.state()["runs"][0]
    saved = json.loads((Path(run["directory"]) / "result.json").read_text())
    assert saved["schema_version"] == 2
    assert result["deferred"] == saved["deferred"]
    assert (
        d.control.status(d.now("2026-10-22T09:00:00+09:00"))["judgements"][0][
            "deferred"
        ]
        == saved["deferred"]
    )
    if outcome != "recorded":
        assert d.control.assign() == []
        assert d.h.store.parents == {}
    # A lost controller response must recognize every terminal v2 outcome
    # without another judgement turn or turning a deferral into a reservation.
    with d.h.store.locked():
        state = d.schedule.read()
        state["active"] = run["id"]
        state["runs"][0].update(phase="running", controller=None)
        d.schedule.write(state)
    assert d.run()["result"] == "not_due"
    assert d.state()["active"] is None
    assert d.state()["runs"][0]["phase"] == outcome
    assert d.control.assign() == ([1] if outcome == "recorded" else [])
    with d.h.store.locked():
        assert list(d.h.store.parents) == (["o/r#1"] if outcome == "recorded" else [])
    assert len(d.h.calls.read_text().splitlines()) == 1


def test_scheduled_judgement_uses_split_read_directory_and_output_schema(
    d: DispatchHarness,
) -> None:
    d.prepare()
    directory = Path(d.state()["runs"][0]["directory"])
    contract = json.loads((directory / "agent-runtime/result.json").read_text())[
        "contract"
    ]
    assert contract["read_dirs"] == [str(directory / "agent-input")]
    assert (
        contract["json_schema_sha256"]
        == hashlib.sha256(
            (ROOT / ".agents/ops/dispatcher/judgement.schema.json").read_bytes()
        ).hexdigest()
    )
    args = json.loads(d.h.calls.read_text())["args"]
    assert args[args.index("--add-dir") + 1] == str(directory / "agent-input")
    assert json.loads(args[args.index("--json-schema") + 1])["$schema"] == (
        "http://json-schema.org/draft-07/schema#"
    )


@pytest.mark.parametrize("stage", ["preflight", "publication", "final_check"])
def test_mixed_deferral_reserves_fresh_subject_and_defers_changed_subject_to_next_slot(
    d: DispatchHarness, stage: str
) -> None:
    d.h.gh.issues[2] = issue(2)
    d.h.response(decision(1), decision(2))

    def edit_after_judgement(endpoint: str) -> None:
        if d.h.calls.exists():
            d.h.gh.on_get = None
            d.h.gh.issues[1]["body"] += "\n要件変更"

    def edit_during_publication() -> None:
        if stage == "publication" or len(d.h.gh.writes) == 2:
            d.h.gh.on_write = None
            d.h.gh.issues[1]["body"] += "\n要件変更"

    if stage == "preflight":
        d.h.gh.on_get = edit_after_judgement
    else:
        d.h.gh.on_write = edit_during_publication
    result = d.run()
    assert result["result"] == "recorded", result
    assert [(x["issue"], x["stage"]) for x in result["deferred"]] == [(1, stage)]
    saved = json.loads(
        (Path(d.state()["runs"][0]["directory"]) / "result.json").read_text()
    )
    assert [c["priority"] for c in saved["candidates"]] == [2]  # Keep the gap.
    assert d.control.assign() == [2]
    assert "o/r#1" not in d.h.store.parents
    assert d.run("2026-10-22T19:59:59+09:00")["result"] == "not_due"
    assert len(d.h.calls.read_text().splitlines()) == 1
    # Free only the runtime slot; the existing owner remains assigned.
    d.control.reconcile(stop=True, parent=2)
    d.h.response(decision(1), decision(2, "existing_owner"))
    next_result = d.run("2026-10-22T20:00:00+09:00")
    assert next_result["result"] == "recorded", next_result
    assert next_result["deferred"] == []
    assert d.control.assign() == [1]
    assert len(d.h.calls.read_text().splitlines()) == 2


@pytest.mark.parametrize("outcome", ["deferred", "preview", "failed"])
def test_nonrecorded_result_cannot_be_promoted_by_journal_or_injected_candidates(
    d: DispatchHarness, outcome: str
) -> None:
    d.prepare()
    path = Path(d.state()["runs"][0]["directory"]) / "result.json"
    saved = json.loads(path.read_text())
    saved["result"] = outcome
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="only recorded"):
        d.control.assign()
    assert d.h.store.parents == {}


@pytest.mark.parametrize("version", [1, True, 999])
def test_unknown_judgement_result_versions_cannot_reserve(
    d: DispatchHarness, version: int | bool
) -> None:
    d.prepare()
    path = Path(d.state()["runs"][0]["directory"]) / "result.json"
    saved = json.loads(path.read_text())
    saved["schema_version"] = version
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="result contract"):
        d.control.assign()
    assert d.h.store.parents == {}


def test_fresh_validate_reserve_order_and_manual_registration_share_lock(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    d.prepare()
    order = []
    original_fresh = d.mod.judgement.assert_candidate_fresh
    original_validate = d.mod.validate_candidate

    def fresh(*args: Any) -> None:
        order.append("fresh")
        # The same flock remains held through both validations and reservation.
        with importlib.import_module("dispatcher.runtime").gate(
            d.h.store.directory / "registry.lock"
        ) as fd:
            assert fd is None
        original_fresh(*args)

    def validate(*args: Any) -> Any:
        order.append("validate")
        assert d.h.store.parents == {}
        return original_validate(*args)

    monkeypatch.setattr(d.mod.judgement, "assert_candidate_fresh", fresh)
    monkeypatch.setattr(d.mod, "validate_candidate", validate)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: d.control.assign(), range(2)))
    assert sum(len(x) for x in results) == 1
    assert order == ["fresh", "validate"]
    with d.h.store.locked(), pytest.raises(ValueError, match="別セッション"):
        session = importlib.import_module("parent.session").ParentSession(
            d.h.gh, d.h.store, 1, "claude:manual"
        )
        session.register()


@pytest.mark.parametrize("change", ["body", "comment", "code", "owner", "basis"])
def test_changed_candidate_never_starts(d: DispatchHarness, change: str) -> None:
    d.prepare()
    if change == "body":
        d.h.gh.issues[1]["body"] += "\nrequirement changed"
    elif change == "comment":
        d.h.gh.comment_rows[1].append(
            {"id": 999, "user": {"login": "human"}, "body": "stop"}
        )
    elif change == "code":
        (d.h.work / "README.md").write_text("different code")
    elif change == "owner":
        d.update(
            {
                "session": "claude:manual",
                "status": "active",
                "generation": 1,
                "tasks": {},
                "reviews": [],
            }
        )
    else:
        marker = next(
            c for c in d.h.gh.comment_rows[1] if '"execution_basis"' in c["body"]
        )
        first, rest = marker["body"].split("\n", 1)
        event = json.loads(first[len("<!-- parent-session:v1 ") : -4])
        event["basis_sha256"] = "f" * 64
        marker["body"] = "<!-- parent-session:v1 " + json.dumps(event) + " -->\n" + rest
    assert d.control.assign() == []
    assert d.state()["runs"][0]["candidates"]["1"]["state"] in {
        "stale",
        "existing_owner",
    }


def test_freshness_failure_skips_authorization_and_all_reservations(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    d.prepare()
    d.h.gh.issues[1]["body"] += "\n要件変更"

    def unexpected_authorization(*args: Any) -> None:
        pytest.fail("validate_candidate must not run after failed freshness")

    monkeypatch.setattr(d.mod, "validate_candidate", unexpected_authorization)
    assert d.control.assign() == []
    assert d.h.store.parents == {}
    assert "次の予定回" in d.state()["runs"][0]["candidates"]["1"]["reason"]
    assert len(d.h.calls.read_text().splitlines()) == 1


def test_current_authorization_is_required_after_external_freshness_check(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    d.prepare()
    checked = []
    original = d.mod.judgement.assert_candidate_fresh

    def fresh(*args: Any) -> None:
        original(*args)
        checked.append("fresh")

    def revoked(*args: Any) -> None:
        checked.append("authorization")
        raise ValueError("execution basis revoked")

    monkeypatch.setattr(d.mod.judgement, "assert_candidate_fresh", fresh)
    monkeypatch.setattr(d.mod, "validate_candidate", revoked)
    assert d.control.assign() == []
    assert checked == ["fresh", "authorization"]
    assert d.h.store.parents == {}
    assert not (d.schedule.directory / "workers").exists()


def test_slot_waiting_and_sequential_candidates_keep_owner(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_runtime(d, monkeypatch)
    d.prepare(1, 2)
    assert d.control.assign() == [1]
    env = d.accepted()
    assert env["CLAUDE_CODE_SESSION_ID"] == "fixture-session"
    assert d.control.assign() == []
    job = d.job()
    d.runtime(job)
    d.control.reconcile()
    assert d.row()["dispatch"]["phase"] == "waiting"
    assert d.row()["status"] == "active"
    assert d.control.assign() == [2]
    assert (
        d.control.status(d.now("2026-10-22T09:00:00+09:00"))["used_parent_slots"] == 1
    )
    d.control.resume(1, "CI完了")
    d.control.reconcile()
    assert d.row()["dispatch"]["phase"] == "resume_pending"
    assert len(d.row()["dispatch"]["jobs"]) == 1


def test_crash_after_reservation_before_schedule_save_reuses_same_claim(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    d.prepare()
    original = d.schedule.write
    monkeypatch.setattr(
        d.schedule, "write", lambda _: (_ for _ in ()).throw(OSError("crash"))
    )
    with pytest.raises(OSError):
        d.control.assign()
    first = d.row()
    monkeypatch.setattr(d.schedule, "write", original)
    assert d.control.assign() == []
    assert d.row() == first
    assert d.state()["runs"][0]["candidates"]["1"]["state"] == "assigned"


def test_startup_without_result_holds_slot_and_never_invents_session(
    d: DispatchHarness,
) -> None:
    d.prepare(1, 2)
    assert d.control.assign() == [1]
    row = d.row()
    d.job(row)["phase"] = "starting"
    d.update(row)
    d.control.reconcile()
    assert d.row()["dispatch"]["phase"] == "recovery_required"
    assert d.control.assign() == []
    with pytest.raises(ValueError):
        d.control.resume(1, "unknown outcome")


def test_inherited_launcher_lock_prevents_startup_recovery(d: DispatchHarness) -> None:
    d.prepare()
    d.control.assign()
    row = d.row()
    job = d.job(row)
    job["phase"] = "starting"
    d.update(row)
    with importlib.import_module("dispatcher.runtime").gate(
        Path(job["directory"]) / "lifetime.lock"
    ):
        d.control.reconcile()
        assert d.row()["dispatch"]["phase"] == "reserved"


def test_children_redelegation_cannot_hide_live_workers(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_runtime(d, monkeypatch)
    d.prepare()
    d.control.assign()
    env = d.accepted()
    row = d.row()
    for n in (10, 11, 12):
        row["tasks"][str(n)] = {
            "delegated": True,
            "delegation_id": str(n),
            "agent": "codex",
            "authorization": row["dispatch"]["authorization"],
            "worktree": str(d.h.work),
            "branch": "test",
        }
    # Reservation policy normally prevents the third; the runtime entry refuses
    # even if another caller changes that policy directly.
    d.update(row)
    with pytest.raises(ValueError, match="reservation limit"):
        d.control.start_child(1, 10, "10", "task", env)
    row["tasks"].pop("12")
    d.update(row)
    first = d.control.start_child(1, 10, "10", "task", env)
    d.control.start_child(1, 11, "11", "task", env, "gpt-6-astra")
    row = d.row()
    row["tasks"]["10"]["delegation_id"] = "new-delegation"
    row["tasks"]["10"]["worker_ended"] = {
        "delegation_id": "new-delegation",
        "evidence": "claimed elsewhere",
    }
    d.update(row)
    with pytest.raises(ValueError):
        d.control.start_child(1, 10, "new-delegation", "task", env)
    row = d.row()
    assert row["dispatch"]["jobs"][first]["phase"] == "reserved"
    d.runtime(d.job(row))
    d.control.reconcile()
    assert (
        d.control.status(d.now("2026-10-22T09:00:00+09:00"))["used_parent_slots"] == 1
    )
    assert d.mod.children_pending(d.row()) == [10, 11]


def test_child_end_event_releases_only_matching_delegation_and_resumes_same_parent(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_runtime(d, monkeypatch)
    d.prepare()
    d.control.assign()
    env = d.accepted()
    original_parent = d.job()
    row = d.row()
    row["tasks"]["10"] = {
        "delegated": True,
        "delegation_id": "d1",
        "agent": "codex",
        "authorization": row["dispatch"]["authorization"],
        "worktree": str(d.h.work),
        "branch": "test",
    }
    d.update(row)
    child_id = d.control.start_child(1, 10, "d1", "task", env)
    row = d.row()
    child = row["dispatch"]["jobs"][child_id]
    child["phase"] = "starting"
    d.update(row)
    d.runtime(d.job(row))
    d.runtime(child, live=True)
    d.control.reconcile()
    assert d.row()["dispatch"]["phase"] == "waiting"
    d.runtime(child)
    d.control.reconcile()
    row = d.row()
    assert row["tasks"]["10"]["worker_ended"]["delegation_id"] == "d1"
    assert len(row["dispatch"]["jobs"]) == 3
    assert d.job(row)["resume_from"] == str(
        Path(original_parent["directory"]) / "runtime"
    )
    assert row["session"] == "claude:fixture-session"


@pytest.mark.parametrize("which", ["schedule", "registry", "extension"])
def test_corrupt_state_is_not_reset(d: DispatchHarness, which: str) -> None:
    d.prepare()
    d.control.assign()
    path = d.schedule.path if which == "schedule" else d.h.store.path
    if which == "extension":
        row = d.row()
        row["dispatch"]["schema_version"] = 999
        d.update(row)
    else:
        path.write_text('{"version":1,"version":1}')
    before = path.read_bytes()
    with pytest.raises((ValueError, KeyError)):
        d.control.reconcile()
    assert path.read_bytes() == before


def test_disable_and_shutdown_do_not_abandon_live_child(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_runtime(d, monkeypatch)
    d.prepare()
    d.control.assign()
    d.accepted()
    d.service.stop_judgement(d.h.store, d.schedule)
    assert d.run("2026-10-23T08:00:00+09:00")["result"] == "disabled"
    d.control.reconcile(stop=True)
    assert d.row()["dispatch"]["phase"] == "stopped"
    assert d.row()["status"] == "active"


def test_lost_judgment_response_reconciles_saved_result(d: DispatchHarness) -> None:
    d.prepare()
    with d.h.store.locked():
        state = d.schedule.read()
        state["active"] = state["runs"][0]["id"]
        state["runs"][0].update(phase="running", controller=None)
        d.schedule.write(state)
    assert d.run()["result"] == "not_due"
    assert d.state()["active"] is None
    assert d.control.assign() == [1]
    assert len(d.h.calls.read_text().splitlines()) == 1


def test_unconfirmed_previous_judgment_blocks_next_slot_until_explicit_recovery(
    d: DispatchHarness,
) -> None:
    d.prepare()
    with d.h.store.locked():
        state = d.schedule.read()
        run = state["runs"][0]
        run.update(phase="running", controller=None)
        state["active"] = run["id"]
        d.schedule.write(state)
    result_path = Path(run["directory"]) / "agent-runtime/result.json"
    record = json.loads(result_path.read_text())
    record.update(result="running", cleanup_complete=False)
    result_path.write_text(json.dumps(record))
    assert d.run("2026-10-22T20:00:00+09:00")["result"] == "recovery_required"
    assert len(d.h.calls.read_text().splitlines()) == 1
    # The current runner requires explicit descendant-cleanup evidence after
    # losing its subreaper; an empty CLI group does not release this gate.
    with pytest.raises(ValueError, match="cleanup remains unconfirmed"):
        d.service.recover_judgement(d.h.store, d.schedule, "CLI group is empty")
    assert d.state()["active"] == run["id"]
    record = json.loads(result_path.read_text())
    confirmation = result_path.parent / "supervisor-cleanup.json"
    confirmation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                **{key: record[key] for key in ("run_id", "runner", "process")},
                "cleanup_complete": True,
                "evidence": "offline fixture CLI returned and has no descendant tools",
            }
        )
    )
    confirmed = subprocess.run(
        [
            str(d.h.mod.RUNNER),
            "stop",
            "--run-dir",
            str(result_path.parent),
            "--cleanup-confirmation",
            str(confirmation),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert confirmed.returncode == 0, confirmed.stderr
    d.service.recover_judgement(
        d.h.store,
        d.schedule,
        "inspected old runner/CLI/group; runner stop confirmed cleanup",
    )
    assert d.state()["active"] is None
    assert d.run("2026-10-22T20:00:00+09:00")["result"] == "recorded"
    assert len(d.h.calls.read_text().splitlines()) == 2


def test_parent_completion_does_not_free_unconfirmed_child_reservation(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_runtime(d, monkeypatch)
    d.prepare(1, 2)
    d.control.assign()
    d.accepted()
    row = d.row()
    row["tasks"]["10"] = {
        "delegated": True,
        "delegation_id": "still-live",
        "stopped": True,
    }
    d.update(row)
    d.runtime(d.job(row), state="completed")
    d.control.reconcile()
    assert d.row()["dispatch"]["phase"] == "failed"
    assert d.row()["status"] == "active"
    assert d.mod.occupies(d.row())
    assert d.control.assign() == []


def test_bootstrap_does_not_bind_a_different_cli_session(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_runtime(d, monkeypatch)
    d.prepare()
    d.control.assign()
    row = d.row()
    job = d.job(row)
    job["phase"] = "starting"
    d.update(row)
    d.runtime(job, live=True)
    before = d.row()
    with pytest.raises(ValueError, match="identity"):
        d.control.accept(
            json.loads((Path(job["directory"]) / "input.json").read_text()),
            {
                "CLAUDE_CODE_SESSION_ID": "other-session",
                "AGENT_RUNTIME_RUN_DIR": str(Path(job["directory"]) / "runtime"),
                "AGENT_RUNTIME_TASK_ID": job["task_id"],
            },
        )
    assert d.row() == before


def test_new_child_delegation_resumes_previous_child_session(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_runtime(d, monkeypatch)
    d.prepare()
    d.control.assign()
    env = d.accepted()
    row = d.row()
    row["tasks"]["10"] = {
        "delegated": True,
        "delegation_id": "d1",
        "agent": "codex",
        "authorization": row["dispatch"]["authorization"],
        "worktree": str(d.h.work),
        "branch": "test",
    }
    d.update(row)
    job_id = d.control.start_child(1, 10, "d1", "initial", env)
    row = d.row()
    child = row["dispatch"]["jobs"][job_id]
    child["phase"] = "starting"
    d.update(row)
    d.runtime(child)
    d.control.reconcile()
    row = d.row()
    row["tasks"]["10"]["delegation_id"] = "d2"
    row["tasks"]["10"].pop("worker_ended")
    d.update(row)
    resumed = d.control.start_child(1, 10, "d2", "fix", env)
    assert d.row()["dispatch"]["jobs"][resumed]["resume_from"] == str(
        Path(child["directory"]) / "runtime"
    )


def wait_until(predicate: Any, timeout: float = 12) -> None:
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if predicate():
            return
        time.sleep(0.03)
    raise AssertionError("fixture did not reach the expected state")


def test_real_runner_worker_claim_stop_and_same_session_resume(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    d.prepare()
    d.control.assign()
    for name in ("claude", "codex"):
        fake = d.h.root / "bin" / name
        shutil.copyfile(Path(__file__).with_name("fake_dispatch_cli.py"), fake)
        fake.chmod(0o700)
    monkeypatch.setenv("FAKE_DISPATCH_CALLS", str(d.h.root / "runtime-calls.jsonl"))
    monkeypatch.setenv("FAKE_DISPATCH_HOLD", "1")
    monkeypatch.setenv("CODEX_HOME", str(d.h.root / "codex-home"))
    job = d.job()
    directory = Path(job["directory"])
    runtime = importlib.import_module("dispatcher.runtime").Runtime(Path(job["runner"]))
    # Duplicate launchers race, but only one claims and starts a real wrapper.
    runtime.spawn(d.h.store.directory, "o/r#1", job["id"], directory)
    runtime.spawn(d.h.store.directory, "o/r#1", job["id"], directory)

    def initialized() -> bool:
        path = directory / "runtime/result.json"
        return path.exists() and bool(
            json.loads(path.read_text())["runtime"]["session_id"]
        )

    try:
        wait_until(initialized)
        record = json.loads((directory / "runtime/result.json").read_text())
        env = {
            "CLAUDE_CODE_SESSION_ID": record["runtime"]["session_id"],
            "AGENT_RUNTIME_RUN_DIR": str(directory / "runtime"),
            "AGENT_RUNTIME_TASK_ID": job["task_id"],
        }
        d.control.accept(json.loads((directory / "input.json").read_text()), env)
        (directory / "release").touch()

        def waiting() -> bool:
            d.control.reconcile()
            return bool(d.row()["dispatch"]["phase"] == "waiting")

        wait_until(waiting)
        first_identity = d.row()["session"]
        d.control.resume(1, "CI event")
        d.control.reconcile()
        resumed = d.job()
        d.control.launch_reserved()
        wait_until(
            lambda: (
                (Path(resumed["directory"]) / "runtime/result.json").exists()
                and len((d.h.root / "runtime-calls.jsonl").read_text().splitlines())
                == 2
            )
        )
        d.control.reconcile(stop=True)
        wait_until(
            lambda: (
                not runtime.inspect(Path(resumed["directory"]) / "runtime")["summary"][
                    "runner_alive"
                ]
            )
        )
        d.control.reconcile()
        assert d.row()["session"] == first_identity
        calls = [
            json.loads(line)
            for line in (d.h.root / "runtime-calls.jsonl").read_text().splitlines()
        ]
        assert len(calls) == 2
        assert "--resume" in calls[1]["args"]
        assert (
            calls[1]["args"][calls[1]["args"].index("--model") + 1] == "claude-opus-5-5"
        )
    finally:
        d.control.reconcile(stop=True)


def test_real_implementation_slots_and_sol_astra_wiring(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    d.prepare()
    d.control.assign()
    for name in ("claude", "codex"):
        fake = d.h.root / "bin" / name
        shutil.copyfile(Path(__file__).with_name("fake_dispatch_cli.py"), fake)
        fake.chmod(0o700)
    monkeypatch.setenv("FAKE_DISPATCH_CALLS", str(d.h.root / "runtime-calls.jsonl"))
    monkeypatch.setenv("FAKE_DISPATCH_HOLD", "1")
    monkeypatch.setenv("CODEX_HOME", str(d.h.root / "codex-home"))
    parent = d.job()
    d.control.launch_reserved()

    def initialized(job: dict[str, Any]) -> bool:
        path = Path(job["directory"]) / "runtime/result.json"
        return path.exists() and bool(
            json.loads(path.read_text())["runtime"]["session_id"]
        )

    try:
        wait_until(lambda: initialized(parent))
        record = json.loads(
            (Path(parent["directory"]) / "runtime/result.json").read_text()
        )
        env = {
            "CLAUDE_CODE_SESSION_ID": record["runtime"]["session_id"],
            "AGENT_RUNTIME_RUN_DIR": str(Path(parent["directory"]) / "runtime"),
            "AGENT_RUNTIME_TASK_ID": parent["task_id"],
        }
        d.control.accept(
            json.loads((Path(parent["directory"]) / "input.json").read_text()), env
        )
        row = d.row()
        for child in (10, 11):
            row["tasks"][str(child)] = {
                "delegated": True,
                "delegation_id": str(child),
                "agent": "codex",
                "authorization": row["dispatch"]["authorization"],
                "worktree": str(d.h.work),
                "branch": "test",
            }
        d.update(row)
        jobs = [
            d.control.start_child(1, 10, "10", "implementation", env),
            d.control.start_child(1, 11, "11", "implementation", env, "gpt-6-astra"),
        ]
        d.control.launch_reserved()
        wait_until(
            lambda: all(initialized(d.row()["dispatch"]["jobs"][j]) for j in jobs)
        )
        row = d.row()
        row["tasks"]["12"] = {**row["tasks"]["10"], "delegation_id": "12"}
        d.update(row)
        with pytest.raises(ValueError, match="runtime limit"):
            d.control.start_child(1, 12, "12", "too many", env)
        assert len((d.h.root / "runtime-calls.jsonl").read_text().splitlines()) == 3
        d.control.reconcile(stop=True)
        wait_until(
            lambda: (
                not any(
                    d.mod.alive(j["worker"])
                    for j in d.row()["dispatch"]["jobs"].values()
                )
            )
        )
        d.control.reconcile()
        selected = []
        for job_id in jobs:
            job = d.row()["dispatch"]["jobs"][job_id]
            evidence = json.loads(
                (Path(job["directory"]) / "runtime/result.json").read_text()
            )
            selected.append(evidence["contract"]["settings"]["selected"]["model"])
            assert evidence["contract"]["settings"]["selected"]["effort"] == "max"
            assert evidence["cleanup_complete"] is True
            assert job["phase"] == "ended"
        assert selected == ["gpt-6.1-sol", "gpt-6-astra"]
    finally:
        d.control.reconcile(stop=True)
        wait_until(
            lambda: (
                not any(
                    d.mod.alive(j["worker"])
                    for j in d.row()["dispatch"]["jobs"].values()
                )
            )
        )


def test_installer_disabled_install_and_main_source_only(tmp_path: Path) -> None:
    repo = tmp_path / "checkout"
    source = repo / ".agents/ops"
    shutil.copytree(ROOT / ".agents/ops", source)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "systemctl"
    fake.write_text('#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$FAKE_SYSTEMCTL_LOG"\n')
    fake.chmod(0o700)
    log = tmp_path / "systemctl.log"
    env = {
        **os.environ,
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "FAKE_SYSTEMCTL_LOG": str(log),
    }
    install = subprocess.run(
        [str(source / "install_units.sh")], env=env, capture_output=True, text=True
    )
    assert install.returncode == 0, install.stderr
    assert log.read_text().splitlines() == ["--user daemon-reload"]
    unit = tmp_path / "config/systemd/user/tennis-lab-dispatch-judgement.timer"
    assert unit.is_symlink() and unit.resolve().is_relative_to(repo)
    enabled = subprocess.run(
        [str(source / "install_units.sh"), "--enable"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert enabled.returncode == 0, enabled.stderr
    assert (
        "--user enable --now tennis-lab-dispatch-supervisor.service" in log.read_text()
    )
    refusal = subprocess.run(
        [str(ROOT / ".agents/ops/install_units.sh"), "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert refusal.returncode == 2 and "linked worktree" in refusal.stderr


def test_cli_disabled_setup_configuration_and_diagnostics(d: DispatchHarness) -> None:
    schedule = d.h.root / "cli-schedule"
    parent = d.h.root / "cli-registry"

    def cli(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / ".agents/ops/dispatcher/dispatch.py"),
                "--state-dir",
                str(schedule),
                "--parent-state-dir",
                str(parent),
                *args,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    initial = cli(
        "init",
        "--repo",
        "o/r",
        "--cwd",
        str(d.h.work),
        "--now",
        "2026-10-22T07:00:00+09:00",
    )
    assert initial.returncode == 0, initial.stderr
    assert json.loads(initial.stdout)["enabled"] is False
    assert (
        json.loads(cli("run", "--now", "2026-10-22T08:00:00+09:00").stdout)["result"]
        == "disabled"
    )
    assert cli("configure", "--max-parents", "2", "--max-children", "3").returncode == 0
    state = json.loads(cli("status").stdout)
    assert state["max_development_parents"] == 2
    assert state["default_max_implementation_children"] == 3
    assert cli("enable").returncode == 0
    assert cli("disable").returncode == 0
    error = cli("configure", "--max-parents", "0", "--max-children", "3")
    assert error.returncode == 2
    assert "positive integers" in error.stderr
    assert str(schedule) not in error.stdout


def test_stop_during_popen_stops_launcher_before_runtime_record(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    d.prepare()
    d.control.assign()
    module = importlib.import_module("dispatcher.runtime")
    runner = d.h.root / "fake-launcher"
    runner.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    runner.chmod(0o700)
    runtime = module.Runtime(runner)
    original = subprocess.Popen
    processes: list[subprocess.Popen[bytes]] = []

    def launch(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        proc: subprocess.Popen[bytes] = original(*args, **kwargs)
        processes.append(proc)
        # Deliver the worker's signal after fork but before the Popen object
        # has been assigned to Runtime.launcher.
        runtime.request_stop()
        return proc

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    job = d.job()
    with module.gate(Path(job["directory"]) / "lifetime.lock") as fd:
        status = runtime.execute(job, fd)
        assert status in {-15, 143}
        assert runtime.execute(job, fd) == 143
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert not (Path(job["directory"]) / "runtime/result.json").exists()


def test_bootstrap_schema_paths_and_limits_are_bound_to_saved_reservation(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_runtime(d, monkeypatch)
    d.prepare()
    d.control.assign()
    row = d.row()
    job = d.job(row)
    job["phase"] = "starting"
    d.update(row)
    d.runtime(job, live=True)
    env = {
        "CLAUDE_CODE_SESSION_ID": "fixture-session",
        "AGENT_RUNTIME_RUN_DIR": str(Path(job["directory"]) / "runtime"),
        "AGENT_RUNTIME_TASK_ID": job["task_id"],
    }
    inputs = json.loads((Path(job["directory"]) / "input.json").read_text())
    before = d.row()
    for key, value in (
        ("human_confirmation", "not granted"),
        ("schema_version", True),
        ("parent_state_dir", str(d.h.root / "other-registry")),
        ("dispatcher_state_dir", str(d.h.root / "other-schedule")),
        ("runtime_dir", str(d.h.root / "other-runtime")),
        ("max_implementation_children", 999),
    ):
        with pytest.raises(ValueError, match="bootstrap"):
            d.control.accept({**inputs, key: value}, env)
        assert d.row() == before
    assert d.control.accept(inputs, env)["accepted"] is True


def test_invalid_child_worktree_never_defaults_to_parent_checkout(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_runtime(d, monkeypatch)
    d.prepare()
    d.control.assign()
    env = d.accepted()
    initial = d.row()
    task = {
        "delegated": True,
        "delegation_id": "d1",
        "agent": "codex",
        "authorization": initial["dispatch"]["authorization"],
        "worktree": str(d.h.work),
        "branch": "test",
    }
    for missing in ("worktree", "branch"):
        row = copy.deepcopy(initial)
        row["tasks"]["10"] = {k: v for k, v in task.items() if k != missing}
        d.update(row)
        with pytest.raises(ValueError, match="delegated"):
            d.control.start_child(1, 10, "d1", "task", env)
        assert d.row() == row
    row = copy.deepcopy(initial)
    row["tasks"]["10"] = {**task, "worktree": "relative-checkout"}
    d.update(row)
    with pytest.raises(ValueError, match="absolute"):
        d.control.start_child(1, 10, "d1", "task", env)
    assert d.row() == row
