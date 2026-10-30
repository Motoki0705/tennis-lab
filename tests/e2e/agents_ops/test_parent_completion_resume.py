"""Resume closed standalone work through the real gate and dispatcher claims."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from tests.e2e.agents_ops.test_dispatcher_supervisor import FixtureRuntime
from tests.e2e.agents_ops.test_merge_gate import GateEnv, make_gate_env
from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


@dataclass
class CompletionWorkflow:
    gate: GateEnv
    control: Any
    session: Any
    parent: dict[str, Any]
    implementation: dict[str, Any]

    def row(self) -> dict[str, Any]:
        with self.control.store.locked():
            return cast(dict[str, Any], json.loads(json.dumps(self.session.row)))

    def runtime(
        self, job: dict[str, Any], *, live: bool, state: str = "waiting"
    ) -> None:
        directory = Path(job["directory"]) / "runtime"
        directory.mkdir(exist_ok=True)
        parent = job["role"] == "parent"
        record = {
            "schema_version": 1,
            "run_id": job["id"],
            "result": "running"
            if live
            else "failed"
            if state == "failed"
            else "completed",
            "turn_end_reason": None
            if live
            else "interrupted"
            if state == "failed"
            else "completed",
            "cleanup_complete": not live,
            "_alive": live,
            "runtime": {
                "session_id": self.gate.review.p.env["CLAUDE_CODE_SESSION_ID"]
                if parent
                else "fixture-worker"
            },
            "contract": {
                "cwd": job["cwd"],
                "task_id": job["task_id"],
                "mode": "write",
                "settings": {
                    "role": job["role"],
                    "agent": "claude" if parent else "codex",
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
                    "reason": "終了照合への復帰待ち。",
                    "resume_condition": "mergeと終了を照合して同じ親へ戻す。",
                }
            )
        )

    def environment(self, job: dict[str, Any]) -> dict[str, str]:
        return {
            "CLAUDE_CODE_SESSION_ID": self.gate.review.p.env["CLAUDE_CODE_SESSION_ID"],
            "AGENT_RUNTIME_RUN_DIR": str(Path(job["directory"]) / "runtime"),
            "AGENT_RUNTIME_TASK_ID": job["task_id"],
        }

    def accept(self, job: dict[str, Any]) -> None:
        self.runtime(job, live=True)
        self.control.accept(
            json.loads((Path(job["directory"]) / "input.json").read_text()),
            self.environment(job),
        )

    def finish(
        self, job: dict[str, Any], state: str, final: dict[str, Any] | None = None
    ) -> None:
        self.runtime(job, live=False, state=state)
        if final is not None:
            (Path(job["directory"]) / "runtime/final.md").write_text(json.dumps(final))
        with self.control.store.locked():
            self.session.row["dispatch"]["jobs"][job["id"]]["worker"] = None
            self.control.store.save()
        self.control.reconcile()

    def close_and_reserve(
        self, state: str = "waiting", *, lost_response: bool = False
    ) -> dict[str, Any]:
        merged = self.gate.cli(
            "merge", **({"PARENT_MERGE_RESPONSE_LOST": "1"} if lost_response else {})
        )
        assert merged.returncode == 0, merged.stderr + merged.stdout
        self.finish(self.parent, state)
        assert self.row()["dispatch"]["phase"] == state
        assert self.row()["status"] == "active"
        self.control.resume(1, "確認済みのmerge/closeの後、完了照合へ戻る。")
        self.control.reconcile()
        row = self.row()
        return dict(row["dispatch"]["jobs"][row["dispatch"]["active_job"]])


@pytest.fixture
def completion_workflow(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> CompletionWorkflow:
    p = parent_env
    policy = importlib.import_module("parent.policy")
    execution = importlib.import_module("parent.execution")

    def autonomous_basis() -> None:
        p.cli("register")
        inputs = json.loads(p.cli("basis-input").stdout)
        p.cli(
            "basis-record",
            "--spec",
            p.spec(
                {
                    "basis": {
                        "schema_version": 1,
                        "kind": "autonomous_decision",
                        "purpose": "単独作業を完了する。",
                        "scope": "実装と終了照合。",
                        "behavior": "同じ親へ戻る。",
                        "acceptance": "実mergeと終了を確認する。",
                        "rationale": "既存issueから範囲と受入条件を確定できる。",
                        "sources": ["https://github.com/o/r/issues/1"],
                        "input_sha256": policy.digest(inputs),
                        "policy_version": execution.POLICY_VERSION,
                    },
                    "inputs": inputs,
                    "ignored_comments": [],
                }
            ),
        )

    monkeypatch.setattr(p, "agreed", autonomous_basis)
    g = make_gate_env(p, monkeypatch, "standalone")
    state = p.state()
    for issue in state["issues"].values():
        issue["assignees"] = []
    p.save(state)
    for key, value in p.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(p.repo)
    mod = importlib.import_module("dispatcher.supervisor")
    development = importlib.import_module("parent.development")
    monkeypatch.setattr(mod, "Runtime", FixtureRuntime)
    monkeypatch.setattr(development, "Runtime", FixtureRuntime)
    clock = importlib.import_module("dispatcher.schedule")
    schedule = clock.Schedule(p.root / "dispatcher")
    store = development.workflow_store(
        Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions",
        schedule.directory,
    )
    gh = importlib.import_module("parent.github").GitHub("o/r")
    control = mod.Supervisor(gh, store, schedule)
    with store.locked():
        schedule.initialize(
            {
                "repo": "o/r",
                "cwd": str(p.repo),
                "runner": str(importlib.import_module("dispatcher.agent").RUNNER),
                "max_parents": 1,
                "max_children": 2,
                "judgement_timeout": 10,
            },
            clock.local_time("2026-10-24T08:00:00+09:00"),
        )
        scheduled = schedule.read()
        scheduled["enabled"] = True
        schedule.write(scheduled)
        session = importlib.import_module("parent.session").ParentSession(
            gh, store, 1, "claude:" + p.env["CLAUDE_CODE_SESSION_ID"]
        )
        auth = execution.authorization(session)
        collect = importlib.import_module("dispatcher.collect")
        snapshot = collect.collect(gh, store, p.repo, [1])
        basis = execution.current_basis(session)
        candidate = {
            "schema_version": 1,
            "repo": "o/r",
            "parent": 1,
            "route": "standalone",
            "priority": 1,
            "basis_sha256": auth["execution_basis_sha256"],
            "generation": auth["generation"],
            "input_sha256": policy.digest(collect.material(snapshot)),
            "target_sha256": policy.digest(
                collect.material(snapshot, 1, basis["sources"])
            ),
            "code_commit": snapshot["code"]["commit"],
            "requirements": {
                k: basis[k] for k in ("purpose", "scope", "behavior", "acceptance")
            },
            "dependencies": [],
            "rationale": basis["rationale"],
            "sources": basis["sources"],
        }
        session.row["dispatch"] = {
            "schema_version": 1,
            "assignment_id": "fixture-assignment",
            "source_run": "fixture-judgement",
            "phase": "running",
            "session": session.row["session"],
            "candidate": candidate,
            "authorization": auth,
            "child_limit": 2,
            "jobs": {},
            "active_job": None,
            "events": [],
        }
        pr = p.state()["prs"][0]
        session.row["tasks"]["1"] = {
            "delegated": True,
            "delegation_id": "fixture-delegation",
            "agent": "codex",
            "authorization": auth,
            "worktree": str(p.repo / ".claude/worktrees/pr-0"),
            "branch": pr["head"]["ref"],
            "requirements_sha256": policy.digest(
                execution.requirement_body(p.state()["issues"]["1"]["body"])
            ),
        }
        parent = control._job(session.key, session.row)
        implementation = control._job(
            session.key,
            session.row,
            child=1,
            delegation_id="fixture-delegation",
            prompt="検証済みの実装。",
        )
        implementation["phase"] = "ended"
        session.row["tasks"]["1"]["worker_ended"] = {
            "delegation_id": "fixture-delegation",
            "evidence": "fixture workerの終了・回収を確認した。",
        }
        store.save()
    flow = CompletionWorkflow(g, control, session, parent, implementation)
    flow.runtime(implementation, live=False)
    assert control.claim("o/r#1", parent["id"]) is not None
    flow.accept(parent)
    p.cli(
        "record-result",
        "--child",
        "1",
        "--evidence-file",
        p.note("親が差分・受入条件・テストを確認した。"),
    )
    return flow


@pytest.mark.parametrize("state", ["waiting", "failed"])
@pytest.mark.parametrize("lost_response", [False, True])
def test_closed_standalone_parent_resumes_same_session_and_finishes(
    completion_workflow: CompletionWorkflow, state: str, lost_response: bool
) -> None:
    flow = completion_workflow
    p, control = flow.gate.review.p, flow.control
    previous = flow.row()
    resumed = flow.close_and_reserve(state, lost_response=lost_response)
    assert resumed["resume_from"] == str(Path(flow.parent["directory"]) / "runtime")
    assert control.claim("o/r#1", resumed["id"]) is not None
    flow.accept(resumed)
    row = flow.row()
    assert (
        row["session"] == previous["session"]
        and row["generation"] == previous["generation"]
    )
    assert row["tasks"]["1"]["delegation_id"] == previous["tasks"]["1"]["delegation_id"]
    blocked = p.cli(
        "--state-dir",
        str(control.store.directory),
        "--dispatcher-state-dir",
        str(control.schedule.directory),
        "start-worker",
        "--child",
        "1",
        "--branch",
        "codex/extra",
        "--worktree",
        str(p.repo / ".claude/worktrees/pr-0"),
        "--prompt-file",
        p.note("新しい実装を起動してはいけない。"),
        extra=flow.environment(resumed),
        ok=False,
    )
    assert "parent issue is not open" in blocked.stderr
    development = importlib.import_module("parent.development")
    with pytest.raises(ValueError, match="parent issue is not open"):
        development.start_worker(
            flow.session,
            control.schedule.directory,
            flow.environment(resumed),
            child=1,
            prompt="closed issueで修正を再開してはいけない。",
            model=None,
            continuation="完了照合への復帰だけが許可されている。",
        )
    assert len(flow.row()["dispatch"]["jobs"]) == len(row["dispatch"]["jobs"])
    with control.store.locked():
        result = development.report(
            flow.session,
            {
                "schema_version": 1,
                "state": "completed",
                "reason": "実merge/closeと全体受入を照合した。",
                "resume_condition": "完了。追加要件は別issueで扱う。",
                "evidence": "同じ親が保存済みのレビュー・検証・終了を再確認した。",
            },
        )
    flow.finish(resumed, "completed", result)
    assert flow.row()["status"] == "ended"
    assert flow.row()["dispatch"]["phase"] == "completed"
    assert flow.row()["session"] == previous["session"]
    assert len(flow.gate.merge_calls()) == 1
    assert p.state()["issues"]["1"]["state"] == "closed"


@pytest.mark.parametrize(
    "change",
    [
        "cancelled",
        "unmerged",
        "head",
        "journal",
        "verification",
        "delegation",
        "generation",
        "basis",
        "source",
        "session",
        "worker",
    ],
)
def test_closed_parent_resume_rejects_unconfirmed_completion(
    completion_workflow: CompletionWorkflow, change: str
) -> None:
    flow = completion_workflow
    p, control = flow.gate.review.p, flow.control
    resumed = flow.close_and_reserve()
    state = p.state()
    if change == "cancelled":
        state["issues"]["1"]["state_reason"] = "not_planned"
    elif change == "unmerged":
        state["prs"][0]["merged"] = False
    elif change == "head":
        state["prs"][0]["head"]["sha"] = "f" * 40
    p.save(state)
    if change == "basis":
        p.cli("basis-revoke", "--reason-file", p.note("実行根拠を取り消す。"))
    elif change == "journal":
        attempts = list((control.store.directory / "merges").rglob("*.json"))
        assert len(attempts) == 1
        for path in attempts:
            path.unlink()
    elif change in {"source", "session", "worker"}:
        if change == "source":
            with control.store.locked():
                flow.session.row["dispatch"]["jobs"][resumed["id"]]["resume_from"] = (
                    "unrelated-runtime"
                )
                control.store.save()
        else:
            job = flow.parent if change == "session" else flow.implementation
            path = Path(job["directory"]) / "runtime/result.json"
            record = json.loads(path.read_text())
            if change == "session":
                record["runtime"]["session_id"] = "unrelated-session"
            else:
                record["cleanup_complete"] = False
            path.write_text(json.dumps(record))
    elif change in {"verification", "delegation", "generation"}:
        with control.store.locked():
            row = flow.session.row
            if change == "verification":
                row["tasks"]["1"]["verification"] = {}
            elif change == "delegation":
                row["tasks"]["1"]["delegation_id"] = "unrelated-delegation"
                row["tasks"]["1"]["worker_ended"]["delegation_id"] = (
                    "unrelated-delegation"
                )
            else:
                row["generation"] += 1
            control.store.save()
    assert control.claim("o/r#1", resumed["id"]) is None
    row = flow.row()
    assert row["status"] == "active"
    rejected = row["dispatch"]["jobs"][resumed["id"]]
    assert rejected["cancelled_before_start"] and rejected["diagnostic"]
    assert not (Path(resumed["directory"]) / "runtime").exists()
    assert len(flow.gate.merge_calls()) == 1


@pytest.mark.parametrize("role", ["initial_parent", "implementation"])
def test_closed_issue_never_allows_initial_parent_or_implementation_start(
    completion_workflow: CompletionWorkflow, role: str
) -> None:
    flow = completion_workflow
    resumed = flow.close_and_reserve()
    with flow.control.store.locked():
        row = flow.session.row
        if role == "initial_parent":
            job = row["dispatch"]["jobs"][resumed["id"]]
            job["resume_from"] = None
        else:
            row["tasks"]["1"].pop("worker_ended")
            job = flow.control._job(
                row=row,
                key="o/r#1",
                child=1,
                delegation_id="fixture-delegation",
                prompt="closed issueでの実装は禁止。",
            )
        flow.control.store.save()
    assert flow.control.claim("o/r#1", job["id"]) is None
    rejected = flow.row()["dispatch"]["jobs"][job["id"]]
    assert rejected["cancelled_before_start"]
    expected = (
        "accepted resume source"
        if role == "initial_parent"
        else "parent issue is not open"
    )
    assert expected in rejected["diagnostic"]
