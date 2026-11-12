"""Resume closed standalone work through the real gate and dispatcher claims."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from tests.e2e.agents_ops.test_dispatcher_supervisor import FixtureRuntime
from tests.e2e.agents_ops.test_merge_gate import GateEnv, make_gate_env
from tests.e2e.agents_ops.test_parent_session import (
    ParentEnv,
    evidence_comments,
    evidence_number,
    run_git,
    untracked_legacy_fixture,
)
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
            "exit_code": None if live else 1 if state == "failed" else 0,
            "task_success": None,
            "runner": None,
            "process": None,
            "_alive": live,
            "runtime": {
                "session_id": self.gate.review.p.env["CLAUDE_CODE_SESSION_ID"]
                if parent
                else "fixture-worker"
            },
            "contract": {
                "cwd": job["cwd"],
                "branch": job["branch"],
                "task_id": job["task_id"],
                "mode": "write",
                "settings": {
                    "role": job["role"],
                    "agent": "claude" if parent else "codex",
                    "profile": None,
                },
            },
            "prompt_sha256": hashlib.sha256(
                (Path(job["directory"]) / "prompt.md").read_bytes()
            ).hexdigest(),
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
    parent_env: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
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
    variant = getattr(request, "param", "standalone")
    direct = variant == "parent_self_direct"
    g = make_gate_env(
        p, monkeypatch, "parent_self" if variant != "standalone" else variant
    )
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
            "route": "standalone" if variant == "standalone" else "parent",
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
        # Seed the pinned judgement and initial reservation, as assign() does.
        # Continuations must read an existing parent's start-input, never create
        # an initial job by silently collecting over an absent prior snapshot.
        directory = schedule.directory / "judgements/fixture-judgement"
        store.write_json(directory / "input.json", snapshot)
        store.write_json(
            directory / "result.json",
            {
                "schema_version": 2,
                "result": "recorded",
                "input_sha256": candidate["input_sha256"],
                "decisions": [{"issue": 1, "disposition": "startable"}],
                "deferred": [],
                "candidates": [candidate],
                "publications": [],
                "diagnostics": [],
            },
        )
        scheduled["runs"].append(
            {
                "id": "fixture-judgement",
                "slot": scheduled["cursor"],
                "explicit": False,
                "phase": "recorded",
                "controller": None,
                "directory": str(directory),
                "diagnostic": None,
                "candidates": {"1": {"state": "assigned", "reason": "fixture"}},
            }
        )
        schedule.write(scheduled)
        session.row["session"] = "dispatcher-reservation:fixture-assignment"
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
        parent = control._job(session.key, session.row)
        implementation = {}
        if not direct:
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
            preflight = importlib.import_module("parent.worker_preflight")
            task = session.row["tasks"]["1"]
            task["preflight"] = preflight.new(task["delegation_id"])
            prepared, _ = preflight.prepare_turn(
                session,
                1,
                "検証済みの実装。",
                resume=False,
            )
            preflight.commit_turn(task, prepared)
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
    if implementation:
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


@pytest.mark.parametrize(
    "completion_workflow", ["parent_self", "parent_self_direct"], indirect=True
)
def test_labelled_parent_recovers_completion_with_or_without_a_worker(
    completion_workflow: CompletionWorkflow,
) -> None:
    flow = completion_workflow
    before = flow.row()
    p = flow.gate.review.p
    assert {label["name"] for label in p.state()["issues"]["1"]["labels"]} == {
        "issue:parent"
    }
    resumed = flow.close_and_reserve()
    assert flow.control.claim("o/r#1", resumed["id"]) is not None
    flow.accept(resumed)
    p.announce_completion()
    from tests.support.agents_ops.improvements import finish_for

    acceptance = finish_for(p, state_dir=flow.control.store.directory)
    with flow.control.store.locked():
        result = importlib.import_module("parent.development").report(
            flow.session,
            {
                "schema_version": 2,
                "state": "completed",
                "finish": acceptance,
                "reason": "親自身の成果を再照合した。",
                "resume_condition": "完了。",
                "evidence": "対象HEADのCI・正式レビュー・merge journal・受入証拠と実行終了を確認した。",
            },
        )
    flow.finish(resumed, "completed", result)
    after = flow.row()
    assert after["status"] == "ended" and after["dispatch"]["phase"] == "completed"
    assert (
        after["session"] == before["session"]
        and after["generation"] == before["generation"]
    )
    if not flow.implementation:
        assert not after["tasks"]["1"].get("delegated")
        assert "worker_ended" not in after["tasks"]["1"]
        assert all(
            job["role"] == "parent" for job in after["dispatch"]["jobs"].values()
        )
    assert p.state()["parents"] == {}


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
    assert "existing worker" in blocked.stderr
    development = importlib.import_module("parent.development")
    assert len(flow.row()["dispatch"]["jobs"]) == len(row["dispatch"]["jobs"])
    p.announce_completion()
    from tests.support.agents_ops.improvements import finish_for

    acceptance = finish_for(p, state_dir=control.store.directory)
    with control.store.locked():
        result = development.report(
            flow.session,
            {
                "schema_version": 2,
                "state": "completed",
                "finish": acceptance,
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
def test_closed_parent_resume_checks_execution_evidence(
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
    if change == "generation":
        # A new owner generation alone does not invalidate saved execution facts.
        previous = flow.row()
        assert control.claim("o/r#1", resumed["id"]) is not None
        flow.accept(resumed)
        row = flow.row()
        assert row["generation"] != resumed["authorization"]["generation"]
        assert (
            row["tasks"]["1"]["verification"] == previous["tasks"]["1"]["verification"]
        )
        assert (
            row["dispatch"]["jobs"][resumed["id"]]["authorization"]
            == resumed["authorization"]
        )
        assert len(flow.gate.merge_calls()) == 1
        return
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
            # A stale caller can still construct a reservation; the public
            # claim must reject it because the standalone issue is closed.
            row["tasks"]["1"]["preflight"]["turns"][-1]["source"] = None
            job = flow.control._job(
                row=row,
                key="o/r#1",
                child=1,
                delegation_id="fixture-delegation",
                prompt=(
                    Path(flow.implementation["directory"]) / "prompt.md"
                ).read_text(),
            )
        flow.control.store.save()
    assert flow.control.claim("o/r#1", job["id"]) is None
    rejected = flow.row()["dispatch"]["jobs"][job["id"]]
    assert rejected["cancelled_before_start"]
    expected = "accepted resume source" if role == "initial_parent" else "完了済み"
    assert expected in rejected["diagnostic"]


@pytest.mark.parametrize("merged", [False, True], ids=["reviewing", "merged"])
def test_legacy_result_migration_preserves_review_and_completion(
    completion_workflow: CompletionWorkflow, merged: bool
) -> None:
    """An explicit migration can adopt proven old work without inventing turns."""
    flow = completion_workflow
    p, control = flow.gate.review.p, flow.control
    # Review fixtures use short local branch names; a real worker's checkout
    # must also match its recorded remote branch, not only its commit.
    pr = p.state()["prs"][0]
    worktree = p.repo / ".claude/worktrees/pr-0"
    run_git(worktree, "switch", "-C", pr["head"]["ref"])
    if merged:
        (worktree / "change.txt").write_text("reviewed fix\n")
        run_git(worktree, "add", "change.txt")
        run_git(worktree, "commit", "-m", "verified legacy fix")
        head = run_git(worktree, "rev-parse", "HEAD")
        state = p.state()
        state["prs"][0]["head"]["sha"] = head
        p.save(state)
        flow.gate.review.cli(
            "resolve",
            "--run",
            flow.gate.review.request["review_run_id"],
            "--pr",
            "100",
            "--fixed-head",
            head,
            "--finding",
            "changes",
            "--evidence",
            p.spec(
                {
                    "ci": "対象headのPython成功。",
                    "regression_tests": "代表回帰が成功。",
                    "resolution": "親が旧委任の修正差分を確認。",
                    "reviewed_paths": ["change.txt"],
                }
            ),
        )
        response = flow.gate.cli("merge")
        assert response.returncode == 0, response.stderr + response.stdout
    else:
        # The reviewer has returned its result, but the parent has not yet
        # accepted/confirmed it: the existing PR is genuinely still in review.
        state = p.state()
        policy = importlib.import_module("parent.policy")
        state["comments"][str(evidence_number(state))] = [
            comment
            for comment in evidence_comments(state)
            if not any(
                event.get("event") == "review"
                and event.get("review_run_id")
                == flow.gate.review.request["review_run_id"]
                and event.get("state") in {"result_received", "fixes_verified"}
                for event in policy.events([comment], "operator", 1)
            )
        ]
        p.save(state)
        # The legacy fixture predates posting these acknowledgement events.
        untracked_legacy_fixture(p)
    with control.store.locked():
        task = flow.session.row["tasks"]["1"]
        task.pop("preflight")
        task.pop("review_preflight", None)
        job = flow.session.row["dispatch"]["jobs"][flow.implementation["id"]]
        job.pop("preflight", None)
        if not merged:
            flow.session.row["reviews"][-1]["state"] = "instructed"
        control.store.save()
    old_request = flow.gate.review.request_path.read_bytes()
    old_result = (
        Path(flow.implementation["directory"]) / "runtime/result.json"
    ).read_bytes()
    status = json.loads(p.cli("status").stdout)
    assert status["tasks"]["1"]["worker_phase"]["status"] == "unrecorded"
    before = flow.row()
    result_path = Path(flow.implementation["directory"]) / "runtime/result.json"
    unclean = json.loads(old_result)
    unclean["cleanup_complete"] = False
    result_path.write_text(json.dumps(unclean))
    rejected = p.cli(
        "record-result", "--child", "1", "--evidence-file", p.note(), ok=False
    )
    assert "termination" in rejected.stderr
    assert flow.row() == before
    result_path.write_bytes(old_result)
    p.cli(
        "record-result",
        "--child",
        "1",
        "--evidence-file",
        p.note("保存済みの成果と現条件を照合した。"),
    )
    assert "preflight" not in flow.row()["tasks"]["1"]
    assert flow.gate.review.request_path.read_bytes() == old_request
    assert result_path.read_bytes() == old_result
    assert json.loads(p.cli("reconcile").stdout)["tasks"][0]["artifacts_accepted"]
    if merged:
        with control.store.locked():
            importlib.import_module("parent.development").verify_completion(
                flow.session
            )
    else:
        response = flow.gate.cli("merge", AGENT_AUTO_MERGE="0")
        assert response.returncode == 1
        assert not flow.gate.merge_calls()
