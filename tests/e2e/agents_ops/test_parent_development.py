"""Managed development with offline CLIs, disposable worktrees and fake GitHub."""

from __future__ import annotations

import copy
import importlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, cast

import pytest

from tests.e2e.agents_ops.test_dispatcher_supervisor import (
    DispatchHarness,
    FixtureRuntime,
    fixture_runtime,
    wait_until,
)
from tests.e2e.agents_ops.test_dispatcher_supervisor import d as d
from tests.e2e.agents_ops.test_merge_gate import GateEnv
from tests.e2e.agents_ops.test_merge_gate import gate_env as gate_env
from tests.e2e.agents_ops.test_parent_session import (
    ParentEnv,
    make_review_env,
    review_policy_module,
)
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


def workflow(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Any, dict[str, str]]:
    """Real bootstrap/claims, with an offline parent held alive for child actions."""
    gh_class = type(d.h.gh)
    github = importlib.import_module("issues.cli").GitHub
    monkeypatch.setattr(gh_class, "family", github.family, raising=False)
    original = gh_class.write

    def write(self: Any, method: str, endpoint: str, payload: Any) -> Any:
        if method == "PATCH" and re.fullmatch(r"repos/o/r/issues/\d+", endpoint):
            if self.on_write:
                self.on_write()
            self.issues[int(endpoint.rsplit("/", 1)[1])].update(payload)
            self.writes.append(
                {"method": method, "endpoint": endpoint, "payload": payload}
            )
            return copy.deepcopy(self.issues[int(endpoint.rsplit("/", 1)[1])])
        return original(self, method, endpoint, payload)

    monkeypatch.setattr(gh_class, "write", write)
    d.prepare()
    d.control.assign()
    fixture_runtime(d, monkeypatch)
    env = d.accepted()
    parent_runtime = str(Path(d.job()["directory"]) / "runtime")
    runtime_class = importlib.import_module("dispatcher.runtime").Runtime

    class MixedRuntime(runtime_class):  # type: ignore[misc,valid-type]
        def inspect(self, directory: Path, *, stop: bool = False) -> dict[str, Any]:
            if str(directory) == parent_runtime:
                return dict(FixtureRuntime(self.runner).inspect(directory, stop=stop))
            return dict(super().inspect(directory, stop=stop))

    monkeypatch.setattr(d.mod, "Runtime", MixedRuntime)
    monkeypatch.chdir(d.h.work)
    for name in ("claude", "codex"):
        path = d.h.root / "bin" / name
        shutil.copyfile(Path(__file__).with_name("fake_dispatch_cli.py"), path)
        path.chmod(0o700)
    monkeypatch.setenv("FAKE_DISPATCH_CALLS", str(d.h.root / "runtime-calls.jsonl"))
    monkeypatch.setenv("CODEX_HOME", str(d.h.root / "codex-home"))
    mod = importlib.import_module("parent.development")
    session = importlib.import_module("parent.session").ParentSession(
        d.h.gh, d.h.store, 1, "claude:fixture-session"
    )
    return mod, session, env


def child_tree(d: DispatchHarness, name: str = "parser") -> Path:
    tree = Path(d.h.work) / ".claude/worktrees" / name
    d.h.git("worktree", "add", "-b", "codex/" + name, str(tree))
    return tree


def drain(d: DispatchHarness, job_id: str) -> dict[str, Any]:
    def done() -> bool:
        d.control.reconcile()
        return bool(d.row()["dispatch"]["jobs"][job_id]["phase"] == "ended")

    wait_until(done)
    return cast(dict[str, Any], d.row()["dispatch"]["jobs"][job_id])


@pytest.mark.parametrize("model", [None, "gpt-6-astra"])
def test_bootstrap_delegates_standalone_sol_or_astra_and_does_not_trust_exit_zero(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch, model: str | None
) -> None:
    mod, session, env = workflow(d, monkeypatch)
    tree = child_tree(d)
    result = mod.start_worker(
        session,
        d.schedule.directory,
        env,
        child=1,
        prompt="境界を検証しPRを作る。",
        model=model,
        branch="codex/parser",
        worktree=tree,
    )
    try:
        d.control.launch_reserved()
        job = drain(d, result["job_id"])
        record = json.loads(
            (Path(job["directory"]) / "runtime/result.json").read_text()
        )
        assert record["result"] == "completed" and record["task_success"] is None
        assert record["contract"]["settings"]["selected"]["model"] == (
            model or "gpt-6.1-sol"
        )
        assert record["contract"]["settings"]["selected"]["effort"] == "max"
        assert record["contract"]["cwd"] == str(tree)
        assert record["contract"]["branch"] == "codex/parser"
        assert job["child"] == 1 and d.h.gh.parents == {}
        assert "担当issue: #1" in (Path(job["directory"]) / "prompt.md").read_text()
        assert "pytest_locked.sh" in (Path(job["directory"]) / "prompt.md").read_text()
        with d.h.store.locked():
            state = mod.inspect_worker(session, 1)
            assert state["termination_confirmed"] and state["task_success"] is None
            with pytest.raises(ValueError, match="close|verification"):
                mod.report(
                    session,
                    {
                        "schema_version": 1,
                        "state": "completed",
                        "reason": "子が成功を報告",
                        "resume_condition": "完了",
                        "evidence": "exit 0",
                    },
                )
        assert d.row()["status"] == "active"
    finally:
        d.control.reconcile(stop=True)


def test_existing_pr_fix_continues_same_session_and_model_after_end_evidence(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod, session, env = workflow(d, monkeypatch)
    tree = child_tree(d)
    first = mod.start_worker(
        session,
        d.schedule.directory,
        env,
        child=1,
        prompt="実装。",
        model="gpt-6-astra",
        branch="codex/parser",
        worktree=tree,
    )
    try:
        d.control.launch_reserved()
        previous = drain(d, first["job_id"])
        d.h.gh.prs.append(
            {
                "number": 50,
                "state": "open",
                "body": "Closes #1",
                "merged_at": None,
                "head": {
                    "ref": "codex/parser",
                    "sha": d.h.git("rev-parse", "HEAD"),
                    "repo": {"full_name": "o/r"},
                },
            }
        )
        with pytest.raises(ValueError, match="model"):
            mod.start_worker(
                session,
                d.schedule.directory,
                env,
                child=1,
                prompt="修正。",
                model="gpt-6.1-sol",
                continuation="終了・差分を確認した。",
            )
        with pytest.raises(ValueError, match="existing worker"):
            mod.start_worker(
                session,
                d.schedule.directory,
                env,
                child=1,
                prompt="別sessionは禁止。",
                model=None,
                branch="codex/parser",
                worktree=tree,
            )
        second = mod.start_worker(
            session,
            d.schedule.directory,
            env,
            child=1,
            prompt="指摘と回帰テストを修正。",
            model=None,
            continuation="前turn終了・同じPRの差分とHEADを確認。",
        )
        d.control.launch_reserved()
        current = drain(d, second["job_id"])
        before = json.loads(
            (Path(previous["directory"]) / "runtime/result.json").read_text()
        )
        after = json.loads(
            (Path(current["directory"]) / "runtime/result.json").read_text()
        )
        assert current["resume_from"] == str(Path(previous["directory"]) / "runtime")
        assert before["runtime"]["session_id"] == after["runtime"]["session_id"]
        assert after["contract"]["settings"]["selected"]["model"] == "gpt-6-astra"
        assert first["delegation_id"] != second["delegation_id"]
        assert len((d.h.root / "runtime-calls.jsonl").read_text().splitlines()) == 2
        calls = [
            json.loads(line)
            for line in (d.h.root / "runtime-calls.jsonl").read_text().splitlines()
        ]
        assert "resume" in calls[1]["args"]
    finally:
        d.control.reconcile(stop=True)


def test_wrong_state_or_stale_basis_does_not_reserve_worker(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod, session, env = workflow(d, monkeypatch)
    tree = child_tree(d)
    with pytest.raises(ValueError, match="another parent runtime"):
        mod.start_worker(
            session,
            d.schedule.directory,
            {},
            child=1,
            prompt="実装。",
            model=None,
            branch="codex/parser",
            worktree=tree,
        )
    d.h.gh.issues[1]["body"] += "\n要件変更。"
    with pytest.raises(ValueError, match="inputs changed"):
        mod.start_worker(
            session,
            d.schedule.directory,
            env,
            child=1,
            prompt="実装。",
            model=None,
            branch="codex/parser",
            worktree=tree,
        )
    assert d.row()["tasks"] == {}
    assert not (d.h.root / "runtime-calls.jsonl").exists()


def test_parent_cli_uses_bootstrap_state_for_start_inspect_and_targeted_stop(
    d: DispatchHarness,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, _, env = workflow(d, monkeypatch)
    tree = child_tree(d)
    cli = importlib.import_module("parent.cli")
    monkeypatch.setattr(cli, "GitHub", lambda repo: d.h.gh)
    for key in ("CODEX_SESSION_ID", "CODEX_THREAD_ID", "CLAUDE_CODE_SESSION_ID"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("XDG_STATE_HOME", str(d.h.root / "unused-state"))
    common = [
        "--repo",
        "o/r",
        "--parent",
        "1",
        "--state-dir",
        str(d.h.store.directory),
        "--dispatcher-state-dir",
        str(d.schedule.directory),
    ]
    task = d.h.root / "task.md"
    task.write_text("境界を実装し検証する。")
    assert (
        cli.main(
            [
                *common,
                "start-worker",
                "--child",
                "1",
                "--branch",
                "codex/parser",
                "--worktree",
                str(tree),
                "--prompt-file",
                str(task),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["result"] == "reserved"
    assert cli.main([*common, "inspect-worker", "--child", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["termination_confirmed"] is False and result["task_success"] is None
    spec = d.h.root / "stop.json"
    spec.write_text(
        json.dumps(
            {
                key: "入力待ち。証拠を確認したら再開する。"
                for key in (
                    "reason",
                    "facts",
                    "impact",
                    "resume_condition",
                    "next_action",
                )
            }
        )
    )
    assert cli.main([*common, "stop-worker", "--child", "1", "--spec", str(spec)]) == 0
    assert json.loads(capsys.readouterr().out)["result"] == "stop_requested"
    assert d.row()["tasks"]["1"]["worker_ended"]
    assert not (d.h.root / "unused-state").exists()
    assert not (d.h.root / "runtime-calls.jsonl").exists()


def test_stop_propagates_only_to_dependencies_and_reserved_worker_is_cancelled(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    p.agreed()
    for key, deps in (("upstream", []), ("dependent", [10]), ("independent", [])):
        p.child(key, deps)
    for n in (10, 12):
        p.cli(
            "delegate",
            "--child",
            str(n),
            "--agent",
            "codex",
            "--branch",
            f"codex/worker-{n}",
            "--worktree",
            str(p.worktree(f"worker-{n}")),
        )
    state_mod = importlib.import_module("parent.state")
    session_mod = importlib.import_module("parent.session")
    gh = importlib.import_module("parent.github").GitHub("o/r")
    monkeypatch.setenv("PATH", p.env["PATH"])
    for name in ("PARENT_GH_STATE", "PARENT_GH_CALLS"):
        monkeypatch.setenv(name, p.env[name])
    directory = Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    with state_mod.Store(directory).locked() as store:
        identity = store.parents["o/r#1"]["session"]
        session = session_mod.ParentSession(gh, store, 1, identity)
        authorization = importlib.import_module("parent.execution").authorization(
            session
        )
        jobs = {
            str(n): {
                "id": str(n),
                "sequence": n,
                "role": "implementation",
                "child": n,
                "delegation_id": session.row["tasks"][str(n)]["delegation_id"],
                "phase": "reserved",
                "directory": str(p.root / f"runtime-{n}"),
                "cwd": str(p.repo),
                "runner": "fake",
                "task_id": f"worker-{n}",
            }
            for n in (10, 12)
        }
        jobs["parent"] = {
            "id": "parent",
            "sequence": 0,
            "role": "parent",
            "child": None,
            "phase": "starting",
            "directory": str(p.root / "parent-runtime"),
            "cwd": str(p.repo),
            "runner": "fake",
            "task_id": "parent",
        }
        session.row["dispatch"] = {
            "schema_version": 1,
            "assignment_id": "fixture",
            "session": identity,
            "authorization": authorization,
            "phase": "running",
            "child_limit": 2,
            "jobs": jobs,
            "events": [],
            "active_job": "parent",
        }
        store.save()
        mod = importlib.import_module("parent.development")
        detail = {
            key: "入力仕様の人間判断待ち。独立作業は継続。"
            for key in ("reason", "facts", "impact", "resume_condition", "next_action")
        }
        response = mod.stop_worker(session, 10, detail)
        planned = {t["child"]: t for t in response["launch_plan"]["tasks"]}
        assert [planned[n]["state"] for n in (10, 11, 12)] == [
            "stopped",
            "held",
            "continue",
        ]
        assert jobs["10"]["phase"] == "ended" and jobs["12"]["phase"] == "reserved"
        assert session.row["tasks"]["10"]["worker_ended"]
        session.stop_or_resume(10, detail, True)
        assert session.launch_plan()["tasks"][0]["state"] == "continue"


def test_explicit_state_path_and_waiting_report_keep_owner_and_use_note(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    actual = Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    alternate = p.root / "explicit-registry"
    shutil.copytree(actual, alternate)
    response = p.cli(
        "--state-dir",
        str(alternate),
        "workflow-report",
        "--spec",
        p.spec(
            {
                "schema_version": 1,
                "state": "waiting",
                "reason": "CI待ち",
                "resume_condition": "CI完了の証拠で明示resume",
                "evidence": "PR #100 のCIが未完了。",
            }
        ),
        extra={"XDG_STATE_HOME": str(p.root / "unused-state")},
    )
    assert json.loads(response.stdout) == {
        "schema_version": 1,
        "state": "waiting",
        "reason": "CI待ち",
        "resume_condition": "CI完了の証拠で明示resume",
    }
    assert not (p.root / "unused-state").exists()
    row = json.loads((alternate / "registry.json").read_text())["parents"]["o/r#1"]
    assert row["status"] == "active" and row["development_result"]["prs"] == []
    marker = importlib.import_module("parent.policy").events(
        p.state()["comments"]["1"], "operator", 1
    )[-1]
    assert marker["event"] == "note"
    assert "human_confirmation" not in json.dumps(marker)


def test_standalone_external_dependency_is_rechecked_without_a_fake_child(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    state = p.state()
    state["issues"]["1"].update(
        labels=[], body="## 目的\n実装する。\n\n## 依存関係\n#2\n"
    )
    state["issues"]["2"].update(state="closed", state_reason="completed")
    p.save(state)
    p.agreed()
    planned = json.loads(p.cli("status").stdout)["launch_plan"]
    assert planned["launchable_children"] == [1]
    assert planned["tasks"][0]["dependencies"] == [2]
    p.cli(
        "delegate",
        "--child",
        "1",
        "--agent",
        "codex",
        "--branch",
        "codex/parser",
        "--worktree",
        str(p.worktree("parser")),
    )
    state = p.state()
    state["issues"]["2"].update(state_reason="not_planned")
    p.save(state)
    planned = json.loads(p.cli("status").stdout)["launch_plan"]
    assert planned["launchable_children"] == []
    assert "dependency_not_completed" in {
        r["code"] for r in planned["tasks"][0]["reasons"]
    }
    assert p.state()["parents"] == {}


def test_workflow_report_rejects_duplicate_json_and_empty_parent_completion(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    spec = p.root / "report.json"
    spec.write_text(
        '{"schema_version":1,"schema_version":1,"state":"waiting","reason":"CI待ち","resume_condition":"明示resume","evidence":"未完了"}'
    )
    rejected = p.cli("workflow-report", "--spec", str(spec), ok=False)
    assert "duplicate" in rejected.stderr
    rejected = p.cli(
        "workflow-report",
        "--spec",
        p.spec(
            {
                "schema_version": 1,
                "state": "completed",
                "reason": "完了",
                "resume_condition": "なし",
                "evidence": "全体確認した",
            }
        ),
        ok=False,
    )
    assert "no implementation targets" in rejected.stderr


@pytest.mark.parametrize("gate_env", ["standalone"], indirect=True)
@pytest.mark.parametrize(
    "remote", ["success", "dry_run", "not_closed", "wrong_head", "lost_response"]
)
def test_completed_report_requires_gate_real_merge_close_and_parent_verification(
    gate_env: GateEnv, remote: str
) -> None:
    g, p = gate_env, gate_env.review.p
    p.cli(
        "record-result",
        "--child",
        "1",
        "--evidence-file",
        p.note("親がdiff・受入条件・テストとPRを確認した。"),
    )
    if remote == "dry_run":
        response = g.cli("merge", AGENT_AUTO_MERGE="0")
    else:
        response = g.cli(
            "merge",
            **(
                {"PARENT_MERGE_SKIP_CLOSE": "1"}
                if remote == "not_closed"
                else {"PARENT_MERGE_RESPONSE_LOST": "1"}
                if remote == "lost_response"
                else {}
            ),
        )
    assert response.returncode in {0, 1}, response.stderr
    if remote == "wrong_head":
        state = p.state()
        state["prs"][0]["head"]["sha"] = "f" * 40
        p.save(state)
    completed = p.cli(
        "workflow-report",
        "--spec",
        p.spec(
            {
                "schema_version": 1,
                "state": "completed",
                "reason": "受入と実merge/closeを確認した。",
                "resume_condition": "完了。追加要件は別途判断。",
                "evidence": "全体の受入条件をdiffとテストで確認。PR #100 の実mergeとissue closeを照合。",
            }
        ),
        ok=None,
    )
    assert completed.returncode == (
        0 if remote in {"success", "lost_response"} else 2
    ), completed.stderr
    if completed.returncode == 0:
        assert json.loads(completed.stdout)["state"] == "completed"
    assert len(g.merge_calls()) == (0 if remote == "dry_run" else 1)
    assert len(g.review.calls()) == 1


def bind_adoption(p: ParentEnv, target: int) -> None:
    props = importlib.import_module("opsreport.proposals")
    policy = importlib.import_module("parent.policy")
    proposal = props.Proposal(
        section="その他の助言",
        title="境界を検証する",
        evidence="fake gh",
        purpose="保守性",
        scope="parserとテスト",
        expected_effect="不正入力の拒否",
        acceptance=("境界を検証できる",),
    )
    markdown, _ = props.render_proposals([proposal], "20261019")
    body = (
        "<!-- ops-report:v2 slot=20261019 since=2026-10-16T00:00:00+00:00 until=2026-10-19T00:00:00+00:00 proposals=1 status=complete -->\n"
        + props.PROPOSALS_BEGIN
        + "\n"
        + markdown.replace("- [ ] **", "- [x] **")
        + props.PROPOSALS_END
        + "\n"
    )
    _, parsed = importlib.import_module("parent.proposals").parse_report(body)
    state = p.state()
    state["issues"]["90"] = {
        **state["issues"]["1"],
        "number": 90,
        "id": 10090,
        "title": "運用レポート",
        "body": body,
        "labels": [{"name": "ops-report"}],
    }
    state["issues"][str(target)]["body"] = p.formats.bind_task(
        state["issues"][str(target)]["body"],
        plan="a" * 64,
        node="parser",
        bindings=[
            {
                "source": {
                    "proposal_id": "R-20261019-01",
                    "report_number": 90,
                    "option": None,
                },
                "proposal_sha256": policy.digest(parsed[0].snapshot()),
            }
        ],
    )
    p.save(state)


@pytest.mark.parametrize("target", [1, 10])
def test_delegate_checks_context_adoption_on_parent_and_child(
    parent_env: ParentEnv, target: int
) -> None:
    p = parent_env
    p.agreed()
    p.child("parser")
    bind_adoption(p, target)
    state = p.state()
    state["issues"]["90"]["body"] = state["issues"]["90"]["body"].replace(
        "- [x] **", "- [ ] **"
    )
    p.save(state)
    response = p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/parser",
        "--worktree",
        str(p.worktree("parser")),
        ok=False,
    )
    assert "adoption" in response.stderr
    assert not json.loads(p.cli("status").stdout)["tasks"]["10"].get("delegated")


@pytest.mark.parametrize("target", [1, 10])
def test_merge_checks_revoked_adoptions_and_keeps_independent_prs_eligible(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch, target: int
) -> None:
    p = parent_env
    review = make_review_env(p, independent=True)
    bind_adoption(p, target)
    manifest = json.loads(
        p.cli(
            "review-prepare",
            "--prs",
            "100",
            "101",
            "102",
            "--reason-file",
            p.note("採用出典を含めて対象を固定。まだレビューは実行していない。"),
        ).stdout
    )
    review.request = json.loads(
        p.cli(
            "review-request",
            "--run-id",
            manifest["review_run_id"],
            "--instruction-file",
            p.note("この対象集合を1回でレビュー。"),
        ).stdout
    )
    review.request_path.write_text(json.dumps(review.request))
    state = p.state()
    for pr in state["prs"]:
        pr.update(
            labels=[],
            mergeable=True,
            statusCheckRollup=[
                {
                    "__typename": "CheckRun",
                    "name": "Python",
                    "workflowName": "CI",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                }
            ],
        )
    p.save(state)
    review.run()
    g = GateEnv(review, review_policy_module(monkeypatch))
    g.confirm()
    before = g.cli("merge", AGENT_AUTO_MERGE="0")
    assert all(row["decision"] == "merge" for row in g.results(before).values())
    state = p.state()
    state["issues"]["90"]["body"] = state["issues"]["90"]["body"].replace(
        "- [x] **", "- [ ] **"
    )
    p.save(state)
    after = g.cli("merge")
    rows = g.results(after)
    assert "task_adoption_changed" in rows[100]["reasons"]
    assert rows[100]["action"] == "blocked"
    if target == 10:
        assert rows[101]["action"] == rows[102]["action"] == "merged"
        assert len(g.merge_calls()) == 2
    else:
        assert g.merge_calls() == []
    assert len(review.calls()) == 1
