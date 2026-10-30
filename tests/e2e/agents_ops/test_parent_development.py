"""Managed development with offline CLIs, disposable worktrees and fake GitHub."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import signal
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
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
from tests.e2e.agents_ops.test_merge_gate import GateEnv, GateSeed, make_gate_env
from tests.e2e.agents_ops.test_merge_gate import gate_env as gate_env
from tests.e2e.agents_ops.test_parent_session import (
    ParentEnv,
    make_review_env,
    review_policy_module,
)
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


def workflow(
    d: DispatchHarness,
    monkeypatch: pytest.MonkeyPatch,
    *,
    family: bool = False,
    dependencies: dict[int, list[int]] | None = None,
) -> tuple[Any, Any, dict[str, str]]:
    """Real bootstrap/claims, with an offline parent held alive for child actions."""
    if family:
        d.prepare_family(10, 11, 12, dependencies=dependencies)
    else:
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
        d.h.gh,
        mod.workflow_store(d.h.store.directory, d.schedule.directory),
        1,
        "claude:fixture-session",
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


def launch(d: DispatchHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    d.publish_github(monkeypatch)
    d.control.launch_reserved()


@pytest.mark.parametrize("model", [None, "gpt-6-astra"])
@pytest.mark.parametrize("family", [False, True])
def test_bootstrap_delegates_standalone_sol_or_astra_and_does_not_trust_exit_zero(
    d: DispatchHarness,
    monkeypatch: pytest.MonkeyPatch,
    model: str | None,
    family: bool,
) -> None:
    mod, session, env = workflow(d, monkeypatch, family=family)
    target = 10 if family else 1
    tree = child_tree(d)
    env["PYTEST_LOCK"] = str(d.h.root / "shared-pytest.lock")
    result = mod.start_worker(
        session,
        d.schedule.directory,
        env,
        child=target,
        prompt="境界を検証しPRを作る。",
        model=model,
        branch="codex/parser",
        worktree=tree,
    )
    try:
        launch(d, monkeypatch)
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
        assert job["child"] == target
        assert d.h.gh.parents == ({10: 1, 11: 1, 12: 1} if family else {})
        prompt = (Path(job["directory"]) / "prompt.md").read_text()
        assert f"担当issue: #{target}" in prompt
        assert "pytest_locked.sh" in prompt
        assert "PYTEST_LOCK:" not in prompt
        assert "pytestは共有lockなし" in prompt
        with session.store.locked():
            state = mod.inspect_worker(session, target)
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


@pytest.mark.parametrize("cancel_fix", [False, True])
def test_existing_pr_fix_continues_same_session_and_model_after_end_evidence(
    d: DispatchHarness,
    monkeypatch: pytest.MonkeyPatch,
    cancel_fix: bool,
) -> None:
    mod, session, env = workflow(d, monkeypatch, family=True)
    tree = child_tree(d)
    first = mod.start_worker(
        session,
        d.schedule.directory,
        env,
        child=10,
        prompt="実装。",
        model="gpt-6-astra",
        branch="codex/parser",
        worktree=tree,
    )
    try:
        launch(d, monkeypatch)
        previous = drain(d, first["job_id"])
        d.h.gh.prs.append(
            {
                "number": 50,
                "state": "open",
                "body": "Closes #10",
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
                child=10,
                prompt="修正。",
                model="gpt-6.1-sol",
                continuation="終了・差分を確認した。",
            )
        with pytest.raises(ValueError, match="existing worker"):
            mod.start_worker(
                session,
                d.schedule.directory,
                env,
                child=10,
                prompt="別sessionは禁止。",
                model=None,
                branch="codex/parser",
                worktree=tree,
            )
        second = mod.start_worker(
            session,
            d.schedule.directory,
            env,
            child=10,
            prompt="指摘と回帰テストを修正。",
            model=None,
            continuation="前turn終了・同じPRの差分とHEADを確認。",
        )
        if cancel_fix:
            cancelled = d.row()["dispatch"]["jobs"][second["job_id"]]
            with session.store.locked():
                mod.stop_worker(session, 10, stop_detail())
                session.stop_or_resume(10, stop_detail(), True)
            assert d.row()["dispatch"]["jobs"][cancelled["id"]][
                "cancelled_before_start"
            ]
            assert not (Path(cancelled["directory"]) / "runtime").exists()
            corrupted = d.row()
            corrupted["dispatch"]["jobs"][cancelled["id"]]["resume_from"] = None
            d.update(corrupted)
            with pytest.raises(ValueError, match="saved resume source"):
                mod.start_worker(
                    session,
                    d.schedule.directory,
                    env,
                    child=10,
                    prompt="壊れた再開元を新sessionで置き換えない。",
                    model=None,
                    continuation="取消予約と起動済みrunを照合した。",
                )
            corrupted["dispatch"]["jobs"][cancelled["id"]]["resume_from"] = cancelled[
                "resume_from"
            ]
            d.update(corrupted)
            second = mod.start_worker(
                session,
                d.schedule.directory,
                env,
                child=10,
                prompt="未起動の修正予約を取り消した。元sessionで修正を継続する。",
                model=None,
                continuation="取消記録と元run・PR/HEADを再確認した。",
            )
        launch(d, monkeypatch)
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


def stop_detail() -> dict[str, str]:
    return {
        key: "入力を確認するまで停止。確認後に同じworkerへ戻す。"
        for key in ("reason", "facts", "impact", "resume_condition", "next_action")
    }


def test_cancelled_initial_reservation_keeps_model_and_restarts_only_if_unstarted(
    d: DispatchHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod, session, env = workflow(d, monkeypatch, family=True)
    first = mod.start_worker(
        session,
        d.schedule.directory,
        env,
        child=10,
        prompt="実装する。",
        model="gpt-6-astra",
        branch="codex/parser",
        worktree=child_tree(d),
    )
    with session.store.locked():
        mod.stop_worker(session, 10, stop_detail())
        state = mod.inspect_worker(session, 10)
        assert state["cancelled_before_start"] and state["termination_confirmed"]
        assert state["task_success"] is None and "runtime_result" not in state
        session.stop_or_resume(10, stop_detail(), True)
    assert any(
        e["event"] == "runtime_cancelled" and e["job_id"] == first["job_id"]
        for e in d.row()["dispatch"]["events"]
    )
    second = mod.start_worker(
        session,
        d.schedule.directory,
        env,
        child=10,
        prompt="未起動の取消予約を照合した。同じ担当で実装する。",
        model=None,
        continuation="実CLI未起動、PRなし、branch/worktreeを確認した。",
    )
    try:
        launch(d, monkeypatch)
        job = drain(d, second["job_id"])
        record = json.loads(
            (Path(job["directory"]) / "runtime/result.json").read_text()
        )
        assert job["resume_from"] is None
        assert record["contract"]["settings"]["selected"]["model"] == "gpt-6-astra"
        assert len((d.h.root / "runtime-calls.jsonl").read_text().splitlines()) == 1
    finally:
        d.control.reconcile(stop=True)


def test_standalone_stop_resume_does_not_reuse_changed_autonomous_basis(
    d: DispatchHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod, session, env = workflow(d, monkeypatch)
    first = mod.start_worker(
        session,
        d.schedule.directory,
        env,
        child=1,
        prompt="実装する。",
        model=None,
        branch="codex/parser",
        worktree=child_tree(d),
    )
    with session.store.locked():
        mod.stop_worker(session, 1, stop_detail())
        session.stop_or_resume(1, stop_detail(), True)
    with pytest.raises(ValueError, match="inputs changed"):
        mod.start_worker(
            session,
            d.schedule.directory,
            env,
            child=1,
            prompt="根拠の再判断が必要。",
            model=None,
            continuation="取消後の状態を確認した。",
        )
    jobs = d.row()["dispatch"]["jobs"]
    assert jobs[first["job_id"]]["cancelled_before_start"]
    assert len(jobs) == 2 and not (d.h.root / "runtime-calls.jsonl").exists()


def test_lost_runner_holds_capacity_until_cleanup_confirmation_then_resumes(
    d: DispatchHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod, session, env = workflow(d, monkeypatch, family=True)
    monkeypatch.setenv("FAKE_DISPATCH_HOLD", "1")
    first = mod.start_worker(
        session,
        d.schedule.directory,
        env,
        child=10,
        prompt="実装する。",
        model=None,
        branch="codex/parser",
        worktree=child_tree(d),
    )
    job = d.row()["dispatch"]["jobs"][first["job_id"]]
    directory = Path(job["directory"]) / "runtime"

    def initialized() -> bool:
        path = directory / "result.json"
        return path.exists() and bool(
            json.loads(path.read_text())["runtime"]["session_id"]
        )

    try:
        launch(d, monkeypatch)
        wait_until(initialized)
        record = json.loads((directory / "result.json").read_text())
        assert d.mod.alive(record["runner"])
        os.kill(record["runner"]["pid"], signal.SIGKILL)
        wait_until(
            lambda: (
                not d.mod.alive(d.row()["dispatch"]["jobs"][first["job_id"]]["worker"])
            )
        )
        with session.store.locked():
            mod.stop_worker(session, 10, stop_detail())
        d.control.reconcile()
        interrupted = json.loads((directory / "result.json").read_text())
        assert interrupted["runner_lost"] and not interrupted["cleanup_complete"]
        assert d.mod.children_pending(d.row()) == [10]
        with pytest.raises(ValueError, match="termination"):
            mod.start_worker(
                session,
                d.schedule.directory,
                env,
                child=10,
                prompt="回収未確認のまま進めない。",
                model=None,
                continuation="runnerの喪失を確認。全子孫の回収は未確認。",
            )
        confirmation = d.h.root / "supervisor-cleanup.json"
        confirmation.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    **{k: interrupted[k] for k in ("run_id", "runner", "process")},
                    "cleanup_complete": True,
                    "evidence": "fixture CLIはtoolを起動せず、保存したCLI/groupとworkerの終了を確認した。",
                }
            )
        )
        confirmed = subprocess.run(
            [
                job["runner"],
                "stop",
                "--run-dir",
                str(directory),
                "--cleanup-confirmation",
                str(confirmation),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert confirmed.returncode == 0, confirmed.stderr
        drain(d, first["job_id"])
        recovered = json.loads((directory / "result.json").read_text())
        assert recovered["cleanup_complete"] and "recovery_error" not in recovered
        with session.store.locked():
            session.stop_or_resume(10, stop_detail(), True)
        second = mod.start_worker(
            session,
            d.schedule.directory,
            env,
            child=10,
            prompt="確認済みの元sessionへ戻す。",
            model=None,
            continuation="全子孫の回収、元run/session、差分を照合した。",
        )
        monkeypatch.delenv("FAKE_DISPATCH_HOLD")
        launch(d, monkeypatch)
        resumed = drain(d, second["job_id"])
        result = json.loads(
            (Path(resumed["directory"]) / "runtime/result.json").read_text()
        )
        assert result["result"] == "completed"
        assert result["runtime"]["session_id"] == recovered["runtime"]["session_id"]
        assert len((d.h.root / "runtime-calls.jsonl").read_text().splitlines()) == 2
    finally:
        d.control.reconcile(stop=True)


def test_stop_does_not_release_a_reservation_with_launch_evidence(
    d: DispatchHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod, session, env = workflow(d, monkeypatch)
    first = mod.start_worker(
        session,
        d.schedule.directory,
        env,
        child=1,
        prompt="実装する。",
        model=None,
        branch="codex/parser",
        worktree=child_tree(d),
    )
    job = d.row()["dispatch"]["jobs"][first["job_id"]]
    (Path(job["directory"]) / "runner.stdout").touch()
    with session.store.locked():
        mod.stop_worker(session, 1, stop_detail())
    row = d.row()
    current = row["dispatch"]["jobs"][first["job_id"]]
    assert current["phase"] == row["dispatch"]["phase"] == "recovery_required"
    assert not current.get("cancelled_before_start")
    assert not row["tasks"]["1"].get("worker_ended")
    assert not (d.h.root / "runtime-calls.jsonl").exists()


@pytest.mark.parametrize("change", ["requirements", "stopped"])
def test_worker_claim_rechecks_workflow_reservation_before_running_cli(
    d: DispatchHarness,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    mod, session, env = workflow(d, monkeypatch)
    first = mod.start_worker(
        session,
        d.schedule.directory,
        env,
        child=1,
        prompt="実装する。",
        model=None,
        branch="codex/parser",
        worktree=child_tree(d),
    )
    if change == "requirements":
        d.h.gh.issues[1]["body"] += "\n## 追加条件\n\n受入条件を変更した。\n"
    else:
        with session.store.locked():
            session.stop_or_resume(1, stop_detail(), False)
    launch(d, monkeypatch)
    job = drain(d, first["job_id"])
    assert job["cancelled_before_start"] and job["diagnostic"]
    assert any(
        e["event"] == "runtime_cancelled" and e["job_id"] == first["job_id"]
        for e in d.row()["dispatch"]["events"]
    )
    assert not (d.h.root / "runtime-calls.jsonl").exists()


@pytest.mark.parametrize("loss", ["missing", "empty"])
def test_workflow_cli_never_recreates_registry_with_dispatcher_history(
    d: DispatchHarness,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    loss: str,
) -> None:
    _, _, env = workflow(d, monkeypatch)
    cli = importlib.import_module("parent.cli")
    monkeypatch.setattr(cli, "GitHub", lambda repo, **kwargs: d.h.gh)
    for name in ("CODEX_SESSION_ID", "CODEX_THREAD_ID", "CLAUDE_CODE_SESSION_ID"):
        monkeypatch.delenv(name, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    tree = child_tree(d)
    prompt = d.h.root / "task.md"
    prompt.write_text("実装する。")
    report = d.h.root / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "waiting",
                "reason": "状態確認待ち",
                "resume_condition": "担当を復元する",
                "evidence": "registryの復旧待ち。",
            }
        )
    )
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
    original_schedule = d.schedule.path.read_bytes()
    if loss == "missing":
        d.h.store.path.unlink()
        damaged = None
    else:
        damaged = b'{"version":1,"parents":{}}'
        d.h.store.path.write_bytes(damaged)
    for command in (
        [
            "start-worker",
            "--child",
            "1",
            "--branch",
            "codex/parser",
            "--worktree",
            str(tree),
            "--prompt-file",
            str(prompt),
        ],
        ["workflow-report", "--spec", str(report)],
    ):
        assert cli.main([*common, *command]) == 2
        assert "registry" in capsys.readouterr().err
        assert (
            d.h.store.path.read_bytes() if d.h.store.path.exists() else None
        ) == damaged
        assert d.schedule.path.read_bytes() == original_schedule
    assert not (d.h.root / "runtime-calls.jsonl").exists()


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
    original_get = d.h.gh.get
    operations = []
    operation = cli.Store.operation

    @contextmanager
    def traced_operation(store: Any, key: str) -> Iterator[None]:
        operations.append(key)
        with operation(store, key):
            yield

    def bound_github(repo: str, *, store: Any) -> Any:
        def get(endpoint: str, *, missing_parent: bool = False) -> object:
            with store.unlocked_read():
                return original_get(endpoint, missing_parent=missing_parent)

        monkeypatch.setattr(d.h.gh, "get", get)
        return d.h.gh

    monkeypatch.setattr(cli, "GitHub", bound_github)
    monkeypatch.setattr(cli.Store, "operation", traced_operation)
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
    assert operations == ["o/r#1"]
    assert cli.main([*common, "inspect-worker", "--child", "1"]) == 0
    assert operations == ["o/r#1"]
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
    assert operations == ["o/r#1", "o/r#1"]
    assert json.loads(capsys.readouterr().out)["result"] == "stop_requested"
    assert d.row()["tasks"]["1"]["worker_ended"]
    assert not (d.h.root / "unused-state").exists()
    assert not (d.h.root / "runtime-calls.jsonl").exists()


@pytest.mark.parametrize("change", ["requirements", "adoption"])
def test_changed_inputs_after_delegation_keep_claim_but_do_not_reserve_runtime(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    mod, session, env = workflow(d, monkeypatch)
    delegate = session.delegate

    def changed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        claim = cast(dict[str, Any], delegate(*args, **kwargs))
        if change == "requirements":
            d.h.gh.issues[1]["body"] += (
                "\n## 追加の受入条件\n\n境界の検証を追加する。\n"
            )
        else:
            report, bindings = adoption_source()
            d.h.gh.issues[90] = {
                **d.h.gh.issues[1],
                "number": 90,
                "id": 10090,
                "body": report.replace("- [x] **", "- [ ] **"),
                "labels": [{"name": "ops-report"}],
            }
            formats = importlib.import_module("issues")
            d.h.gh.issues[1]["body"] = formats.bind_task(
                d.h.gh.issues[1]["body"],
                plan="a" * 64,
                node="parser",
                bindings=bindings,
            )
        return claim

    monkeypatch.setattr(session, "delegate", changed)
    with pytest.raises(ValueError, match="inputs changed|adoption"):
        mod.start_worker(
            session,
            d.schedule.directory,
            env,
            child=1,
            prompt="予約後も根拠を確認してから実装する。",
            model=None,
            branch="codex/parser",
            worktree=child_tree(d),
        )
    row = d.row()
    assert row["tasks"]["1"]["delegated"] is True
    assert all(j["role"] == "parent" for j in row["dispatch"]["jobs"].values())
    assert not (d.h.root / "runtime-calls.jsonl").exists()


def test_stop_propagates_only_to_dependencies_and_reserved_worker_is_cancelled(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    p.agreed()
    for key, deps in (("upstream", []), ("dependent", [10]), ("independent", [])):
        p.child(key, deps)
    for n in (10, 11, 12):
        if n == 11:
            state = p.state()
            state["prs"].append(p.pr(50, 10, branch="codex/worker-10"))
            p.save(state)
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
                "worker": None,
                "runner_exit": None,
                "resume_from": None,
            }
            for n in (10, 11, 12)
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
            "worker": None,
            "runner_exit": None,
            "resume_from": None,
        }
        session.row["dispatch"] = {
            "schema_version": 1,
            "assignment_id": "fixture",
            "session": identity,
            "authorization": authorization,
            "phase": "running",
            "child_limit": 3,
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
        assert jobs["11"]["phase"] == "ended"
        assert session.row["tasks"]["11"]["worker_ended"]
        assert session.row["tasks"]["10"]["worker_ended"]
        session.stop_or_resume(10, detail, True)
        assert session.launch_plan()["tasks"][0]["state"] == "continue"


def test_stop_terminates_running_dependency_but_keeps_independent_worker(
    d: DispatchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod, session, env = workflow(d, monkeypatch, family=True, dependencies={11: [10]})
    upstream = mod.start_worker(
        session,
        d.schedule.directory,
        env,
        child=10,
        prompt="先行実装。",
        model=None,
        branch="codex/upstream",
        worktree=child_tree(d, "upstream"),
    )
    try:
        launch(d, monkeypatch)
        drain(d, upstream["job_id"])
        d.h.gh.prs.append(
            {
                "number": 50,
                "state": "open",
                "body": "Closes #10",
                "merged_at": None,
                "head": {
                    "ref": "codex/upstream",
                    "sha": d.h.git("rev-parse", "HEAD"),
                    "repo": {"full_name": "o/r"},
                },
            }
        )
        monkeypatch.setenv("FAKE_DISPATCH_HOLD", "1")
        reservations = {
            n: mod.start_worker(
                session,
                d.schedule.directory,
                env,
                child=n,
                prompt="実装を継続。",
                model=None,
                branch=f"codex/worker-{n}",
                worktree=child_tree(d, f"worker-{n}"),
            )
            for n in (11, 12)
        }
        launch(d, monkeypatch)
        jobs = {
            n: d.row()["dispatch"]["jobs"][r["job_id"]] for n, r in reservations.items()
        }
        paths = {
            n: Path(j["directory"]) / "runtime/result.json" for n, j in jobs.items()
        }

        def running() -> bool:
            return all(
                p.exists() and json.loads(p.read_text())["runtime"]["session_id"]
                for p in paths.values()
            )

        wait_until(running)
        processes = {n: json.loads(p.read_text())["process"] for n, p in paths.items()}
        with session.store.locked():
            result = mod.stop_worker(session, 10, stop_detail())
            states = {t["child"]: t["state"] for t in result["launch_plan"]["tasks"]}
            assert states == {10: "stopped", 11: "held", 12: "continue"}
            # A stop request never manufactures end evidence or frees a slot.
            assert d.mod.children_pending(session.row) == [11, 12]
            assert "worker_ended" not in session.row["tasks"]["11"]
        drain(d, reservations[11]["job_id"])
        assert not d.mod.alive(processes[11])
        assert d.mod.alive(processes[12])
        assert d.mod.children_pending(d.row()) == [12]
        assert json.loads(paths[11].read_text())["cleanup_complete"]
        assert json.loads(paths[12].read_text())["result"] == "running"
        (Path(jobs[12]["directory"]) / "release").touch()
        drain(d, reservations[12]["job_id"])
        assert json.loads(paths[12].read_text())["result"] == "completed"
    finally:
        d.control.reconcile(stop=True)


COMPLETION_SEED = pytest.StashKey[GateSeed]()


@pytest.fixture
def completed_gate_env(
    parent_env: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> GateEnv:
    # One real merged implementation plus one cancelled sibling exercises the
    # completion/cancellation interaction; independent merge ordering is separate.
    if COMPLETION_SEED not in request.config.stash:
        gate = make_gate_env(parent_env, monkeypatch)
        cache = request.getfixturevalue("tmp_path_factory").mktemp("completion-seed")
        request.config.stash[COMPLETION_SEED] = GateSeed.capture(
            gate, cache / "snapshot"
        )
        return gate
    gate = request.config.stash[COMPLETION_SEED].restore(parent_env, monkeypatch)
    parent_env.calls_path.write_text("")
    return gate


@pytest.mark.parametrize(
    "change", ["none", "missing", "body", "state_reason", "open_pr", "worker"]
)
def test_completed_parent_distinguishes_confirmed_cancellation_from_success(
    completed_gate_env: GateEnv, change: str
) -> None:
    g, p = completed_gate_env, completed_gate_env.review.p
    p.child("old-plan")
    state = p.state()
    state["issues"]["11"].update(state="closed", state_reason="not_planned")
    p.save(state)
    p.cli("sync")
    for n in (10,):
        p.cli(
            "record-result",
            "--child",
            str(n),
            "--evidence-file",
            p.note("親が受入条件・差分・テストを確認した。"),
        )
    spec = {
        "schema_version": 1,
        "reason": "旧案を代替実装へ統合した。",
        "handoff": "残作業は #10 へ引き継いだ。",
        "evidence": "各issueの範囲・受入条件と旧案との差分を親が照合した。",
    }
    if change != "missing":
        receipt = p.cli("record-cancellation", "--child", "11", "--spec", p.spec(spec))
        assert json.loads(receipt.stdout)["state"] == "cancelled"
    merged = g.cli("merge")
    assert merged.returncode == 0, merged.stderr + merged.stdout
    assert len(g.merge_calls()) == 1
    state = p.state()
    if change == "body":
        state["issues"]["11"]["body"] += "\n\n引継ぎ先が変更された。\n"
    elif change == "state_reason":
        state["issues"]["11"]["state_reason"] = "duplicate"
    elif change == "open_pr":
        state["prs"].append(p.pr(200, 11))
    p.save(state)
    directory = Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    if change == "worker":
        registry = directory / "registry.json"
        data = json.loads(registry.read_text())
        data["parents"]["o/r#1"]["tasks"]["11"].update(
            delegated=True, delegation_id="unfinished-cancelled-worker"
        )
        registry.write_text(json.dumps(data))
    result = p.cli(
        "workflow-report",
        "--spec",
        p.spec(
            {
                "schema_version": 1,
                "state": "completed",
                "reason": "全体受入を確認した。",
                "resume_condition": "完了。追加要件は別途判断。",
                "evidence": "実merge/closeと取消理由・引継ぎ先を照合した。",
            }
        ),
        ok=None,
    )
    assert result.returncode == (0 if change == "none" else 2), result.stderr
    if change == "none":
        report = json.loads(result.stdout)
        assert set(report) == {"schema_version", "state", "reason", "resume_condition"}
        assert report["state"] == "completed"
        row = json.loads((directory / "registry.json").read_text())["parents"]["o/r#1"]
        assert row["development_result"]["cancelled"] == [11]
        assert row["development_result"]["prs"] == [100]
        assert "- [ ] #11 old-plan（取消）" in p.state()["issues"]["1"]["body"]
    elif change == "worker":
        assert "termination" in result.stderr
    else:
        assert "cancellation" in result.stderr


@pytest.mark.parametrize("problem", ["handoff", "worker", "open", "all_cancelled"])
def test_cancellation_confirmation_never_hides_unresolved_work(
    parent_env: ParentEnv, problem: str
) -> None:
    p = parent_env
    p.agreed()
    p.child("old-plan")
    if problem == "worker":
        p.cli(
            "delegate",
            "--child",
            "10",
            "--agent",
            "codex",
            "--branch",
            "codex/old-plan",
            "--worktree",
            str(p.worktree("old-plan")),
        )
    state = p.state()
    if problem != "open":
        state["issues"]["10"].update(state="closed", state_reason="duplicate")
    p.save(state)
    spec = {
        "schema_version": 1,
        "reason": "既存の実装と重複した。",
        "handoff": "残作業なし。既存実装との差分と受入条件を確認した。",
        "evidence": "親が本文と終了を確認した。",
    }
    if problem == "handoff":
        spec["handoff"] = ""
    result = p.cli(
        "record-cancellation", "--child", "10", "--spec", p.spec(spec), ok=None
    )
    assert result.returncode == (0 if problem == "all_cancelled" else 2)
    if problem == "all_cancelled":
        result = p.cli(
            "workflow-report",
            "--spec",
            p.spec(
                {
                    "schema_version": 1,
                    "state": "completed",
                    "reason": "取消だけ。",
                    "resume_condition": "要件を再判断する。",
                    "evidence": "実装完了はない。",
                }
            ),
            ok=False,
        )
        assert "cancellation is not success" in result.stderr


@pytest.mark.parametrize("state", ["waiting", "failed"])
def test_explicit_state_path_and_report_keep_owner_and_use_note(
    parent_env: ParentEnv, state: str
) -> None:
    p = parent_env
    p.agreed()
    p.child("parser")
    actual = Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    alternate = p.root / "explicit-registry"
    shutil.copytree(actual, alternate)
    reason = "CI待ち" if state == "waiting" else "検証失敗"
    response = p.cli(
        "--state-dir",
        str(alternate),
        "workflow-report",
        "--spec",
        p.spec(
            {
                "schema_version": 1,
                "state": state,
                "reason": reason,
                "resume_condition": "CI完了の証拠で明示resume",
                "evidence": "PR #100 のCIが未完了。",
            }
        ),
        extra={"XDG_STATE_HOME": str(p.root / "unused-state")},
    )
    assert json.loads(response.stdout) == {
        "schema_version": 1,
        "state": state,
        "reason": reason,
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
    # Workflow progress must leave the agreement usable for the next action.
    result = p.cli(
        "--state-dir",
        str(alternate),
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/parser",
        "--worktree",
        str(p.worktree("parser")),
    )
    assert json.loads(result.stdout)["action"] == "launch"


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


def adoption_source() -> tuple[str, list[dict[str, Any]]]:
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
    return body, [
        {
            "source": {
                "proposal_id": "R-20261019-01",
                "report_number": 90,
                "option": None,
            },
            "proposal_sha256": policy.digest(parsed[0].snapshot()),
        }
    ]


def bind_adoption(p: ParentEnv, target: int) -> None:
    body, bindings = adoption_source()
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
        bindings=bindings,
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
    if target == 1:
        p.agreed()  # Freeze the fixture's provenance before revoking its adoption.
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

    def configure_adoption(p: ParentEnv) -> None:
        bind_adoption(p, target)
        if target == 1:
            p.agreed()
        for child in (10, 11, 12):
            p.cli(
                "record-result",
                "--child",
                str(child),
                "--evidence-file",
                p.note("出典を含む現在の要件で差分・テスト・PRを確認した。"),
            )

    review = make_review_env(p, independent=True, before_review=configure_adoption)
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
