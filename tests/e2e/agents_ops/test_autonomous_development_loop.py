"""Cross-component development loops with the real runner, gate and disposable git."""

from __future__ import annotations

import importlib
import json
import os
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest

from tests.e2e.agents_ops.test_dispatcher_judgement import decision, issue
from tests.e2e.agents_ops.test_dispatcher_supervisor import wait_until
from tests.e2e.agents_ops.test_merge_gate import GateEnv
from tests.e2e.agents_ops.test_parent_session import (
    OPS,
    ParentEnv,
    request_review,
    review_policy_module,
    run_git,
)
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


class DevelopmentLoop:
    """One GitHub file and registry from scheduled collection to assignment end."""

    def __init__(self, p: ParentEnv, monkeypatch: pytest.MonkeyPatch) -> None:
        self.p = p
        self.mod = importlib.import_module("dispatcher.supervisor")
        self.service = importlib.import_module("dispatcher.service")
        self.clock = importlib.import_module("dispatcher.schedule")
        self.development = importlib.import_module("parent.development")
        self.schedule = self.clock.Schedule(p.root / "dispatcher")
        self.store = self.development.workflow_store(
            Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions",
            self.schedule.directory,
        )
        # Dispatch collection needs a code source; review needs the fixed base rules.
        (p.repo / "README.md").write_text("# Parser\nStrict input boundaries.\n")
        (p.repo / "AGENTS.md").write_text("# Rules\nReject invalid inputs.\n")
        run_git(p.repo, "add", ".")
        run_git(p.repo, "commit", "-m", "fixture rules")
        for key in list(os.environ):
            if key.startswith(
                ("HERDR_", "FAKE_", "OPS_REPORT_", "AGENT_RUNTIME_")
            ) or key in {
                "CODEX_THREAD_ID",
                "CODEX_SESSION_ID",
                "CLAUDE_CODE_SESSION_ID",
                "CLAUDE_CODE_PARENT_SESSION_ID",
                "CLAUDE_CODE_AGENT_ID",
            }:
                monkeypatch.delenv(key, raising=False)
        p.env = {
            k: v
            for k, v in p.env.items()
            if not k.startswith(("FAKE_", "OPS_REPORT_", "AGENT_RUNTIME_"))
            and k != "CLAUDE_CODE_SESSION_ID"
        }
        p.env.update(
            CODEX_HOME=str(p.root / "codex-home"),
            CLAUDE_CONFIG_DIR=str(p.root / "claude-home"),
            FAKE_DISPATCH_CALLS=str(p.root / "runtime-calls.jsonl"),
            FAKE_DISPATCH_HOLD="1",
            FAKE_JUDGEMENT_RESPONSE=str(p.root / "judgement.json"),
            FAKE_JUDGEMENT_CALLS=str(p.root / "judgement-calls.jsonl"),
            FAKE_ORGANIZATION_PLAN=str(p.root / "organization.json"),
            FAKE_ORGANIZATION_CALLS=str(p.root / "organization-calls.jsonl"),
            AGENT_AUTO_MERGE="0",
        )
        for key, value in p.env.items():
            monkeypatch.setenv(key, value)
        # All three fake CLIs still run inside the production agent_exec.sh.
        for name in ("claude", "codex"):
            path = p.root / "bin" / name
            path.write_text(
                f"#!{sys.executable}\n"
                "import os, runpy\nfrom pathlib import Path\n"
                "task = Path(os.environ['AGENT_RUNTIME_RUN_DIR']).parent\n"
                "name = ('fake_judgement_cli.py' if (task / 'agent-input/index.json').exists()\n"
                "        else 'fake_organization_cli.py' if (task / 'organization-input.json').exists()\n"
                "        else 'fake_dispatch_cli.py')\n"
                f"module = runpy.run_path(str(Path({str(Path(__file__).parent)!r}) / name))\n"
                "module['main']()\n"
            )
            path.chmod(0o700)
        monkeypatch.chdir(p.repo)
        self.gh = importlib.import_module("parent.github").GitHub("o/r")
        with self.store.locked():
            self.store.save()
            self.schedule.initialize(
                {
                    "repo": "o/r",
                    "cwd": str(p.repo),
                    "runner": str(OPS / "lib/agent_exec.sh"),
                    "max_parents": 1,
                    "max_children": 2,
                    "judgement_timeout": 20,
                },
                self.clock.local_time("2026-10-24T07:00:00+09:00"),
            )
            state = self.schedule.read()
            state["enabled"] = True
            self.schedule.write(state)
        self.restart()

    def restart(self) -> None:
        self.store = self.development.workflow_store(
            self.store.directory, self.schedule.directory
        )
        self.control = self.mod.Supervisor(self.gh, self.store, self.schedule)

    def inputs(
        self,
        children: tuple[int, ...] = (),
        *,
        second: bool = False,
        dependencies: dict[int, list[int]] | None = None,
    ) -> None:
        state = self.p.state()
        state.update(issues={"1": issue(1)}, parents={})
        if children:
            render = importlib.import_module("issues.cli").render_spec
            for child in children:
                sections = {
                    h: h + "を検証する。" for h in self.p.formats.CHILD_REQUIRED
                }
                sections["依存関係"] = (
                    "\n".join(f"- #{n}" for n in (dependencies or {}).get(child, []))
                    or "なし。"
                )
                sections["Acceptance checklist"] = (
                    "- [ ] 境界を検証する\n  文脈: parser。\n  下流への影響: 整合。\n  検証方法・証拠: CPU試験。"
                )
                rendered = render(
                    {
                        "role": "child",
                        "summary": f"parser {child}",
                        "parent": {"number": 1, "identifier": "開発"},
                        "sections": sections,
                    }
                )
                state["issues"][str(child)] = {**issue(child), **rendered}
                state["parents"][str(child)] = 1
            sections = {
                h: h + "を検証する。"
                for h in self.p.formats.PARENT_REQUIRED
                if h != self.p.formats.TASKS_HEADING
            }
            rendered = render(
                {
                    "role": "parent",
                    "identifier": "開発",
                    "summary": "parser改善",
                    "sections": sections,
                    "children": [
                        {"number": n, "title": state["issues"][str(n)]["title"]}
                        for n in children
                    ],
                }
            )
            state["issues"]["1"].update(rendered)
        if second:
            state["issues"]["2"] = issue(2)
        for key, row in state["issues"].items():
            row.update(
                id=10000 + int(key),
                labels=[
                    {"name": v} if isinstance(v, str) else v for v in row["labels"]
                ],
            )
        self.p.save(state)
        self.response(
            decision(1),
            *(
                decision(
                    n,
                    "existing_parent",
                    target=1,
                    deps=tuple((dependencies or {}).get(n, [])),
                )
                for n in children
            ),
            *([decision(2, priority=2)] if second else []),
        )

    def response(self, *rows: dict[str, Any]) -> None:
        for row in rows:
            if row["requirements"] is not None:
                row["requirements"]["scope"] = (
                    "parserとCPU試験。親が実装・レビュー・mergeを照合する。"
                )
        Path(self.p.env["FAKE_JUDGEMENT_RESPONSE"]).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "input_sha256": "INPUT_DIGEST",
                    "decisions": rows,
                },
                ensure_ascii=False,
            )
        )

    def due(self, now: str = "2026-10-24T08:00:00+09:00") -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self.service.run_due(
                self.gh, self.store, self.schedule, self.clock.local_time(now)
            ),
        )

    def schedule_state(self) -> dict[str, Any]:
        with self.store.locked():
            return cast(dict[str, Any], self.schedule.read())

    def row(self, parent: int = 1) -> dict[str, Any]:
        with self.store.locked():
            return cast(
                dict[str, Any],
                json.loads(json.dumps(self.store.parents[f"o/r#{parent}"])),
            )

    def job(self, identifier: str | None = None) -> dict[str, Any]:
        dispatch = self.row()["dispatch"]
        return cast(
            dict[str, Any], dispatch["jobs"][identifier or dispatch["active_job"]]
        )

    def record(self, job: dict[str, Any]) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            json.loads((Path(job["directory"]) / "runtime/result.json").read_text()),
        )

    def calls(self, kind: str = "runtime") -> list[dict[str, Any]]:
        path = self.p.root / f"{kind}-calls.jsonl"
        return (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )

    def running(self, job: dict[str, Any]) -> None:
        def initialized() -> bool:
            path = Path(job["directory"]) / "runtime/result.json"
            return path.exists() and bool(self.record(job)["runtime"]["session_id"])

        wait_until(initialized)

    def ended(self, job: dict[str, Any]) -> None:
        def terminal() -> bool:
            self.control.reconcile()
            return bool(self.job(job["id"])["phase"] == "ended")

        wait_until(terminal, timeout=20)
        assert self.record(job)["cleanup_complete"]

    def release(self, job: dict[str, Any], final: dict[str, Any] | None = None) -> None:
        if final is not None:
            (Path(job["directory"]) / "final.json").write_text(json.dumps(final))
        (Path(job["directory"]) / "release").touch()
        self.ended(job)

    def accept(self, job: dict[str, Any]) -> None:
        self.running(job)
        env = {
            "CLAUDE_CODE_SESSION_ID": self.record(job)["runtime"]["session_id"],
            "AGENT_RUNTIME_RUN_DIR": str(Path(job["directory"]) / "runtime"),
            "AGENT_RUNTIME_TASK_ID": job["task_id"],
        }
        self.control.accept(
            json.loads((Path(job["directory"]) / "input.json").read_text()), env
        )
        self.p.env.update(env)

    def begin(self) -> dict[str, Any]:
        result = self.due()
        assert result["result"] == "recorded", self.schedule_state()["runs"][-1]
        self.control.assign()
        job = self.job()
        self.control.launch_reserved()
        self.accept(job)
        return job

    def start(
        self, child: int = 1, model: str | None = None, *, base: str | None = None
    ) -> dict[str, Any]:
        name = f"parser-{child}"
        tree = self.p.repo / ".claude/worktrees" / name
        run_git(
            self.p.repo,
            "worktree",
            "add",
            "-b",
            f"codex/{name}",
            str(tree),
            *([base] if base else []),
        )
        result = json.loads(
            self.p.cli(
                "--dispatcher-state-dir",
                str(self.schedule.directory),
                "start-worker",
                "--child",
                str(child),
                "--branch",
                f"codex/{name}",
                "--worktree",
                str(tree),
                "--prompt-file",
                self.p.note("境界を実装・検証する。"),
                *(["--model", model] if model else []),
            ).stdout
        )
        job = self.job(result["job_id"])
        self.control.launch_reserved()
        self.running(job)
        return job

    def publish_pr(self, child: int = 1) -> Path:
        tree = self.p.repo / f".claude/worktrees/parser-{child}"
        (tree / "change.txt").write_text("boundary implementation\n")
        run_git(tree, "add", ".")
        run_git(tree, "commit", "-m", "implement boundary")
        state = self.p.state()
        pr = self.p.pr(
            100, child, run_git(tree, "rev-parse", "HEAD"), f"codex/parser-{child}"
        )
        pr.update(labels=[], mergeable=True, statusCheckRollup=[])
        state["prs"].append(pr)
        self.p.save(state)
        return tree

    def verify(self, child: int = 1) -> None:
        inspected = json.loads(
            self.p.cli(
                "--dispatcher-state-dir",
                str(self.schedule.directory),
                "inspect-worker",
                "--child",
                str(child),
            ).stdout
        )
        assert inspected["termination_confirmed"] and inspected["task_success"] is None
        self.p.cli(
            "record-result",
            "--child",
            str(child),
            "--evidence-file",
            self.p.note("親が実diff・受入条件・CPU試験の結果を確認した。"),
        )

    def report(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            json.loads(
                self.p.cli(
                    "--dispatcher-state-dir",
                    str(self.schedule.directory),
                    "workflow-report",
                    "--spec",
                    self.p.spec(
                        {
                            "schema_version": 1,
                            "state": "completed",
                            "reason": "全体受入を確認した。",
                            "resume_condition": "完了。追加要件は別途判断する。",
                            "evidence": "実merge/close・差分・試験・レビュー・worker終了と残課題を照合した。",
                        }
                    ),
                ).stdout
            ),
        )


@pytest.fixture
def loop(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> Iterator[DevelopmentLoop]:
    flow = DevelopmentLoop(parent_env, monkeypatch)
    try:
        yield flow
    finally:
        # Stop only workers belonging to this test's private registry, even on failure.
        with flow.store.locked():
            jobs = [
                j
                for r in flow.store.parents.values()
                if "dispatch" in r
                for j in r["dispatch"]["jobs"].values()
            ]
        if any(j["phase"] != "ended" or flow.mod.alive(j.get("worker")) for j in jobs):
            flow.control.reconcile(stop=True)
        wait_until(
            lambda: not any(flow.mod.alive(j.get("worker")) for j in jobs), timeout=20
        )


@pytest.mark.parametrize(
    ("model", "interrupt_after_close"), [(None, False), ("gpt-6-astra", True)]
)
def test_scheduled_development_review_fix_merge_and_assignment_end(
    loop: DevelopmentLoop,
    monkeypatch: pytest.MonkeyPatch,
    model: str | None,
    interrupt_after_close: bool,
) -> None:
    f, p = loop, loop.p
    f.inputs()
    parent = f.begin()
    assert len(f.calls("judgement")) == 1
    worker = f.start(model=model)
    tree = f.publish_pr()
    f.release(worker)
    f.verify()
    review = request_review(p, [100])
    result = json.loads(review.output.read_text())
    result["verdict"] = "request_changes"
    result["prs"][0]["findings"] = [
        {
            "severity": "major",
            "path": "change.txt",
            "line": 1,
            "rule": "正しさ",
            "detail": "入力境界を修正する。",
        }
    ]
    review.output.write_text(json.dumps(result))
    review.run()
    p.cli(
        "review-state",
        "--run-id",
        review.request["review_run_id"],
        "--state",
        "result_received",
        "--evidence-file",
        p.note("親がmajor指摘を確認した。"),
    )
    gate = GateEnv(review, review_policy_module(monkeypatch))
    blocked = gate.cli("merge")
    assert gate.results(blocked)[100]["decision"] == "blocked"
    assert gate.merge_calls() == []
    continued = json.loads(
        p.cli(
            "--dispatcher-state-dir",
            str(f.schedule.directory),
            "continue-worker",
            "--child",
            "1",
            "--prompt-file",
            p.note("指摘の境界を修正し回帰試験する。"),
            "--evidence-file",
            p.note("終了・既存PRとHEAD・指摘・今回の修正範囲を照合した。"),
        ).stdout
    )
    fix = f.job(continued["job_id"])
    f.control.launch_reserved()
    f.running(fix)
    (tree / "change.txt").write_text("fixed boundary\n")
    (tree / "regression.txt").write_text("invalid input rejected\n")
    run_git(tree, "add", ".")
    run_git(tree, "commit", "-m", "fix reviewed boundary")
    head = run_git(tree, "rev-parse", "HEAD")
    state = p.state()
    state["prs"][0]["head"]["sha"] = head
    p.save(state)
    f.release(fix)
    assert (
        f.record(worker)["runtime"]["session_id"]
        == f.record(fix)["runtime"]["session_id"]
    )
    assert (
        continued["delegation_id"]
        != f.row()["dispatch"]["jobs"][worker["id"]]["delegation_id"]
    )
    f.verify()
    proof = p.spec(
        {
            "ci": "CIは未実行の観測値。",
            "regression_tests": "CPU境界試験成功。",
            "resolution": "修正差分と回帰結果を親が確認した。",
            "reviewed_paths": ["change.txt", "regression.txt"],
        }
    )
    review.cli(
        "resolve",
        "--run",
        review.request["review_run_id"],
        "--pr",
        "100",
        "--fixed-head",
        head,
        "--finding",
        "pr-100-1",
        "--evidence",
        proof,
    )
    gate.confirm()
    assert len(review.calls()) == 1
    dry = gate.cli("merge", AGENT_AUTO_MERGE="0")
    assert gate.results(dry)[100]["decision"] == "merge" and gate.merge_calls() == []
    merged = gate.cli(
        "merge",
        **({"PARENT_MERGE_RESPONSE_LOST": "1"} if interrupt_after_close else {}),
    )
    assert merged.returncode == 0, merged.stderr + merged.stdout
    assert len(gate.merge_calls()) == 1
    assert p.state()["issues"]["1"]["state_reason"] == "completed"
    identity, generation = f.row()["session"], f.row()["generation"]
    if interrupt_after_close:
        f.control.reconcile(stop=True, parent=1)
        f.ended(parent)
        f.restart()
        f.control.resume(1, "実merge/close後に中断。保存証拠を同じ親で完了照合する。")
        f.control.reconcile()
        parent = f.job()
        f.control.launch_reserved()
        f.accept(parent)
        assert "--resume" in f.calls()[-1]["args"]
    final = f.report()
    f.release(parent, final)
    assert f.row()["status"] == "ended" and f.row()["dispatch"]["phase"] == "completed"
    assert (f.row()["session"], f.row()["generation"]) == (identity, generation)
    assert f.row()["development_result"]["prs"] == [100]
    for job in (worker, fix):
        record = f.record(job)
        settings = record["contract"]["settings"]
        assert settings["defaults"]["model"] == "gpt-6.1-sol"
        assert settings["overrides"] == ({"model": model} if model else {})
        assert settings["selected"]["model"] == (model or "gpt-6.1-sol")
        assert settings["selected"]["effort"] == "max"
        assert (
            record["runtime"]["effective"]["model"]["value"]
            == settings["selected"]["model"]
        )
        assert record["runtime"]["effective"]["effort"]["value"] == "max"
        assert record["task_success"] is None
    parent_record = f.record(parent)
    assert parent_record["contract"]["settings"]["defaults"] == {
        "model": "claude-opus-5-5",
        "effort": "xhigh",
    }
    assert (
        parent_record["contract"]["settings"]["selected"]
        == parent_record["contract"]["settings"]["defaults"]
    )
    assert parent_record["runtime"]["effective"]["model"]["value"] == "claude-opus-5-5"
    assert parent_record["runtime"]["effective"]["effort"]["status"] == "unavailable"
    assert len(f.calls("judgement")) == 1 and len(review.calls()) == 1
    assert p.state()["parents"] == {}


def test_half_day_hold_comments_updates_restart_and_missed_slots(
    loop: DevelopmentLoop,
) -> None:
    f, p = loop, loop.p
    f.inputs()
    f.response(decision(1, "information_needed"))
    assert f.due("2026-10-24T07:59:59+09:00")["result"] == "not_due"
    assert f.calls("judgement") == []
    assert f.due()["result"] == "recorded"
    first = p.state()["comments"]["1"]
    assert len(first) == 1
    assert "期待する入力形式" in first[0]["body"] and "再開条件" in first[0]["body"]
    assert f.due("2026-10-24T20:00:00+09:00")["result"] == "recorded"
    assert p.state()["comments"]["1"] == first
    assert len(f.calls("judgement")) == 2  # No changed input is required.
    for minute in range(1, 4):
        state = p.state()
        state["issues"]["1"]["body"] += f"\n入力についての追記 {minute}。"
        p.save(state)
        f.restart()
        f.control.reconcile()
        assert f.control.assign() == []
        f.control.launch_reserved()
        assert f.due(f"2026-10-24T20:0{minute}:00+09:00")["result"] == "not_due"
    assert len(f.calls("judgement")) == 2 and f.calls() == []
    # Simulate a powered-off PC across five scheduled slots, then duplicate wakeups.
    assert f.due("2026-10-27T09:00:00+09:00")["result"] == "recorded"
    assert f.due("2026-10-27T09:00:01+09:00")["result"] == "not_due"
    assert len(f.calls("judgement")) == 3
    assert [r["slot"] for r in f.schedule_state()["runs"]] == [
        "2026-10-24T08:00:00+09:00",
        "2026-10-24T20:00:00+09:00",
        "2026-10-27T08:00:00+09:00",
    ]
    assert f.control.assign() == []
    assert len(p.state()["comments"]["1"]) == 1


def test_controller_response_loss_reuses_judgement_and_parent_after_restart(
    loop: DevelopmentLoop,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    f = loop
    f.inputs()
    original = f.service.judgement.run

    def lose_response(*args: Any, **kwargs: Any) -> Any:
        original(*args, **kwargs)
        raise ConnectionError("controller lost its completed response")

    with monkeypatch.context() as patch:
        patch.setattr(f.service.judgement, "run", lose_response)
        with pytest.raises(ConnectionError, match="lost its completed response"):
            f.due()
    with f.store.locked():
        state = f.schedule.read()
        assert state["active"] == state["runs"][0]["id"]
        # The fake controller has disappeared after persisting the real runner output.
        state["runs"][0]["controller"] = None
        f.schedule.write(state)
    f.restart()
    assert f.due()["result"] == "not_due"
    assert "保存済み結果" in f.schedule_state()["runs"][0]["diagnostic"]
    assert f.control.assign() == [1]
    parent = f.job()
    f.control.launch_reserved()
    f.accept(parent)
    identity = f.row()["session"]
    for _ in range(2):
        f.restart()
        f.control.reconcile()
        f.control.assign()
        f.control.launch_reserved()
    assert f.row()["session"] == identity
    assert len(f.row()["dispatch"]["jobs"]) == 1
    assert len(f.calls()) == len(f.calls("judgement")) == 1
    f.release(parent)
    assert f.row()["dispatch"]["phase"] == "waiting"
    assert f.row()["status"] == "active"


def test_parent_startup_failure_retains_reservation_without_replacement(
    loop: DevelopmentLoop,
) -> None:
    f = loop
    f.inputs()
    assert f.due()["result"] == "recorded"
    assert f.control.assign() == [1]
    parent = f.job()
    (Path(parent["directory"]) / "cli-failure").touch()
    f.control.launch_reserved()
    f.ended(parent)
    record = f.record(parent)
    assert record["turn_end_reason"] == "cli_error"
    assert record["task_success"] is None and record["runtime"]["session_id"] is None
    assert f.row()["dispatch"]["phase"] == "failed"
    assert f.row()["status"] == "active"
    f.restart()
    f.control.reconcile()
    f.control.assign()
    f.control.launch_reserved()
    assert len(f.calls()) == len(f.calls("judgement")) == 1
    assert len(f.row()["dispatch"]["jobs"]) == 1


def test_scheduled_parent_child_limits_and_running_dependency_stop(
    loop: DevelopmentLoop,
) -> None:
    f, p = loop, loop.p
    f.inputs((10, 11, 12, 13), second=True, dependencies={11: [10]})
    parent = f.begin()
    upstream = f.start(10)
    f.publish_pr(10)
    f.release(upstream)
    dependent = f.start(11, base="codex/parser-10")
    independent = f.start(12, "gpt-6-astra")
    extra_tree = p.worktree("extra")
    denied = p.cli(
        "--dispatcher-state-dir",
        str(f.schedule.directory),
        "start-worker",
        "--child",
        "13",
        "--worktree",
        str(extra_tree),
        "--branch",
        "codex/extra",
        "--prompt-file",
        p.note("上限を超える。"),
        ok=False,
    )
    assert "implementation slots are occupied" in denied.stderr
    assert f.mod.children_pending(f.row()) == [11, 12]
    # Parent waiting keeps live children in its occupied parent slot.
    f.release(parent)
    f.restart()
    f.control.reconcile()
    f.control.assign()
    with f.store.locked():
        assert "o/r#2" not in f.store.parents
    assert f.schedule_state()["runs"][0]["candidates"]["2"]["state"] == "capacity_wait"
    assert len(f.calls()) == 4
    # The earlier child-end event returns to the same parent without a new judgment.
    resumed = f.job()
    assert resumed["id"] != parent["id"]
    assert resumed["resume_from"] == str(Path(parent["directory"]) / "runtime")
    f.control.launch_reserved()
    f.accept(resumed)
    detail = {
        "reason": "先行子の仕様確認待ち。",
        "facts": "既存PRを確認した。",
        "impact": "先行子と依存子。独立子は継続。",
        "resume_condition": "仕様の判断を記録する。",
        "next_action": "親が確認し同じworkerを再開する。",
    }
    args = (
        "--dispatcher-state-dir",
        str(f.schedule.directory),
        "stop-worker",
        "--child",
        "10",
        "--spec",
        p.spec(detail),
    )
    stop = p.cli(*args, ok=None)
    if stop.returncode:
        # The terminating worker may save its result while the CLI releases the
        # registry lock for GitHub. Honor that explicit guard, then fetch again.
        assert stop.returncode == 2
        assert "parent registry changed during GitHub retrieval" in stop.stderr
        f.ended(dependent)
        stop = p.cli(*args)
    stopped = json.loads(stop.stdout)
    states = {r["child"]: r["state"] for r in stopped["launch_plan"]["tasks"]}
    assert states[10] == "stopped" and states[11] == "held" and states[12] == "continue"
    f.ended(dependent)
    assert f.record(dependent)["turn_end_reason"] == "stopped"
    assert f.record(independent)["result"] == "running"
    assert f.mod.children_pending(f.row()) == [12]
    assert (
        f.record(independent)["contract"]["settings"]["selected"]["model"]
        == "gpt-6-astra"
    )
    f.release(independent)
    assert len(f.calls("judgement")) == 1
    assert len([c for c in f.calls() if c["agent"] == "codex"]) == 3


def test_cancelled_child_history_survives_real_gate_and_parent_completion(
    loop: DevelopmentLoop,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    f, p = loop, loop.p
    f.inputs((10, 11))
    parent = f.begin()
    worker = f.start(10)
    f.publish_pr(10)
    f.release(worker)
    f.verify(10)
    state = p.state()
    state["issues"]["11"].update(state="closed", state_reason="not_planned")
    p.save(state)
    p.cli(
        "--dispatcher-state-dir",
        str(f.schedule.directory),
        "record-cancellation",
        "--child",
        "11",
        "--spec",
        p.spec(
            {
                "schema_version": 1,
                "reason": "同じ境界を先行実装に統合した。",
                "handoff": "残作業は #10 で実装・検証済み。",
                "evidence": "元の範囲・受入条件と実装差分、起動していないことを親が確認した。",
            }
        ),
    )
    review = request_review(p, [100])
    review.run()
    gate = GateEnv(review, review_policy_module(monkeypatch))
    gate.confirm()
    merged = gate.cli("merge")
    assert merged.returncode == 0, merged.stderr + merged.stdout
    f.release(parent, f.report())
    assert f.row()["status"] == "ended"
    assert f.row()["development_result"]["cancelled"] == [11]
    assert f.row()["development_result"]["prs"] == [100]
    assert p.state()["parents"] == {"10": 1, "11": 1}
    assert "- [ ] #11 parser 11（取消）" in p.state()["issues"]["1"]["body"]
    assert p.state()["issues"]["11"]["state_reason"] == "not_planned"
    assert (
        len(gate.merge_calls()) == len(review.calls()) == len(f.calls("judgement")) == 1
    )


def test_adopted_report_triage_reuses_formal_family_until_scheduled_assignment(
    loop: DevelopmentLoop,
) -> None:
    from tests.e2e.agents_ops.test_adopted_proposal_issues import (
        decision as organization_decision,
    )
    from tests.e2e.agents_ops.test_adopted_proposal_issues import (
        parent_node,
        plan,
        raw_issue,
        task_node,
    )

    f, p = loop, loop.p
    props = importlib.import_module("opsreport.proposals")
    triage = importlib.import_module("opsreport.triage")
    proposal = props.Proposal(
        "コード負債",
        "parserの責務を分割する",
        evidence="READMEと入力境界。",
        purpose="保守性を高める。",
        scope="parserとCPU試験。",
        expected_effect="境界を明確にする。",
        acceptance=("CPUの境界試験が通る。",),
    )
    text, _ = props.render_proposals([proposal], "20261019")
    marker = props.report_marker(
        slot="20261019",
        since=datetime.fromisoformat("2026-10-16T08:00:00+09:00"),
        until=datetime.fromisoformat("2026-10-19T08:00:00+09:00"),
        proposals=1,
        status="complete",
    )
    report = raw_issue(
        90,
        title="運用レポート",
        labels=[{"name": "ops-report"}],
        body=f"{marker}\n{props.PROPOSALS_BEGIN}\n{text.replace('- [ ] **', '- [x] **')}\n{props.PROPOSALS_END}\n",
    )
    state = p.state()
    state.update(issues={"90": report}, parents={}, next=1)
    p.save(state)
    Path(p.env["FAKE_ORGANIZATION_PLAN"]).write_text(
        json.dumps(
            plan(
                [
                    parent_node(),
                    task_node("parser", "child", "improvements"),
                    task_node("tests", "child", "improvements"),
                ],
                [organization_decision("90/R-20261019-01", "improvements")],
            )
        )
    )
    for run in (1, 2):
        directory = p.root / f"triage-{run}"
        directory.mkdir()
        result = triage.triage(repo_root=p.repo, work_dir=directory, dry_run=False)
        assert not result.errors, result.errors
        assert p.state()["parents"] == {"2": 1, "3": 1}
        assert f.calls("judgement") == f.calls() == []
        with f.store.locked():
            assert f.store.parents == {}
    formats = importlib.import_module("issues")
    assert not formats.check_consistency(f.gh.family(1))
    assert formats.index_proposals(
        (int(n), r["body"]) for n, r in p.state()["issues"].items()
    ) == {
        (90, "R-20261019-01"): 1,
    }
    assert len(f.calls("organization")) == 1
    creates = [
        c
        for c in p.calls()
        if c[0] == "api" and "POST" in c and "repos/o/r/issues" in c
    ]
    assert len(creates) == 3  # One parent and two children, even after repeated triage.
    decisions = [
        decision(1),
        decision(2, "existing_parent", target=1),
        decision(3, "existing_parent", target=1),
        decision(90, "out_of_scope"),
    ]
    for row in decisions[:3]:
        row["sources"].append("issue:90")
    f.response(*decisions)
    assert f.due("2026-10-24T07:59:59+09:00")["result"] == "not_due"
    parent = f.begin()
    assert f.row()["dispatch"]["candidate"]["route"] == "parent"
    assert f.row()["dispatch"]["candidate"]["parent"] == 1
    assert (
        f.row()["dispatch"]["authorization"]["execution_basis_kind"]
        == "autonomous_decision"
    )
    index = json.loads(
        (
            Path(f.schedule_state()["runs"][0]["directory"]) / "agent-input/index.json"
        ).read_text()
    )
    assert index["issues"]["1"]["source_reports"] == [90]
    assert index["issues"]["2"]["source_reports"] == [90]
    comments = p.state()["comments"]["1"]
    assert all(
        '"event": "agreed"' not in row["body"] and "merge_permission" not in row["body"]
        for row in comments
    )
    assert len(f.calls("judgement")) == len(f.calls()) == 1
    f.release(parent)
