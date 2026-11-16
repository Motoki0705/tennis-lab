"""Review findings exercised through real CLI/adapters and bounded fake GitHub."""

from __future__ import annotations

import copy
import fcntl
import importlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_dispatcher_supervisor import d as d
from tests.e2e.agents_ops.test_dispatcher_supervisor import (
    registered_repair as registered_repair,
)
from tests.e2e.agents_ops.test_manual_worker_runs import reserved as reserved
from tests.e2e.agents_ops.test_parent_session import (
    ParentEnv,
    evidence_number,
    make_review_env,
)
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_progression_cli import prepared, request


def registry(p: ParentEnv) -> Any:
    return importlib.import_module("parent.state").Store(
        Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    )


@pytest.mark.parametrize("source", ["registry", "body"])
def test_stopped_direct_work_never_merges_and_independent_pr_finishes(
    parent_env: ParentEnv,
    source: str,
) -> None:
    p = parent_env
    review = make_review_env(p, independent=True, pr_count=2)
    p.cli("stop", "--child", "10", "--spec", p.detail())
    if source == "body":
        with registry(p).locked() as saved:
            saved.parents["o/r#1"]["tasks"]["10"]["stopped"] = False
            saved.save()
    else:
        remote = p.state()
        policy = importlib.import_module("parent.policy")
        remote["issues"]["10"]["body"] = policy.replace_section(
            remote["issues"]["10"]["body"], "作業状況・停止事項", "継続する。"
        )
        p.save(remote)
    value = json.loads(
        p.cli(
            "merge-continue",
            "--instruction-file",
            p.note("独立した成果だけ進める。"),
            "--wait",
        ).stdout
    )
    rows = {r["number"]: r for r in value["prs"]}
    assert rows[100]["stage"] == "parent_wait" and "stopped" in rows[100]["reason"]
    path = (
        registry(p).directory
        / "progressions"
        / review.request["review_run_id"]
        / "state.json"
    )
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text())
    assert state["targets"]["101"]["stage"] == "completed"
    assert [c[2] for c in p.calls() if c[:2] == ["pr", "merge"]] == ["101"]


@pytest.mark.parametrize(
    "fault", [None, "open_pr", "issue_open", "main_ci", "missing_commit"]
)
def test_legacy_merged_prs_can_be_confirmed_without_review_or_merge(
    parent_env: ParentEnv,
    fault: str | None,
) -> None:
    p = parent_env
    numbers, _, calls = prepared(p, count=2)
    for n in (10, 11):
        p.cli(
            "record-result",
            "--child",
            str(n),
            "--evidence-file",
            p.note("旧方式の受入済み成果。"),
        )
    for n in numbers:
        done = subprocess.run(
            [
                "gh",
                "pr",
                "merge",
                str(n),
                "--merge",
                "--match-head-commit",
                next(pr["head"]["sha"] for pr in p.state()["prs"] if pr["number"] == n),
            ],
            cwd=p.repo,
            env=p.env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 0, done.stderr
    archived = {
        "state": "complete",
        "request": {
            "schema_version": 1,
            "review_run_id": "parent-1-legacy",
            "prs": [{"number": n} for n in numbers],
        },
    }
    with registry(p).locked() as saved:
        saved.parents["o/r#1"]["reviews"].append(copy.deepcopy(archived))
        saved.save()
    remote = p.state()
    if fault == "open_pr":
        remote["prs"][0].update(
            state="open", merged=False, merged_at=None, merge_commit_sha=None
        )
    elif fault == "issue_open":
        remote["issues"]["10"].update(state="open", state_reason=None)
    elif fault == "main_ci":
        remote["prs"][0]["statusCheckRollup"][0]["conclusion"] = "FAILURE"
    p.save(remote)
    p.env.pop("AGENT_AUTO_MERGE", None)  # Confirmation has no merge permission.
    if fault == "missing_commit":
        p.env["PARENT_FAIL_GET_PATH"] = (
            "repos/o/r/commits/" + remote["prs"][0]["merge_commit_sha"]
        )
    before = len(p.calls())
    response = p.cli(
        "merge-confirm",
        "--prs",
        *map(str, numbers),
        "--wait",
        ok=None,
    )
    if fault in {"open_pr", "missing_commit"}:
        assert response.returncode == 2
        assert ("merged" if fault == "open_pr" else "503") in response.stderr
    else:
        assert response.returncode == 0, response.stderr
        value = json.loads(response.stdout)
        if fault is None:
            assert value["status"] == "completed"
        else:
            assert value["status"] in {"parent_wait", "running"}
            assert (
                next(t for t in value["prs"] if t["number"] == 100)["stage"]
                == "parent_wait"
            )
        if fault is None:
            with registry(p).locked() as saved:
                # The production completion consumer reads the new receipt, not legacy approval.
                check = importlib.import_module("review.merge_gate").completion
                for n in numbers:
                    assert check(saved, "o/r", 1, n)["issues_confirmed"]
    if fault is None:
        capture_file = (
            registry(p).directory
            / "progressions"
            / value["review_run_id"]
            / "decision-capture.json"
        )
        captured = importlib.import_module("decision_logs.capture").Operation.load(
            Path(json.loads(capture_file.read_text())["operation"])
        )
        assert captured.row["operation"] == "verify"
        results = [
            entry["record"]
            for entry in captured.row["entries"]
            if entry["record"]["kind"] == "result"
        ]
        assert results and all(r["data"]["observer"] == "machine" for r in results)
    if fault is None:
        import sys

        script = p.root / "confirm-completion.py"
        script.write_text(
            "from parent.state import Store,state_directory\nfrom parent.github import GitHub\nfrom parent.session import ParentSession\nfrom parent.development import verify_completion\nimport json,os\ns=Store(state_directory())\nwith s.locked():\n p=ParentSession(GitHub('o/r',store=s),s,1,'claude:'+os.environ['CLAUDE_CODE_SESSION_ID'])\n print(json.dumps(verify_completion(p)))\n"
        )
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=p.repo,
            env=p.env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == numbers
    assert calls.read_text() == ""
    assert not any(c[:2] == ["pr", "merge"] for c in p.calls()[before:])
    assert registry(p).read_data()["parents"]["o/r#1"]["reviews"] == [archived]


def test_review_context_and_mcp_include_bound_evidence_and_dependency_pr(
    parent_env: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = parent_env
    numbers, fake, calls = prepared(p, count=2)
    remote = p.state()
    policy = importlib.import_module("parent.policy")
    remote["issues"]["10"]["body"] = policy.replace_section(
        remote["issues"]["10"]["body"], "依存関係", "#11 の後。"
    )
    remote["prs"][1].update(
        merged=True, state="closed", merged_at="2026-11-16T00:00:00Z"
    )
    remote["issues"]["11"].update(state="closed", state_reason="completed")
    p.save(remote)
    with registry(p).locked() as saved:
        saved.parents["o/r#1"]["tasks"]["10"]["dependencies"] = [11]
        saved.save()
    evidence = evidence_number(remote)
    original = [c["body"] for c in remote["comments"][str(evidence)]]
    value = request(p, [numbers[0]], fake)
    assert value["status"] == "completed", value
    context = json.loads(calls.read_text().splitlines()[0])["context"]
    by_issue = {r["issue"]["number"]: r for r in context["issues"]}
    assert evidence in by_issue, (
        "reviewer must receive the agreement's actual evidence container"
    )
    assert [c["body"] for c in by_issue[evidence]["comments"]] == original
    assert {r["pr"]["number"] for r in context["prs"]} == set(numbers)
    config = json.loads(
        (
            registry(p).directory
            / "progressions"
            / value["review_run_id"]
            / "review-1/retrieval-config.json"
        ).read_text()
    )
    assert evidence in config["github_scope"]["issues"]
    assert set(numbers) <= set(config["github_scope"]["prs"])
    dependency_head = remote["prs"][1]["head"]["sha"]
    assert dependency_head in config["github_scope"]["ci_heads"]
    for k, v in p.env.items():
        monkeypatch.setenv(k, v)
    server = importlib.import_module("lib.review_tools.review_server").ReviewRetrieval(
        config
    )
    response = server.call("github_item", {"kind": "issue", "number": evidence})
    assert response["ok"], response
    assert server.call("github_item", {"kind": "pr", "number": numbers[1]})["ok"]
    dependency_ci = server.call("ci_runs", {"head_sha": dependency_head})
    assert dependency_ci["ok"], dependency_ci
    assert (
        server.call("github_item", {"kind": "issue", "number": 999})["error"]["code"]
        == "FORBIDDEN_SCOPE"
    )


def test_progression_dispatcher_adapter_starts_repair_without_repository_listing(
    registered_repair: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    d, runtime_type = registered_repair
    with d.h.store.locked():
        d.h.gh.evidence_issues().ensure(10, "operator", store=d.h.store)
    d.publish_github(monkeypatch)
    endpoints = d.h.root / "repair-http.jsonl"
    monkeypatch.setenv("FAKE_DISPATCH_HTTP_CALLS", str(endpoints))
    row = d.row()
    saved = row["repair_progressions"]["pipeline-1"]
    request_value = {
        "schema_version": 4,
        "review_run_id": "parent-1-" + "a" * 32,
        "repo": "o/r",
        "parent_issue": 1,
        "authorization": saved["authorization"],
        "prs": [
            {
                "number": 42,
                "issues": [10],
                "worker": {
                    "kind": "dispatcher",
                    "child": 10,
                    "delegation_id": saved["delegation_id"],
                    "progression_id": "pipeline-1",
                },
            }
        ],
    }
    state = importlib.import_module("review.progression").new_state(
        request_value, owner=row["session"], generation=row["generation"]
    )
    state["options"] = {"dispatcher_state_dir": str(d.schedule.directory)}
    backend = importlib.import_module("review.progression_backend").LiveBackend(
        state, d.h.store
    )
    assert backend.repair(
        state,
        state["targets"]["42"],
        {
            "id": "repair-through-progression",
            "details": {"reason": "ci_failure", "log": "failure"},
        },
    ) == {"outcome": "running"}
    job = next(
        j
        for j in d.row()["dispatch"]["jobs"].values()
        if j.get("repair", {}).get("repair_id") == "repair-through-progression"
    )
    control = backend.dispatcher()
    claimed = control.claim("o/r#1", job["id"])
    assert claimed is not None
    captured = d.h.root / "actual-repair-command.json"
    runner = d.h.root / "actual-repair-runner"
    runner.write_text(
        f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\nPath({str(captured)!r}).write_text(json.dumps(sys.argv[1:]))\n"
    )
    runner.chmod(0o700)
    with (d.h.root / "repair-execute.lock").open("a") as lock:
        assert runtime_type(runner).execute(claimed, lock.fileno()) == 0
    argv = json.loads(captured.read_text())
    assert argv[0] == "resume" and argv[argv.index("--effort") + 1] == "high"
    assert argv[argv.index("--resume-from") + 1] == job["resume_from"]
    d.runtime(job)
    record_path = Path(job["directory"]) / "runtime/result.json"
    record = json.loads(record_path.read_text())
    settings = importlib.import_module("lib.agent_settings")
    record["contract"]["settings"] = settings.resolve(
        settings.DEFAULT_CONFIG, "codex", None, "gpt-6-astra", None, "write"
    )
    record["contract"]["settings"]["selected"]["effort"] = "high"
    record["contract"]["repair"] = job["repair"]
    record_path.write_text(json.dumps(record))
    report = {
        "schema_version": 1,
        "repair_id": "repair-through-progression",
        "pr": 42,
        "outcome": "repaired",
        "head_sha": "b" * 40,
        "summary": "Fixed failing import",
        "verification": ["fixture check passed"],
        "parent_request": None,
    }
    (record_path.parent / "final.md").write_text(json.dumps(report))
    remote_file = d.h.root / "github.json"
    remote = json.loads(remote_file.read_text())
    remote["prs"][0]["head"]["sha"] = report["head_sha"]
    remote_file.write_text(json.dumps(remote))
    row = d.row()
    row["dispatch"]["jobs"][job["id"]]["worker"] = None
    d.update(row)
    control.reconcile()
    assert d.row()["dispatch"]["jobs"][job["id"]]["phase"] == "ended"
    assert (
        backend.recover_repair(
            state, state["targets"]["42"], {"id": "repair-through-progression"}
        )
        == report
    )
    requests = [json.loads(line) for line in endpoints.read_text().splitlines()]
    assert requests and not any(
        r.partition("?")[0] in {"repos/o/r/issues", "repos/o/r/pulls"} for r in requests
    )


def major_progression(session: Any, pr: int, options: dict[str, Any]) -> Path:
    gate = importlib.import_module("review.merge_gate")
    settings = importlib.import_module("lib.review_settings")
    choice = settings.choose(
        settings.load_policy(), model="gpt-6-astra", reason="Fixture broad review"
    )
    with session.store.locked():
        response = gate.register(
            session, [pr], "Review requirement", None, choice, options
        )
    path = gate.state_path(session.store, response["review_run_id"])
    state = json.loads(path.read_text())
    state["status"] = "parent_wait"
    state["review"] = {
        "stage": "parent_wait",
        "round": 1,
        "result": {
            "verdict": "major",
            "summary": "Fix behavior",
            "major": [
                {
                    "prs": [pr],
                    "title": "Incorrect behavior",
                    "detail": "Repair with the creator",
                }
            ],
        },
    }
    gate.save(path, state)
    return Path(path)


def test_dispatcher_major_continuation_rebinds_same_worker_then_merges(
    d: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import sys

    from tests.e2e.agents_ops.test_parent_development import (
        child_tree,
        drain,
        launch,
        workflow,
    )
    from tests.e2e.agents_ops.test_parent_session import FAKE_GH

    mod, session, env = workflow(d, monkeypatch, family=True)
    tree = child_tree(d)
    first = mod.start_worker(
        session,
        d.schedule.directory,
        env,
        child=10,
        prompt="Initial implementation",
        model=None,
        branch="codex/parser",
        worktree=tree,
    )
    try:
        launch(d, monkeypatch)
        previous = drain(d, first["job_id"])
        head = d.h.git("rev-parse", "HEAD")
        d.h.gh.prs = [
            {
                "number": 50,
                "title": "Implementation",
                "state": "open",
                "body": "Closes #10",
                "merged_at": None,
                "draft": False,
                "labels": [],
                "mergeable": True,
                "head": {
                    "ref": "codex/parser",
                    "sha": head,
                    "repo": {"full_name": "o/r"},
                },
                "base": {
                    "ref": "test",
                    "sha": head,
                    "repo": {"full_name": "o/r", "default_branch": "test"},
                },
                "statusCheckRollup": [
                    {
                        "__typename": "CheckRun",
                        "name": "Python",
                        "status": "COMPLETED",
                        "conclusion": "SUCCESS",
                    }
                ],
            }
        ]
        path = major_progression(
            session, 50, {"dispatcher_state_dir": str(d.schedule.directory)}
        )
        second = mod.start_worker(
            session,
            d.schedule.directory,
            env,
            child=10,
            prompt="Fix major",
            model=None,
            continuation="Major fix on the same session",
        )
        launch(d, monkeypatch)
        current = drain(d, second["job_id"])
        assert current["delegation_id"] != previous["delegation_id"]
        assert current["resume_from"] == str(Path(previous["directory"]) / "runtime")
        # Exercise the public continuation/controller with the actual scoped HTTP adapter.
        state_file = d.h.root / "continuation-github.json"
        state_file.write_text(
            json.dumps(
                {
                    "issues": d.h.gh.issues,
                    "parents": d.h.gh.parents,
                    "comments": d.h.gh.comment_rows,
                    "prs": d.h.gh.prs,
                    "default_branch": "test",
                    "next_number": 9000,
                    "next_comment_id": 100000,
                }
            )
        )
        calls_file = d.h.root / "continuation-gh-calls.jsonl"
        calls_file.touch()
        fake = d.h.root / "bin/gh"
        fake.write_text(f"#!{sys.executable}\n" + FAKE_GH)
        fake.chmod(0o700)
        p = ParentEnv(
            d.h.root,
            d.h.work,
            {
                **os.environ,
                "PARENT_GH_STATE": str(state_file),
                "PARENT_GH_CALLS": str(calls_file),
                "XDG_STATE_HOME": str(d.h.store.directory.parent.parent),
                "CLAUDE_CODE_SESSION_ID": "fixture-session",
                "CODEX_SESSION_ID": "",
                "CODEX_THREAD_ID": "",
                "AGENT_AUTO_MERGE": "1",
                "PYTHONPATH": str(Path(__file__).resolve().parents[3] / ".agents/ops"),
            },
            state_file,
            calls_file,
            importlib.import_module("issues"),
        )
        # This fixture's Store is not the usual state_directory location; bind the public flag.
        response = p.cli(
            "--state-dir",
            str(d.h.store.directory),
            "merge-continue",
            "--run-id",
            path.parent.name,
            "--instruction-file",
            p.note("修正を確認。"),
            "--wait",
            ok=None,
        )
        assert response.returncode == 0, response.stderr
        value = json.loads(response.stdout)
        assert value["status"] == "completed", value
        after = json.loads(path.read_text())
        worker = after["targets"]["50"]["worker"]
        assert worker["delegation_id"] == current["delegation_id"]
        registration = d.row()["repair_progressions"][worker["progression_id"]]
        assert registration["delegation_id"] == current["delegation_id"]
        assert len((d.h.root / "runtime-calls.jsonl").read_text().splitlines()) == 2
    finally:
        d.control.reconcile(stop=True)


@pytest.mark.parametrize("changed_session", [False, True])
def test_manual_major_continuation_checks_ended_creator_session(
    reserved: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
    changed_session: bool,
) -> None:
    from tests.e2e.agents_ops.test_manual_worker_runs import (
        latest,
        manual,
        run,
        store,
        task,
    )
    from tests.e2e.agents_ops.test_repair_workers import creating_worker

    p = reserved
    creating_worker(p)
    remote = p.state()
    remote["prs"][0].update(
        mergeable=True,
        statusCheckRollup=[
            {
                "__typename": "CheckRun",
                "name": "Python",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
            }
        ],
    )
    p.save(remote)
    for k, v in p.env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.chdir(p.repo)
    monkeypatch.setenv("AGENT_AUTO_MERGE", "1")
    api = importlib.import_module("parent.manual_workers")
    monkeypatch.setattr(api, "RUNNER", p.root / "fake-runner")
    gh = importlib.import_module("parent.github").GitHub("o/r", store=store(p))
    session_type = importlib.import_module("parent.session").ParentSession
    with gh._store.locked():
        session = session_type(
            gh, gh._store, 1, "claude:" + p.env["CLAUDE_CODE_SESSION_ID"]
        )
    path = major_progression(session, 42, {})
    run(p, "--resume")
    if changed_session:
        record_path = latest(p) / "result.json"
        record = json.loads(record_path.read_text())
        record["runtime"]["session_id"] = "different-worker"
        record_path.write_text(json.dumps(record))
    response = manual(
        p,
        "merge-continue",
        "--run-id",
        path.parent.name,
        "--instruction-file",
        p.note("同じworkerのmajor修正を確認した。"),
        "--wait",
        AGENT_AUTO_MERGE="1",
        ok=not changed_session,
    )
    if changed_session:
        assert "session" in response.stderr
        assert not any(c[:2] == ["pr", "merge"] for c in p.calls())
    else:
        assert json.loads(response.stdout)["status"] == "completed", response.stdout
        assert (
            task(p)["preflight"]["accepted_turn_id"]
            == task(p)["preflight"]["turns"][-1]["snapshot"]["turn_id"]
        )


def test_revoked_adoption_holds_only_the_affected_reviewed_pr(
    parent_env: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.e2e.agents_ops.test_merge_gate import make_gate_env
    from tests.e2e.agents_ops.test_merge_gate_reads import bind_task_adoption

    g = make_gate_env(parent_env, monkeypatch, "independent")
    p = g.review.p
    bind_task_adoption(g, 10)
    remote = p.state()
    remote["issues"]["77"]["body"] = remote["issues"]["77"]["body"].replace(
        "- [x] **", "- [ ] **"
    )
    p.save(remote)
    response = g.cli("merge")
    assert response.returncode == 0, response.stderr
    path = (
        registry(p).directory
        / "progressions"
        / g.review.request["review_run_id"]
        / "state.json"
    )
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        saved = json.loads(path.read_text())
    assert saved["targets"]["100"]["stage"] == "parent_wait"
    assert all(
        t["stage"] == "completed" for n, t in saved["targets"].items() if n != "100"
    )
    assert "100" not in [c[2] for c in g.merge_calls()]
    assert len(g.review.calls()) == 1
