"""An explicit, cleanup-verified closure releases PRs without granting evidence."""

from __future__ import annotations

import fcntl
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_parent_session import ReviewEnv, review_policy_module
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_parent_session import (
    single_review_env as single_review_env,
)
from tests.e2e.agents_ops.test_review_concurrency import subset_request
from tests.e2e.agents_ops.test_review_inspection import inspect
from tests.e2e.agents_ops.test_review_recovery import (
    gate_for,
    inject_lost_response,
    json_edit,
    recover,
    recovery_request,
)


def interrupted(r: ReviewEnv, *, cleanup: bool = True) -> None:
    """Model a crash before final/postcheck persistence; retain independent files."""
    execution = json.loads((r.run_dir / "execution.json").read_text())
    execution["status"] = "running"
    execution["stages"]["agent"]["status"] = "running"
    for name in ("output", "postcheck", "publication"):
        execution["stages"][name] = {"status": "not_started"}
    (r.run_dir / "execution.json").write_text(json.dumps(execution))
    json_edit(r.run_dir / "agent-runtime/result.json", cleanup_complete=cleanup)


def close(r: ReviewEnv, *, ok: bool = True, confirmation: Path | None = None) -> Any:
    return r.cli(
        "close-interrupted",
        "--run",
        r.request["review_run_id"],
        "--reason-file",
        r.p.note("本体終了と全子孫のcleanup、workspace撤去を確認して中断を終結。"),
        *(["--cleanup-confirmation", str(confirmation)] if confirmation else []),
        ok=ok,
    )


@pytest.mark.parametrize("missing", [False, True], ids=["unclean", "missing-runtime"])
def test_close_interrupted_requires_runtime_cleanup(
    single_review_env: ReviewEnv, missing: bool
) -> None:
    r = single_review_env
    r.run()
    interrupted(r, cleanup=False)
    if missing:
        (r.run_dir / "agent-runtime/result.json").unlink()
    before = (r.run_dir / "execution.json").read_bytes(), r.p.state()
    response = close(r, ok=False)
    assert response.returncode == 2
    assert "cleanup" in response.stderr and "agent_exec.sh stop" in response.stderr
    assert ((r.run_dir / "execution.json").read_bytes(), r.p.state()) == before
    assert len(r.calls()) == 1


def test_close_interrupted_checks_agent_stage_even_when_counter_is_zero(
    single_review_env: ReviewEnv,
) -> None:
    r = single_review_env
    r.run()
    interrupted(r, cleanup=False)
    json_edit(r.run_dir / "execution.json", agent_invocations=0)
    assert "cleanup" in close(r, ok=False).stderr


def test_close_interrupted_without_final_allows_a_new_same_pr_run(
    single_review_env: ReviewEnv,
) -> None:
    r = single_review_env
    r.run()
    interrupted(r)
    json_edit(
        r.run_dir / "agent-runtime/result.json",
        result="failed",
        exit_code=143,
        turn_end_reason="stopped",
    )
    for name in (
        "result.json",
        "output.json",
        "agent_output.md",
        "agent-runtime/final.md",
    ):
        (r.run_dir / name).unlink()
    before, calls = r.p.state(), len(r.calls())
    _, value = inspect(r, 100)
    assert value["prs"][0]["history"][0]["execution"]["status"] == "running"
    response = json.loads(close(r).stdout)
    assert response["status"] == "interrupted_closed"
    assert response["publication"] == "not_published"
    assert response["agent_invocations"] == 0
    execution = json.loads((r.run_dir / "execution.json").read_text())
    assert execution["agent_invocations"] == 1  # original invocation remains counted
    assert (
        execution["interruption"]["reason"]
        and execution["interruption"]["agent_started"]
    )
    assert r.p.state() == before and len(r.calls()) == calls
    assert not (r.run_dir / "result.json").exists()
    assert json.loads(close(r).stdout) == response  # identical retry is inert
    assert "closed" in r.run(ok=False).stderr
    assert (
        "closed"
        in r.cli(
            "resume-publication", "--run", r.request["review_run_id"], ok=False
        ).stderr
    )
    _, value = inspect(r, 100)
    entry = value["prs"][0]["history"][0]
    assert entry["execution"]["interruption"] == execution["interruption"]
    assert entry["execution"]["status"] == "interrupted_closed"
    assert entry["coverage_check"] is None and entry["result"] is None
    next_run = subset_request(r, 100)
    next_run.run()
    assert len(r.calls()) == calls + 1


def test_closed_run_never_grants_gate_coverage_or_recovery(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = single_review_env
    r.run()
    interrupted(r)
    saved = (r.run_dir / "result.json").read_bytes()
    g = gate_for(r, monkeypatch)
    g.confirm()
    close(r)
    blocked = g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))[100]
    assert blocked["decision"] != "merge"
    assert "parent_review_interrupted_closed" in blocked["reasons"]
    assert not g.merge_calls()
    _, value = inspect(r, 100)
    entry = value["prs"][0]["history"][0]
    assert entry["result"]["retained_only"] is True
    assert entry["result"]["verdict"] == "approve"
    assert entry["coverage_check"] is None
    target = recovery_request(r)
    assert "closed" in recover(r, target, ok=False).stderr
    assert (r.run_dir / "result.json").read_bytes() == saved
    assert len(r.calls()) == 1


@pytest.mark.parametrize("finding", [False, True], ids=["old-approve", "old-finding"])
def test_closed_request_still_supersedes_older_evidence(
    single_review_env: ReviewEnv, finding: bool
) -> None:
    r = single_review_env
    if finding:
        output = json.loads(r.output.read_text())
        output["verdict"] = "request_changes"
        output["prs"][0]["findings"] = [
            {
                "severity": "major",
                "path": "change.txt",
                "line": 1,
                "rule": "正しさ",
                "detail": "後続の中断終結で隠してはいけない指摘。",
            }
        ]
        r.output.write_text(json.dumps(output))
    r.run()
    original = (r.run_dir / "result.json").read_bytes()
    later = subset_request(r, 100)
    later.run()
    interrupted(later)
    close(later)
    target = recovery_request(r)
    assert "superseded" in recover(r, target, ok=False).stderr
    assert not (target.run_dir / "execution.json").exists()
    assert (r.run_dir / "result.json").read_bytes() == original
    _, value = inspect(r, 100)
    history = value["prs"][0]["history"]
    older = next(x for x in history if x["review_run_id"] == r.request["review_run_id"])
    assert later.request["review_run_id"] in older["subsequent_requests"]
    assert bool(older["result"]["findings"]) is finding
    assert len(r.calls()) == 2


def test_close_interrupted_rejects_a_held_run_lock(
    single_review_env: ReviewEnv,
) -> None:
    r = single_review_env
    r.run()
    interrupted(r)
    with (r.run_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        response = close(r, ok=False)
        assert "active" in response.stderr
    assert json.loads((r.run_dir / "execution.json").read_text())["status"] == "running"


def test_close_interrupted_rejects_a_held_pr_lease(
    single_review_env: ReviewEnv,
) -> None:
    r = single_review_env
    r.run()
    interrupted(r)
    path = next((r.run_dir.parents[1] / "review-targets").glob("*-pr-100.lock"))
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert "active" in close(r, ok=False).stderr
    assert json.loads((r.run_dir / "execution.json").read_text())["status"] == "running"


def test_close_cannot_revoke_published_review_or_use_empty_reason(
    single_review_env: ReviewEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r = single_review_env
    r.run()
    g = gate_for(r, monkeypatch)
    g.confirm()
    original = json.loads((r.run_dir / "execution.json").read_text())
    result_before, public_before = (r.run_dir / "result.json").read_bytes(), r.p.state()
    before = g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))[100]
    assert before["decision"] == "merge"
    assert json.loads(close(r).stdout)["status"] == "published"
    after = json.loads((r.run_dir / "execution.json").read_text())
    assert after.pop("exit_confirmation")["workspace_cleanup_complete"] is True
    assert after == original
    assert (r.run_dir / "result.json").read_bytes() == result_before
    assert r.p.state() == public_before
    gate_after = g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))[100]
    assert (gate_after["decision"], gate_after["reasons"]) == (before["decision"], before["reasons"])
    interrupted(r)
    response = r.cli(
        "close-interrupted",
        "--run",
        r.request["review_run_id"],
        "--reason-file",
        r.p.note(" \n"),
        ok=False,
    )
    assert "reason" in response.stderr
    assert original["status"] == "published"
    assert json.loads((r.run_dir / "execution.json").read_text())["status"] == "running"


@pytest.mark.parametrize("obstacle", ["workspace", "foreign-runtime"])
def test_close_interrupted_rejects_unverified_cleanup_identity(
    single_review_env: ReviewEnv, obstacle: str
) -> None:
    r = single_review_env
    r.run()
    interrupted(r)
    runtime = r.run_dir / "agent-runtime/result.json"
    if obstacle == "workspace":
        cwd = Path(json.loads(runtime.read_text())["contract"]["cwd"])
        cwd.mkdir(parents=True)
    else:
        json_edit(runtime, run_dir=str(r.p.root / "foreign-runtime"))
    response = close(r, ok=False)
    assert (
        "workspace" if obstacle == "workspace" else "another run"
    ) in response.stderr
    assert json.loads((r.run_dir / "execution.json").read_text())["status"] == "running"


def test_close_pre_agent_interruption_needs_no_runtime(
    single_review_env: ReviewEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r = single_review_env
    execution = {
        "status": "running",
        "request_sha256": review_policy_module(monkeypatch).digest(
            json.loads((r.run_dir / "request.json").read_text())
        ),
        "agent_invocations": 0,
        "stage_version": 1,
        "stages": {
            name: {"status": "not_started"}
            for name in ("agent", "output", "postcheck", "publication")
        },
    }
    (r.run_dir / "execution.json").write_text(json.dumps(execution))
    result = json.loads(close(r).stdout)
    assert result["status"] == "interrupted_closed"
    closure = json.loads((r.run_dir / "execution.json").read_text())["interruption"]
    assert closure["agent_started"] is False and closure["runtime_sha256"] is None
    assert not (r.run_dir / "result.json").exists() and len(r.calls()) == 0


@pytest.mark.parametrize("status", ["published", "publication_failed"])
@pytest.mark.parametrize("cause", ["unconfirmed", "workspace"])
def test_publication_does_not_prove_exit_before_another_run(
    single_review_env: ReviewEnv,
    status: str,
    cause: str,
) -> None:
    r = single_review_env
    extra = {
        "REVIEW_UNCONFIRMED_CLEANUP"
        if cause == "unconfirmed"
        else "REVIEW_LEAVE_WORKSPACE": "1"
    }
    if status == "publication_failed":
        # Keep remote publication verifiable while independently testing exit
        # evidence. An absent sent POST must stop at the common reader.
        inject_lost_response(r, target="repos/o/r/issues/100/comments", method="POST")
    r.run(ok=False, extra=extra)
    execution_path = r.run_dir / "execution.json"
    execution = json.loads(execution_path.read_text())
    assert execution["status"] == status
    assert execution["stages"]["agent"]["status"] == (
        "unconfirmed" if cause == "unconfirmed" else "finished"
    )
    next_run = subset_request(r, 100)
    blocked = next_run.run(ok=False)
    assert blocked.returncode == 2 and "cleanup" in blocked.stderr
    assert len(r.calls()) == 1 and not (next_run.run_dir / "execution.json").exists()
    original_result, original_public = (
        (r.run_dir / "result.json").read_bytes(),
        r.p.state(),
    )
    if cause == "workspace":
        assert "workspace" in close(r, ok=False).stderr
        runtime = json.loads((r.run_dir / "agent-runtime/result.json").read_text())
        shutil.rmtree(Path(runtime["contract"]["cwd"]))
    else:
        assert "cleanup" in close(r, ok=False).stderr
        json_edit(r.run_dir / "agent-runtime/result.json", cleanup_complete=True)
    confirmed = json.loads(close(r).stdout)
    assert confirmed["status"] == status and confirmed["publication"] == "unchanged"
    saved = json.loads(execution_path.read_text())
    assert saved.pop("exit_confirmation")["runtime_sha256"]
    assert saved == execution
    assert (r.run_dir / "result.json").read_bytes() == original_result
    assert r.p.state() == original_public
    next_run.run()
    assert len(r.calls()) == 2


def launch_gap(r: ReviewEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    execution = {
        "status": "running",
        "request_sha256": review_policy_module(monkeypatch).digest(
            json.loads((r.run_dir / "request.json").read_text())
        ),
        "agent_invocations": 1,
        "stage_version": 1,
        "stages": {
            name: {"status": "running" if name == "agent" else "not_started"}
            for name in ("agent", "output", "postcheck", "publication")
        },
    }
    (r.run_dir / "execution.json").write_text(json.dumps(execution))


def supervisor_confirmation(r: ReviewEnv, **updates: Any) -> Path:
    path: Path = r.p.root / "cleanup-confirmation.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": r.request["review_run_id"],
                "runner": None,
                "process": None,
                "cleanup_complete": True,
                "evidence": "supervisorの隔離scopeでwrapperと全子孫を回収し、当該review requestと照合した。native identityは初回保存前で未取得。",
                **updates,
            }
        )
    )
    return path


def test_missing_runtime_requires_explicit_supervisor_proof_and_then_allows_new_run(
    single_review_env: ReviewEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r = single_review_env
    launch_gap(r, monkeypatch)
    assert "cleanup" in close(r, ok=False).stderr
    assert not (r.run_dir / "agent-runtime/result.json").exists()
    before = r.p.state()
    confirmation = supervisor_confirmation(r)
    value = json.loads(close(r, confirmation=confirmation).stdout)
    assert value["status"] == "interrupted_closed"
    assert (
        value["interruption"]["cleanup_confirmation"]["run_id"]
        == r.request["review_run_id"]
    )
    assert value["interruption"]["runtime_sha256"] is None
    assert r.p.state() == before and len(r.calls()) == 0
    assert json.loads(close(r, confirmation=confirmation).stdout) == value
    assert (
        "confirmation"
        in close(
            r, ok=False, confirmation=supervisor_confirmation(r, cleanup_complete=False)
        ).stderr
    )
    assert not (r.run_dir / "agent-runtime/result.json").exists()
    next_run = subset_request(r, 100)
    next_run.run()
    assert len(r.calls()) == 1


@pytest.mark.parametrize(
    "bad", ["other_run", "unclean", "empty_evidence", "unknown_identity"]
)
def test_missing_runtime_rejects_unbound_or_incomplete_supervisor_proof(
    single_review_env: ReviewEnv,
    monkeypatch: pytest.MonkeyPatch,
    bad: str,
) -> None:
    r = single_review_env
    launch_gap(r, monkeypatch)
    cases: dict[str, dict[str, Any]] = {
        "other_run": {"run_id": "parent-1-" + "f" * 32},
        "unclean": {"cleanup_complete": False},
        "empty_evidence": {"evidence": " "},
        "unknown_identity": {"runner": {"pid": 1}},
    }
    changes = cases[bad]
    path = supervisor_confirmation(r, **changes)
    before = (r.run_dir / "execution.json").read_bytes()
    response = close(r, ok=False, confirmation=path)
    assert "confirmation" in response.stderr
    assert (r.run_dir / "execution.json").read_bytes() == before
    assert len(r.calls()) == 0


def test_missing_runtime_confirmation_cannot_override_a_held_runner_lock(
    single_review_env: ReviewEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r = single_review_env
    launch_gap(r, monkeypatch)
    runtime = r.run_dir / "agent-runtime"
    runtime.mkdir()
    with (runtime / "session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        response = close(r, ok=False, confirmation=supervisor_confirmation(r))
        assert "active" in response.stderr
    assert json.loads((r.run_dir / "execution.json").read_text())["status"] == "running"
