"""Interrupted decisions/completions, fresh admission and generation boundaries."""

from __future__ import annotations

import importlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_parent_improvements import (
    create_approved_child,
    decide,
    decision_spec,
    post_count,
    publish,
    record,
    source_finished,
)
from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.support.agents_ops.improvements import candidate_spec, finish_for


def test_decision_response_loss_and_parallel_retry_preserve_one_event(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    publish(p)
    source_finished(p)
    p.env["PARENT_FAIL_AFTER_COMMENT_CONTAINING"] = '"event": "decision"'
    decide(p, "approve", ok=False)
    del p.env["PARENT_FAIL_AFTER_COMMENT_CONTAINING"]
    p.env["CLAUDE_CODE_SESSION_ID"] = "replacement-recording-session"
    before = len(p.state()["comments"]["1"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda _: json.loads(decide(p, "approve").stdout), range(2))
        )
    assert len({r["decision_id"] for r in results}) == 1
    assert len(p.state()["comments"]["1"]) == before
    assert json.loads(p.cli("status").stdout)["assignment"] == "ended"
    assert post_count(p) == 1


@pytest.mark.parametrize("lost_event", ["completion", "ended"])
def test_finish_response_loss_repairs_only_missing_record(
    parent_env: ParentEnv, lost_event: str
) -> None:
    from tests.e2e.agents_ops.test_parent_completion_records import agree

    p = parent_env
    agree(p)
    spec = finish_for(p)
    p.env["PARENT_FAIL_AFTER_COMMENT_CONTAINING"] = f'"event": "{lost_event}"'
    p.cli("release", "--ended", "--spec", p.spec(spec), ok=False)
    del p.env["PARENT_FAIL_AFTER_COMMENT_CONTAINING"]
    p.cli("release", "--ended", "--spec", p.spec(spec))
    bodies = [c["body"] for c in p.state()["comments"]["1"]]
    assert sum('"event": "completion"' in b for b in bodies) == 1
    assert sum('"event": "ended"' in b for b in bodies) == 1
    assert json.loads(p.cli("status").stdout)["completion"]["acceptance"] == "completed"
    assert post_count(p) == 0


def test_manual_revocation_between_preparation_and_spawn_never_starts_runner(
    parent_env: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = parent_env
    approved = create_approved_child(p)
    for k, v in p.env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.chdir(p.repo)
    manual = importlib.import_module("parent.manual_workers")
    admission = importlib.import_module("parent.improvement_runtime")
    real_execute = admission.execute_manual
    value = decision_spec(
        previous=approved, request="human-stop-before-spawn", quote="起動前に取り消す。"
    )

    def revoke_then_execute(*args: Any, **kwargs: Any) -> Any:
        assert (
            json.loads(decide(p, "revoke", value, ok=False).stdout)["state"]
            == "revoked"
        )
        return real_execute(*args, **kwargs)

    monkeypatch.setattr(admission, "execute_manual", revoke_then_execute)
    runner = p.root / "never-start"
    calls = p.root / "unexpected-runner"
    runner.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(calls)!r}).touch()\n"
    )
    runner.chmod(0o700)
    monkeypatch.setattr(manual, "RUNNER", runner)
    store = importlib.import_module("parent.state").Store(
        Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    )
    gh = importlib.import_module("parent.github").GitHub("o/r", store=store)
    code, result = manual.run_worker(
        gh,
        store,
        10,
        "claude:" + p.env["CLAUDE_CODE_SESSION_ID"],
        child=11,
        prompt=Path(p.note()),
        resume=False,
        restart_evidence=None,
    )
    assert code == 2 and result["launch_state"] == "rejected_before_start"
    assert not calls.exists()
    assert json.loads(decide(p, "revoke", value).stdout)["stop"]["state"] == "stopped"


def test_claimed_dispatch_job_rechecks_new_proposal_before_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.e2e.agents_ops.test_dispatcher_supervisor import DispatchHarness

    d = DispatchHarness(tmp_path, monkeypatch)
    d.prepare()
    d.control.assign()
    job = d.job()
    assert d.control.claim("o/r#1", job["id"]) is not None
    # A new protected source appears after claim: cached authorization cannot start it.
    api = importlib.import_module("issues.improvement")
    source = api.ImprovementSource(2, "next")
    row = api.event(
        source,
        "candidate",
        1,
        None,
        {"source_issue": 2, "spec": candidate_spec(), "legacy": False},
    )
    d.h.gh.issues[1]["body"] = source.marker() + "\n\n" + d.h.gh.issues[1]["body"]
    d.h.gh.comment_rows[2] = [
        {"id": 9000, "user": {"login": "operator"}, "body": api.render_event(row)}
    ]
    runtime = importlib.import_module("dispatcher.runtime").Runtime(Path(job["runner"]))
    runtime.launch_context = lambda: d.control.launch_admission("o/r#1", job["id"])
    calls = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: calls.append(a))
    with pytest.raises(ValueError, match="I-2-next"):
        runtime.execute(job, 0)
    assert calls == [] and d.job()["cancelled_before_start"]
    assert d.row()["status"] == "released"


def test_completed_first_generation_does_not_authorize_the_next(
    parent_env: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.e2e.agents_ops.test_merge_gate import make_gate_env

    p = parent_env
    original_agree = p.agreed
    first_approval = []

    def agreed_improvement() -> None:
        original_agree()
        api = importlib.import_module("issues.improvement")
        source = api.ImprovementSource(2, "first")
        candidate = api.event(
            source,
            "candidate",
            1,
            None,
            {"source_issue": 2, "spec": candidate_spec(), "legacy": False},
        )
        published = api.event(
            source,
            "published",
            1,
            candidate["id"],
            {"issue": 1, "mode": "created", "draft_sha256": "a" * 64},
        )
        state = p.state()
        state["issues"]["1"]["body"] = (
            source.marker() + "\n\n" + state["issues"]["1"]["body"]
        )
        state["comments"]["2"] = [
            {
                "id": 9000 + i,
                "user": {"login": "operator"},
                "body": api.render_event(row),
            }
            for i, row in enumerate((candidate, published))
        ]
        p.save(state)
        response = p.cli(
            "improvement-decide",
            "--candidate",
            "I-2-first",
            "--action",
            "approve",
            "--spec",
            p.spec(decision_spec()),
            parent="2",
        )
        approval = json.loads(response.stdout)["decision_id"]
        first_approval.append(approval)
        p.cli("agree", "--improvement-approval", approval)

    monkeypatch.setattr(p, "agreed", agreed_improvement)
    g = make_gate_env(p, monkeypatch)
    p.cli(
        "record-result",
        "--child",
        "10",
        "--evidence-file",
        p.note("親が差分・受入・テストを確認。"),
    )
    merged = g.cli("merge")
    assert merged.returncode == 0, merged.stderr
    before = post_count(p)
    record(p)
    assert post_count(p) == before  # Organizing never recursively publishes.
    finish = finish_for(p, candidates=["I-1-result-access"])
    p.cli(
        "workflow-report",
        "--spec",
        p.spec(
            {
                "schema_version": 2,
                "state": "completed",
                "finish": finish,
                "reason": "第1世代の受入完了。",
                "resume_condition": "次世代は独立した人間判断。",
                "evidence": "PR #100 の実merge・子close・全体受入を照合。",
            }
        ),
    )
    status = json.loads(p.cli("release", "--ended", "--spec", p.spec(finish)).stdout)
    assert status["completion"]["acceptance"] == "completed"
    first = json.loads(p.cli("status", parent="2").stdout)["improvements"][0]
    assert first["execution"] == "accepted"
    result = json.loads(publish(p).stdout)
    second = str(result["issue"])
    p.cli("register", parent=second)
    denied = p.cli(
        "agree", "--improvement-approval", first_approval[0], parent=second, ok=False
    )
    assert "pending" in denied.stderr
    status = json.loads(p.cli("status").stdout)
    assert (
        status["assignment"] == "ended"
        and status["completion"]["acceptance"] == "completed"
    )
    assert status["improvements"][0]["human_decision"] == "pending"
    assert len(g.merge_calls()) == len(g.review.calls()) == 1


def test_declined_closed_proposal_is_reproposed_in_place_with_fresh_human_consent(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    publish(p)
    declined = json.loads(decide(p, "decline").stdout)["decision_id"]
    source_finished(p)
    state = p.state()
    state["issues"]["10"].update(state="closed", state_reason="not_planned")
    p.save(state)
    revised = candidate_spec()
    revised["scope"] += " 読取専用の取得一覧を含む。"
    response = p.cli("improvement", "--key", "result-access", "--spec", p.spec(revised))
    assert json.loads(response.stdout)["revision"] == 2
    assert json.loads(publish(p).stdout)["issue"] == 10 and post_count(p) == 1
    assert p.state()["issues"]["10"]["state"] == "closed"
    current = json.loads(p.cli("status").stdout)["improvements"][0]
    assert current["human_decision"] == "pending"
    state = p.state()
    state["issues"]["10"]["body"] += "\n## 追加の境界\n\n読取専用の取得一覧を含む。\n"
    p.save(state)
    approved = json.loads(
        decide(
            p,
            "approve",
            decision_spec(
                previous=declined,
                request="reconsider-2",
                revision=2,
                quote="更新された境界でこの同じissueの実行を承認します。",
            ),
        ).stdout
    )
    assert "needs-human" in {v["name"] for v in p.state()["issues"]["10"]["labels"]}
    p.cli("register", parent="10", ok=False)  # Consent does not reopen an issue.
    state = p.state()
    state["issues"]["10"].update(state="open", state_reason="reopened")
    p.save(state)
    p.cli("register", parent="10")
    p.cli("agree", "--improvement-approval", approved["decision_id"], parent="10")
    assert json.loads(p.cli("status").stdout)["assignment"] == "ended"
    assert post_count(p) == 1
