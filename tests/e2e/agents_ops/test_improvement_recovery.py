"""Interrupted decisions/completions, fresh admission and generation boundaries."""

from __future__ import annotations

import importlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_merge_gate import GateEnv
from tests.e2e.agents_ops.test_merge_gate import gate_env as gate_env
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


@pytest.mark.parametrize("interruption", ["before_result", "post_response"])
def test_publication_with_decision_link_recovers_after_response_loss_and_concurrent_retry(
    parent_env: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
    interruption: str,
) -> None:
    p = parent_env
    p.agreed()
    candidate = record(p)
    args = (
        "improvement-propose",
        "--decision-id",
        candidate["decision_id"],
    )
    if interruption == "post_response":
        p.env["PARENT_FAIL_AFTER_CREATE"] = "1"
        p.cli(*args, ok=False)
        del p.env["PARENT_FAIL_AFTER_CREATE"]
    else:
        for key, value in p.env.items():
            monkeypatch.setenv(key, value)
        from tests.e2e.agents_ops.test_manual_worker_runs import state_dir

        store = importlib.import_module("parent.state").Store(state_dir(p))
        parsed = (
            importlib.import_module("parent.cli")
            .parser()
            .parse_args(["--repo", "o/r", "--parent", "1", *args])
        )
        previous = importlib.import_module("decision_logs.parent_commands").begin(
            parsed, store, "o/r", 1
        )
        assert previous is not None and previous.records("result") == []
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: json.loads(p.cli(*args).stdout), range(2)))
    assert {r["issue"] for r in results} == {10} and post_count(p) == 1
    logs = json.loads(p.cli("decision-log-status").stdout)
    assert logs["status"] == "captured"
    rows = [entry["record"] for entry in logs["records"]]
    api = importlib.import_module("decision_logs.graph")
    api.validate_graph(rows)
    actions = {row["id"]: row for row in rows if row["kind"] == "action"}
    retry = [
        r
        for r in rows
        if r["kind"] == "result"
        and actions[r["data"]["action_id"]]["data"]["operation"] == "retry_publication"
    ]
    assert retry and all(r["data"]["previous_result_id"] for r in retry)
    assert any(
        a["data"]["decision_id"] == candidate["decision_id"] for a in actions.values()
    )


@pytest.mark.parametrize("lost_event", ["completion", "ended"])
@pytest.mark.parametrize("gate_env", ["parent_self"], indirect=True)
def test_finish_response_loss_repairs_only_missing_record(
    gate_env: GateEnv, lost_event: str
) -> None:
    g, p = gate_env, gate_env.review.p
    merged = g.cli("merge")
    assert merged.returncode == 0, merged.stderr
    p.announce_completion()
    spec = finish_for(p)
    p.env["PARENT_FAIL_AFTER_COMMENT_CONTAINING"] = f'"event": "{lost_event}"'
    p.cli("release", "--ended", "--spec", p.spec(spec), ok=False)
    del p.env["PARENT_FAIL_AFTER_COMMENT_CONTAINING"]
    p.cli("release", "--ended", "--spec", p.spec(spec))
    bodies = [c["body"] for c in p.comments()]
    assert sum('"event": "completion"' in b for b in bodies) == 1
    assert sum('"event": "ended"' in b for b in bodies) == 1
    assert json.loads(p.cli("status").stdout)["completion"]["acceptance"] == "completed"
    assert post_count(p) == 0
    assert len(g.merge_calls()) == len(g.review.calls()) == 1


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
        11,
        "claude:" + p.env["CLAUDE_CODE_SESSION_ID"],
        child=12,
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


def test_unapproved_managed_parent_can_handoff_without_granting_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.e2e.agents_ops.test_dispatcher_supervisor import (
        DispatchHarness,
        fixture_runtime,
    )

    d = DispatchHarness(tmp_path, monkeypatch)
    fixture_runtime(d, monkeypatch)
    d.prepare()
    d.control.assign()
    d.accepted()
    before = d.row()
    api = importlib.import_module("issues.improvement")
    source = api.ImprovementSource(2, "next")
    candidate = api.event(
        source,
        "candidate",
        1,
        None,
        {"source_issue": 2, "spec": candidate_spec(), "legacy": False},
    )
    formats = importlib.import_module("issues")
    rendered = importlib.import_module("issues.cli").render_spec(
        {
            "role": "parent",
            "identifier": "改善",
            "summary": "未承認の改善",
            "children": [],
            "improvement_source": {"parent": 2, "key": "next"},
            "sections": {
                h: "人間との議論待ち。"
                for h in formats.PARENT_REQUIRED
                if h != formats.TASKS_HEADING
            },
        }
    )
    d.h.gh.issues[1].update(
        title=rendered["title"],
        body=rendered["body"],
        labels=[{"name": label} for label in rendered["labels"]],
    )
    d.h.gh.comment_rows[2] = [
        {"id": 9000, "user": {"login": "operator"}, "body": api.render_event(candidate)}
    ]
    parent = importlib.import_module("parent.session")
    execution = importlib.import_module("parent.execution")
    with d.h.store.locked():
        old = parent.ParentSession(d.h.gh, d.h.store, 1, before["session"])
        old.lifecycle("handoff_pending", "未承認の改善について議論する担当を移す。")
        current = parent.ParentSession(
            d.h.gh, d.h.store, 1, "codex:new-discussion-owner"
        )
        current.accept("保存済みworkerを引き継ぎ、実装は承認まで再開しない。")
        assert current.row["status"] == "active"
        assert current.row["generation"] == before["generation"] + 1
        assert (
            current.row["dispatch"]["context_rebindings"][-1]["authorization"] is None
        )
        assert current.row["dispatch"]["controller_mode"] == "handoff"
        with pytest.raises(execution.ExecutionBasisStale, match="pending"):
            execution.authorization(current)
        assert set(current.row["dispatch"]["jobs"]) == set(before["dispatch"]["jobs"])


def approved_gate(
    p: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
    variant: str | None = None,
) -> tuple[GateEnv, str]:
    from tests.e2e.agents_ops.test_merge_gate import make_gate_env

    original_agree = p.agreed
    first_approval = []

    def agreed_improvement() -> None:
        original_agree()
        api = importlib.import_module("issues.improvement")
        provenance = importlib.import_module("issues.improvement_proposals")
        snapshots = importlib.import_module("parent.improvement_proposals")
        state = p.state()
        state["issues"]["1"]["body"] = (
            provenance.Extraction(2, 3, ("first", "second")).marker()
            + "\n\n"
            + state["issues"]["1"]["body"]
        )
        state["issues"]["3"] = {
            **state["issues"]["2"],
            "id": 10003,
            "number": 3,
            "body": provenance.Proposal(2, 1).marker(),
            "created_at": "2099-01-01T00:00:00Z",
            "state": "closed",
            "state_reason": "completed",
            "labels": [{"name": "issue:proposal"}],
        }
        decisions = {}
        chains = {}
        for k in ("first", "second"):
            source = api.ImprovementSource(2, k)
            candidate = api.event(
                source,
                "candidate",
                1,
                None,
                {"source_issue": 2, "spec": candidate_spec(), "legacy": False},
            )
            proposed = api.event(
                source,
                "proposed",
                1,
                candidate["id"],
                {"issue": 3, "generation": 1, "created_at": "2099-01-01T00:00:00Z"},
            )
            decision = api.event(
                source,
                "decision",
                1,
                proposed["id"],
                {
                    "target": "candidate",
                    "request_key": "human-1",
                    "request_sha256": "b" * 64,
                    "action": "approve",
                    "issue": 3,
                    "human_confirmation": decision_spec()["human_confirmation"],
                    "reason": "会話の承認を記録。",
                    "requirements": snapshots.candidate_requirements(candidate_spec()),
                    "previous_decision": None,
                },
            )
            decisions[k] = decision["id"]
            chains[k] = [candidate, proposed, decision]
        rows = []
        for k, chain in chains.items():
            chain.append(
                api.event(
                    api.ImprovementSource(2, k),
                    "extracted",
                    1,
                    chain[-1]["id"],
                    {
                        "issue": 1,
                        "proposal": 3,
                        "decision_ids": decisions,
                        "requirements": api.contract(state["issues"]["1"]["body"]),
                    },
                )
            )
            rows.extend(chain)
        state["comments"]["2"] = [
            {
                "id": 9000 + i,
                "user": {"login": "operator"},
                "body": api.render_event(row),
            }
            for i, row in enumerate(rows)
        ]
        p.save(state)
        first_approval.append(decisions["first"])
        p.cli("agree", "--improvement-approval", decisions["first"])

    monkeypatch.setattr(p, "agreed", agreed_improvement)
    g = make_gate_env(p, monkeypatch, variant)
    return g, first_approval[0]


def test_completed_first_generation_does_not_authorize_the_next(
    parent_env: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = parent_env
    g, first_approval = approved_gate(p, monkeypatch)
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
    p.announce_completion()
    finish = finish_for(p, candidates=["I-1-result-access"])
    p.announce_completion()
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
    assert "proposal" in p.cli("register", parent=second, ok=False).stderr
    denied = p.cli(
        "agree", "--improvement-approval", first_approval, parent=second, ok=False
    )
    assert "proposal" in denied.stderr or "担当" in denied.stderr
    status = json.loads(p.cli("status").stdout)
    assert (
        status["assignment"] == "ended"
        and status["completion"]["acceptance"] == "completed"
    )
    assert status["improvements"][0]["human_decision"] == "pending"
    assert len(g.merge_calls()) == len(g.review.calls()) == 1


def test_changed_extracted_parent_loses_acceptance_and_cannot_reapprove_closed_proposal(
    parent_env: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = parent_env
    g, first_approval = approved_gate(p, monkeypatch, "parent_self")
    merged = g.cli("merge")
    assert merged.returncode == 0, merged.stderr
    p.announce_completion()
    p.cli(
        "workflow-report",
        "--spec",
        p.spec(
            {
                "schema_version": 2,
                "state": "completed",
                "finish": finish_for(p),
                "reason": "元の条件を受け入れた。",
                "resume_condition": "完了。",
                "evidence": "元の条件についてPR #100 のレビュー・mergeと結果を確認。",
            }
        ),
    )
    saved = json.loads(p.cli("status").stdout)["completion"]["record"]
    assert saved["improvement_approval"]["decision_id"] == first_approval
    completed_body = p.state()["issues"]["1"]["body"]
    assert "現在の段階: workflow完了照合済み" in completed_body
    assert "未達・残件: なし" in completed_body
    assert (
        json.loads(p.cli("status", parent="2").stdout)["improvements"][0]["execution"]
        == "accepted"
    )

    state = p.state()
    state["issues"]["1"]["body"] = state["issues"]["1"]["body"].replace(
        "\n\n", "\n\n\n"
    )
    p.save(state)
    assert (
        json.loads(p.cli("status", parent="2").stdout)["improvements"][0]["execution"]
        == "accepted"
    )
    policy = importlib.import_module("parent.policy")
    body = state["issues"]["1"]["body"]
    state["issues"]["1"].update(
        body=policy.replace_section(
            body,
            "全体の完了条件",
            policy.raw_section(body, "全体の完了条件")
            + "\n- [ ] 新しい形式の結果も取得できる。",
        ),
        state="open",
        state_reason="reopened",
    )
    p.save(state)
    changed = json.loads(p.cli("status", parent="2").stdout)["improvements"][0]
    response = p.cli(
        "improvement-decide",
        "--candidate",
        "I-2-first",
        "--action",
        "approve",
        "--spec",
        p.spec(
            decision_spec(
                previous=first_approval,
                request="expanded-requirements",
                quote="追加した条件で承認。",
            )
        ),
        parent="2",
        ok=False,
    )
    assert "closed" in response.stderr
    assert changed["execution"] != "accepted"
    target = json.loads(p.cli("status").stdout)["completion"]
    assert target["record"] == saved and target["acceptance"] == "unverified"
    denied = p.cli("agree", "--improvement-approval", first_approval, ok=False)
    assert "requirements changed" in denied.stderr
    assert len(g.merge_calls()) == len(g.review.calls()) == 1
