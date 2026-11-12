"""Public CLI contracts using fake GitHub and disposable registry/git only."""

from __future__ import annotations

import copy
import importlib
import json
import os
import select
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.support.agents_ops.improvements import candidate_spec


def record(p: ParentEnv, *, fixed_key: str = "result-access") -> dict[str, Any]:
    return dict(
        json.loads(
            p.cli(
                "improvement", "--key", fixed_key, "--spec", p.spec(candidate_spec())
            ).stdout
        )
    )


@pytest.mark.parametrize(
    "entry",
    [
        "agree",
        "create-child",
        "delegate",
        "run-worker",
        "resume-worker",
        "start-worker",
        "continue-worker",
        "basis-record",
    ],
)
def test_every_execution_entry_rejects_unapproved_proposal_after_label_removal(
    parent_env: ParentEnv, entry: str
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    publish(p)
    source_finished(p)
    p.cli("register", parent="10")
    state = p.state()
    state["issues"]["10"]["labels"] = [{"name": "issue:parent"}]
    p.save(state)
    if entry in {"agree", "create-child"}:
        args = [entry, "--spec", p.spec({})]
    elif entry in {"delegate", "start-worker"}:
        args = [
            entry,
            "--child",
            "11",
            "--branch",
            "codex/not-started",
            "--worktree",
            str(p.repo),
        ]
        args += (
            ["--agent", "codex"] if entry == "delegate" else ["--prompt-file", p.note()]
        )
    elif entry in {"run-worker", "resume-worker", "continue-worker"}:
        args = [
            "run-worker" if entry == "resume-worker" else entry,
            "--child",
            "11",
            "--prompt-file",
            p.note(),
        ]
        if entry == "resume-worker":
            args.append("--resume")
    else:
        inputs = json.loads(p.cli("basis-input", parent="10").stdout)
        digest = importlib.import_module("parent.policy").digest
        basis = {
            "schema_version": 1,
            "kind": "autonomous_decision",
            "purpose": "self adopted",
            "scope": "all",
            "behavior": "implement",
            "acceptance": "done",
            "rationale": "AI selected",
            "sources": ["https://github.com/o/r/issues/10"],
            "input_sha256": digest(inputs),
            "policy_version": "autonomous-start/v1",
        }
        args = [
            entry,
            "--spec",
            p.spec({"basis": basis, "inputs": inputs, "ignored_comments": []}),
        ]
    denied = p.cli(*args, parent="10", ok=False)
    assert "I-1-result-access" in denied.stderr and "pending" in denied.stderr
    assert not json.loads(p.cli("status", parent="10").stdout)["tasks"]
    assert not (p.root / "runner-calls.jsonl").exists()


def create_approved_child(p: ParentEnv) -> str:
    p.agreed()
    record(p)
    publish(p)
    approved = json.loads(decide(p, "approve").stdout)["decision_id"]
    source_finished(p)
    p.cli("register", parent="10")
    p.cli("agree", "--improvement-approval", approved, parent="10")
    sections = {name: name + "を確認する。" for name in p.formats.CHILD_REQUIRED}
    sections["依存関係"] = "なし。"
    sections["Acceptance checklist"] = (
        "- [ ] 保存済み結果を取得する\n  文脈: 合意した改善。\n  下流への影響: 重複実行を防ぐ。\n  検証方法・証拠: fake runner。"
    )
    p.cli(
        "create-child",
        "--spec",
        p.spec(
            {
                "key": "retrieval",
                "dependencies": [],
                "spec": {
                    "role": "child",
                    "parent": {"identifier": "継続改善", "number": 10},
                    "summary": "取得を簡素化する",
                    "sections": sections,
                    "labels": ["enhancement"],
                },
            }
        ),
        parent="10",
    )
    p.cli(
        "delegate",
        "--child",
        "11",
        "--agent",
        "codex",
        "--branch",
        "codex/approved",
        "--worktree",
        str(p.worktree("approved")),
        parent="10",
    )
    return str(approved)


def run_improvement_worker(
    p: ParentEnv, *, ok: bool = True, code: str = "0"
) -> subprocess.CompletedProcess[str]:
    from tests.e2e.agents_ops.test_manual_worker_runs import ENTRY, state_dir

    runner = p.root / "improvement-runner"
    runner.write_text(
        f"#!{sys.executable}\n"
        + Path(__file__).with_name("fake_manual_worker_runner.py").read_text()
    )
    runner.chmod(0o700)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            ENTRY,
            str(runner),
            "--repo",
            "o/r",
            "--parent",
            "10",
            "run-worker",
            "--child",
            "11",
            "--prompt-file",
            p.note("承認済みの取得改善を実装する。"),
        ],
        cwd=p.repo,
        env={
            **p.env,
            "MANUAL_STATE_DIR": str(state_dir(p)),
            "MANUAL_PARENT_KEY": "o/r#10",
            "MANUAL_CHILD": "11",
            "MANUAL_RUNNER_CALLS": str(p.root / "runner-calls.jsonl"),
            "MANUAL_RUNNER_EXIT": code,
        },
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert (result.returncode == 0) == ok, result.stderr + result.stdout
    return result


def test_approved_manual_delegation_runs_once_and_failed_execution_is_not_accepted(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    create_approved_child(p)
    status = json.loads(p.cli("status").stdout)["improvements"][0]
    assert status["execution"] == "reserved"
    result = run_improvement_worker(p, ok=False, code="17")
    assert result.returncode == 17
    run_improvement_worker(p, ok=False)
    assert len((p.root / "runner-calls.jsonl").read_text().splitlines()) == 1
    assert (
        json.loads(p.cli("status").stdout)["improvements"][0]["execution"] != "accepted"
    )


@pytest.mark.parametrize("action", ["hold", "decline", "revoke"])
def test_human_stop_after_delegation_blocks_runner_without_releasing_source_completion(
    parent_env: ParentEnv, action: str
) -> None:
    p = parent_env
    previous = create_approved_child(p)
    decide(
        p,
        action,
        decision_spec(
            previous=previous,
            request="human-2",
            quote="この改善を実行せず止めてください。",
        ),
    )
    run_improvement_worker(p, ok=False)
    assert not (p.root / "runner-calls.jsonl").exists()
    assert json.loads(p.cli("status").stdout)["assignment"] == "ended"


def test_candidate_reason_and_approval_are_linked_without_writing_ended_source_journal(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    recorded = record(p)
    assert recorded["decision_id"] and len(recorded["plan_ids"]) == 1
    p.cli(
        "improvement-publish",
        "--candidate",
        recorded["candidate_id"],
        "--decision-id",
        recorded["decision_id"],
    )
    source_finished(p)
    before = json.loads(p.cli("decision-log-status").stdout)
    human = json.loads(decide(p, "approve").stdout)
    after = json.loads(p.cli("decision-log-status").stdout)
    assert before == after
    p.cli("register", parent="10")
    p.cli("agree", "--improvement-approval", human["decision_id"], parent="10")
    observed = json.loads(p.cli("decision-log-status", parent="10").stdout)
    source_rows = [item["record"] for item in after["records"]]
    target_rows = [item["record"] for item in observed["records"]]
    cut = next(row for row in source_rows if row["id"] == recorded["decision_id"])
    assert cut["data"]["inputs"]
    assert (
        cut["data"]["reason"]
        == "See existing improvement reason in referenced evidence"
    )
    assert any(
        row["kind"] == "action"
        and row["data"]["decision_id"] == cut["id"]
        and row["data"]["operation"] == "publish"
        for row in source_rows
    )
    approval = next(
        row
        for row in target_rows
        if row["kind"] == "result"
        and "Human approval observed" in row["data"]["summary"]
    )
    assert (
        approval["scope"]["parent"] == 10 and approval["data"]["observer"] == "machine"
    )
    assert any(
        e.get("url", "").startswith("https://github.com/o/r/issues/1#issuecomment-")
        for e in approval["data"]["evidence"]
    )
    assert "execution not started" in approval["data"]["summary"]
    from tests.e2e.agents_ops.test_manual_worker_runs import state_dir

    journal = importlib.import_module("decision_logs.journal").Journal(
        state_dir(p), "o/r", 1
    )
    # Generated decision IDs stay public even though worker-plan IDs are private.
    for row in source_rows:
        journal.prepare(
            row["id"], key="prepare-" + row["id"], at="2026-11-11T00:00:00+00:00"
        )
    assert any(row["id"] == recorded["decision_id"] for row in journal.public_records())


def test_human_revocation_stops_a_running_offline_worker_and_confirms_cleanup(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    previous = create_approved_child(p)
    fake_cli = p.root / "bin/codex"
    shutil.copyfile(Path(__file__).with_name("fake_dispatch_cli.py"), fake_cli)
    fake_cli.chmod(0o700)
    ready, block = p.root / "ready.fifo", p.root / "block.fifo"
    os.mkfifo(ready)
    os.mkfifo(block)
    fd = os.open(ready, os.O_RDWR | os.O_NONBLOCK)
    p.env.update(
        CODEX_HOME=str(p.root / "offline-codex-home"),
        FAKE_DISPATCH_READY_FIFO=str(ready),
        FAKE_DISPATCH_BLOCK_FIFO=str(block),
        FAKE_DISPATCH_CALLS=str(p.root / "offline-calls.jsonl"),
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "parent",
            "--repo",
            "o/r",
            "--parent",
            "10",
            "run-worker",
            "--child",
            "11",
            "--prompt-file",
            p.note("承認範囲のworker。終了は試験側の停止イベントで制御する。"),
        ],
        cwd=p.repo,
        env=p.env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    directory: Path | None = None
    try:
        readable, _, _ = select.select([fd], [], [], 45)
        assert readable, "offline worker did not report its startup event"
        directory = Path(os.read(fd, 4096).decode().strip())
        assert directory.is_relative_to(p.root)
        decision = decision_spec(
            previous=previous,
            request="stop-running",
            quote="実行中の改善を停止してください。",
        )
        response = decide(p, "revoke", decision, ok=None)
        result = json.loads(response.stdout)
        assert result["state"] == "revoked"
        assert result["stop"]["state"] in {"stopping", "stopped"}
        assert response.returncode == (result["stop"]["state"] == "stopping")
        # The stop request can precede result.json. Wait on the actual launcher
        # exit, then replay the same decision to capture the cleanup proof.
        _, stderr = proc.communicate(timeout=30)
        assert proc.returncode != 0, stderr
        record = json.loads((directory / "result.json").read_text())
        assert record["result"] == "failed" and record["cleanup_complete"] is True
        replayed = json.loads(decide(p, "revoke", decision).stdout)
        assert replayed["decision_id"] == result["decision_id"]
        assert replayed["reused"] is True and replayed["stop"]["state"] == "stopped"
        assert len((p.root / "offline-calls.jsonl").read_text().splitlines()) == 1
        run_improvement_worker(p, ok=False)
        assert not (p.root / "runner-calls.jsonl").exists()
    finally:
        os.close(fd)
        if directory is not None:
            runner = importlib.import_module("parent.manual_workers").RUNNER

            subprocess.run(
                [str(runner), "stop", "--run-dir", str(directory)],
                env=p.env,
                capture_output=True,
                timeout=30,
                check=False,
            )
        if proc.poll() is None:
            proc.terminate()
            proc.communicate(timeout=30)


def publish(p: ParentEnv, *, ok: bool = True, existing: int | None = None) -> Any:
    return p.cli(
        "improvement-publish",
        "--candidate",
        "I-1-result-access",
        *(["--existing", str(existing)] if existing else []),
        ok=ok,
    )


def decision_spec(
    *,
    previous: str | None = None,
    revision: int = 1,
    request: str = "human-1",
    quote: str = "この改善案の範囲で実行してください。",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "request_key": request,
        "revision": revision,
        "previous_decision": previous,
        "human_confirmation": {
            "quote": quote,
            "source": "2026-11-11 人間との会話。候補 I-1-result-access を確認。",
        },
        "reason": "元発言を確認し、示された候補の範囲に記録する。",
    }


def decide(
    p: ParentEnv,
    action: str,
    spec: dict[str, Any] | None = None,
    *,
    ok: bool | None = True,
) -> Any:
    return p.cli(
        "improvement-decide",
        "--candidate",
        "I-1-result-access",
        "--action",
        action,
        "--spec",
        p.spec(spec or decision_spec()),
        ok=ok,
    )


def source_finished(p: ParentEnv) -> None:
    p.cli("release", "--ended", "--note-file", p.note("担当終了。成果受入は未確認。"))
    state = p.state()
    state["issues"]["1"].update(state="closed", state_reason="completed")
    p.save(state)


def post_count(p: ParentEnv) -> int:
    return sum(
        "repos/o/r/issues" in call and call[call.index("--method") + 1] == "POST"
        for call in p.calls()
        if "--method" in call
    )


def test_parent_without_children_records_candidates_and_posts_one_independent_parent(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    original = copy.deepcopy(p.state()["issues"]["1"])
    first = record(p)
    assert record(p) == first
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: json.loads(publish(p).stdout), range(2)))
    assert {r["issue"] for r in results} == {10}
    assert post_count(p) == 1
    state = p.state()
    created = state["issues"]["10"]
    assert {label["name"] for label in created["labels"]} == {
        "enhancement",
        "issue:parent",
        "needs-human",
    }
    assert created["body"].startswith(
        "<!-- parent-improvement-source:v1 parent=1 key=result-access -->"
    )
    assert (
        "親issue:" not in created["body"]
        and "ops-report-proposal" not in created["body"]
    )
    assert state["parents"].get("10") is None
    assert state["issues"]["1"] == original
    assert not any("sub_issues" in " ".join(c) and "POST" in c for c in p.calls())
    # The only target write was the initial POST, so labels were never attached later.
    assert not any("repos/o/r/issues/10" in c and "PATCH" in c for c in p.calls())
    status = json.loads(p.cli("status").stdout)["improvements"][0]
    assert (status["human_decision"], status["execution"]) == ("pending", "not_started")


@pytest.mark.parametrize("failure", ["create_response", "link"])
def test_publication_recovers_only_missing_links_after_source_close(
    parent_env: ParentEnv, failure: str
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    p.env[
        "PARENT_FAIL_AFTER_CREATE"
        if failure == "create_response"
        else "PARENT_FAIL_PATH"
    ] = "1" if failure == "create_response" else "repos/o/r/issues/1/comments"
    publish(p, ok=False)
    p.env.pop("PARENT_FAIL_AFTER_CREATE", None)
    p.env.pop("PARENT_FAIL_PATH", None)
    source_finished(p)
    # Recovery must not require the old owner or revive its assignment.
    p.env["CLAUDE_CODE_SESSION_ID"] = "different-recording-session"
    result = json.loads(
        p.cli(
            "--repo",
            "O/R",
            "improvement-publish",
            "--candidate",
            "I-1-result-access",
        ).stdout
    )
    assert result["issue"] == 10 and post_count(p) == 1
    source = json.loads(p.cli("status").stdout)
    assert source["assignment"] == "ended"
    assert len(p.state()["issues"]) == 3


def test_unknown_creation_does_not_repeat_post_when_lookup_temporarily_has_no_issue(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    p.env["PARENT_FAIL_AFTER_CREATE"] = "1"
    publish(p, ok=False)
    del p.env["PARENT_FAIL_AFTER_CREATE"]
    state = p.state()
    saved = state["issues"].pop("10")
    p.save(state)
    assert "outcome is unknown" in publish(p, ok=False).stderr
    assert post_count(p) == 1
    state["issues"]["10"] = {
        **saved,
        "title": "renamed",
        "state": "closed",
        "state_reason": "not_planned",
    }
    p.save(state)
    publish(p)
    assert (
        post_count(p) == 1
        and p.state()["issues"]["10"]["state_reason"] == "not_planned"
    )


@pytest.mark.parametrize(
    "state,reason", [("open", None), ("closed", "completed"), ("closed", "not_planned")]
)
def test_existing_purpose_is_referenced_without_changing_its_scope_or_decision(
    parent_env: ParentEnv, state: str, reason: str | None
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    remote = p.state()
    remote["issues"]["2"].update(state=state, state_reason=reason)
    p.save(remote)
    before = copy.deepcopy(p.state()["issues"]["2"])
    result = json.loads(publish(p, existing=2).stdout)
    assert result["issue"] == 2 and result["publication"] == "reference"
    publish(p, existing=2)
    assert p.state()["issues"]["2"] == before and post_count(p) == 0
    assert "does not extend" in decide(p, "approve", ok=False).stderr


def test_register_allows_discussion_but_no_generic_agreement_or_delegation_bypasses_approval(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    publish(p)
    source_finished(p)
    p.cli("register", parent="10")
    state = p.state()
    state["issues"]["10"]["labels"] = [{"name": "issue:parent"}]
    p.save(state)
    agreement = {
        k: "元親の合意を流用する。"
        for k in ("purpose", "scope", "behavior", "acceptance", "human_confirmation")
    }
    denied = p.cli("agree", "--spec", p.spec(agreement), parent="10", ok=False)
    assert "I-1-result-access" in denied.stderr and "pending" in denied.stderr
    denied = p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/proposal",
        "--worktree",
        str(p.worktree("proposal")),
        parent="10",
        ok=False,
    )
    assert "pending" in denied.stderr or "実行根拠" in denied.stderr
    fake_confirmation = {**decision_spec(), "source_kind": "parent_choice"}
    decide(p, "approve", fake_confirmation, ok=False)
    assert not json.loads(p.cli("status", parent="10").stdout)["tasks"]


def test_explicit_approval_is_imported_once_and_old_replay_cannot_revoke_a_stop(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    publish(p)
    approved = json.loads(decide(p, "approve").stdout)
    source_finished(p)
    p.cli("register", parent="10")
    p.cli("agree", "--improvement-approval", approved["decision_id"], parent="10")
    count = len(p.state()["comments"]["10"])
    p.cli("agree", "--improvement-approval", approved["decision_id"], parent="10")
    assert len(p.state()["comments"]["10"]) == count
    assert "needs-human" in {
        label["name"] for label in p.state()["issues"]["10"]["labels"]
    }
    revoked = json.loads(
        decide(
            p,
            "revoke",
            decision_spec(
                previous=approved["decision_id"],
                request="human-2",
                quote="この改善の実行を取り消してください。",
            ),
        ).stdout
    )
    replay = json.loads(decide(p, "approve").stdout)
    assert (
        replay["state"] == "revoked"
        and replay["decision_id"] == approved["decision_id"]
    )
    reused_consent = decision_spec(
        previous=revoked["decision_id"], request="replayed-old-consent"
    )
    assert (
        "new human instruction" in decide(p, "approve", reused_consent, ok=False).stderr
    )
    assert (
        "revoked"
        in p.cli(
            "agree",
            "--improvement-approval",
            approved["decision_id"],
            parent="10",
            ok=False,
        ).stderr
    )
    source = json.loads(p.cli("status").stdout)
    assert (
        source["assignment"] == "ended"
        and source["improvements"][0]["decision_id"] == revoked["decision_id"]
    )


def test_approval_waits_for_supplement_and_material_changes_need_a_new_decision(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    publish(p)
    state = p.state()
    state["supplement_enabled_since"] = "2026-10-30T00:00:00Z"
    p.save(state)
    assert "pending" in decide(p, "approve", ok=False).stderr
    state["issues"]["10"]["labels"].append({"name": "supplement:done"})
    p.save(state)
    approval = json.loads(decide(p, "approve").stdout)
    source_finished(p)
    p.cli("register", parent="10")
    state = p.state()
    state["issues"]["10"]["body"] = state["issues"]["10"]["body"].replace(
        "\n\n", "\n\n\n"
    )
    p.save(state)
    p.cli("agree", "--improvement-approval", approval["decision_id"], parent="10")
    state = p.state()
    state["issues"]["10"]["body"] += "\n## 追加要件\n\nGPU学習を100回実行する。\n"
    p.save(state)
    assert (
        "requirements changed"
        in p.cli(
            "agree",
            "--improvement-approval",
            approval["decision_id"],
            parent="10",
            ok=False,
        ).stderr
    )


def test_preamble_requirement_change_invalidates_existing_authorization(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    publish(p)
    approval = json.loads(decide(p, "approve").stdout)["decision_id"]
    source_finished(p)
    p.cli("register", parent="10")
    p.cli("agree", "--improvement-approval", approval, parent="10")
    state = p.state()
    marker, remaining = state["issues"]["10"]["body"].split("\n", 1)
    state["issues"]["10"]["body"] = (
        marker + "\n\n追加で未処理の動画100本も処理する。\n" + remaining
    )
    p.save(state)
    denied = p.cli("agree", "--improvement-approval", approval, parent="10", ok=False)
    assert (
        "requirements changed" in denied.stderr and "I-1-result-access" in denied.stderr
    )
    assert json.loads(p.cli("status", parent="10").stdout)["tasks"] == {}
