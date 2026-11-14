"""Public CLI contracts using fake GitHub and disposable registry/git only."""

from __future__ import annotations

import copy
import hashlib
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

from tests.e2e.agents_ops.test_parent_session import ParentEnv, evidence_number
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.support.agents_ops.improvements import candidate_spec


def record(p: ParentEnv, *, fixed_key: str = "result-access") -> dict[str, Any]:
    p.env["PARENT_CREATED_AT"] = "2099-01-01T00:00:00Z"
    return dict(
        json.loads(
            p.cli(
                "improvement", "--key", fixed_key, "--spec", p.spec(candidate_spec())
            ).stdout
        )
    )


def publish(p: ParentEnv, *, ok: bool = True) -> Any:
    return p.cli("improvement-propose", ok=ok)


def extract(p: ParentEnv, *keys: str, ok: bool = True) -> Any:
    return p.cli(
        "improvement-extract", "--candidates", *(keys or ("I-1-result-access",)), ok=ok
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
    return int(
        sum(
            "repos/o/r/issues" in call and call[call.index("--method") + 1] == "POST"
            for call in p.calls()
            if "--method" in call
        )
        - sum(
            row.get("labels") == [{"name": "parent-evidence"}]
            for row in p.state()["issues"].values()
        )
    )


def create_approved_child(p: ParentEnv) -> str:
    p.agreed()
    record(p)
    publish(p)
    approved = json.loads(decide(p, "approve").stdout)["decision_id"]
    extract(p)
    source_finished(p)
    p.cli("register", parent="11")
    p.cli("agree", "--improvement-approval", approved, parent="11")
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
                    "parent": {"identifier": "継続改善", "number": 11},
                    "summary": "取得を簡素化する",
                    "sections": sections,
                    "labels": ["enhancement"],
                },
            }
        ),
        parent="11",
    )
    p.cli(
        "delegate",
        "--child",
        "12",
        "--agent",
        "codex",
        "--branch",
        "codex/approved",
        "--worktree",
        str(p.worktree("approved")),
        parent="11",
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
            "11",
            "run-worker",
            "--child",
            "12",
            "--prompt-file",
            p.note("承認済みの取得改善を実装する。"),
        ],
        cwd=p.repo,
        env={
            **p.env,
            "MANUAL_STATE_DIR": str(state_dir(p)),
            "MANUAL_PARENT_KEY": "o/r#11",
            "MANUAL_CHILD": "12",
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
        "improvement-propose",
        "--decision-id",
        recorded["decision_id"],
    )
    source_finished(p)
    before = json.loads(p.cli("decision-log-status").stdout)
    human = json.loads(decide(p, "approve").stdout)
    after = json.loads(p.cli("decision-log-status").stdout)
    assert before == after
    extract(p)
    p.cli("register", parent="11")
    p.cli("agree", "--improvement-approval", human["decision_id"], parent="11")
    observed = json.loads(p.cli("decision-log-status", parent="11").stdout)
    source_rows = [item["record"] for item in after["records"]]
    target_rows = [item["record"] for item in observed["records"]]
    cut = next(row for row in source_rows if row["id"] == recorded["decision_id"])
    assert cut["data"]["inputs"]
    assert (
        cut["data"]["reason"]
        == "See existing improvement reason in referenced evidence"
    )
    candidate_body = next(
        c
        for c in p.state()["comments"][str(evidence_number(p.state()))]
        if '"event": "candidate"' in c["body"]
    )
    assert cut["data"]["source"]["evidence"] == [
        {
            "kind": "public",
            "url": f"https://github.com/o/r/issues/{evidence_number(p.state())}#issuecomment-{candidate_body['id']}",
            "revision": None,
            "content_sha256": hashlib.sha256(
                candidate_body["body"].encode()
            ).hexdigest(),
        }
    ]
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
        approval["scope"]["parent"] == 11 and approval["data"]["observer"] == "machine"
    )
    assert any(
        e.get("url", "").startswith(
            f"https://github.com/o/r/issues/{evidence_number(p.state())}#issuecomment-"
        )
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
            "11",
            "run-worker",
            "--child",
            "12",
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


@pytest.mark.parametrize(
    "entry",
    ["register", "agree", "create-child", "delegate", "run-worker", "start-worker"],
)
def test_proposal_never_becomes_a_work_target(
    parent_env: ParentEnv, entry: str
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    publish(p)
    # Even malformed mixed labels or removed labels cannot turn a proposal into work.
    source_finished(p)
    if entry == "register":
        args = [entry]
    elif entry in {"agree", "create-child"}:
        args = [entry, "--spec", p.spec({})]
    elif entry in {"delegate", "start-worker"}:
        args = [
            entry,
            "--child",
            "10",
            "--branch",
            "codex/blocked",
            "--worktree",
            str(p.repo),
        ]
        args += (
            ["--agent", "codex"] if entry == "delegate" else ["--prompt-file", p.note()]
        )
    else:
        args = [entry, "--child", "10", "--prompt-file", p.note()]
    if entry != "register":
        # Exercise the real gate after a previously registered work Issue has
        # been converted to proposal; no new registration may bypass it.
        state = p.state()
        proposal = copy.deepcopy(state["issues"]["10"])
        state["issues"]["10"].update(
            title="[開発] 提案になる前の親",
            body=state["issues"]["1"]["body"],
            labels=[{"name": "issue:parent"}],
        )
        p.save(state)
        p.cli("register", parent="10")
        state = p.state()
        state["issues"]["10"] = proposal
        p.save(state)
    denied = p.cli(*args, parent="10", ok=False)
    assert "proposal" in denied.stderr or "実行根拠" in denied.stderr
    assert not (p.root / "runner-calls.jsonl").exists()


def test_propose_adds_candidates_and_replaces_edited_projection(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    original = copy.deepcopy(p.state()["issues"]["1"])
    record(p)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: json.loads(publish(p).stdout), range(2)))
    assert {r["issue"] for r in results} == {10} and post_count(p) == 1
    created = p.state()["issues"]["10"]
    assert {x["name"] for x in created["labels"]} == {
        "issue:proposal",
        "needs-human",
        "enhancement",
    }
    assert created["body"].startswith(
        "<!-- parent-improvement-proposal:v1 parent=1 generation=1 -->"
    )
    assert "2099-01-08T00:00:00Z" in created["body"]
    state = p.state()
    state["issues"]["10"]["body"] = created["body"].replace(
        "状態: 未判断", "状態: 承認"
    )
    p.save(state)
    extract(p, ok=False)
    record(p, fixed_key="second")
    assert json.loads(publish(p).stdout)["issue"] == 10
    body = p.state()["issues"]["10"]["body"]
    assert "I-1-second" in body and "状態: 承認" not in body
    assert p.state()["issues"]["1"] == original and post_count(p) == 1
    assert p.state()["parents"].get("10") is None


@pytest.mark.parametrize("failure", ["create_response", "membership"])
def test_propose_response_loss_is_recovered_without_a_second_post(
    parent_env: ParentEnv, failure: str
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    fault = (
        "PARENT_FAIL_AFTER_CREATE"
        if failure == "create_response"
        else "PARENT_FAIL_AFTER_COMMENT_CONTAINING"
    )
    p.env[fault] = "1" if failure == "create_response" else '"event": "proposed"'
    publish(p, ok=False)
    del p.env[fault]
    source_finished(p)
    p.env["CLAUDE_CODE_SESSION_ID"] = "replacement-recorder"
    assert json.loads(publish(p).stdout)["issue"] == 10
    assert post_count(p) == 1
    assert json.loads(p.cli("status").stdout)["assignment"] == "ended"


def test_unknown_creation_never_reposts_while_the_index_is_incomplete(
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
    state["issues"]["10"] = saved
    p.save(state)
    publish(p)
    assert post_count(p) == 1


def approve_many(p: ParentEnv, keys: list[str]) -> dict[str, Any]:
    spec = decision_spec()
    del spec["revision"], spec["previous_decision"]
    spec["candidates"] = {k: {"revision": 1, "previous_decision": None} for k in keys}
    args = ["improvement-decide"]
    for k in keys:
        args += ["--candidate", k]
    return dict(
        json.loads(p.cli(*args, "--action", "approve", "--spec", p.spec(spec)).stdout)
    )


def test_multi_approval_requires_whole_set_and_extracts_one_parent(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    record(p, fixed_key="second")
    record(p, fixed_key="unselected")
    publish(p)
    result = approve_many(p, ["I-1-result-access", "I-1-second"])
    assert len(result["decisions"]) == 2
    assert "missing=['second']" in extract(p, ok=False).stderr
    assert (
        "extra=['unselected']"
        in extract(
            p, "I-1-result-access", "I-1-second", "I-1-unselected", ok=False
        ).stderr
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        created = list(
            pool.map(
                lambda _: json.loads(
                    extract(p, "I-1-result-access", "I-1-second").stdout
                ),
                range(2),
            )
        )
    assert {r["issue"] for r in created} == {11} and post_count(p) == 2
    raw = p.state()["issues"]["11"]
    assert {x["name"] for x in raw["labels"]} == {"issue:parent", "enhancement"}
    assert raw["body"].startswith(
        "<!-- parent-improvement-extraction:v1 parent=1 proposal=10 candidates=I-1-result-access,I-1-second -->"
    )
    assert "I-1-unselected" not in raw["body"] and "issues/10" in raw["body"]
    proposal = p.state()["issues"]["10"]
    assert proposal["state"] == "closed" and proposal["state_reason"] == "completed"
    assert (
        "取り出し済み → #11" in proposal["body"]
        and "状態: 提案終了" in proposal["body"]
    )
    source_finished(p)
    p.cli("register", parent="11")
    p.cli(
        "agree",
        "--improvement-approval",
        result["decisions"][1]["decision_id"],
        parent="11",
    )
    saved = next(
        e
        for e in reversed(
            importlib.import_module("parent.policy").events(
                p.comments(11), "operator", 11
            )
        )
        if e["event"] == "agreed"
    )["improvement_approval"]
    assert len(saved["approvals"]) == 2
    revoked = p.cli(
        "improvement-decide",
        "--candidate",
        "I-1-second",
        "--action",
        "revoke",
        "--spec",
        p.spec(
            decision_spec(
                previous=result["decisions"][1]["decision_id"],
                request="stop-second",
                quote="2件目を撤回して全体を停止する。",
            )
        ),
    )
    assert json.loads(revoked.stdout)["stop"]["state"] == "stopped"
    denied = p.cli(
        "agree",
        "--improvement-approval",
        result["decisions"][0]["decision_id"],
        parent="11",
        ok=False,
    )
    assert "I-1-second" in denied.stderr and "revoked" in denied.stderr


def test_revision_and_replayed_human_instruction_do_not_restore_approval(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    publish(p)
    approval = json.loads(decide(p, "approve").stdout)
    revoked = json.loads(
        decide(
            p,
            "revoke",
            decision_spec(
                previous=approval["decision_id"], request="human-2", quote="撤回する。"
            ),
        ).stdout
    )
    replayed = json.loads(decide(p, "approve").stdout)
    assert (
        replayed["state"] == "revoked"
        and replayed["decision_id"] == approval["decision_id"]
    )
    assert (
        "new human instruction"
        in decide(
            p,
            "approve",
            decision_spec(previous=revoked["decision_id"], request="reuse-quote"),
            ok=False,
        ).stderr
    )
    revised = candidate_spec()
    revised["scope"] += " 取得一覧も含む。"
    p.cli("improvement", "--key", "result-access", "--spec", p.spec(revised))
    assert (
        "different candidate revision"
        in decide(
            p,
            "approve",
            decision_spec(
                previous=revoked["decision_id"],
                request="stale-revision",
                quote="修正版を承認する。",
            ),
            ok=False,
        ).stderr
    )
    extract(p, ok=False)
    revised_approval = json.loads(
        decide(
            p,
            "approve",
            decision_spec(
                previous=revoked["decision_id"],
                revision=2,
                request="approve-revision-2",
                quote="取得一覧を追加したrevision 2を承認する。",
            ),
        ).stdout
    )
    extract(p)
    assert "取得一覧も含む" in p.state()["issues"]["11"]["body"]
    assert revised_approval["decision_id"] != approval["decision_id"]


@pytest.mark.parametrize("fault", ["post", "record", "close"])
def test_extract_recovers_parent_and_proposal_close(
    parent_env: ParentEnv, fault: str
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    publish(p)
    decide(p, "approve")
    key = (
        "PARENT_FAIL_AFTER_CREATE"
        if fault == "post"
        else "PARENT_FAIL_AFTER_COMMENT_CONTAINING"
        if fault == "record"
        else "PARENT_FAIL_PATCH_ISSUE"
    )
    p.env[key] = (
        "1"
        if fault == "post"
        else '"event": "extracted"'
        if fault == "record"
        else "10"
    )
    extract(p, ok=False)
    del p.env[key]
    assert json.loads(extract(p).stdout)["issue"] == 11
    assert p.state()["issues"]["10"]["state"] == "closed" and post_count(p) == 2
    assert (
        "different fixed extraction" in extract(p, "I-1-other", ok=False).stderr
        or "recorded proposal" in extract(p, "I-1-other", ok=False).stderr
    )


def test_expiry_uses_created_at_and_retries_comment_and_close(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    publish(p)
    decide(p, "approve")
    before = p.cli("improvement-expire", "--now", "2099-01-07T23:59:59Z")
    assert json.loads(before.stdout)["closed"] == []
    state = p.state()
    state["issues"]["10"]["body"] = state["issues"]["10"]["body"].replace(
        "2099-01-08", "2199-01-08"
    )
    p.save(state)
    assert json.loads(
        p.cli("improvement-expire", "--now", "2099-01-08T00:00:00Z", "--dry-run").stdout
    )["pending"] == [10]
    p.env["PARENT_FAIL_PATCH_ISSUE"] = "10"
    failed = p.cli("improvement-expire", "--now", "2099-01-08T00:00:00Z", ok=False)
    assert json.loads(failed.stdout)["errors"]
    del p.env["PARENT_FAIL_PATCH_ISSUE"]
    assert json.loads(
        p.cli("improvement-expire", "--now", "2099-01-08T00:00:00Z").stdout
    )["closed"] == [10]
    assert (
        json.loads(p.cli("improvement-expire", "--now", "2099-01-09T00:00:00Z").stdout)[
            "closed"
        ]
        == []
    )
    proposal = p.state()["issues"]["10"]
    assert (
        proposal["state_reason"] == "not_planned"
        and "状態: 期限切れ" in proposal["body"]
    )
    assert (
        sum(
            "parent-improvement-expired:v1" in c["body"]
            for c in p.state()["comments"]["10"]
        )
        == 1
    )
    assert "承認済み・未取り出し（承認失効）" in p.state()["comments"]["10"][0]["body"]
    decide(
        p,
        "hold",
        decision_spec(
            previous=json.loads(decide(p, "approve").stdout)["decision_id"],
            request="late",
            quote="保留。",
        ),
        ok=False,
    )
    extract(p, ok=False)
    assert json.loads(publish(p).stdout)["issue"] is None
    record(p, fixed_key="new-generation")
    assert json.loads(publish(p).stdout)["generation"] == 2
    assert "I-1-result-access" not in p.state()["issues"]["11"]["body"]


def test_extracted_parent_agreement_keeps_snapshot_when_supplement_enabled(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    publish(p)
    state = p.state()
    state["supplement_enabled_since"] = "2026-10-30T00:00:00Z"
    state["issues"]["1"]["labels"].append({"name": "supplement:done"})
    p.save(state)
    approved = json.loads(decide(p, "approve").stdout)
    extract(p)
    source_finished(p)
    p.cli("register", parent="11")
    p.cli("agree", "--improvement-approval", approved["decision_id"], parent="11")
    state = p.state()
    state["issues"]["11"]["body"] += "\n## 追加要件\n\n無承認の追加作業。\n"
    p.save(state)
    assert (
        "requirements changed"
        in p.cli(
            "agree",
            "--improvement-approval",
            approved["decision_id"],
            parent="11",
            ok=False,
        ).stderr
    )


def test_multi_decision_response_loss_resumes_only_missing_candidate_events(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    record(p, fixed_key="second")
    publish(p)
    keys = ["I-1-result-access", "I-1-second"]
    value = decision_spec()
    del value["revision"], value["previous_decision"]
    value["candidates"] = {k: {"revision": 1, "previous_decision": None} for k in keys}
    args = [
        "improvement-decide",
        "--candidate",
        keys[0],
        "--candidate",
        keys[1],
        "--action",
        "approve",
        "--spec",
        p.spec(value),
    ]
    p.env["PARENT_FAIL_AFTER_COMMENT_CONTAINING"] = '"event": "decision"'
    p.cli(*args, ok=False)
    del p.env["PARENT_FAIL_AFTER_COMMENT_CONTAINING"]
    response = json.loads(p.cli(*args).stdout)
    assert [r["reused"] for r in response["decisions"]] == [True, False]
    comments = p.comments()
    decisions = [
        json.loads(c["body"].splitlines()[0][len("<!-- parent-improvement:v1 ") : -4])
        for c in comments
        if '"event": "decision"' in c["body"]
    ]
    assert (
        len(decisions) == 2
        and decisions[0]["data"]["human_confirmation"]
        == decisions[1]["data"]["human_confirmation"]
    )
    assert "状態: 承認" in p.state()["issues"]["10"]["body"]


def test_expiry_reports_partial_success_and_detects_removed_source_marker(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    publish(p)
    source_finished(p)
    p.cli("register", parent="2")
    p.cli(
        "improvement", "--key", "next", "--spec", p.spec(candidate_spec()), parent="2"
    )
    p.cli("improvement-propose", parent="2")
    p.env["PARENT_FAIL_PATCH_ISSUE"] = "11"
    failed = json.loads(
        p.cli("improvement-expire", "--now", "2099-01-08T00:00:00Z", ok=False).stdout
    )
    assert failed["closed"] == [10] and failed["errors"]
    del p.env["PARENT_FAIL_PATCH_ISSUE"]
    state = p.state()
    before = state["issues"]["11"]["body"]
    state["issues"]["11"]["body"] = "目印を手編集で削除した。"
    p.save(state)
    denied = p.cli("improvement-expire", "--now", "2099-01-08T00:00:00Z", ok=False)
    assert json.loads(denied.stdout)["errors"]
    state["issues"]["11"]["body"] = before
    p.save(state)
    assert json.loads(
        p.cli("improvement-expire", "--now", "2099-01-08T00:00:00Z").stdout
    )["closed"] == [11]


def test_expiry_needs_neither_assignment_nor_session_identity(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[3] / ".agents/ops/checkout_job.py"),
            "--repo",
            str(p.repo),
            "--",
            sys.executable,
            "-m",
            "parent",
            "--repo",
            "o/r",
            "improvement-expire",
            "--now",
            "2099-01-08T00:00:00Z",
        ],
        cwd=p.repo,
        env={
            k: v
            for k, v in p.env.items()
            if k
            not in {
                "CLAUDE_CODE_SESSION_ID",
                "CODEX_THREAD_ID",
                "CODEX_SESSION_ID",
                "PARENT_SESSION_ID",
            }
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["closed"] == []


def test_lost_propose_response_includes_candidates_recorded_before_recovery(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    p.env["PARENT_FAIL_AFTER_CREATE"] = "1"
    publish(p, ok=False)
    del p.env["PARENT_FAIL_AFTER_CREATE"]
    record(p, fixed_key="recorded-during-recovery")
    result = json.loads(publish(p).stdout)
    assert set(result["candidates"]) == {
        "I-1-result-access",
        "I-1-recorded-during-recovery",
    }
    assert post_count(p) == 1


def test_partial_proposal_membership_recovers_after_published_candidate_revision(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    record(p, fixed_key="second")
    p.env["PARENT_FAIL_AFTER_COMMENT_CONTAINING"] = '"event": "proposed"'
    publish(p, ok=False)
    del p.env["PARENT_FAIL_AFTER_COMMENT_CONTAINING"]
    assert sum('"event": "proposed"' in c["body"] for c in p.comments()) == 1
    approval = json.loads(decide(p, "approve").stdout)["decision_id"]

    revised = candidate_spec()
    revised["scope"] += " 一覧からの取得も対象にする。"
    p.cli("improvement", "--key", "result-access", "--spec", p.spec(revised))
    recovered = json.loads(publish(p).stdout)
    assert recovered["issue"] == 10
    assert set(recovered["candidates"]) == {"I-1-result-access", "I-1-second"}
    body = p.state()["issues"]["10"]["body"]
    assert revised["scope"] in body and "revision 2" in body
    assert "状態: 承認" not in body
    assert (
        "different candidate revision"
        in decide(
            p,
            "approve",
            decision_spec(previous=approval, request="stale-revision"),
            ok=False,
        ).stderr
    )
    decide(
        p,
        "approve",
        decision_spec(
            previous=approval,
            revision=2,
            request="revised-approval",
            quote="一覧からの取得を含む更新後の候補を承認する。",
        ),
    )
    assert "状態: 承認" in p.state()["issues"]["10"]["body"]
    publish(p)
    assert post_count(p) == 1
    assert sum('"event": "proposed"' in c["body"] for c in p.comments()) == 2


def test_removed_extraction_marker_before_first_agreement_cannot_unbind_approval(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    publish(p)
    approval = json.loads(decide(p, "approve").stdout)["decision_id"]
    extract(p)
    decide(
        p,
        "revoke",
        decision_spec(
            previous=approval, request="human-stop", quote="取り出した改善を撤回する。"
        ),
    )
    source_finished(p)
    p.cli("register", parent="11")
    inputs = json.loads(p.cli("basis-input", parent="11").stdout)
    state = p.state()
    state["issues"]["11"]["body"] = state["issues"]["11"]["body"].split("\n", 1)[1]
    p.save(state)
    # Supply the valid input a body-only caller could obtain after the edit.
    # The missing marker must veto the basis independently of input freshness.
    inputs["issue"].pop("improvement")
    inputs["issue"]["body_sha256"] = importlib.import_module("parent.policy").digest(
        importlib.import_module("parent.execution").requirement_body(
            state["issues"]["11"]["body"]
        )
    )
    basis = {
        "schema_version": 1,
        "kind": "autonomous_decision",
        "purpose": "未担当の具体的な作業を進める",
        "scope": "issueに記載した変更範囲を守る",
        "behavior": "共通APIから委任とレビューを要求する",
        "acceptance": "fake CLIで境界を確認する",
        "rationale": "元の承認を本文から削除している。",
        "sources": ["https://github.com/o/r/issues/11"],
        "input_sha256": importlib.import_module("parent.policy").digest(inputs),
        "policy_version": "autonomous-start/v1",
    }
    for command, spec in (
        ("basis-record", {"basis": basis, "inputs": inputs, "ignored_comments": []}),
        (
            "agree",
            {
                **{k: basis[k] for k in ("purpose", "scope", "behavior", "acceptance")},
                "human_confirmation": "本文に記載した通常parentの範囲で進めてください。",
            },
        ),
    ):
        denied = p.cli(command, "--spec", p.spec(spec), parent="11", ok=False)
        assert "extraction" in denied.stderr and "provenance" in denied.stderr
    assert not any(
        '"event": "execution_basis"' in c["body"] or '"event": "agreed"' in c["body"]
        for c in p.comments(11)
    )


def test_reopened_proposal_does_not_restore_its_candidates(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    publish(p)
    decide(p, "approve")
    state = p.state()
    state["issues"]["10"]["state_reason"] = "reopened"
    p.save(state)
    assert "reopened" in publish(p, ok=False).stderr
    extract(p, ok=False)
    assert (
        "closed"
        in decide(
            p,
            "hold",
            decision_spec(
                previous=json.loads(decide(p, "approve").stdout)["decision_id"],
                request="after-reopen",
                quote="保留する。",
            ),
            ok=False,
        ).stderr
    )


def test_expiry_without_checkout_launcher_is_rejected_before_github(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    result = subprocess.run(
        [sys.executable, "-m", "parent", "--repo", "o/r", "improvement-expire"],
        cwd=p.repo,
        env=p.env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 75 and "checkout_job.py" in result.stderr
    assert p.calls() == []


def test_extraction_response_loss_then_revocation_repairs_existing_parent_without_authority(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    record(p)
    publish(p)
    approval = json.loads(decide(p, "approve").stdout)["decision_id"]
    p.env["PARENT_FAIL_AFTER_CREATE"] = "1"
    extract(p, ok=False)
    del p.env["PARENT_FAIL_AFTER_CREATE"]
    decide(
        p,
        "revoke",
        decision_spec(
            previous=approval,
            request="stop-during-recovery",
            quote="応答が途切れた改善の承認を撤回する。",
        ),
    )
    result = json.loads(extract(p).stdout)
    assert result["issue"] == 11 and result["reused"] is True and post_count(p) == 2
    assert p.state()["issues"]["10"]["state"] == "closed"
    source_finished(p)
    p.cli("register", parent="11")
    assert (
        "revoked"
        in p.cli(
            "agree", "--improvement-approval", approval, parent="11", ok=False
        ).stderr
    )
