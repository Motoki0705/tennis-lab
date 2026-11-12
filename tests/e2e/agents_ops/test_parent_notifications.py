"""Explicit parent decisions, durable evidence, and one completion notification."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from tests.e2e.agents_ops.test_parent_session import ParentEnv, evidence_number
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


def notification(
    p: ParentEnv, kind: str = "impact", key: str = "training-wait"
) -> dict:
    note = p.root / "reason.md"
    note.write_text(
        "Implementation and integration complete; human demonstration unverified."
        if kind == "completion"
        else "Evaluation changed; the full training run must wait."
    )
    source = json.loads(p.cli("note", "--body-file", str(note)).stdout)["url"]
    details = {
        "impact": {
            "result": "Evaluation changed",
            "decision": "Wait",
            "affected": ["full training"],
            "next_action": "Check the evaluation",
            "achievements": [],
            "remaining": ["Full training after evaluation repair"],
        },
        "completion": {
            "achievements": ["Nine child implementations and integration"],
            "remaining": [
                "Human demonstration unverified",
                "Camera comparison unpublished",
            ],
            "artifacts": ["https://github.com/o/r/pull/100"],
        },
        "question": {
            "question": "May the full training run resume?",
            "resume_condition": "Human approval",
        },
        "agreement": {
            "change": "CI policy changed",
            "impact": "Every push runs CI",
            "answers": [
                {"source": "conversation:2026-11-12", "quote": "Run CI on every push"}
            ],
        },
    }[kind]
    return {
        "schema_version": 1,
        "key": key,
        "kind": kind,
        "summary": "A parent decision",
        "evidence": [source],
        "resolves": [],
        "details": details,
    }


def notify(p: ParentEnv, spec: dict, **kwargs):
    path = p.root / "notification.json"
    path.write_text(json.dumps(spec))
    return p.cli("notify", "--spec", str(path), **kwargs)


def test_explicit_notifications_keep_sources_and_question_state(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.cli("register")
    assert not p.state()["comments"].get("1")
    question = notification(p, "question", "resume-training")
    first = json.loads(notify(p, question).stdout)
    assert len(p.state()["comments"]["1"]) == 1
    assert first["url"] in p.state()["issues"]["1"]["body"]
    changed = notification(p, "agreement", "ci-policy")
    changed["resolves"] = [question["key"]]
    result = json.loads(notify(p, changed).stdout)
    body = p.state()["issues"]["1"]["body"]
    assert "判断待ち: なし" in body
    assert result["url"] in body
    assert "Run CI on every push" in p.state()["comments"]["1"][-1]["body"]
    notify(p, changed)
    assert len(p.state()["comments"]["1"]) == 2


def test_completion_is_fixed_across_retry_release_and_new_generation(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.cli("register")
    spec = notification(p, "completion", "completion")
    first = json.loads(notify(p, spec).stdout)
    note = p.root / "ended.md"
    note.write_text(
        "Five training and nine evaluation conditions; human acceptance remains."
    )
    notify(p, spec)
    p.cli("release", "--ended", "--note-file", str(note))
    p.cli("release", "--ended", "--note-file", str(note))
    p.cli(
        "handoff",
        "--accept",
        "--note-file",
        str(note),
        extra={"CLAUDE_CODE_SESSION_ID": "next-owner"},
    )
    again = json.loads(
        notify(p, spec, extra={"CLAUDE_CODE_SESSION_ID": "next-owner"}).stdout
    )
    assert again["url"] == first["url"]
    assert len(p.state()["comments"]["1"]) == 1
    assert "Human demonstration unverified" in p.state()["issues"]["1"]["body"]
    spec["details"]["remaining"] = []
    assert (
        "conflict"
        in notify(
            p, spec, extra={"CLAUDE_CODE_SESSION_ID": "next-owner"}, ok=False
        ).stderr
    )


def test_notification_response_loss_reconciles_without_reposting(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.cli("register")
    spec = notification(p)
    notify(p, spec, extra={"PARENT_FAIL_AFTER_COMMENT": "1"}, ok=False)
    notify(p, spec)
    assert len(p.state()["comments"]["1"]) == 1


@pytest.mark.parametrize(
    "fault", ["missing_evidence", "unknown_kind", "unresolved_key"]
)
def test_invalid_notification_never_posts_to_parent(
    parent_env: ParentEnv, fault: str
) -> None:
    p = parent_env
    p.cli("register")
    spec = notification(p)
    if fault == "missing_evidence":
        spec["evidence"] = ["https://github.com/o/r/issues/1#issuecomment-999999"]
    elif fault == "unknown_kind":
        spec["kind"] = "routine"
    else:
        spec["resolves"] = ["unknown-question"]
    notify(p, spec, ok=False)
    assert not p.state()["comments"].get("1")


def test_uncertain_parent_post_never_retries_without_remote_evidence(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.cli("register")
    spec = notification(p)
    notify(p, spec, extra={"PARENT_FAIL_PATH": "repos/o/r/issues/1/comments"}, ok=False)
    for _ in range(2):
        assert "uncertain" in notify(p, spec, ok=False).stderr
    assert not p.state()["comments"]["1"]
    assert (
        len(
            [c for c in p.calls() if "POST" in c and "repos/o/r/issues/1/comments" in c]
        )
        == 1
    )


def test_later_human_close_approval_is_a_difference_and_keeps_unverified_work(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    done = json.loads(notify(p, notification(p, "completion", "completion")).stdout)
    quote = "これについては人間の合意済みとしてクローズしちゃっていいです。"
    state = p.state()
    state["comments"]["1"].append(
        {"id": 10000, "user": {"login": "operator"}, "body": quote}
    )
    p.save(state)
    spec = {
        "schema_version": 1,
        "key": "human-close",
        "kind": "agreement",
        "summary": "クローズ承認の差分。実演確認を実施した記録ではない。",
        "evidence": [done["record_url"]],
        "resolves": [],
        "details": {
            "change": "人間がクローズを承認",
            "impact": "完了報告の未確認事項・残課題は保持",
            "answers": [
                {
                    "source": "https://github.com/o/r/issues/1#issuecomment-10000",
                    "quote": quote,
                }
            ],
        },
    }
    approved = json.loads(notify(p, spec).stdout)
    rows = p.state()["comments"]["1"]
    marked = [
        json.loads(c["body"].splitlines()[0][len("<!-- parent-session:v1 ") : -4])
        for c in rows
        if c["body"].startswith("<!-- parent-session:")
    ]
    assert [r["key"] for r in marked].count("completion") == 1
    assert len(marked) == 2 and len(rows) == 3
    summary = p.state()["issues"]["1"]["body"]
    assert done["url"] in summary and approved["url"] in summary
    assert "Human demonstration unverified" in summary
    assert any(c["body"] == quote for c in rows)
    assert 10000 in [
        c["id"] for c in json.loads(p.cli("status").stdout)["input_changes"]["comments"]
    ]


def test_progress_preserves_human_notes_and_unchanged_sync_does_not_patch(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.cli("register")
    state = p.state()
    state["issues"]["1"]["body"] += "\nHuman note outside generated progress.\n"
    p.save(state)
    spec = notification(p)
    notify(p, spec)
    assert "Human note outside generated progress." in p.state()["issues"]["1"]["body"]
    assert "Full training after evaluation repair" in p.state()["issues"]["1"]["body"]
    offset = len(p.calls())
    p.cli("sync")
    p.cli("sync")
    assert not any("PATCH" in c for c in p.calls()[offset:])


def test_corrupted_notification_outbox_is_not_reset_to_an_empty_state(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.cli("register")
    spec = notification(p)
    notify(p, spec)
    directory = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/comment-outbox"
    )
    path = next(
        f
        for f in directory.glob("*.json")
        if json.loads(f.read_text())["key"] == "notification:training-wait"
    )
    path.write_text("{}")
    before = p.state()
    assert "outbox conflict" in notify(p, spec, ok=False).stderr
    assert p.state() == before and path.read_text() == "{}"


def test_legacy_only_parent_continues_without_copying_or_filling_fields(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    policy = importlib.import_module("parent.policy")
    agreement = {
        "purpose": "Legacy experiment",
        "scope": "Parent and child",
        "behavior": "Continue with fixed inputs",
        "acceptance": "Published evidence",
        "human_confirmation": "Human: proceed with these conditions",
    }
    old = [
        {
            "id": 1,
            "user": {"login": "operator"},
            "body": policy.marker({"parent": 1, "event": "assigned", "generation": 1})
            + "\n\n担当開始。担当指定と要件合意は別段階です。要件を確認してから分割・委任します。",
        },
        {
            "id": 2,
            "user": {"login": "operator"},
            "body": policy.marker(
                {
                    "parent": 1,
                    "event": "agreed",
                    "revision": 1,
                    "agreement": agreement,
                    "agreement_sha256": policy.digest(agreement),
                }
            ),
        },
    ]
    state = p.state()
    state["comments"]["1"] = old
    p.save(state)
    registry = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "parents": {
                    "o/r#1": {
                        "session": "claude:" + p.env["CLAUDE_CODE_SESSION_ID"],
                        "status": "active",
                        "generation": 1,
                        "tasks": {},
                        "reviews": [],
                    }
                },
            }
        )
    )
    assert json.loads(p.cli("status").stdout)["phase"] == "要件合意済み"
    p.cli(
        "note",
        "--body-file",
        p.note("Continuing the old parent with new evidence storage."),
    )
    p.child("legacy-next")
    assert p.state()["comments"]["1"] == old
    assert evidence_number(p.state()) != 1
    assert "requirements_sha256" not in json.loads(
        old[1]["body"].splitlines()[0][len(policy.PREFIX) : -4]
    )


@pytest.mark.parametrize("case", ["ci-policy", "training-wait", "training-milestone"])
def test_historical_parent_decisions_remain_reachable(
    parent_env: ParentEnv, case: str
) -> None:
    p = parent_env
    p.cli("register")
    source, kind, details, units = {
        "ci-policy": (
            "https://github.com/Motoki0705/tennis-lab/issues/1349#issuecomment-6522107679",
            "agreement",
            {
                "change": "朝のCI・自律修復を廃止し、CIはPR pushごとに戻す",
                "impact": "#1495を取消、#1461は多段探索を継続",
                "answers": [
                    {
                        "source": "conversation:2026-10-30",
                        "quote": "設計を根本から変更してCIはPR pushごとに戻します。",
                    }
                ],
            },
            ["PR push", "#1495", "#1461"],
        ),
        "training-wait": (
            "https://github.com/Motoki0705/tennis-lab/issues/1367#issuecomment-6379838901",
            "impact",
            {
                "result": "別配布testを評価CLIが拒否する",
                "decision": "修正を#1476へ分離",
                "affected": ["本学習#1377"],
                "next_action": "#1476の修正を含む固定版まで本学習を待機",
                "achievements": [],
                "remaining": ["#1476の修正確認", "#1377の本学習"],
            },
            ["#1476", "#1377", "待機"],
        ),
        "training-milestone": (
            "https://github.com/Motoki0705/tennis-lab/issues/1581#issuecomment-6759472046",
            "impact",
            {
                "result": "新5ペアの学習と9条件の評価を完了",
                "decision": "追加学習は行わない",
                "affected": ["#1586の集計"],
                "next_action": "比較とknowledge/PRを仕上げる",
                "achievements": ["新5ペア学習・9条件評価", "Drive照合と所有VM停止"],
                "remaining": ["#1586の比較・knowledge/PR"],
            },
            ["新5ペア", "9条件", "#1586", "Drive"],
        ),
    }[case]
    evidence = json.loads(
        p.cli(
            "note",
            "--body-file",
            p.note(
                "原出典: " + source + "\n" + json.dumps(details, ensure_ascii=False)
            ),
        ).stdout
    )["url"]
    spec = {
        "schema_version": 1,
        "kind": kind,
        "key": case,
        "summary": "原記録に基づく親の判断",
        "evidence": [evidence],
        "resolves": [],
        "details": details,
    }
    result = json.loads(notify(p, spec).stdout)
    parent = p.state()["comments"]["1"][0]["body"]
    assert all(unit in parent for unit in units)
    assert result["url"] in p.state()["issues"]["1"]["body"]
    assert evidence in parent
    assert source in next(
        c["body"] for c in p.comments() if c["id"] == int(evidence.rsplit("-", 1)[1])
    )
