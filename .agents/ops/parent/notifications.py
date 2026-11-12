"""Parent-selected notifications. The CLI checks structure and evidence, not meaning."""

from __future__ import annotations

import re
from html import escape
from typing import TYPE_CHECKING, Any

from .policy import MAX_NOTE_CHARS, digest, marker, nonempty, object_value

if TYPE_CHECKING:
    from .history import History, Record
    from .session import ParentSession

KINDS = {"question", "agreement", "impact", "completion"}
DETAILS = {
    "question": {"question", "resume_condition"},
    "agreement": {"change", "impact", "answers"},
    "impact": {
        "result",
        "decision",
        "affected",
        "next_action",
        "achievements",
        "remaining",
    },
    "completion": {"achievements", "remaining", "artifacts"},
}


def notice_text(value: Any) -> str:
    text = nonempty(value, "notification text")
    if len(text) > MAX_NOTE_CHARS:
        raise ValueError(f"notification text exceeds {MAX_NOTE_CHARS} characters")
    return text


def strings(value: Any, field: str, *, empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not empty):
        raise ValueError(f"{field} requires a list{' (nonempty)' if not empty else ''}")
    for item in value:
        notice_text(nonempty(item, field))
    if len(value) != len(set(value)):
        raise ValueError(f"duplicate {field}")
    return value


def validate(spec: dict[str, Any]) -> dict[str, Any]:
    if (
        set(spec)
        != {
            "schema_version",
            "key",
            "kind",
            "summary",
            "evidence",
            "resolves",
            "details",
        }
        or type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or not isinstance(spec["kind"], str)
        or spec["kind"] not in KINDS
    ):
        raise ValueError("invalid notification schema/kind")
    key = nonempty(spec["key"], "notification key")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", key) or (
        (spec["kind"] == "completion") != (key == "completion")
    ):
        raise ValueError(
            "completion uses the fixed key 'completion'; other keys must be distinct"
        )
    notice_text(spec["summary"])
    strings(spec["evidence"], "evidence")
    strings(spec["resolves"], "resolves", empty=True)
    if spec["kind"] == "question" and spec["resolves"]:
        raise ValueError("a question cannot resolve questions")
    details = object_value(spec["details"])
    if set(details) != DETAILS[spec["kind"]]:
        raise ValueError("invalid notification details")
    for name, value in details.items():
        if name in {"affected", "achievements", "remaining", "artifacts"}:
            strings(
                value,
                name,
                empty=name == "remaining"
                or (name == "achievements" and spec["kind"] == "impact"),
            )
            if name == "artifacts" and any(not v.startswith("https://") for v in value):
                raise ValueError("artifact links must use https")
        elif name == "answers":
            if not isinstance(value, list) or not value:
                raise ValueError("agreement requires original answers and sources")
            for answer in value:
                if not isinstance(answer, dict) or set(answer) != {"source", "quote"}:
                    raise ValueError("answer requires source/quote")
                notice_text(answer["source"])
                notice_text(answer["quote"])
        else:
            notice_text(value)
    return spec


def render(spec: dict[str, Any]) -> str:
    validate(spec)
    d = spec["details"]
    parts = [escape(spec["summary"], quote=False)]
    labels = {
        "question": "判断してほしいこと",
        "resume_condition": "再開条件",
        "change": "決まったこと・変更",
        "impact": "影響",
        "result": "結果",
        "decision": "親の判断",
        "affected": "影響する作業",
        "next_action": "次の一手",
        "achievements": "達成内容",
        "remaining": "未達・残件",
        "artifacts": "成果物",
    }
    for name in (
        "question",
        "resume_condition",
        "change",
        "impact",
        "answers",
        "result",
        "decision",
        "affected",
        "next_action",
        "achievements",
        "remaining",
        "artifacts",
    ):
        if name not in d:
            continue
        value = d[name]
        if name == "answers":
            parts.extend(
                f"回答の原文（{escape(a['source'], quote=False)}）:\n"
                + "\n".join(
                    "> " + escape(line, quote=False) for line in a["quote"].splitlines()
                )
                for a in value
            )
        else:
            parts.append(
                f"{labels[name]}: "
                + escape(
                    (" / ".join(value) or "なし") if isinstance(value, list) else value,
                    quote=False,
                )
            )
    parts.append("証拠:\n" + "\n".join(f"- {v}" for v in spec["evidence"]))
    return notice_text("\n\n".join(parts))


def saved(history: History) -> dict[str, Record]:
    return {
        r.payload["key"]: r
        for r in history.records
        if r.payload.get("event") == "notification_record"
    }


def completion(session: ParentSession) -> Record | None:
    return saved(session.gh.record_history(session.parent, session.login)).get(
        "completion"
    )


def notify(session: ParentSession, spec: dict[str, Any]) -> dict[str, Any]:
    from .record_writer import post_once, url

    session.owner()
    validate(spec)
    history = session.gh.record_history(session.parent, session.login)
    history.require_valid()
    existing = saved(history)
    previous = existing.get(spec["key"])
    if previous and previous.payload["notification"] != spec:
        raise ValueError(
            "notification key conflict; publish a distinct decision/difference"
        )
    sources = {r.url for r in history.records}
    if any(ref not in sources for ref in spec["evidence"]):
        raise ValueError(
            "notification evidence missing from the verified parent history"
        )
    if any(
        key not in existing
        or existing[key].payload["notification"]["kind"] != "question"
        for key in spec["resolves"]
    ):
        raise ValueError("notification resolves an unknown question")
    sha = digest(spec)
    record = session.post(
        {
            "event": "notification_record",
            "key": spec["key"],
            "notification_sha256": sha,
            "notification": spec,
        },
        render(spec),
    )
    record_url = url(session, record)
    text = render(spec) + f"\n\n[判断・通知の保存記録]({record_url})"
    event = {
        "parent": session.parent,
        "event": "notification",
        "key": spec["key"],
        "notification_sha256": sha,
        "record_url": record_url,
        "body_sha256": digest(text),
    }
    comment = post_once(
        session,
        session.parent,
        marker(event) + "\n\n" + text,
        "notification:" + spec["key"],
    )
    session.sync_progress()
    return {
        "parent": session.parent,
        "key": spec["key"],
        "url": url(session, comment),
        "record_url": record_url,
    }


def progress_lines(session: ParentSession, history: History) -> list[str]:
    from .policy import phase

    events = history.session_events()
    decisions = saved(history)
    notices = {
        r.payload["key"]: r.url
        for r in history.records
        if r.payload.get("event") == "notification"
    }
    resolved = {
        key for r in decisions.values() for key in r.payload["notification"]["resolves"]
    }
    questions = [
        r
        for key, r in decisions.items()
        if r.payload["notification"]["kind"] == "question" and key not in resolved
    ]
    review = next(
        (e["state"] for e in reversed(events) if e.get("event") == "review"), "未指示"
    )
    stage = (
        "workflow完了照合済み"
        if any(e.get("event") == "workflow_completed" for e in events)
        else "全体成果記録済み"
        if "completion" in decisions
        else phase(events)
    )
    lines = [
        f"現在の段階: {stage} / 担当: {session.row.get('status', '未登録')} / レビュー: {review}"
    ]
    lines.append(
        "判断待ち: "
        + (
            " / ".join(
                f"{r.payload['notification']['details']['question']}（[質問]({notices.get(r.payload['key'], r.url)})）"
                for r in questions
            )
            or "なし"
        )
    )
    achieved = next(
        (
            r
            for r in reversed(tuple(decisions.values()))
            if r.payload["notification"]["kind"] in {"completion", "impact"}
        ),
        None,
    )
    if achieved:
        d = achieved.payload["notification"]["details"]
        lines += [
            "主要成果: " + (" / ".join(d["achievements"]) or "未達"),
            "未達・残件: " + (" / ".join(d["remaining"]) or "なし"),
            "全体の結果・判断: " + notices.get(achieved.payload["key"], achieved.url),
        ]
    else:
        lines += [
            "主要成果・未達/残件: 全体の受入は未記録。各作業の成果物・検証を参照。"
        ]
    latest = next(
        (
            r
            for r in reversed(history.records)
            if r.payload.get("event") == "notification"
        ),
        None,
    )
    if latest:
        lines.append(f"最新の通知: {latest.url}")
    agreement = next(
        (
            r
            for r in reversed(history.records)
            if r.payload.get("event")
            in {"agreed", "proposal_agreed", "execution_basis"}
        ),
        None,
    )
    if agreement:
        lines.append(f"最新の合意・実行根拠: {agreement.url}")
    issue = session.gh.evidence_issues().find(session.parent, session.login)
    if issue:
        lines.append(f"機械証拠・判断の詳細: {issue.url}")
    pending = [
        f"[{key}]({r.url})" for key, r in decisions.items() if key not in notices
    ]
    if pending:
        lines.append("親通知の公開待ち: " + " / ".join(pending))
    return [
        escape(line.replace("\r", " ").replace("\n", " "), quote=False)
        for line in lines
    ]
