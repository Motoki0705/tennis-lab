"""Human agreement discussion freshness and explicit irrelevance classification."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .execution import ignored_comment_map, instruction_comments, sha256
from .judgement_comments import owned_comment
from .policy import digest, events, nonempty, number, object_value

if TYPE_CHECKING:
    from .session import ParentSession


def validate_inputs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("agreement comment inputs must be a list")
    for raw in value:
        item = object_value(raw)
        if set(item) != {"id", "login", "sha256"}:
            raise ValueError("malformed agreement comment input")
        number(item["id"])
        nonempty(item["login"], "comment login")
        sha256(item["sha256"])
    if [c["id"] for c in value] != sorted({c["id"] for c in value}):
        raise ValueError("agreement comment inputs must be unique and sorted")
    return value


def snapshot(session: ParentSession) -> list[dict[str, Any]]:
    return _inputs(instruction_comments(session, session.gh.comments(session.parent)))


def _inputs(comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        ({"id": c["id"], "login": c["login"], "sha256": digest(c)} for c in comments),
        key=lambda c: c["id"],
    )


def _observations(
    session: ParentSession, agreement: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    raw = session.gh.comments(session.parent)
    comments = _inputs(instruction_comments(session, raw))
    history = events(raw, session.login, session.parent)
    latest = next(
        (
            e
            for e in reversed(history)
            if e["event"] == "agreement_comments"
            and e["agreement_sha256"] == agreement["agreement_sha256"]
            and e["revision"] == agreement["revision"]
        ),
        agreement,
    )
    legacy_edits = set()
    if "comment_inputs" in latest:
        baseline = validate_inputs(latest["comment_inputs"])
        # Preserve agreements whose snapshots predate the shared display filter.
        displays = {
            c["id"]
            for c in raw
            if owned_comment(c, session.login, session.gh.repo, session.parent)
        }
        baseline = [c for c in baseline if c["id"] not in displays]
    else:
        # Old markers have no snapshot. Preserve them and use GitHub's comment
        # identity/edit time to detect later instructions without rewriting reads.
        anchor = next((c for c in raw if c.get("body") == agreement["body"]), None)
        if anchor is None:
            raise ValueError("human agreement changed during freshness validation")
        baseline = [c for c in comments if c["id"] < number(anchor["id"])]
        if anchor.get("created_at"):
            agreed_at = datetime.fromisoformat(anchor["created_at"])
            legacy_edits = {
                c["id"]
                for c in raw
                if c["id"] in {b["id"] for b in baseline}
                and c.get("updated_at")
                and datetime.fromisoformat(c["updated_at"]) > agreed_at
            }
    before, after = ({c["id"]: c for c in rows} for rows in (baseline, comments))
    changes = {}
    for n in before.keys() | after.keys():
        if n not in before:
            changes[n] = "added"
        elif n not in after:
            changes[n] = "deleted"
        elif before[n] != after[n] or n in legacy_edits:
            changes[n] = "edited"
    return comments, changes


def _stale(session: ParentSession, changes: Mapping[int, str]) -> ValueError:
    refs = ", ".join(
        f"ID={n} https://github.com/{session.gh.repo}/issues/{session.parent}#issuecomment-{n} ({changes[n]})"
        for n in sorted(changes)
    )
    return ValueError(
        f"unclassified agreement comments: {refs}。"
        "要件と無関係と確認できる現存コメントは agreement-ignore-comments "
        "--ignored-comments でID/digest/理由を明示してください。"
        "人間の指示を無関係として除外してはなりません。"
        "要件変更・停止指示・削除は checking → agree で再合意してください"
    )


def require_fresh(session: ParentSession, agreement: Mapping[str, Any]) -> None:
    _, changes = _observations(session, agreement)
    if changes:
        raise _stale(session, changes)


def record_ignored(session: ParentSession, value: Any) -> dict[str, Any]:
    session.owner()
    if session.gh.issue(session.parent)["state"] != "open":
        raise ValueError("parent issue is not open")
    if not isinstance(value, list) or not value:
        raise ValueError("ignored comments must be a nonempty list")
    ignored = ignored_comment_map(value)
    history = session.history()
    agreement = next(
        (
            e
            for e in reversed(history)
            if e["event"] in {"agreed", "checking", "execution_basis", "basis_revoked"}
        ),
        {},
    )
    if agreement.get("event") != "agreed":
        raise ValueError("comment classification requires a human agreement")
    comments, changes = _observations(session, agreement)
    current = {c["id"]: c for c in comments}
    for n, entry in ignored.items():
        if n not in current or entry["sha256"] != current[n]["sha256"]:
            raise ValueError("ignored comment changed or missing; classify it again")
    remaining = {n: kind for n, kind in changes.items() if n not in ignored}
    if remaining:
        raise _stale(session, remaining)
    event = {
        "event": "agreement_comments",
        "agreement_sha256": agreement["agreement_sha256"],
        "revision": agreement["revision"],
        "comment_inputs": comments,
        "ignored_comments": sorted(ignored.values(), key=lambda c: c["id"]),
    }
    previous = next(
        (e for e in reversed(history) if e["event"] == "agreement_comments"), {}
    )
    event["classification_revision"] = (
        previous["classification_revision"]
        if previous and all(previous.get(k) == v for k, v in event.items())
        else max(
            (
                e["classification_revision"]
                for e in history
                if e["event"] == "agreement_comments"
            ),
            default=0,
        )
        + 1
    )
    session.post(
        event,
        "合意後のコメントを確認し、要件と無関係と分類しました。\n\n"
        + "\n".join(f"- ID={n}: {c['reason']}" for n, c in sorted(ignored.items())),
    )
    # Publication is not atomic with discussion edits or a new checking event.
    session.agreed()
    return session.status()
