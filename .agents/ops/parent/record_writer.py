"""Durable, reconciled comments; new machine records have one evidence destination."""

from __future__ import annotations

import subprocess
from contextlib import ExitStack, nullcontext
from typing import TYPE_CHECKING, Any

from worker_logs.journal import atomic, locked

from .history import comment_url, parse_history
from .judgement_comments import read_json
from .policy import digest, number, object_value

if TYPE_CHECKING:
    from .session import ParentSession


def post_once(
    session: ParentSession, issue: int, body: str, key: str
) -> dict[str, Any]:
    """A lost response is reconciled. Absence after a sent intent never causes POST."""
    session.public(body)
    gh = session.gh
    path = (
        session.store.directory
        / "comment-outbox"
        / f"{digest([gh.repo.casefold(), issue, key])}.json"
    )
    identity = {
        "version": 1,
        "repo": gh.repo.casefold(),
        "issue": number(issue),
        "publisher": session.login,
        "body_sha256": digest(body),
        "key": key,
    }
    with ExitStack() as locks, gh.fresh_reads():
        with session.store.unlocked_read() if session.store._lock else nullcontext():
            locks.enter_context(locked(path))
        if path.is_symlink():
            raise ValueError("comment outbox is a symlink")
        saved = object_value(read_json(path.read_text())) if path.exists() else None
        if saved is not None and (
            set(saved) != set(identity) | {"state", "comment_id"}
            or any(saved.get(k) != v for k, v in identity.items())
            or saved["state"] not in {"sent", "bound"}
            or type(saved["version"]) is not int
            or (saved["state"] == "sent" and saved["comment_id"] is not None)
        ):
            raise ValueError("comment outbox conflict; inspect the original operation")
        if saved is not None and saved["state"] == "bound":
            number(saved["comment_id"])
        matches = [
            c
            for c in gh.comments(issue)
            if c.get("user", {}).get("login") == session.login and c.get("body") == body
        ]
        if len(matches) > 1:
            raise ValueError("duplicate comments for one publication")
        if (
            saved
            and saved["state"] == "bound"
            and (not matches or matches[0]["id"] != saved["comment_id"])
        ):
            raise ValueError("published comment changed or disappeared")
        if not matches:
            if saved is not None:
                raise ValueError(
                    "comment publication uncertain; reconcile before any new POST"
                )
            atomic(path, {**identity, "state": "sent", "comment_id": None})
            try:
                gh.write(
                    "POST", f"repos/{gh.repo}/issues/{issue}/comments", {"body": body}
                )
            except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
                raise ValueError(
                    "comment publication uncertain; retry only to reconcile"
                ) from exc
            matches = [
                c
                for c in gh.comments(issue)
                if c.get("user", {}).get("login") == session.login
                and c.get("body") == body
            ]
            if len(matches) != 1:
                raise ValueError("comment publication could not be verified")
        comment = matches[0]
        atomic(
            path, {**identity, "state": "bound", "comment_id": number(comment["id"])}
        )
        return {
            **comment,
            "_parent_record_issue": issue,
            "_parent_record_source": "legacy_parent_comment"
            if issue == session.parent
            else "evidence_issue",
        }


def write_record(session: ParentSession, body: str) -> dict[str, Any]:
    """Legacy bytes are reused, never copied. Validate the prospective full history."""
    history = session.gh.record_history(session.parent, session.login)
    for comment in history.comments:
        if (
            comment.get("user", {}).get("login") == session.login
            and comment.get("body") == body
        ):
            if comment.get("_parent_record_source") == "evidence_issue":
                return post_once(
                    session, number(comment["_parent_record_issue"]), body, digest(body)
                )
            return comment
    issue = session.gh.evidence_issues().ensure(
        session.parent,
        session.login,
        store=session.store if session.store._lock else None,
    )
    rows = session.gh.comments(issue.number)
    cid = 1 + max((c["id"] for c in history.comments + rows), default=0)
    parse_history(
        session.gh.repo,
        session.parent,
        session.login,
        history.parent_comments,
        [*rows, {"id": cid, "body": body, "user": {"login": session.login}}],
        evidence_issue=issue.number,
    ).require_valid()
    return post_once(session, issue.number, body, digest(body))


def url(session: ParentSession, comment: dict[str, Any]) -> str:
    return comment_url(session.gh.repo, session.parent, comment)
