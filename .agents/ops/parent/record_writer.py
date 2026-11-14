"""Durable, reconciled comments; new machine records have one evidence destination."""

from __future__ import annotations

import subprocess
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any

from worker_logs.journal import atomic, locked

from . import comment_journal
from .history import comment_url, parse_history
from .policy import digest, number
from .state import RegistryConflict

if TYPE_CHECKING:
    from .session import ParentSession


def post_once(
    session: ParentSession, issue: int, body: str, key: str, *, projection: bool = False
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
        "version": 2,
        "repo": gh.repo.casefold(),
        "parent": session.parent,
        "issue": number(issue),
        "source": "pr"
        if projection
        else "legacy_parent_comment"
        if issue == session.parent
        else "evidence_issue",
        "publisher": session.login,
        "body_sha256": digest(body),
        "key": key,
    }
    with ExitStack() as locks, gh.fresh_reads():
        with gh.io_scope():
            locks.enter_context(locked(path))
        saved = (
            comment_journal.load(path) if path.exists() or path.is_symlink() else None
        )
        if saved is not None and saved["version"] == 1:
            identity = {
                k: v for k, v in identity.items() if k not in {"parent", "source"}
            }
            identity["version"] = 1
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
            except RegistryConflict:
                raise
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
    with session.gh.fresh_reads():
        return _write_record(session, body)


def _write_record(session: ParentSession, body: str) -> dict[str, Any]:
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
            return post_once(session, session.parent, body, digest(body))
    issue = session.gh.evidence_issues().ensure(
        session.parent,
        session.login,
        store=session.gh._store,
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


def replace_pending(session: ParentSession, comment: dict[str, Any], body: str) -> None:
    """Journal the sole mutable record transition before its idempotent PATCH."""
    from review.parent_review_policy import records

    session.public(body)
    before = records([comment], session.login)
    after = records([{"body": body, "user": {"login": session.login}}], session.login)
    if (
        len(before) != 1
        or len(after) != 1
        or before[0]["kind"] != "pending"
        or after[0]["kind"] not in {"result", "adoption"}
        or any(before[0][k] != after[0][k] for k in ("review_run_id", "request_sha256"))
        or after[0]["request"]["parent_issue"] != session.parent
    ):
        raise ValueError("only a matching pending review can be finalized")
    issue = number(comment.get("_parent_record_issue", session.parent))
    cid = number(comment["id"])
    row = {
        "version": 1,
        "repo": session.gh.repo.casefold(),
        "parent": session.parent,
        "issue": issue,
        "publisher": session.login,
        "comment_id": cid,
        "from_sha256": digest(comment["body"]),
        "to_sha256": digest(body),
    }
    path = (
        session.store.directory
        / "comment-updates"
        / f"{digest([row['repo'], issue, cid])}.json"
    )
    with ExitStack() as locks, session.gh.fresh_reads():
        with session.gh.io_scope():
            locks.enter_context(locked(path))
        saved = (
            comment_journal.load(path, update=True)
            if path.exists() or path.is_symlink()
            else None
        )
        if saved is not None and any(saved[k] != v for k, v in row.items()):
            raise ValueError("comment update journal conflict")
        observed = [c for c in session.gh.comments(issue) if c["id"] == cid]
        if (
            len(observed) != 1
            or observed[0].get("user", {}).get("login") != session.login
        ):
            raise ValueError("pending review changed or disappeared")
        current = digest(observed[0].get("body") or "")
        if current != row["to_sha256"]:
            if current != row["from_sha256"] or (saved and saved["state"] == "bound"):
                raise ValueError("pending review changed after publication")
            atomic(path, {**row, "state": "sent"})
            session.gh.write(
                "PATCH",
                f"repos/{session.gh.repo}/issues/comments/{cid}",
                {"body": body},
            )
            observed = [c for c in session.gh.comments(issue) if c["id"] == cid]
            if (
                len(observed) != 1
                or observed[0].get("user", {}).get("login") != session.login
                or observed[0].get("body") != body
            ):
                raise ValueError("review update could not be verified")
        atomic(path, {**row, "state": "bound"})
