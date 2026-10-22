"""Append-only posts, frozen mentions and explicit read/response operations."""

from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from .access import as_parent, parent_state
from .delivery import Notifiers, dispatch
from .errors import BoardError
from .names import Assignment, load_names, resolve_name, timestamp
from .records import post_row, post_view, require_participant
from .safety import public, text
from .service import _assigned, _reference
from .storage import Store, directory

# Scan code blocks and inline code as well as prose. A word/email/path prefix
# prevents an @ token from being interpreted as a board mention.
MENTION = re.compile(r"(?<![\w@.+/\-])@([A-Za-z]+)(?![\w@\-])", re.ASCII)


def _mentions(body: str, recipients: Sequence[Assignment], broadcast: bool) -> None:
    catalog = {name.casefold() for name in load_names()}
    addressed = {recipient.name.casefold() for recipient in recipients}
    mentioned = {match[1].casefold() for match in MENTION.finditer(body)}
    if (mentioned & catalog) - addressed or ("all" in mentioned and not broadcast):
        raise BoardError("body mentions a board name outside the recipients")


def _recipient(assignment: Assignment, parents: Mapping[str, Any]) -> str:
    owner = parents.get(f"{assignment.repo}#{assignment.parent}")
    if assignment.retired_at is not None or (owner and owner["status"] == "ended"):
        raise BoardError("recipient name is retired")
    if owner is None:
        raise BoardError("recipient has no registered parent")
    engine = owner["session"].partition(":")[0]
    if engine not in {"claude", "codex"}:
        raise BoardError("invalid recipient engine identity")
    return "pending" if owner["status"] == "active" else "held"


def _destinations(
    connection: sqlite3.Connection,
    parents: Mapping[str, Any],
    sender: Assignment,
    to: Sequence[str],
    original: dict[str, Any] | None,
) -> tuple[list[Assignment], bool]:
    destinations: dict[int, Assignment] = {}
    if original is not None and original["sender_assignment_id"] != sender.id:
        row = connection.execute(
            "SELECT * FROM name_assignments WHERE id=?",
            (original["sender_assignment_id"],),
        ).fetchone()
        assignment = Assignment(*row)
        _recipient(assignment, parents)  # Never re-resolve a historical author by name.
        destinations[assignment.id] = assignment
    broadcast = any(value.casefold() == "all" for value in to)
    for value in to:
        if value.casefold() == "all":
            for key, owner in sorted(parents.items()):
                if owner["status"] == "active":
                    repo, parent = _reference(key)
                    if (repo, parent) != (sender.repo, sender.parent):
                        assignment = _assigned(connection, repo, parent)
                        destinations[assignment.id] = assignment
        else:
            assignment = resolve_name(connection, value)
            if assignment.id == sender.id:
                raise BoardError("sender cannot be a recipient")
            _recipient(assignment, parents)
            destinations[assignment.id] = assignment
    if not destinations:
        raise BoardError("post requires at least one recipient")
    return sorted(destinations.values(), key=lambda entry: entry.id), broadcast


def _identity(connection: sqlite3.Connection, now: datetime) -> tuple[str, str]:
    timestamp(now)  # Reject naive time before arithmetic.
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    elapsed = now.astimezone(UTC) - epoch
    micros = (elapsed.days * 86400 + elapsed.seconds) * 1000000 + elapsed.microseconds
    previous = connection.execute(
        "SELECT id FROM posts ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if previous is not None:
        micros = max(micros, int(previous[0][:16], 16) + 1)
    if not 0 <= micros < 2**64:
        raise BoardError("board clock is outside the supported range")
    try:
        stamp = timestamp(epoch + timedelta(microseconds=micros))
    except OverflowError:
        raise BoardError("board clock is outside the supported range") from None
    return f"{micros:016x}{secrets.token_hex(8)}", stamp


def _mark_read(
    connection: sqlite3.Connection, post_id: str, assignment_id: int, stamp: str
) -> None:
    connection.execute(
        "UPDATE recipients SET read_at=COALESCE(read_at, ?) WHERE post_id=? AND assignment_id=?",
        (stamp, post_id, assignment_id),
    )


def _created(env: Mapping[str, str], post_id: str) -> dict[str, Any]:
    with parent_state(env) as parents, Store(directory(env)).read() as connection:
        result = {"post": post_view(connection, post_id)}
        public(result, parents, env)
        return result


def _create(
    env: Mapping[str, str],
    *,
    to: Sequence[str],
    kind: str,
    subject: str,
    body: str,
    original_id: str | None,
    notifiers: Notifiers | None,
    now: datetime | None,
) -> dict[str, Any]:
    if notifiers is not None and set(notifiers) - {"claude", "codex"}:
        raise BoardError("unknown board notification engine")
    with as_parent(env, write=True) as (connection, sender, parents):
        original = (
            post_row(connection, original_id) if original_id is not None else None
        )
        if original is not None:
            require_participant(connection, original, sender.id)
            subject = ("Re: " + original["subject"])[:200]
        text(body, parents, env)
        text(subject, parents, env, subject=True)
        recipients, broadcast = _destinations(connection, parents, sender, to, original)
        _mentions(body, recipients, broadcast)
        post_id, stamp = _identity(
            connection, now if now is not None else datetime.now(UTC)
        )
        connection.execute(
            "INSERT INTO posts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                post_id,
                stamp,
                sender.id,
                sender.name,
                sender.repo,
                sender.parent,
                kind,
                int(broadcast),
                subject,
                body,
                original_id,
                original["thread_id"] if original else post_id,
            ),
        )
        for recipient in recipients:
            state = _recipient(recipient, parents)
            key = hashlib.sha256(f"board:{post_id}:{recipient.id}".encode()).hexdigest()
            connection.execute(
                "INSERT INTO recipients(post_id,assignment_id,name,repo,parent) VALUES(?,?,?,?,?)",
                (
                    post_id,
                    recipient.id,
                    recipient.name,
                    recipient.repo,
                    recipient.parent,
                ),
            )
            connection.execute(
                "INSERT INTO deliveries(post_id,assignment_id,idempotency_key,state,reason,retryable) VALUES(?,?,?,?,?,?)",
                (
                    post_id,
                    recipient.id,
                    key,
                    state,
                    "recipient_unavailable" if state == "held" else None,
                    int(state == "held"),
                ),
            )
        if original is not None:
            _mark_read(connection, original["id"], sender.id, stamp)
            connection.execute(
                """UPDATE recipients SET response_state='answered', responded_at=?
                   WHERE post_id=? AND assignment_id=? AND response_state IS NULL""",
                (stamp, original_id, sender.id),
            )
        public(post_view(connection, post_id), parents, env)
    # Commit the post and all recipient rows before attempting any bells.
    dispatch(env, notifiers if notifiers is not None else {}, post_id=post_id)
    return _created(env, post_id)


def post(
    env: Mapping[str, str],
    *,
    to: Sequence[str],
    kind: str,
    subject: str,
    body: str,
    notifiers: Notifiers | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    if kind not in {"info", "question", "request"}:
        raise BoardError("invalid board post kind")
    return _create(
        env,
        to=to,
        kind=kind,
        subject=subject,
        body=body,
        original_id=None,
        notifiers=notifiers,
        now=now,
    )


def reply(
    env: Mapping[str, str],
    *,
    post_id: str,
    body: str,
    to: Sequence[str] = (),
    notifiers: Notifiers | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    return _create(
        env,
        to=to,
        kind="reply",
        subject="",
        body=body,
        original_id=post_id,
        notifiers=notifiers,
        now=now,
    )


def read(env: Mapping[str, str], post_id: str) -> dict[str, Any]:
    with as_parent(env, write=True) as (connection, caller, parents):
        row = post_row(connection, post_id)
        require_participant(connection, row, caller.id)
        _mark_read(connection, post_id, caller.id, timestamp(datetime.now(UTC)))
        result = {"post": post_view(connection, post_id)}
        public(result, parents, env)
        return result


def ack(env: Mapping[str, str], *, post_id: str, state: str) -> dict[str, Any]:
    if state not in {"accepted", "declined", "done"}:
        raise BoardError("invalid board acknowledgement state")
    with as_parent(env, write=True) as (connection, caller, parents):
        post_row(connection, post_id)
        row = connection.execute(
            "SELECT response_state FROM recipients WHERE post_id=? AND assignment_id=?",
            (post_id, caller.id),
        ).fetchone()
        if row is None:
            raise BoardError("only a post recipient may acknowledge it")
        previous = row[0]
        if previous != state:
            if previous in {"declined", "done"} or (
                previous == "accepted" and state != "done"
            ):
                raise BoardError("board acknowledgement is already final")
            stamp = timestamp(datetime.now(UTC))
            _mark_read(connection, post_id, caller.id, stamp)
            connection.execute(
                "UPDATE recipients SET response_state=?, responded_at=? WHERE post_id=? AND assignment_id=?",
                (state, stamp, post_id, caller.id),
            )
        result = {"post": post_view(connection, post_id)}
        public(result, parents, env)
        return result


def inbox(env: Mapping[str, str], *, unread: bool = False) -> dict[str, Any]:
    with as_parent(env) as (connection, caller, parents):
        ids = connection.execute(
            "SELECT post_id FROM recipients WHERE assignment_id=?"
            + (" AND read_at IS NULL" if unread else "")
            + " ORDER BY post_id",
            (caller.id,),
        ).fetchall()
        result = {
            "posts": [post_view(connection, post_id, body=False) for (post_id,) in ids]
        }
        public(result, parents, env)
        return result


def status(env: Mapping[str, str]) -> dict[str, Any]:
    with as_parent(env) as (connection, caller, parents):
        ids = connection.execute(
            "SELECT id FROM posts WHERE sender_assignment_id=? ORDER BY id",
            (caller.id,),
        ).fetchall()
        result = {
            "posts": [post_view(connection, post_id, body=False) for (post_id,) in ids]
        }
        public(result, parents, env)
        return result


def thread(env: Mapping[str, str], post_id: str) -> dict[str, Any]:
    with as_parent(env, write=True) as (connection, caller, parents):
        root = post_row(connection, post_id)["thread_id"]
        if (
            connection.execute(
                """SELECT 1 FROM posts p WHERE p.thread_id=? AND
               (p.sender_assignment_id=? OR EXISTS
                (SELECT 1 FROM recipients r WHERE r.post_id=p.id AND r.assignment_id=?))""",
                (root, caller.id, caller.id),
            ).fetchone()
            is None
        ):
            raise BoardError("only a thread participant may read this thread")
        ids = connection.execute(
            "SELECT id FROM posts WHERE thread_id=? ORDER BY id", (root,)
        ).fetchall()
        stamp = timestamp(datetime.now(UTC))
        for (item,) in ids:
            _mark_read(connection, item, caller.id, stamp)
        result = {
            "thread_id": root,
            "posts": [post_view(connection, item) for (item,) in ids],
        }
        public(result, parents, env)
        return result
