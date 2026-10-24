"""Public post projections shared by CLI operations and notification dispatch."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from .errors import BoardError
from .reactions import views as reaction_views

REASONS = {
    "recipient_unavailable": "担当不在・配送保留",
    "recipient_retired": "宛先の担当が終了しています",
    "notifier_unconfigured": "通知アダプタが未設定です",
    "receiver_not_accepting": "受信設定が accept ではありません",
    "route_unavailable": "通知経路が利用できません（未送信）",
    "notification_rejected": "通知経路が受理を拒否しました",
    "notification_uncertain": "通知の受理結果が不明です（自動再送を停止）",
}


def post_row(connection: sqlite3.Connection, post_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{32}", post_id):
        raise BoardError("invalid board post ID")
    cursor = connection.execute("SELECT * FROM posts WHERE id=?", (post_id,))
    row = cursor.fetchone()
    if row is None:
        raise BoardError("board post does not exist")
    return dict(zip((field[0] for field in cursor.description), row, strict=True))


def recipient_views(
    connection: sqlite3.Connection, post_id: str
) -> list[dict[str, Any]]:
    cursor = connection.execute(
        """SELECT r.assignment_id, r.name, r.repo, r.parent, r.read_at,
                  r.response_state, r.responded_at, d.state AS delivery_state,
                  d.reason AS delivery_reason, d.notified_at, d.engine, d.attempts,
                  d.retryable
           FROM recipients r JOIN deliveries d
             ON r.post_id=d.post_id AND r.assignment_id IS d.assignment_id
           WHERE r.post_id=? ORDER BY r.assignment_id""",
        (post_id,),
    )
    entries = []
    for row in cursor:
        entry = dict(zip((field[0] for field in cursor.description), row, strict=True))
        entry["state"] = entry["response_state"] or (
            "read" if entry["read_at"] else entry["delivery_state"]
        )
        reason = entry["delivery_reason"]
        if reason is not None and reason not in REASONS:
            raise BoardError("unknown board delivery reason")
        entry["delivery_message"] = REASONS.get(reason)
        entry["retryable"] = bool(entry["retryable"])
        entries.append(entry)
    return entries


def post_view(
    connection: sqlite3.Connection, post_id: str, *, body: bool = True
) -> dict[str, Any]:
    row = post_row(connection, post_id)
    result = {
        "id": row["id"],
        "created_at": row["created_at"],
        "sender": {
            "assignment_id": row["sender_assignment_id"],
            "name": row["sender_name"],
            "repo": row["sender_repo"],
            "parent": row["sender_parent"],
        },
        "kind": row["kind"],
        "broadcast": bool(row["broadcast"]),
        "subject": row["subject"],
        "in_reply_to": row["in_reply_to"],
        "thread_id": row["thread_id"],
        "recipients": recipient_views(connection, post_id),
        "reactions": reaction_views(connection, post_id),
    }
    if body:
        result["body"] = row["body"]
    return result


def require_participant(
    connection: sqlite3.Connection, row: dict[str, Any], assignment_id: int | None
) -> None:
    # The local human can view every public post in the Web UI. Parent CLI
    # access remains limited to the post's saved participants.
    if assignment_id is None:
        return
    if (
        row["sender_assignment_id"] != assignment_id
        and connection.execute(
            "SELECT 1 FROM recipients WHERE post_id=? AND assignment_id=?",
            (row["id"], assignment_id),
        ).fetchone()
        is None
    ):
        raise BoardError("only a post sender or recipient may access this post")


def require_thread_participant(
    connection: sqlite3.Connection, thread_id: str, assignment_id: int | None
) -> None:
    if assignment_id is None:
        return
    if (
        connection.execute(
            """SELECT 1 FROM posts p WHERE p.thread_id=? AND
           (p.sender_assignment_id=? OR EXISTS
            (SELECT 1 FROM recipients r WHERE r.post_id=p.id AND r.assignment_id=?))""",
            (thread_id, assignment_id, assignment_id),
        ).fetchone()
        is None
    ):
        raise BoardError("only a thread participant may access this thread")
