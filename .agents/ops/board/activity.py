"""Assignment-bound unread, response and posting activity queries."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Activity:
    unread_count: int = 0
    awaiting_response: bool = False
    last_post_at: str | None = None
    awaiting_response_count: int = 0


def summary(connection: sqlite3.Connection, assignment_id: int | None) -> Activity:
    unread, waiting = connection.execute(
        """SELECT COUNT(CASE WHEN r.read_at IS NULL THEN 1 END),
                  COUNT(CASE WHEN p.kind IN ('question','request')
                                   AND r.response_state IS NULL THEN 1 END)
           FROM recipients r JOIN posts p ON p.id=r.post_id
           WHERE r.assignment_id IS ?""",
        (assignment_id,),
    ).fetchone()
    last_post = connection.execute(
        "SELECT MAX(created_at) FROM posts WHERE sender_assignment_id IS ?",
        (assignment_id,),
    ).fetchone()[0]
    return Activity(unread, bool(waiting), last_post, waiting)


def has_pending(connection: sqlite3.Connection, assignment_id: int) -> bool:
    state = summary(connection, assignment_id)
    return state.unread_count > 0 or state.awaiting_response
