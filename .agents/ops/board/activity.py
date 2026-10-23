"""Posting/state integration boundary; schema v1 has no posts yet."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Activity:
    unread_count: int = 0
    awaiting_response: bool = False
    last_post_at: str | None = None


def summary(connection: sqlite3.Connection, assignment_id: int) -> Activity:
    """Replace with recipient/post queries together with the posting migration.

    The stable assignment ID identifies one name/parent lifetime. Keep historical
    recipient links when a retired name is later assigned to another parent.
    """
    return Activity()


def has_pending(connection: sqlite3.Connection, assignment_id: int) -> bool:
    state = summary(connection, assignment_id)
    return state.unread_count > 0 or state.awaiting_response
