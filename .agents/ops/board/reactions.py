"""Free reactions, independent of read receipts, responses and notifications."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from .access import as_participant
from .errors import BoardError
from .names import timestamp
from .safety import public

EMOJIS = {
    "+1": "👍",
    "eyes": "👀",
    "white_check_mark": "✅",
    "tada": "🎉",
    "pray": "🙏",
    "question": "❓",
    "warning": "⚠️",
}


def views(connection: sqlite3.Connection, post_id: str) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for (
        code,
        reactor_type,
        assignment_id,
        stamp,
        name,
        repo,
        parent,
    ) in connection.execute(
        """SELECT r.emoji, r.reactor_type, r.assignment_id, r.created_at,
                  a.name, a.repo, a.parent
           FROM reactions r LEFT JOIN name_assignments a ON a.id=r.assignment_id
           WHERE r.post_id=? ORDER BY r.emoji, r.reactor_type, r.assignment_id""",
        (post_id,),
    ):
        if code not in EMOJIS:
            raise BoardError("unknown board reaction emoji")
        if reactor_type == "parent":
            if assignment_id is None or name is None:
                raise BoardError("board reaction has no parent assignment")
        elif reactor_type == "human" and assignment_id is None:
            name = "human"
        else:
            raise BoardError("invalid board reaction participant")
        group = groups.setdefault(
            code, {"code": code, "emoji": EMOJIS[code], "reactors": []}
        )
        group["reactors"].append(
            {
                "type": reactor_type,
                "assignment_id": assignment_id,
                "name": name,
                "repo": repo,
                "parent": parent,
                "created_at": stamp,
            }
        )
    return list(groups.values())


def react(
    env: Mapping[str, str],
    *,
    post_id: str,
    emoji: str,
    remove: bool = False,
    now: datetime | None = None,
    human: bool = False,
) -> dict[str, Any]:
    # Post projections include reactions; import them only when mutating to
    # keep the shared read projection independent of this operation.
    from .records import post_row, post_view, require_thread_participant

    if emoji not in EMOJIS:
        raise BoardError("unknown board reaction emoji")
    with as_participant(env, human=human, write=True) as (connection, caller, parents):
        row = post_row(connection, post_id)
        require_thread_participant(connection, row["thread_id"], caller.id)
        reactor_type = "human" if human else "parent"
        if remove:
            connection.execute(
                "DELETE FROM reactions WHERE post_id=? AND reactor_type=? "
                "AND assignment_id IS ? AND emoji=?",
                (post_id, reactor_type, caller.id, emoji),
            )
        else:
            connection.execute(
                "INSERT INTO reactions(post_id,reactor_type,assignment_id,emoji,created_at) "
                "VALUES(?,?,?,?,?) ON CONFLICT "
                + (
                    "(post_id,emoji) WHERE reactor_type='human' "
                    if human
                    else "(post_id,assignment_id,emoji) WHERE reactor_type='parent' "
                )
                + "DO NOTHING",
                (
                    post_id,
                    reactor_type,
                    caller.id,
                    emoji,
                    timestamp(now if now is not None else datetime.now(UTC)),
                ),
            )
        result = {"post": post_view(connection, post_id)}
        public(result, parents, env)
        return result
