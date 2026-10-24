"""Explicit host-local renames, independent of the operator's conversation."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from .access import parent_state
from .display import update
from .errors import BoardError
from .names import for_parent, rename_name
from .safety import public
from .service import _reference
from .storage import Store, directory


def rename(
    env: Mapping[str, str],
    *,
    parent: int,
    name: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    if parent < 1:
        raise BoardError("rename requires a positive parent number")
    with parent_state(env) as parents:
        matches = [key for key in parents if _reference(key)[1] == parent]
        if len(matches) != 1:
            raise BoardError("rename requires a unique registered parent number")
        key = matches[0]
        if parents[key]["status"] == "ended":
            raise BoardError("cannot rename an ended parent")
        repo, _ = _reference(key)
        with Store(directory(env)).transaction() as connection:
            previous = for_parent(connection, repo, parent)
            assignment = rename_name(
                connection,
                repo,
                parent,
                name=name,
                now=now if now is not None else datetime.now(UTC),
            )
            result = {
                "status": "unchanged" if previous == assignment else "renamed",
                "assignment_id": assignment.id,
                "repo": repo,
                "parent": parent,
                "old_name": previous.name if previous else None,
                "new_name": assignment.name,
            }
            public(result, parents, env)
    # Keep the committed rename if the separate display projection fails.
    return {**result, "display_sync": update(env, [assignment.id], renamed=True)[0]}
