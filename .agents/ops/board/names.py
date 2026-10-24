"""Ordered name catalog and durable parent-bound assignment lifetimes."""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import activity
from .errors import BoardError

REUSE_DELAY = timedelta(days=30)
PendingCheck = Callable[[sqlite3.Connection, int], bool]
CATALOG = Path(__file__).with_name("names.txt")


def one_edit_apart(left: str, right: str) -> bool:
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right, strict=True)) == 1
    if len(left) > len(right):
        left, right = right, left
    if len(right) - len(left) != 1:
        return False
    index = next(
        (i for i, (a, b) in enumerate(zip(left, right, strict=False)) if a != b),
        len(left),
    )
    return left[index:] == right[index + 1 :]


def validate_names(values: Sequence[str]) -> tuple[str, ...]:
    if not values or any(not re.fullmatch(r"[A-Z][a-zA-Z]*", name) for name in values):
        raise BoardError("name set requires ASCII letters and initial capitals")
    folded = [name.casefold() for name in values]
    if "all" in folded:
        raise BoardError("name set contains the reserved name all")
    if len(set(folded)) != len(folded):
        raise BoardError("name set contains a duplicate")
    if any(one_edit_apart(a, b) for i, a in enumerate(folded) for b in folded[i + 1 :]):
        raise BoardError("name set contains spellings one edit apart")
    return tuple(values)


def load_names() -> tuple[str, ...]:
    return validate_names(CATALOG.read_text(encoding="ascii").splitlines())


def canonical_name(value: str, catalog: Sequence[str] | None = None) -> str:
    names = load_names() if catalog is None else validate_names(catalog)
    if value.casefold() == "all":
        raise BoardError("all is reserved, not an individual name")
    for name in names:
        if name.casefold() == value.casefold():
            return name
    raise BoardError("name is outside the configured name set")


def timestamp(now: datetime) -> str:
    if now.tzinfo is None or now.utcoffset() is None:
        raise BoardError("board clock requires a timezone-aware timestamp")
    return now.astimezone(UTC).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class Assignment:
    id: int
    name: str
    repo: str
    parent: int
    identifier: str | None
    assigned_at: str
    retired_at: str | None

    def public(self) -> dict[str, str | int | None]:
        return {
            "name": self.name,
            "repo": self.repo,
            "parent": self.parent,
            "identifier": self.identifier,
            "assigned_at": self.assigned_at,
            "retired_at": self.retired_at,
        }


def for_parent(
    connection: sqlite3.Connection, repo: str, parent: int
) -> Assignment | None:
    row = connection.execute(
        "SELECT * FROM name_assignments WHERE repo=? AND parent=? ORDER BY id DESC LIMIT 1",
        (repo.casefold(), parent),
    ).fetchone()
    return Assignment(*row) if row else None


def resolve_name(
    connection: sqlite3.Connection, value: str, *, catalog: Sequence[str] | None = None
) -> Assignment:
    name = canonical_name(value, catalog)
    row = connection.execute(
        "SELECT * FROM name_assignments WHERE name=? AND retired_at IS NULL", (name,)
    ).fetchone()
    if not row:
        raise BoardError("name is unassigned or retired")
    return Assignment(*row)


def _reserved(
    connection: sqlite3.Connection, name: str, assignment_id: int | None
) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM name_renames WHERE old_name=? AND assignment_id IS NOT ? LIMIT 1",
            (name, assignment_id),
        ).fetchone()
        is not None
    )


def _latest_name(connection: sqlite3.Connection, name: str) -> Assignment | None:
    # A rename can give an older assignment ID a newer name lifetime.
    row = connection.execute(
        "SELECT * FROM name_assignments WHERE name=? "
        "ORDER BY (retired_at IS NULL) DESC, retired_at DESC, id DESC LIMIT 1",
        (name,),
    ).fetchone()
    return Assignment(*row) if row else None


def _available(
    connection: sqlite3.Connection,
    name: str,
    now: datetime,
    pending: PendingCheck,
    assignment_id: int | None = None,
) -> bool:
    if _reserved(connection, name, assignment_id):
        return False
    assignment = _latest_name(connection, name)
    if assignment is None:
        return True
    if assignment.retired_at is None:
        return False
    retired = datetime.fromisoformat(assignment.retired_at)
    if retired.tzinfo is None:
        raise BoardError("invalid name retirement timestamp")
    return now - retired >= REUSE_DELAY and not pending(connection, assignment.id)


def _select_name(
    connection: sqlite3.Connection,
    names: Sequence[str],
    now: datetime,
    pending: PendingCheck,
    assignment_id: int | None = None,
) -> str:
    counts = Counter(
        row[0][0].casefold()
        for row in connection.execute(
            "SELECT name FROM name_assignments WHERE retired_at IS NULL"
        )
    )
    # Stable sorting retains catalog order within the least-used initial.
    for name in sorted(names, key=lambda value: counts[value[0].casefold()]):
        if _available(connection, name, now, pending, assignment_id):
            return name
    raise BoardError("name set exhausted; a human must extend names.txt")


def ensure_name(
    connection: sqlite3.Connection,
    repo: str,
    parent: int,
    identifier: str | None,
    *,
    now: datetime,
    catalog: Sequence[str] | None = None,
    pending: PendingCheck | None = None,
) -> Assignment:
    """Caller holds Store.transaction; selection and insert are one write unit."""
    names = load_names() if catalog is None else validate_names(catalog)
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", repo) or parent < 1:
        raise BoardError("invalid parent reference")
    stamp = timestamp(now)
    previous = for_parent(connection, repo, parent)
    if previous:
        canonical_name(previous.name, names)
        latest = _latest_name(connection, previous.name)
        if (previous.retired_at is None or latest == previous) and not _reserved(
            connection, previous.name, previous.id
        ):
            # Resuming the same parent is not reuse by a different parent.
            connection.execute(
                "UPDATE name_assignments SET retired_at=NULL, identifier=? WHERE id=?",
                (identifier, previous.id),
            )
            return Assignment(
                previous.id,
                previous.name,
                previous.repo,
                previous.parent,
                identifier,
                previous.assigned_at,
                None,
            )
    pending_check = pending if pending is not None else activity.has_pending
    name = _select_name(connection, names, now, pending_check)
    cursor = connection.execute(
        "INSERT INTO name_assignments(name,repo,parent,identifier,assigned_at) VALUES(?,?,?,?,?)",
        (name, repo.casefold(), parent, identifier, stamp),
    )
    if cursor.lastrowid is None:
        raise BoardError("name allocation did not return an assignment")
    return Assignment(
        cursor.lastrowid, name, repo.casefold(), parent, identifier, stamp, None
    )


def rename_name(
    connection: sqlite3.Connection,
    repo: str,
    parent: int,
    *,
    now: datetime,
    name: str | None = None,
    catalog: Sequence[str] | None = None,
) -> Assignment:
    """Rename one live assignment in the caller's transaction, retaining its ID."""
    names = load_names() if catalog is None else validate_names(catalog)
    stamp = timestamp(now)
    previous = for_parent(connection, repo, parent)
    if previous is None or previous.retired_at is not None:
        raise BoardError("parent has no live name to rename")
    canonical_name(previous.name, names)
    if name is None:
        selected = _select_name(
            connection, names, now, activity.has_pending, previous.id
        )
    else:
        selected = canonical_name(name, names)
        if selected == previous.name:
            return previous
        if not _available(connection, selected, now, activity.has_pending, previous.id):
            raise BoardError(
                "name is assigned, reserved by rename history or awaiting reuse"
            )
    connection.execute(
        "INSERT INTO name_renames(assignment_id,old_name,new_name,renamed_at) VALUES(?,?,?,?)",
        (previous.id, previous.name, selected, stamp),
    )
    connection.execute(
        "UPDATE name_assignments SET name=? WHERE id=?", (selected, previous.id)
    )
    return Assignment(
        previous.id,
        selected,
        previous.repo,
        previous.parent,
        previous.identifier,
        previous.assigned_at,
        previous.retired_at,
    )


def retire_name(
    connection: sqlite3.Connection, repo: str, parent: int, *, now: datetime
) -> Assignment | None:
    previous = for_parent(connection, repo, parent)
    if previous is None or previous.retired_at is not None:
        return previous
    stamp = timestamp(now)
    connection.execute(
        "UPDATE name_assignments SET retired_at=? WHERE id=?", (stamp, previous.id)
    )
    return Assignment(
        previous.id,
        previous.name,
        previous.repo,
        previous.parent,
        previous.identifier,
        previous.assigned_at,
        stamp,
    )
