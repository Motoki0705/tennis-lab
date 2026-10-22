"""Ordered name catalog and durable parent-bound assignment lifetimes."""

from __future__ import annotations

import re
import sqlite3
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
        "SELECT * FROM name_assignments WHERE name=? ORDER BY id DESC LIMIT 1", (name,)
    ).fetchone()
    if not row or row[-1] is not None:
        raise BoardError("name is unassigned or retired")
    return Assignment(*row)


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
        latest = connection.execute(
            "SELECT id FROM name_assignments WHERE name=? ORDER BY id DESC LIMIT 1",
            (previous.name,),
        ).fetchone()[0]
        if previous.retired_at is None or latest == previous.id:
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
    for name in names:
        latest_row = connection.execute(
            "SELECT * FROM name_assignments WHERE name=? ORDER BY id DESC LIMIT 1",
            (name,),
        ).fetchone()
        if latest_row:
            latest_assignment = Assignment(*latest_row)
            if latest_assignment.retired_at is None:
                continue
            retired = datetime.fromisoformat(latest_assignment.retired_at)
            if retired.tzinfo is None:
                raise BoardError("invalid name retirement timestamp")
            if now - retired < REUSE_DELAY or pending_check(
                connection, latest_assignment.id
            ):
                continue
        cursor = connection.execute(
            "INSERT INTO name_assignments(name,repo,parent,identifier,assigned_at) VALUES(?,?,?,?,?)",
            (name, repo.casefold(), parent, identifier, stamp),
        )
        if cursor.lastrowid is None:
            raise BoardError("name allocation did not return an assignment")
        return Assignment(
            cursor.lastrowid, name, repo.casefold(), parent, identifier, stamp, None
        )
    raise BoardError("name set exhausted; a human must extend names.txt")


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
