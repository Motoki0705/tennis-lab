"""Fence board operations with the existing parent registry writer lock."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from lib.identity import session_identity
from parent.state import Store as ParentStore

from .errors import BoardError
from .names import Assignment
from .participants import HUMAN, Participant
from .service import _assigned, _reference
from .storage import Store, directory


@contextmanager
def parent_state(env: Mapping[str, str]) -> Iterator[dict[str, Any]]:
    # Never save the registry here. Parent actions release this lock before
    # board synchronization, so the lock order is parent -> board everywhere.
    with ParentStore(directory(env).parent / "parent-sessions").locked() as owners:
        yield owners.parents


@contextmanager
def as_parent(
    env: Mapping[str, str], *, write: bool = False
) -> Iterator[tuple[sqlite3.Connection, Assignment, dict[str, Any]]]:
    identity = session_identity(env)  # Reject nested/missing identity before IO.
    with parent_state(env) as parents:
        owned = [
            key
            for key, row in parents.items()
            if row["session"] == identity and row["status"] == "active"
        ]
        if len(owned) != 1:
            raise BoardError(
                "only the registered active parent session may use board messages"
            )
        repo, number = _reference(owned[0])
        store = Store(directory(env))
        with store.transaction() if write else store.read() as connection:
            yield connection, _assigned(connection, repo, number), parents


@contextmanager
def as_participant(
    env: Mapping[str, str], *, human: bool = False, write: bool = False
) -> Iterator[tuple[sqlite3.Connection, Participant, dict[str, Any]]]:
    if not human:
        with as_parent(env, write=write) as context:
            yield context
        return
    # Web selects this path in code; never derive human identity from a session
    # environment, a name argument, or the parent's registration.
    with parent_state(env) as parents:
        store = Store(directory(env))
        if write:
            # The Web API requires an already migrated board; it cannot
            # initialize state or upgrade an older schema on a browser request.
            with store.read():
                pass
        with store.transaction() if write else store.read() as connection:
            yield connection, HUMAN, parents
