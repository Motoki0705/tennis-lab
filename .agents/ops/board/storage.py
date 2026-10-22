"""Private SQLite storage with explicit, transactional schema migrations."""

from __future__ import annotations

import fcntl
import os
import sqlite3
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from .errors import BoardError


def _names_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE name_assignments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL COLLATE NOCASE,
            repo TEXT NOT NULL,
            parent INTEGER NOT NULL CHECK (parent > 0),
            identifier TEXT,
            assigned_at TEXT NOT NULL,
            retired_at TEXT,
            CHECK (retired_at IS NULL OR retired_at >= assigned_at)
        )"""
    )
    connection.execute(
        "CREATE UNIQUE INDEX live_name ON name_assignments(name) WHERE retired_at IS NULL"
    )
    connection.execute(
        "CREATE UNIQUE INDEX live_parent ON name_assignments(repo, parent) WHERE retired_at IS NULL"
    )


def _posts_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE posts (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            sender_assignment_id INTEGER NOT NULL REFERENCES name_assignments(id),
            sender_name TEXT NOT NULL,
            sender_repo TEXT NOT NULL,
            sender_parent INTEGER NOT NULL,
            kind TEXT NOT NULL CHECK (kind IN ('info','question','request','reply')),
            broadcast INTEGER NOT NULL CHECK (broadcast IN (0,1)),
            subject TEXT NOT NULL,
            body TEXT NOT NULL,
            in_reply_to TEXT REFERENCES posts(id),
            thread_id TEXT NOT NULL REFERENCES posts(id),
            CHECK ((kind = 'reply') = (in_reply_to IS NOT NULL))
        )"""
    )
    connection.execute("CREATE INDEX posts_thread ON posts(thread_id, id)")
    connection.execute("CREATE INDEX posts_sender ON posts(sender_assignment_id, id)")
    connection.execute(
        """CREATE TABLE recipients (
            post_id TEXT NOT NULL REFERENCES posts(id),
            assignment_id INTEGER NOT NULL REFERENCES name_assignments(id),
            name TEXT NOT NULL,
            repo TEXT NOT NULL,
            parent INTEGER NOT NULL,
            read_at TEXT,
            response_state TEXT CHECK (response_state IN ('answered','accepted','declined','done')),
            responded_at TEXT,
            PRIMARY KEY (post_id, assignment_id),
            CHECK ((response_state IS NULL) = (responded_at IS NULL)),
            CHECK (response_state IS NULL OR read_at IS NOT NULL)
        )"""
    )
    connection.execute(
        "CREATE INDEX recipients_inbox ON recipients(assignment_id, post_id)"
    )
    connection.execute(
        """CREATE TABLE deliveries (
            post_id TEXT NOT NULL,
            assignment_id INTEGER NOT NULL,
            idempotency_key TEXT NOT NULL UNIQUE,
            engine TEXT CHECK (engine IN ('claude','codex')),
            state TEXT NOT NULL CHECK (state IN ('pending','held','notified','delivery_failed')),
            reason TEXT,
            retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0,1)),
            attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            attempted_at TEXT,
            notified_at TEXT,
            PRIMARY KEY (post_id, assignment_id),
            FOREIGN KEY (post_id, assignment_id) REFERENCES recipients(post_id, assignment_id),
            CHECK (state != 'notified' OR notified_at IS NOT NULL)
        )"""
    )
    # Text and recipient identity are append-only even for direct SQL clients.
    for table in ("posts", "recipients", "deliveries"):
        connection.execute(
            f"CREATE TRIGGER {table}_no_delete BEFORE DELETE ON {table} "
            "BEGIN SELECT RAISE(ABORT, 'board history is append-only'); END"
        )
    connection.execute(
        "CREATE TRIGGER posts_no_update BEFORE UPDATE ON posts "
        "BEGIN SELECT RAISE(ABORT, 'board posts are immutable'); END"
    )
    # SQLite REPLACE can bypass delete triggers when recursive_triggers is off.
    for table, existing in (
        ("posts", "id=NEW.id"),
        ("recipients", "post_id=NEW.post_id AND assignment_id=NEW.assignment_id"),
        ("deliveries", "post_id=NEW.post_id AND assignment_id=NEW.assignment_id"),
    ):
        connection.execute(
            f"CREATE TRIGGER {table}_no_replace BEFORE INSERT ON {table} "
            f"WHEN EXISTS(SELECT 1 FROM {table} WHERE {existing}) "
            "BEGIN SELECT RAISE(ABORT, 'board history cannot be replaced'); END"
        )
    for table, columns in (
        ("recipients", "post_id, assignment_id, name, repo, parent"),
        ("deliveries", "post_id, assignment_id, idempotency_key"),
    ):
        connection.execute(
            f"CREATE TRIGGER {table}_identity BEFORE UPDATE OF {columns} ON {table} "
            "BEGIN SELECT RAISE(ABORT, 'board recipient identity is immutable'); END"
        )


# Entry N upgrades user_version N to N+1. Append migrations; never edit a
# released migration.
MIGRATIONS: tuple[Callable[[sqlite3.Connection], None], ...] = (
    _names_schema,
    _posts_schema,
)
SCHEMA_VERSION = len(MIGRATIONS)


def directory(env: Mapping[str, str]) -> Path:
    base = Path(env.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    if not base.is_absolute():
        raise BoardError("XDG_STATE_HOME must be absolute")
    return base / "tennis-lab-agents/board"


def _version(connection: sqlite3.Connection) -> int:
    version: int = connection.execute("PRAGMA user_version").fetchone()[0]
    if not 0 <= version <= SCHEMA_VERSION:
        raise BoardError("unknown board schema version")
    return version


def _validate(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise BoardError("board database integrity check failed")
    connection.execute(
        "SELECT id, name, repo, parent, identifier, assigned_at, retired_at FROM name_assignments LIMIT 0"
    )
    indexes = {
        row[1] for row in connection.execute("PRAGMA index_list(name_assignments)")
    }
    if not {"live_name", "live_parent"} <= indexes:
        raise BoardError("board schema is incomplete")
    connection.execute(
        "SELECT id, created_at, sender_assignment_id, sender_name, sender_repo, "
        "sender_parent, kind, broadcast, subject, body, in_reply_to, thread_id FROM posts LIMIT 0"
    )
    connection.execute(
        "SELECT post_id, assignment_id, name, repo, parent, read_at, response_state, "
        "responded_at FROM recipients LIMIT 0"
    )
    connection.execute(
        "SELECT post_id, assignment_id, idempotency_key, engine, state, reason, retryable, "
        "attempts, attempted_at, notified_at FROM deliveries LIMIT 0"
    )


class Store:
    def __init__(self, root: Path) -> None:
        self.directory = root
        self.path = root / "board.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        # Serialize first creation/migration, including the gap between O_EXCL
        # and SQLite's first transaction. Ordinary writes use SQLite's lock.
        fd = os.open(
            self.directory / "schema.lock",
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, "r+") as lock:
            os.fchmod(lock.fileno(), 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX)
            created = False
            try:
                db_fd = os.open(
                    self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600
                )
                created = True
            except FileExistsError:
                db_fd = os.open(self.path, os.O_RDWR | os.O_NOFOLLOW)
            with os.fdopen(db_fd, "rb") as database:
                os.fchmod(database.fileno(), 0o600)
            connection = sqlite3.connect(
                self.path.as_uri() + "?mode=rw",
                uri=True,
                isolation_level=None,
                timeout=20,
            )
            try:
                version = _version(connection)
                if version == 0 and not created:
                    raise BoardError(
                        "unversioned board database; refusing to replace it"
                    )
                connection.execute("PRAGMA foreign_keys=ON")
                if connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
                    raise BoardError("board requires SQLite WAL mode")
                connection.execute("BEGIN IMMEDIATE")
                for index in range(version, SCHEMA_VERSION):
                    MIGRATIONS[index](connection)
                    connection.execute(f"PRAGMA user_version={index + 1}")
                _validate(connection)
                connection.commit()
                return connection
            except BaseException:
                connection.close()
                raise

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """A complete write unit. Name checks and allocation share this lock."""
        connection = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except sqlite3.Error:
            raise BoardError(
                "board database unavailable, corrupt or inconsistent"
            ) from None
        finally:
            if connection is not None:
                connection.close()  # Rolls back an incomplete transaction.

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        """A snapshot that cannot initialize, migrate or mutate board state."""
        connection = None
        try:
            if not self.path.is_file() or self.path.is_symlink():
                raise BoardError(
                    "board database is missing; run parent register to assign a name"
                )
            connection = sqlite3.connect(
                self.path.as_uri() + "?mode=ro",
                uri=True,
                isolation_level=None,
                timeout=20,
            )
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            if _version(connection) != SCHEMA_VERSION:
                raise BoardError(
                    "board schema requires migration through parent register"
                )
            _validate(connection)
            yield connection
        except sqlite3.Error:
            raise BoardError(
                "board database unavailable, corrupt or inconsistent"
            ) from None
        finally:
            if connection is not None:
                connection.close()
