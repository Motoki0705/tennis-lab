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


# Entry N upgrades user_version N to N+1. Append migrations; never edit a
# released migration. Posting/state tables belong in the next migration.
MIGRATIONS: tuple[Callable[[sqlite3.Connection], None], ...] = (_names_schema,)
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
