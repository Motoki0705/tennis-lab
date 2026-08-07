"""Small SQLite persistence layer for OAuth and durable execution metadata."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

_TABLES = {
    "clients",
    "pending_authorizations",
    "authorization_codes",
    "access_tokens",
    "refresh_tokens",
    "revision_workspaces",
    "jobs",
    "training_jobs",
}


class SqliteStore:
    """Persist JSON records using fixed tables and opaque record identifiers."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        self._initialize()
        os.chmod(self.path, 0o600)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{self.path}{suffix}")
            if sidecar.exists():
                os.chmod(sidecar, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            for table in sorted(_TABLES):
                connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        record_key TEXT PRIMARY KEY,
                        payload TEXT NOT NULL,
                        expires_at REAL
                    )
                    """
                )

    @staticmethod
    def _table(name: str) -> str:
        if name not in _TABLES:
            raise ValueError(f"unsupported storage table: {name}")
        return name

    def put(
        self,
        table: str,
        key: str,
        payload: dict[str, Any],
        *,
        expires_at: float | None = None,
    ) -> None:
        table = self._table(table)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                f"""
                INSERT INTO {table}(record_key, payload, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(record_key) DO UPDATE SET
                    payload=excluded.payload,
                    expires_at=excluded.expires_at
                """,
                (key, encoded, expires_at),
            )

    def get(self, table: str, key: str) -> dict[str, Any] | None:
        table = self._table(table)
        now = time.time()
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT payload, expires_at FROM {table} WHERE record_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            expires_at = row["expires_at"]
            if expires_at is not None and float(expires_at) <= now:
                connection.execute(f"DELETE FROM {table} WHERE record_key = ?", (key,))
                return None
            return dict(json.loads(str(row["payload"])))

    def pop(self, table: str, key: str) -> dict[str, Any] | None:
        table = self._table(table)
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"SELECT payload, expires_at FROM {table} WHERE record_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(f"DELETE FROM {table} WHERE record_key = ?", (key,))
            expires_at = row["expires_at"]
            if expires_at is not None and float(expires_at) <= now:
                return None
            return dict(json.loads(str(row["payload"])))

    def delete(self, table: str, key: str) -> None:
        table = self._table(table)
        with self._connect() as connection:
            connection.execute(f"DELETE FROM {table} WHERE record_key = ?", (key,))

    def list(self, table: str, *, limit: int = 100) -> list[dict[str, Any]]:
        table = self._table(table)
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                f"DELETE FROM {table} WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            rows = connection.execute(
                f"SELECT payload FROM {table} ORDER BY rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(json.loads(str(row["payload"]))) for row in rows]
