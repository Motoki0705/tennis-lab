"""Shared SQLite registry. Files hold immutable evidence; rows hold execution state."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from .artifacts import artifact_lock
from .paths import AnnotationLayout


class Registry:
    def __init__(self, layout: AnnotationLayout) -> None:
        self.layout = layout

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.layout.root.mkdir(parents=True, exist_ok=True)
        database = sqlite3.connect(
            self.layout.registry, timeout=30, isolation_level=None
        )
        database.row_factory = sqlite3.Row
        try:
            database.execute("PRAGMA foreign_keys=ON")
            database.execute("BEGIN IMMEDIATE")
            yield database
            database.commit()
        except BaseException:
            database.rollback()
            raise
        finally:
            database.close()

    def initialize(self, component: str, schema: str) -> None:
        with artifact_lock(self.layout, "registry-schema"):
            database = sqlite3.connect(self.layout.registry, timeout=30)
            try:
                database.execute("PRAGMA journal_mode=WAL")
                database.execute(
                    "CREATE TABLE IF NOT EXISTS schema_versions (component TEXT PRIMARY KEY)"
                )
                database.executescript(schema)
                database.execute(
                    "INSERT OR IGNORE INTO schema_versions VALUES (?)", (component,)
                )
                database.commit()
            finally:
                database.close()
