"""Version 6: accept agy recipients while preserving the delivery history."""

from __future__ import annotations

import sqlite3

from .errors import BoardError


def upgrade(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='deliveries'"
    ).fetchone()
    clause = "CHECK (engine IN ('claude','codex'))"
    if not row or clause not in row[0]:
        raise BoardError("inconsistent board delivery schema")
    auxiliaries = connection.execute(
        """SELECT sql FROM sqlite_master
           WHERE tbl_name='deliveries' AND type IN ('index','trigger') AND sql IS NOT NULL
           ORDER BY type, name"""
    ).fetchall()
    connection.execute(
        row[0]
        .replace("CREATE TABLE deliveries (", "CREATE TABLE deliveries_v6 (", 1)
        .replace(clause, "CHECK (engine IN ('claude','codex','agy'))", 1)
        # SQLite can quote a table name that an earlier migration renamed.
        .replace('CREATE TABLE "deliveries" (', "CREATE TABLE deliveries_v6 (", 1)
    )
    connection.execute("INSERT INTO deliveries_v6 SELECT * FROM deliveries")
    connection.execute("DROP TABLE deliveries")
    connection.execute("ALTER TABLE deliveries_v6 RENAME TO deliveries")
    for (sql,) in auxiliaries:
        connection.execute(sql)
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise BoardError("inconsistent board history during agy migration")
