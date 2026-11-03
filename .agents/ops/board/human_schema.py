"""Version 5: nullable parent identities for the single reserved human."""

from __future__ import annotations

import sqlite3

from .errors import BoardError


def upgrade(connection: sqlite3.Connection) -> None:
    if (
        connection.execute(
            "SELECT 1 FROM name_assignments WHERE lower(name)='human'"
        ).fetchone()
        is not None
    ):
        raise BoardError("reserved human name already has a parent assignment")
    # Copy into a complete parallel FK graph before dropping the old graph.
    # Foreign keys remain enabled; no ALTER of a referenced live table is needed.
    tables = ("posts", "recipients", "deliveries", "reactions")
    definitions = dict(
        connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' "
            "AND name IN ('posts','recipients','deliveries','reactions')"
        )
    )
    auxiliaries = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type IN ('index','trigger') "
        "AND tbl_name IN ('posts','recipients','deliveries','reactions') "
        "AND sql IS NOT NULL ORDER BY type, name"
    ).fetchall()
    for table in tables:
        sql = definitions[table].replace(
            f"CREATE TABLE {table} (", f"CREATE TABLE {table}_v5 (", 1
        )
        changes = {
            "posts": (
                (
                    "sender_assignment_id INTEGER NOT NULL",
                    "sender_assignment_id INTEGER",
                ),
                ("sender_repo TEXT NOT NULL", "sender_repo TEXT"),
                ("sender_parent INTEGER NOT NULL", "sender_parent INTEGER"),
            ),
            "recipients": (
                ("assignment_id INTEGER NOT NULL", "assignment_id INTEGER"),
                ("repo TEXT NOT NULL", "repo TEXT"),
                ("parent INTEGER NOT NULL", "parent INTEGER"),
            ),
            "deliveries": (
                ("assignment_id INTEGER NOT NULL", "assignment_id INTEGER"),
                (
                    "'pending','held','notified','delivery_failed'",
                    "'pending','held','notified','delivery_failed','available'",
                ),
            ),
            "reactions": (),
        }[table]
        for before, after in changes:
            if before not in sql:
                raise BoardError("inconsistent board schema for human migration")
            sql = sql.replace(before, after, 1)
        for referenced in tables:
            sql = sql.replace(
                f"REFERENCES {referenced}(", f"REFERENCES {referenced}_v5("
            )
        if table in {"posts", "recipients"}:
            prefix = "sender_" if table == "posts" else ""
            identity = (
                f"CHECK (({prefix}assignment_id IS NULL AND {prefix}name='human' "
                f"AND {prefix}repo IS NULL AND {prefix}parent IS NULL) OR "
                f"({prefix}assignment_id IS NOT NULL AND lower({prefix}name)!='human' "
                f"AND {prefix}repo IS NOT NULL AND {prefix}parent IS NOT NULL))"
            )
            sql = sql.rstrip().removesuffix(")") + ", " + identity + ")"
        elif table == "deliveries":
            sql = sql.rstrip().removesuffix(")") + (
                ", CHECK ((assignment_id IS NULL AND state='available' AND engine IS NULL "
                "AND reason IS NULL AND retryable=0 AND attempts=0 AND attempted_at IS NULL "
                "AND notified_at IS NULL) OR (assignment_id IS NOT NULL AND state!='available')), "
                "FOREIGN KEY (post_id) REFERENCES posts_v5(id))"
            )
        connection.execute(sql)
        connection.execute(f"INSERT INTO {table}_v5 SELECT * FROM {table}")
    # Self references in posts must be deferred during its implicit DROP delete.
    connection.execute("PRAGMA defer_foreign_keys=ON")
    for table in reversed(tables):
        connection.execute(f"DROP TABLE {table}")
    for table in tables:
        connection.execute(f"ALTER TABLE {table}_v5 RENAME TO {table}")
    for (sql,) in auxiliaries:
        # SQLite's NULL equality does not protect the human row from REPLACE.
        connection.execute(
            sql.replace(
                "assignment_id=NEW.assignment_id", "assignment_id IS NEW.assignment_id"
            )
        )
    for table in ("recipients", "deliveries"):
        connection.execute(
            f"CREATE UNIQUE INDEX {table}_human ON {table}(post_id) WHERE assignment_id IS NULL"
        )
    connection.execute(
        "CREATE TRIGGER deliveries_human_recipient BEFORE INSERT ON deliveries "
        "WHEN NEW.assignment_id IS NULL AND NOT EXISTS "
        "(SELECT 1 FROM recipients WHERE post_id=NEW.post_id AND assignment_id IS NULL) "
        "BEGIN SELECT RAISE(ABORT, 'human delivery requires its recipient'); END"
    )
    for operation in ("INSERT", "UPDATE OF name"):
        connection.execute(
            f"CREATE TRIGGER names_no_human_{operation.split()[0].lower()} "
            f"BEFORE {operation} ON name_assignments WHEN lower(NEW.name)='human' "
            "BEGIN SELECT RAISE(ABORT, 'human is reserved'); END"
        )
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise BoardError("inconsistent board history during human migration")
