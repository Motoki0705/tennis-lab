"""Independent, append-only repair observations; never write ci_watch's state."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from ci_watch.store import encode


def digest(value: Any) -> str:
    return hashlib.sha256(encode(value).encode()).hexdigest()


class Store:
    def __init__(self, directory: Path, *, readonly: bool = False) -> None:
        self.directory = directory.resolve()
        path = self.directory / "failures.sqlite3"
        if readonly:
            self.db = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        else:
            directory.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(path)
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1) or (
                version == 0
                and self.db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchone()
            ):
                self.db.close()
                raise ValueError("unsupported CI failure database version")
            if version == 0:
                self.db.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS config (value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS checkpoint (sequence INTEGER, sha256 TEXT);
                    CREATE TABLE IF NOT EXISTS failures (
                        signature TEXT PRIMARY KEY, payload TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS events (
                        sequence INTEGER PRIMARY KEY, signature TEXT NOT NULL,
                        observed_at TEXT NOT NULL, payload TEXT NOT NULL, sha256 TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS job_logs (
                        job_id INTEGER PRIMARY KEY, object TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS event_signature ON events(signature, sequence);
                    PRAGMA user_version = 1;
                    """
                )
        if self.db.execute("PRAGMA user_version").fetchone()[0] != 1:
            self.db.close()
            raise ValueError("unsupported CI failure database version")
        tables = {
            r[0]
            for r in self.db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if tables != {"config", "checkpoint", "failures", "events", "job_logs"}:
            self.db.close()
            raise ValueError("CI failure schema mismatch; preserve and repair state")
        if (
            self.db.execute(
                "SELECT 1 FROM failures f WHERE f.payload IS NOT ("
                "SELECT payload FROM events WHERE signature=f.signature ORDER BY sequence DESC LIMIT 1) LIMIT 1"
            ).fetchone()
            or self.db.execute(
                "SELECT 1 FROM events e WHERE NOT EXISTS(SELECT 1 FROM failures WHERE signature=e.signature) LIMIT 1"
            ).fetchone()
        ):
            self.db.close()
            raise ValueError(
                "CI failure history/projection mismatch; preserve and repair state"
            )

    def configure(self, config: dict[str, Any]) -> None:
        row = self.db.execute("SELECT value FROM config").fetchone()
        if row is None:
            with self.db:
                self.db.execute("INSERT INTO config VALUES (?)", (encode(config),))
        elif json.loads(row[0]) != config:
            raise ValueError("CI failure repository/source contract changed")

    def checkpoint(self) -> tuple[int, str] | None:
        row = self.db.execute("SELECT sequence, sha256 FROM checkpoint").fetchone()
        return (row[0], row[1]) if row else None

    def advance(self, sequence: int, sha256: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM checkpoint")
            self.db.execute("INSERT INTO checkpoint VALUES (?, ?)", (sequence, sha256))

    def save(self, row: dict[str, Any], now: str) -> None:
        payload = encode(row)
        previous = self.db.execute(
            "SELECT payload FROM failures WHERE signature=?", (row["signature"],)
        ).fetchone()
        if previous and previous[0] == payload:
            return
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO failures VALUES (?, ?)",
                (row["signature"], payload),
            )
            self.db.execute(
                "INSERT INTO events(signature, observed_at, payload, sha256) VALUES (?, ?, ?, ?)",
                (
                    row["signature"],
                    now,
                    payload,
                    digest(
                        {
                            "signature": row["signature"],
                            "observed_at": now,
                            "record": row,
                        }
                    ),
                ),
            )

    def stage(self, row: dict[str, Any], now: str) -> None:
        raw = self.db.execute(
            "SELECT payload FROM failures WHERE signature=?", (row["signature"],)
        ).fetchone()
        if raw is None or json.loads(raw[0])["occurrences"] != row["occurrences"]:
            self.save(row, now)

    def reference(self, sequence: int, signature: str, sha256: str) -> dict[str, Any]:
        return {
            "path": str(self.directory / "failures.sqlite3"),
            "entity": "ci_failure",
            "id": signature,
            "sequence": sequence,
            "sha256": sha256,
        }

    def event(self, raw: tuple[Any, ...]) -> dict[str, Any]:
        sequence, signature, at, payload, sha256 = raw
        row = json.loads(payload)
        if (
            not isinstance(row, dict)
            or row.get("signature") != signature
            or digest({"signature": signature, "observed_at": at, "record": row})
            != sha256
        ):
            raise ValueError(
                "CI failure event content mismatch; preserve and repair state"
            )
        return {
            "sequence": sequence,
            "signature": signature,
            "observed_at": at,
            "record": row,
            "ref": self.reference(sequence, signature, sha256),
        }

    def records(self, after: int, limit: int) -> dict[str, Any]:
        raw = self.db.execute(
            "SELECT sequence, signature, observed_at, payload, sha256 FROM events "
            "WHERE sequence>? ORDER BY sequence LIMIT ?",
            (after, limit + 1),
        ).fetchall()
        rows = [self.event(r) for r in raw[:limit]]
        return {
            "schema_version": 1,
            "records": rows,
            "next_after": rows[-1]["sequence"] if rows else after,
            "has_more": len(raw) > limit,
        }

    def show(self, signature: str) -> dict[str, Any]:
        raw = self.db.execute(
            "SELECT sequence, signature, observed_at, payload, sha256 FROM events "
            "WHERE signature=? ORDER BY sequence DESC LIMIT 1",
            (signature,),
        ).fetchone()
        if raw is None:
            raise ValueError("unknown CI failure signature")
        return self.event(raw)

    def original(self, reference: dict[str, Any]) -> dict[str, Any]:
        if (
            not isinstance(reference, dict)
            or set(reference) != {"path", "entity", "id", "sequence", "sha256"}
            or type(reference["sequence"]) is not int
            or reference["sequence"] < 1
        ):
            raise ValueError("invalid CI failure reference")
        raw = self.db.execute(
            "SELECT sequence, signature, observed_at, payload, sha256 FROM events "
            "WHERE sequence=?",
            (reference.get("sequence"),),
        ).fetchone()
        if raw is None:
            raise ValueError("unknown CI failure event")
        event = self.event(raw)
        if reference != event["ref"]:
            raise ValueError("CI failure reference identity/content mismatch")
        return event

    def summaries(self, offset: int, limit: int, unresolved: bool) -> dict[str, Any]:
        condition = (
            "WHERE json_extract(f.payload, '$.status') != 'recovered'"
            if unresolved
            else ""
        )
        total = self.db.execute(
            f"SELECT COUNT(*) FROM failures f {condition}"
        ).fetchone()[0]
        rows = [
            self.event(raw)
            for raw in self.db.execute(
                "SELECT e.sequence, e.signature, e.observed_at, e.payload, e.sha256 "
                "FROM failures f JOIN events e ON e.sequence=("
                "SELECT MAX(sequence) FROM events WHERE signature=f.signature) "
                f"{condition} ORDER BY f.signature LIMIT ? OFFSET ?",
                (limit, offset),
            )
        ]
        return {
            "schema_version": 1,
            "total": total,
            "offset": offset,
            "has_more": offset + limit < total,
            "failures": [
                {
                    "signature": r["signature"],
                    "status": r["record"]["status"],
                    "reason": r["record"]["reason"],
                    "ref": r["ref"],
                }
                for r in rows
            ],
        }

    def put_object(self, text: str) -> str:
        sha256 = hashlib.sha256(text.encode()).hexdigest()
        directory = self.directory / "objects"
        directory.mkdir(exist_ok=True)
        path = directory / sha256
        if path.exists():
            self.object(sha256)
        else:
            temporary = path.with_suffix(".tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
        return sha256

    def job_log(self, job_id: int) -> str | None:
        row = self.db.execute(
            "SELECT object FROM job_logs WHERE job_id=?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        self.object(row[0])
        return str(row[0])

    def remember_log(self, job_id: int, text: str) -> str:
        sha256 = self.put_object(text)
        with self.db:
            self.db.execute("INSERT INTO job_logs VALUES (?, ?)", (job_id, sha256))
        return sha256

    def object(self, sha256: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("invalid CI failure evidence object")
        text = (self.directory / "objects" / sha256).read_text(encoding="utf-8")
        if hashlib.sha256(text.encode()).hexdigest() != sha256:
            raise ValueError("CI failure evidence object content mismatch")
        return text
