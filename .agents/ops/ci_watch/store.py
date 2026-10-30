"""Transactional SHA cursor, immutable observations and content-addressed evidence."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class Store:
    def __init__(self, directory: Path, *, readonly: bool = False) -> None:
        self.directory = directory
        database = directory / "observations.sqlite3"
        if readonly:
            self.db = sqlite3.connect(
                f"{database.resolve().as_uri()}?mode=ro", uri=True
            )
        else:
            self.db = sqlite3.connect(database)
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1) or (
                version == 0
                and self.db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchone()
            ):
                self.db.close()
                raise ValueError("unsupported CI observation database version")
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS config (value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS commits (
                    ordinal INTEGER PRIMARY KEY, sha TEXT UNIQUE NOT NULL,
                    metadata TEXT NOT NULL, fact TEXT NOT NULL, checked_at TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY, kind TEXT NOT NULL, sha TEXT,
                    observed_at TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS event_sha ON events(sha, sequence);
                CREATE TABLE IF NOT EXISTS job_logs (
                    job_id INTEGER PRIMARY KEY, source TEXT NOT NULL
                );
                PRAGMA user_version = 1;
                """
            )
        if self.db.execute("PRAGMA user_version").fetchone()[0] != 1:
            self.db.close()
            raise ValueError("unsupported CI observation database version")

    def close(self) -> None:
        self.db.close()

    def configure(self, config: dict[str, Any]) -> dict[str, Any]:
        row = self.db.execute("SELECT value FROM config").fetchone()
        if row is None:
            if config["start_sha"] is None:
                raise ValueError("first collection requires --start-sha (inclusive)")
            with self.db:
                self.db.execute("INSERT INTO config VALUES (?)", (encode(config),))
            return config
        saved: dict[str, Any] = json.loads(row[0])
        candidate = {**config, "start_sha": config["start_sha"] or saved["start_sha"]}
        if saved != candidate:
            raise ValueError(
                "repository, workflows or start SHA differ from saved config"
            )
        return saved

    def config(self) -> dict[str, Any]:
        row = self.db.execute("SELECT value FROM config").fetchone()
        if row is None:
            raise ValueError("CI observation state is not initialized")
        result: dict[str, Any] = json.loads(row[0])
        return result

    def put_object(self, body: str) -> str:
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        path = self.directory / "objects" / digest
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            self.object(digest)
        else:
            # One collector holds the state lock; replace never exposes a partial blob.
            temporary = path.with_suffix(".tmp")
            temporary.write_text(body, encoding="utf-8")
            temporary.replace(path)
        return digest

    def object(self, digest: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid evidence object ID")
        body = (self.directory / "objects" / digest).read_text(encoding="utf-8")
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != digest:
            raise ValueError(f"corrupt evidence object {digest}")
        return body

    def commits(self) -> list[dict[str, Any]]:
        return [
            {
                "ordinal": n,
                **json.loads(metadata),
                "fact": json.loads(fact),
                "checked_at": checked,
            }
            for n, metadata, fact, checked in self.db.execute(
                "SELECT ordinal, metadata, fact, checked_at FROM commits ORDER BY ordinal"
            )
        ]

    def discover(self, commits: list[dict[str, Any]], now: str) -> None:
        with self.db:
            for commit in commits:
                fact: dict[str, Any] = {
                    "status": "not_collected",
                    "result": None,
                    "workflows": [],
                    "errors": [],
                }
                self.db.execute(
                    "INSERT INTO commits(sha, metadata, fact) VALUES (?, ?, ?)",
                    (commit["sha"], encode(commit), encode(fact)),
                )
                self.append(
                    "observation",
                    commit["sha"],
                    {**commit, **fact, "failures": []},
                    now,
                )

    def fact(self, commit_sha: str, fact: dict[str, Any], now: str) -> None:
        # publish() commits this update together with its classified observation.
        self.db.execute(
            "UPDATE commits SET fact=?, checked_at=? WHERE sha=?",
            (encode(fact), now, commit_sha),
        )

    def job_log(self, identifier: int) -> dict[str, str] | None:
        row = self.db.execute(
            "SELECT source FROM job_logs WHERE job_id=?", (identifier,)
        ).fetchone()
        if row is None:
            return None
        result: dict[str, str] = json.loads(row[0])
        return result

    def remember_log(self, identifier: int, source: dict[str, str]) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO job_logs VALUES (?, ?)", (identifier, encode(source))
            )

    def append(
        self, kind: str, commit_sha: str | None, payload: dict[str, Any], now: str
    ) -> bool:
        previous = self.db.execute(
            "SELECT payload FROM events WHERE kind=? AND sha IS ? ORDER BY sequence DESC LIMIT 1",
            (kind, commit_sha),
        ).fetchone()
        serialized = encode(payload)
        if previous and previous[0] == serialized:
            return False
        self.db.execute(
            "INSERT INTO events(kind, sha, observed_at, payload) VALUES (?, ?, ?, ?)",
            (kind, commit_sha, now, serialized),
        )
        return True

    def records(self, after: int, limit: int, commit_sha: str | None) -> dict[str, Any]:
        rows = self.db.execute(
            "SELECT sequence, kind, sha, observed_at, payload FROM events "
            "WHERE sequence>? AND (? IS NULL OR sha=?) ORDER BY sequence LIMIT ?",
            (after, commit_sha, commit_sha, limit + 1),
        ).fetchall()
        records = [
            {
                "sequence": seq,
                "kind": kind,
                "sha": s,
                "observed_at": at,
                "observation": json.loads(body),
            }
            for seq, kind, s, at, body in rows[:limit]
        ]
        return {
            "schema_version": 1,
            "config": self.config(),
            "records": records,
            "next_after": records[-1]["sequence"] if records else after,
            "has_more": len(rows) > limit,
        }
