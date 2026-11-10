"""Append-only parent histories, with independent preparation/publication entries.

The JSON file is atomically replaced under flock; its entry list only grows.
Storage primitives, JSON duplicate-key rejection and public errors are shared
with worker_logs. No HTTP, Git, runner or parent-registry lock is involved.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from worker_logs.journal import (
    PUBLIC_ERRORS,
    PublicationError,
    atomic,
    decode,
    locked,
    require,
)
from worker_logs.safety import Guard
from worker_logs.schema import digest, fields, json_bytes, sha

from .graph import references, states, validate_graph
from .projection import render, validate_public
from .schema import evidence_list, new_id, public_id, scope, timestamp, validate_record

PRIVATE_FIELDS = {"sessions", "runs", "identifiers", "roots", "checkout_commit"}


def private_context(value: Any) -> None:
    fields(value, PRIVATE_FIELDS)
    for key in ("sessions", "runs", "identifiers", "roots"):
        require(isinstance(value[key], list), "journal_source")
        require(
            all(isinstance(v, str) and v.strip() for v in value[key]), "journal_source"
        )
        require(len(value[key]) == len(set(value[key])), "journal_source")
    require(
        value["checkout_commit"] is None or digest(value["checkout_commit"], 40),
        "journal_source",
    )


def transform_guard(value: Any, repo: str) -> Guard:
    """Validate the private conditions frozen when a projection was prepared."""
    fields(value, {"version", "private", "roots", "repo", "commit"})
    require(type(value["version"]) is int and value["version"] == 1, "journal_version")
    require(value["repo"] == repo, "journal_identity_changed")
    private_context(
        {
            "sessions": [],
            "runs": [],
            "identifiers": value["private"],
            "roots": value["roots"],
            "checkout_commit": value["commit"],
        }
    )
    return Guard(value["private"], value["roots"], repo=repo, commit=value["commit"])


def preparation_sha(entry: dict[str, Any]) -> str:
    """Integrity of the fixed envelope and conditions, without re-rendering."""
    return sha(
        json_bytes(
            {
                key: value
                for key, value in entry.items()
                if key not in {"sequence", "snapshot_sha256"}
            }
        )
    )


class Journal:
    """Use the existing parent store.directory as state_dir (including tests).

    capture accepts an otherwise complete record without an ID and allocates
    uuid4 inside the lock. Retrying the same private key returns that same ID.
    A caller may instead supply a previously allocated new_id().
    """

    def __init__(self, state_dir: Path, repo: str, parent: int) -> None:
        # Validate namespace without depending on a live parent session.
        scope(
            {
                "repo": repo,
                "parent": parent,
                "issue": None,
                "activity_key": new_id(),
                "delegation": None,
                "review_key": None,
            }
        )
        self.repo, self.parent = repo, parent
        self.path = (
            Path(state_dir)
            / "decision-logs"
            / sha(repo.casefold().encode())
            / f"parent-{parent}"
            / "journal.json"
        )

    def _safe_path(self) -> None:
        require(
            not any(p.is_symlink() for p in (self.path, *self.path.parents)),
            "private_storage_symlink",
        )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._safe_path()
        try:
            with locked(self.path):
                yield
        except OSError:
            raise PublicationError("journal_unavailable") from None

    def _empty(self) -> dict[str, Any]:
        return {"version": 1, "repo": self.repo, "parent": self.parent, "entries": []}

    @staticmethod
    def _captures(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {e["record"]["id"]: e for e in row["entries"] if e["event"] == "capture"}

    @staticmethod
    def _preparations(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {e["record_id"]: e for e in row["entries"] if e["event"] == "prepare"}

    def _guard(
        self, row: dict[str, Any], context: dict[str, Any] | None = None
    ) -> Guard:
        private = {
            v
            for e in self._captures(row).values()
            for k in ("sessions", "runs", "identifiers")
            for v in e["private"][k]
        }
        return Guard(
            private,
            (context or {}).get("roots", []),
            repo=self.repo,
            commit=(context or {}).get("checkout_commit"),
        )

    def _validate(self, row: Any) -> dict[str, Any]:
        fields(row, {"version", "repo", "parent", "entries"})
        require(type(row["version"]) is int and row["version"] == 1, "journal_version")
        require(
            row["repo"] == self.repo
            and type(row["parent"]) is int
            and row["parent"] == self.parent,
            "journal_identity_changed",
        )
        require(isinstance(row["entries"], list), "journal_invalid")
        keys: set[str] = set()
        captured: dict[str, dict[str, Any]] = {}
        contexts: dict[str, dict[str, Any]] = {}
        known_private: set[str] = set()
        prepared: dict[str, dict[str, Any]] = {}
        published: set[str] = set()
        # Collect no values from an entry until that entry has passed validation.
        for sequence, entry in enumerate(row["entries"], 1):
            require(isinstance(entry, dict), "journal_invalid")
            event = entry.get("event")
            common = {"sequence", "event", "key"}
            if event == "capture":
                fields(entry, common | {"record", "private"})
            elif event == "prepare":
                fields(
                    entry,
                    common
                    | {
                        "record_id",
                        "status",
                        "payload",
                        "error",
                        "at",
                        "transform",
                        "snapshot_sha256",
                    },
                )
            elif event == "publication":
                fields(
                    entry, common | {"record_id", "status", "evidence", "error", "at"}
                )
            else:
                require(False, "journal_invalid")
            require(
                type(entry["sequence"]) is int and entry["sequence"] == sequence,
                "journal_invalid",
            )
            key = entry["key"]
            require(
                isinstance(key, str) and bool(key.strip()) and key not in keys,
                "journal_key",
            )
            keys.add(key)
            if event == "capture":
                record = validate_record(entry["record"])
                require(
                    record["scope"]["repo"] == self.repo
                    and record["scope"]["parent"] == self.parent,
                    "journal_identity_changed",
                )
                private_context(entry["private"])
                require(record["id"] not in captured, "record_already_exists")
                require(
                    all(ref in captured for ref in references(record)),
                    "invalid_log_reference",
                )
                captured[record["id"]] = record
                contexts[record["id"]] = entry["private"]
                known_private.update(
                    v
                    for name in ("sessions", "runs", "identifiers")
                    for v in entry["private"][name]
                )
                continue
            record_id = entry["record_id"]
            require(
                isinstance(record_id, str) and record_id in captured,
                "invalid_log_reference",
            )
            timestamp(entry["at"])
            status, error = entry["status"], entry["error"]
            require(
                error is None or isinstance(error, str) and error in PUBLIC_ERRORS,
                "invalid_error_code",
            )
            if event == "prepare":
                checker = transform_guard(entry["transform"], self.repo)
                context = contexts[record_id]
                require(
                    set(entry["transform"]["private"]) == known_private
                    and set(entry["transform"]["roots"])
                    == {p.rstrip("/") for p in context["roots"] if p}
                    and entry["transform"]["commit"] == context["checkout_commit"],
                    "journal_source",
                )
                require(
                    digest(entry["snapshot_sha256"])
                    and entry["snapshot_sha256"] == preparation_sha(entry),
                    "prepared_content_changed",
                )
                require(
                    record_id not in prepared
                    or prepared[record_id]["status"] == "blocked",
                    "prepared_content_changed",
                )
                require(status in ("ready", "blocked"), "journal_state")
                if status == "ready":
                    payload = validate_public(entry["payload"], checker)
                    require(
                        error is None and payload["record"]["id"] == record_id,
                        "journal_source",
                    )
                    # A projection may change prose, not record identity or causality.
                    for field in ("schema_version", "kind", "scope", "time"):
                        require(
                            payload["record"][field] == captured[record_id][field],
                            "prepared_content_changed",
                        )
                    require(
                        references(payload["record"])
                        == references(captured[record_id]),
                        "prepared_content_changed",
                    )
                    require(
                        all(
                            ref in prepared and prepared[ref]["status"] == "ready"
                            for ref in references(payload["record"])
                        ),
                        "invalid_log_reference",
                    )
                else:
                    require(
                        entry["payload"] is None and error is not None, "journal_state"
                    )
                prepared[record_id] = entry
            else:
                require(
                    record_id in prepared and prepared[record_id]["status"] == "ready",
                    "final_not_ready",
                )
                require(status in ("pending", "published", "blocked"), "journal_state")
                require(record_id not in published, "publication_identity_changed")
                evidence_list(entry["evidence"])
                Guard().check(entry["evidence"])
                require((status == "blocked") == (error is not None), "journal_state")
                if status == "published":
                    require(
                        any(e["kind"] == "public" for e in entry["evidence"]),
                        "record_source",
                    )
                    published.add(record_id)
        validate_graph(list(captured.values()))
        return dict(row)

    def read(self) -> dict[str, Any]:
        """Read-only; absence is empty, corruption is an error, never a reset."""
        self._safe_path()
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return self._empty()
        except OSError:
            raise PublicationError("journal_unavailable") from None
        try:
            return self._validate(decode(raw))
        except (PublicationError, TypeError, KeyError, ValueError, OverflowError):
            raise PublicationError("journal_invalid") from None

    @staticmethod
    def _key(row: dict[str, Any], key: str) -> dict[str, Any] | None:
        require(isinstance(key, str) and bool(key.strip()), "journal_key")
        return next((e for e in row["entries"] if e["key"] == key), None)

    def _append(self, row: dict[str, Any], entry: dict[str, Any]) -> None:
        row["entries"].append({"sequence": len(row["entries"]) + 1, **entry})
        self._validate(row)
        atomic(self.path, row)

    def capture(
        self, key: str, record: dict[str, Any], *, private: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        context = (
            copy.deepcopy(private)
            if private is not None
            else {
                "sessions": [],
                "runs": [],
                "identifiers": [],
                "roots": [],
                "checkout_commit": None,
            }
        )
        private_context(context)
        require(isinstance(record, dict), "record_schema")
        candidate = copy.deepcopy(record)
        with self._locked():
            row = self.read()
            previous = self._key(row, key)
            if previous is not None:
                require(previous["event"] == "capture", "record_already_exists")
                candidate.setdefault("id", previous["record"]["id"])
                candidate = validate_record(candidate)
                require(
                    candidate == previous["record"] and context == previous["private"],
                    "source_changed",
                )
                return copy.deepcopy(previous["record"])
            candidate.setdefault("id", new_id())
            candidate = validate_record(candidate)
            same_id = self._captures(row).get(candidate["id"])
            if same_id is not None:
                require(
                    same_id["record"] == candidate and same_id["private"] == context,
                    "record_already_exists",
                )
                # Exact record replay is idempotent even with a new delivery key.
                return copy.deepcopy(candidate)
            self._append(
                row,
                {
                    "event": "capture",
                    "key": key,
                    "record": candidate,
                    "private": context,
                },
            )
            return copy.deepcopy(candidate)

    def prepare(
        self, record_id: str, *, key: str, at: str, retry: bool = False
    ) -> dict[str, Any]:
        """Fix a safe projection, or append blocked while retaining raw input.

        Prepare dependencies first. A blocked dependency blocks this record too.
        A blocked projection is retried explicitly, using the same captured input.
        """
        public_id(record_id)
        timestamp(at)
        require(type(retry) is bool, "record_schema")
        with self._locked():
            row = self.read()
            existing = self._key(row, key)
            if existing is not None:
                require(
                    existing["event"] == "prepare"
                    and existing["record_id"] == record_id
                    and existing["at"] == at,
                    "source_changed",
                )
                return copy.deepcopy(existing)
            capture = self._captures(row).get(record_id)
            if capture is None:
                raise PublicationError("invalid_log_reference")
            previous = self._preparations(row).get(record_id)
            if previous is not None and (previous["status"] == "ready" or not retry):
                return copy.deepcopy(previous)
            checker = self._guard(row, capture["private"])
            entry: dict[str, Any] = {
                "event": "prepare",
                "key": key,
                "record_id": record_id,
                "status": "ready",
                "payload": None,
                "error": None,
                "at": at,
                "transform": {
                    "version": 1,
                    "private": sorted(checker.private, key=len, reverse=True),
                    "roots": list(checker.roots),
                    "repo": checker.repo,
                    "commit": checker.commit,
                },
            }
            try:
                entry["payload"] = render(capture["record"], checker)
                ready_records = [
                    e["payload"]["record"]
                    for e in self._preparations(row).values()
                    if e["status"] == "ready"
                ]
                validate_graph([*ready_records, entry["payload"]["record"]])
            except PublicationError as error:
                entry.update(status="blocked", payload=None, error=str(error))
            entry["snapshot_sha256"] = preparation_sha(entry)
            self._append(row, entry)
            return copy.deepcopy(row["entries"][-1])

    def publication(
        self,
        record_id: str,
        *,
        key: str,
        status: str,
        at: str,
        evidence: list[dict[str, Any]],
        error: str | None = None,
    ) -> dict[str, Any]:
        """Record only publication facts; callers perform Git/HTTP separately."""
        public_id(record_id)
        entry = {
            "event": "publication",
            "key": key,
            "record_id": record_id,
            "status": status,
            "evidence": copy.deepcopy(evidence),
            "error": error,
            "at": at,
        }
        with self._locked():
            row = self.read()
            existing = self._key(row, key)
            if existing is not None:
                require(
                    {k: v for k, v in existing.items() if k != "sequence"} == entry,
                    "source_changed",
                )
                return copy.deepcopy(existing)
            checker = self._guard(row)
            checker.check(entry["evidence"])
            if status in ("pending", "published"):
                prepared = self._preparations(row).get(record_id)
                require(
                    prepared is not None and prepared["status"] == "ready",
                    "final_not_ready",
                )
                if prepared is not None:
                    validate_public(prepared["payload"], checker)
            self._append(row, entry)
            return copy.deepcopy(row["entries"][-1])

    def public_payloads(self) -> list[dict[str, Any]]:
        """Publication units in dependency order, including sanitization counts."""
        row = self.read()
        checker = self._guard(row)
        payloads = {
            e["record_id"]: validate_public(e["payload"], checker)
            for e in self._preparations(row).values()
            if e["status"] == "ready"
        }
        ordered = validate_graph([p["record"] for p in payloads.values()])
        return [payloads[key] for key in ordered]

    def public_records(self) -> list[dict[str, Any]]:
        """Safe records for graph/grouping consumers; publication uses payloads."""
        return [p["record"] for p in self.public_payloads()]

    def status(self) -> list[dict[str, Any]]:
        row = self.read()
        captures = self._captures(row)
        result_states = states([e["record"] for e in captures.values()])
        latest = {e["record_id"]: e for e in row["entries"] if e["event"] != "capture"}
        result = []
        for record_id, capture in captures.items():
            publication = latest.get(record_id)
            result.append(
                {
                    "id": record_id,
                    "kind": capture["record"]["kind"],
                    "capture_status": "captured",
                    "publication_status": publication["status"]
                    if publication
                    else "unprepared",
                    "error": publication["error"] if publication else None,
                    "result": result_states.get(record_id),
                }
            )
        self._guard(row).check(result)
        return result
