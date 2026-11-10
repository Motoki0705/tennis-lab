"""Best-effort observation delivery. A failed journal never reruns the operation.

Each invocation first saves a private outbox. Entries freeze IDs, times, sources
and bytes before delivery to Journal. The outbox is also the recovery receipt;
its mutable delivery cursor is not an alternative public record contract.
"""

from __future__ import annotations

import copy
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from worker_logs.journal import atomic, decode, locked, require
from worker_logs.schema import json_bytes, sha
from worker_logs.sources import IDENTITIES

from .graph import references
from .journal import Journal, private_context
from .schema import CHANGE_FIELDS, new_id, public_id, validate_record

CURRENT: ContextVar[Operation | None] = ContextVar("decision_capture", default=None)
RUNTIME_ENV = "DECISION_LOG_CAPTURE"
RETRIEVAL_ENV = "DECISION_LOG_RETRIEVAL"


def now() -> str:
    return datetime.now(UTC).isoformat()


def missing(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def observed(value: Any) -> dict[str, Any]:
    return {"status": "observed", "value": value}


def source(entrypoint: str, artifact: str, payload: Any) -> dict[str, Any]:
    # A fingerprint identifies the private snapshot; it does not promise public
    # retrieval. Do not put local paths or runtime IDs in this public reference.
    return {
        "kind": "unavailable",
        "reason": f"capture_source={entrypoint}; artifact={artifact}; content_sha256={sha(json_bytes(payload))}; private evidence",
    }


def empty_auxiliary() -> dict[str, Any]:
    return {
        key: missing("not reported by this source")
        for key in (
            "selected_model",
            "effective_model",
            "elapsed_seconds",
            "wait_seconds",
            "usage",
        )
    }


def comparison(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    value = decode(path.read_bytes())
    require(
        set(value) == CHANGE_FIELDS
        and all(v in {"changed", "unchanged", "unconfirmed"} for v in value.values()),
        "record_schema",
    )
    return dict(value)


def record(
    kind: str, scope: dict[str, Any], data: dict[str, Any], *, at: str | None = None
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": new_id(),
        "kind": kind,
        "scope": copy.deepcopy(scope),
        "time": {
            "occurred_at": observed(at)
            if at
            else missing("source occurrence time unavailable"),
            "received_at": observed(now()),
            "recorded_at": now(),
        },
        "data": data,
    }


def warn() -> None:
    with suppress(OSError, ValueError):
        print(
            "decision-log: capture incomplete; inspect parent decision-log-status and retry capture only",
            file=sys.stderr,
        )


def safe_path(path: Path) -> None:
    require(
        not any(p.is_symlink() for p in (path, *path.parents)),
        "private_storage_symlink",
    )


class Operation:
    """Private delivery state for one action and its append-only snapshots."""

    def __init__(self, path: Path, row: dict[str, Any]) -> None:
        self.path, self.row = path, row
        self.failed = False
        self.held = False

    @classmethod
    def begin(
        cls,
        directory: Path,
        repo: str,
        parent: int,
        entrypoint: str,
        operation: str,
        *,
        issue: int | None = None,
        decision_id: str | None = None,
        plan_id: str | None = None,
        previous_result_id: str | None = None,
        changes: dict[str, str] | None = None,
        scope: dict[str, Any] | None = None,
    ) -> Operation:
        directory = Path(directory).absolute()
        journal = Journal(directory, repo, parent)
        require(
            previous_result_id is not None
            or not changes
            or all(v == "unconfirmed" for v in changes.values()),
            "invalid_log_reference",
        )
        scope = scope or {
            "repo": repo,
            "parent": parent,
            "issue": issue,
            "activity_key": new_id(),
            "delegation": None,
            "review_key": None,
        }
        row: dict[str, Any] = {
            "version": 1,
            "directory": str(directory),
            "repo": repo,
            "parent": parent,
            "entrypoint": entrypoint,
            "scope": scope,
            "operation": operation,
            "decision_id": decision_id,
            "plan_id": plan_id,
            "previous_result_id": previous_result_id,
            "changes": changes or {key: "unconfirmed" for key in CHANGE_FIELDS},
            "changes_explicit": changes is not None,
            "entries": [],
            "delivered": [],
            "closed": False,
            "private": {
                "sessions": [],
                "runs": [],
                "identifiers": sorted(
                    {
                        v
                        for k, v in os.environ.items()
                        if v
                        and (
                            k in IDENTITIES
                            or re.search(
                                r"TOKEN|PASSWORD|SECRET|API_KEY", k, re.IGNORECASE
                            )
                        )
                    }
                ),
                "roots": [],
                "checkout_commit": None,
            },
            "runtime_directory": None,
            "counter_key": new_id(),
        }
        op = cls(journal.path.parent / "captures" / (new_id() + ".json"), row)
        op.action("requested", f"{entrypoint}: requested")
        return op

    @classmethod
    def load(cls, path: Path) -> Operation:
        path = path.absolute()
        safe_path(path)
        row = decode(path.read_bytes())
        require(
            type(row.get("version")) is int and row["version"] == 1, "journal_version"
        )
        require(
            isinstance(row.get("entrypoint"), str)
            and bool(re.fullmatch(r"[a-z][a-z0-9_.-]*", row["entrypoint"])),
            "journal_source",
        )
        require(
            isinstance(row.get("entries"), list)
            and isinstance(row.get("delivered"), list),
            "journal_invalid",
        )
        journal = Journal(Path(row["directory"]), row["repo"], row["parent"])
        require(
            path.parent == journal.path.parent / "captures", "journal_identity_changed"
        )
        private_context(row["private"])
        require(
            type(row.get("closed")) is bool and bool(row["entries"]), "journal_invalid"
        )
        record_ids = []
        keys = []
        for entry in row["entries"]:
            require(
                set(entry) == {"key", "source", "record", "private", "observation"}
                and entry["source"] == row["entrypoint"],
                "journal_invalid",
            )
            validated = validate_record(entry["record"])
            require(validated["scope"] == row["scope"], "journal_identity_changed")
            private_context(entry["private"])
            record_ids.append(validated["id"])
            keys.append(entry["key"])
        require(
            len(record_ids) == len(set(record_ids))
            and len(keys) == len(set(keys))
            and len(row["delivered"]) == len(set(row["delivered"]))
            and set(row["delivered"]) <= set(record_ids),
            "journal_invalid",
        )
        return cls(path, row)

    @property
    def journal(self) -> Journal:
        return Journal(
            Path(self.row["directory"]), self.row["repo"], self.row["parent"]
        )

    def records(self, kind: str) -> list[dict[str, Any]]:
        return [e["record"] for e in self.row["entries"] if e["record"]["kind"] == kind]

    def _save(self) -> None:
        safe_path(self.path)
        atomic(self.path, self.row)

    def mark_failed(self, *, lost_source: str | None = None) -> None:
        self.failed = True
        warn()
        # Keep a diagnostic on a separate path when the journal/outbox directory
        # itself is broken. If the whole state filesystem is unavailable, stderr
        # is the remaining observable channel; never report capture success.
        try:
            path = (
                failure_directory(
                    Path(self.row["directory"]), self.row["repo"], self.row["parent"]
                )
                / self.path.name
            )
            safe_path(path)
            with locked(path):
                previous = decode(path.read_bytes()) if path.exists() else {}
                atomic(
                    path,
                    {
                        "source": self.row["entrypoint"],
                        "expected_ids": sorted(
                            set(previous.get("expected_ids", []))
                            | {e["record"]["id"] for e in self.row["entries"]}
                        ),
                        "lost_sources": sorted(
                            set(previous.get("lost_sources", []))
                            | ({lost_source} if lost_source else set())
                        ),
                    },
                )
        except Exception:
            pass

    def source_restored(self, name: str) -> None:
        path = (
            failure_directory(
                Path(self.row["directory"]), self.row["repo"], self.row["parent"]
            )
            / self.path.name
        )
        if path.exists() and len(self.row["entries"]) == len(self.row["delivered"]):
            with locked(path):
                marker = decode(path.read_bytes())
                marker["lost_sources"] = [
                    v for v in marker.get("lost_sources", []) if v != name
                ]
                atomic(path, marker)

    def _deliver(self) -> None:
        for entry in self.row["entries"]:
            key = entry["record"]["id"]
            if key not in self.row["delivered"]:
                self.journal.capture(key, entry["record"], private=entry["private"])
                self.row["delivered"].append(key)
                self._save()

    @contextmanager
    def transaction(self) -> Iterator[Operation]:
        with locked(self.path):
            if self.path.exists():
                self.row = Operation.load(self.path).row
            self.held = True
            try:
                yield self
            finally:
                self.held = False

    def emit(self, row: dict[str, Any], key: str, *, payload: Any = None) -> str:
        previous = next((e for e in self.row["entries"] if e["key"] == key), None)
        if previous:
            return str(previous["record"]["id"])
        entry = {
            "key": key,
            "source": self.row["entrypoint"],
            "record": row,
            "private": copy.deepcopy(self.row["private"]),
            "observation": copy.deepcopy(payload),
        }
        try:
            if self.held:
                self.row["entries"].append(entry)
                self._save()
                self._deliver()
            else:
                with self.transaction():
                    self.row["entries"].append(entry)
                    self._save()
                    self._deliver()
        except Exception:
            if not any(e["record"]["id"] == row["id"] for e in self.row["entries"]):
                self.row["entries"].append(entry)
            self.mark_failed()
        return str(row["id"])

    def action(
        self,
        state: str,
        summary: str,
        *,
        key: str | None = None,
        at: str | None = "now",
    ) -> str:
        actions = self.records("action")
        results = self.records("result")
        return self.emit(
            record(
                "action",
                self.row["scope"],
                {
                    "decision_id": self.row["decision_id"],
                    "plan_id": self.row["plan_id"],
                    "decision_missing_reason": None
                    if self.row["decision_id"]
                    else "parent decision not recorded for this operation",
                    "previous_action_id": actions[-1]["id"] if actions else None,
                    "operation": self.row["operation"],
                    "state": state,
                    "summary": summary,
                    "result_ids": [r["id"] for r in results]
                    if state in {"ended", "rejected_before_start", "unknown"}
                    else [],
                },
                at=now() if at == "now" else at,
            ),
            key or "action:" + state + ":" + str(len(actions)),
        )

    def result(
        self,
        payload: Any,
        *,
        artifact: str,
        status: str,
        summary: str,
        key: str | None = None,
        observer: str = "machine",
        verifies: list[str] | None = None,
        final_report: dict[str, Any] | None = None,
        auxiliary: dict[str, Any] | None = None,
        at: str | None = None,
        evidence: list[dict[str, Any]] | None = None,
        entrypoint: str | None = None,
    ) -> str:
        results = self.records("result")
        previous = results[-1]["id"] if results else self.row["previous_result_id"]
        return self.emit(
            record(
                "result",
                self.row["scope"],
                {
                    "action_id": self.records("action")[-1]["id"],
                    "previous_result_id": previous,
                    "verifies": verifies or [],
                    "status": status,
                    "summary": summary,
                    "changes": self.row["changes"]
                    if previous and not results
                    else {k: "unconfirmed" for k in CHANGE_FIELDS},
                    "evidence": [
                        source(entrypoint or self.row["entrypoint"], artifact, payload),
                        *(
                            [
                                source(
                                    "parent.worker-context",
                                    "saved preflight comparison inputs",
                                    self.row["worker_context"],
                                )
                            ]
                            if self.row.get("worker_context")
                            else []
                        ),
                        *(evidence or []),
                    ],
                    "observer": observer,
                    "final_report": final_report
                    or missing("not a final-report observation"),
                    "auxiliary": auxiliary or empty_auxiliary(),
                    "repeat": None,
                },
                at=at,
            ),
            key or "result:" + artifact + ":" + sha(json_bytes(payload)),
            payload=payload,
        )

    def finish(
        self,
        payload: Any,
        *,
        status: str = "success",
        state: str | None = "ended",
        summary: str | None = None,
        observer: str = "machine",
        verifies: list[str] | None = None,
        artifact: str = "command response",
        evidence: list[dict[str, Any]] | None = None,
        final_report: dict[str, Any] | None = None,
        entrypoint: str | None = None,
    ) -> None:
        self.result(
            payload,
            artifact=artifact,
            status=status,
            summary=summary or f"{self.row['entrypoint']}: {status}",
            observer=observer,
            verifies=verifies,
            evidence=evidence,
            final_report=final_report,
            entrypoint=entrypoint,
        )
        if state is not None:
            self.row["closed"] = True
            self.action(state, f"{self.row['entrypoint']}: {state}")

    def view(self) -> dict[str, Any]:
        return {
            "status": "incomplete"
            if self.failed or len(self.row["entries"]) != len(self.row["delivered"])
            else "captured",
            "action_ids": [r["id"] for r in self.records("action")],
            "result_ids": [r["id"] for r in self.records("result")],
        }


def current() -> Operation | None:
    return CURRENT.get()


def runtime_environment(
    directory: Path,
    *,
    operation: Operation | None = None,
    actual_operation: str | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    env.pop(RUNTIME_ENV, None)
    op = operation or current()
    if op is not None:
        try:
            origin = op.row.get("model_specified_by")
            worker_context = op.row.get("worker_context")
            if (
                actual_operation is not None
                and op.row["operation"] in {"start_ai", "continue_ai"}
                and op.row["operation"] != actual_operation
            ):
                with op.transaction():
                    op.finish(
                        {"actual_operation": actual_operation},
                        status="partial",
                        summary=f"{op.row['entrypoint']}: request resolved to {actual_operation}; no AI launched by the requested action",
                    )
                    previous = op.records("result")[-1]["id"]
                op = Operation.begin(
                    Path(op.row["directory"]),
                    op.row["repo"],
                    op.row["parent"],
                    "runner.launch",
                    actual_operation,
                    scope=op.row["scope"],
                    previous_result_id=previous,
                )
            with locked(op.path):
                # The runner and caller can write at different times. Always load
                # the last durable receipt before updating its private binding.
                if op.path.exists():
                    op.row = Operation.load(op.path).row
                op.row["runtime_directory"] = str(directory.resolve())
                if origin is not None:
                    op.row.setdefault("model_specified_by", origin)
                if worker_context is not None:
                    op.row.setdefault("worker_context", worker_context)
                op._save()
            env[RUNTIME_ENV] = str(op.path)
        except Exception:
            op.mark_failed(lost_source="runtime.binding")
    return env


def observe(
    payload: Any, *, status: str = "success", state: str | None = "ended", **kwargs: Any
) -> dict[str, Any] | None:
    op = current()
    if op is None:
        return None
    try:
        with op.transaction():
            op.finish(payload, status=status, state=state, **kwargs)
    except Exception:
        op.mark_failed(lost_source="command.response")
    return op.view()


def failure_directory(directory: Path, repo: str, parent: int) -> Path:
    namespace: str = sha(repo.casefold().encode())
    return directory / "decision-capture-failures" / namespace / str(parent)


def replay(journal: Journal) -> None:
    """Deliver the frozen DAG, even when dependent outboxes sort before inputs."""
    pending: dict[str, tuple[Path, dict[str, Any]]] = {}
    receipts = []
    for path in sorted((journal.path.parent / "captures").glob("*.json")):
        try:
            with locked(path):
                op = Operation.load(path)
                receipts.append(path)
                for entry in op.row["entries"]:
                    pending[entry["record"]["id"]] = (path, entry)
        except Exception:
            continue  # status reports the unreadable receipt without resetting it.
    known = journal._captures(journal.read())
    for key in list(pending):
        if key in known:
            require(
                known[key]["record"] == pending[key][1]["record"]
                and known[key]["private"] == pending[key][1]["private"],
                "source_changed",
            )
            del pending[key]
    while pending:
        ready = [
            key
            for key, (_, entry) in pending.items()
            if set(references(entry["record"])) <= set(known)
        ]
        if not ready:
            break
        for key in ready:
            _, entry = pending.pop(key)
            try:
                journal.capture(key, entry["record"], private=entry["private"])
                known[key] = entry
            except Exception:
                # Independent records can still be restored. Failed deliveries
                # stay pending and are attempted only once in this invocation.
                continue
    for path in receipts:
        with locked(path):
            op = Operation.load(path)
            require(
                all(
                    e["record"]["id"] not in known
                    or (
                        e["record"] == known[e["record"]["id"]]["record"]
                        and e["private"] == known[e["record"]["id"]]["private"]
                    )
                    for e in op.row["entries"]
                ),
                "source_changed",
            )
            op.row["delivered"] = [
                e["record"]["id"]
                for e in op.row["entries"]
                if e["record"]["id"] in known
            ]
            op._save()


def status(
    directory: Path, repo: str, parent: int, *, retry: bool = False
) -> dict[str, Any]:
    directory = Path(directory).absolute()
    journal = Journal(directory, repo, parent)
    if retry:
        try:
            replay(journal)
        except Exception:
            warn()
    receipts = []
    inputs = []
    for path in sorted((journal.path.parent / "captures").glob("*.json")):
        try:
            with locked(path):
                op = Operation.load(path)
                receipt = {
                    "source": op.row["entrypoint"],
                    "closed": op.row["closed"],
                    **op.view(),
                }
                receipts.append(receipt)
                inputs.append((receipt, op.row["entries"]))
        except Exception:
            receipts.append(
                {
                    "source": "unavailable",
                    "status": "incomplete",
                    "error": "capture_receipt_unavailable",
                }
            )
    try:
        records = journal.status()
        from .projection import render

        raw = journal.read()
        by_id = journal._captures(raw)
        for receipt, entries in inputs:
            if receipt["status"] == "captured" and any(
                e["record"]["id"] not in by_id for e in entries
            ):
                receipt.update(status="incomplete", error="capture_delivery_missing")
            if any(
                e["record"]["id"] in by_id
                and (
                    e["record"] != by_id[e["record"]["id"]]["record"]
                    or e["private"] != by_id[e["record"]["id"]]["private"]
                )
                for e in entries
            ):
                receipt.update(status="incomplete", error="capture_source_changed")
        for entry in records:
            capture = by_id[entry["id"]]
            try:
                entry["record"] = render(
                    capture["record"], journal._guard(raw, capture["private"])
                )["record"]
                entry["view_status"] = "ready"
            except Exception:
                entry["view_status"] = "blocked"
                entry["view_error"] = "capture_view_unavailable"
    except Exception:
        return {
            "status": "incomplete",
            "error": "journal_unavailable",
            "receipts": receipts,
            "records": [],
        }
    captured_ids = {r["id"] for r in records}
    failures = []
    for path in sorted(failure_directory(directory, repo, parent).glob("*.json")):
        try:
            safe_path(path)
            marker = decode(path.read_bytes())
            require(
                isinstance(marker.get("source"), str)
                and bool(re.fullmatch(r"[a-z][a-z0-9_.-]*", marker["source"])),
                "journal_source",
            )
            if (
                marker.get("lost_sources")
                or not set(marker["expected_ids"]) <= captured_ids
            ):
                failures.append(
                    {"source": marker["source"], "error": "capture_delivery_missing"}
                )
        except Exception:
            failures.append(
                {"source": "unavailable", "error": "capture_diagnostic_unavailable"}
            )
    return {
        "status": "incomplete"
        if failures or any(r["status"] == "incomplete" for r in receipts)
        else "captured",
        "receipts": receipts,
        "records": records,
        "failures": failures,
    }


def checked_link(
    journal: Journal, decision_id: str | None, plan_id: str | None, operation: str
) -> str | None:
    """Reject an explicit bad link before invoking work, never invent a plan."""
    if decision_id is None:
        require(plan_id is None, "invalid_log_reference")
        return None
    public_id(decision_id)
    rows = journal.read()["entries"]
    decision = next(
        (
            e["record"]
            for e in rows
            if e["event"] == "capture" and e["record"]["id"] == decision_id
        ),
        None,
    )
    require(
        decision is not None and decision["kind"] == "decision", "invalid_log_reference"
    )
    assert decision is not None
    plans = [
        p
        for p in decision["data"]["next_actions"]
        if p["operation"] == operation and (plan_id is None or p["id"] == plan_id)
    ]
    require(len(plans) == 1, "invalid_log_reference")
    require(
        not any(
            e["event"] == "capture"
            and e["record"]["kind"] == "action"
            and e["record"]["data"]["decision_id"] == decision_id
            and e["record"]["data"]["plan_id"] == plans[0]["id"]
            for e in rows
        ),
        "invalid_log_reference",
    )
    return str(plans[0]["id"])
