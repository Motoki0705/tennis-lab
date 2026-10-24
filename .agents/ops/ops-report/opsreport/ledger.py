"""Append-only audit records, replay, validation and bounded parent summaries.

This module only records observations. Native execution, source acquisition,
publication and approval belong to callers; a recorded agent ID proves none of
them. Each write preserves the event prefix and atomically replaces one file.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

VERSION = 1
USAGE_PATHS = frozenset(
    {
        "dynamic-import",
        "registry",
        "string-reference",
        "config-cli",
        "generated-code",
        "training-evaluation-reproduction",
        "public-entrypoint",
    }
)
_TASK_TRANSITIONS = {
    "planned": {"planned", "running", "failed", "interrupted", "cancelled"},
    "running": {"running", "completed", "failed", "interrupted", "cancelled"},
    "failed": {"running", "cancelled"},
    "interrupted": {"running", "cancelled"},
    "completed": set(),
    "cancelled": set(),
}
_FINDING_TRANSITIONS = {
    "suspected": {"suspected", "confirmed", "refuted", "unverified"},
    "unverified": {"unverified", "suspected", "confirmed", "refuted"},
    "confirmed": {"confirmed", "unverified", "refuted"},
    "refuted": {"refuted", "unverified", "suspected"},
}
_TABLES = {
    "task": "tasks",
    "finding": "findings",
    "evidence": "evidence",
    "handoff": "handoffs",
}
_RECOVERY = (
    "preserve the file and reconcile it with a verified backup and source records; "
    "restore or repair explicitly, then reload with the original slot/base_sha; "
    "do not reinitialize or discard events"
)


class LedgerError(ValueError):
    """A record or reference violates the audit contract."""


class LedgerCorruptError(LedgerError):
    """The stored history cannot be replayed; explicit recovery is required."""


class LedgerConflictError(LedgerError):
    """Another writer committed since this instance was loaded."""


def _fields(value: Any, fields: str, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(fields.split()):
        raise LedgerError(f"{name}: expected fields {fields}")
    return value


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise LedgerError(
            f"{name}: expected {'a string' if empty else 'non-empty text'}"
        )
    return value


def _texts(value: Any, name: str, *, required: bool = False) -> list[str]:
    if not isinstance(value, list) or (required and not value):
        raise LedgerError(
            f"{name}: expected a list{' with entries' if required else ''}"
        )
    result = [_text(item, name) for item in value]
    if len(set(result)) != len(result):
        raise LedgerError(f"{name}: duplicate entries")
    return result


def _time(value: Any, name: str) -> datetime:
    text = _text(value, name)
    try:
        result = datetime.fromisoformat(text)
    except ValueError as exc:
        raise LedgerError(f"{name}: invalid timestamp") from exc
    if result.tzinfo is None:
        raise LedgerError(f"{name}: timezone required")
    return result


def _sha(value: Any, name: str) -> str:
    text = _text(value, name)
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", text):
        raise LedgerError(f"{name}: expected a full lowercase commit SHA")
    return text


def _contract(slot: Any, base_sha: Any) -> None:
    text = _text(slot, "slot")
    try:
        valid = datetime.strptime(text, "%Y%m%d").strftime("%Y%m%d") == text
    except ValueError:
        valid = False
    if not valid:
        raise LedgerError("slot: expected a valid YYYYMMDD date")
    _sha(base_sha, "base_sha")


def _issue(value: Any) -> None:
    if type(value) is not int or value <= 0:
        raise LedgerError("issue: expected a positive integer")


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _question_key(question: str, scope: list[str]) -> tuple[str, tuple[str, ...]]:
    return _normalized(question).casefold(), tuple(
        sorted(_normalized(s) for s in scope)
    )


def _gaps(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise LedgerError(f"{name}: expected scope/reason/resume entries")
    for item in value:
        _fields(item, "scope reason resume", name)
        for key in item:
            _text(item[key], f"{name}.{key}")
    return value


def _evidence_refs(
    value: Any, state: dict[str, Any], name: str, *, required: bool = False
) -> list[str]:
    refs = _texts(value, name, required=required)
    for ref in refs:
        if ref not in state["evidence"]:
            raise LedgerError(f"{name}: unknown evidence {ref!r}")
    return refs


def _origin(value: Any, state: dict[str, Any], slot: str) -> tuple[str, str]:
    row = _fields(value, "kind id", "origin")
    kind, key = _text(row["kind"], "origin.kind"), _text(row["id"], "origin.id")
    if kind == "goal":
        if key != slot:
            raise LedgerError("origin: goal must refer to this slot")
    elif kind in {"task", "finding"}:
        if key not in state[_TABLES[kind]]:
            raise LedgerError(f"origin: unknown {kind} {key!r}")
    else:
        raise LedgerError(f"origin: unknown kind {kind!r}")
    return kind, key


def _acyclic(state: dict[str, Any], slot: str) -> None:
    # Iterative traversal also supports histories deeper than Python's recursion limit.
    edges = {}
    for key, row in state["tasks"].items():
        edges[("task", key)] = _origin(row["origin"], state, slot)
    for key, row in state["findings"].items():
        parent = row["task_id"]
        if parent not in state["tasks"]:
            raise LedgerError(f"finding: unknown task {parent!r}")
        edges[("finding", key)] = ("task", parent)
    done: set[tuple[str, str]] = set()
    for node in edges:
        trail: set[tuple[str, str]] = set()
        while node in edges and node not in done:
            if node in trail:
                raise LedgerError(f"origin cycle at {node[0]} {node[1]!r}")
            trail.add(node)
            node = edges[node]
        done.update(trail)


def _task(
    row: dict[str, Any], old: dict[str, Any] | None, state: dict[str, Any], at: datetime
) -> None:
    _fields(
        row,
        "id question scope origin agent_id status started_at finished_at conclusion evidence next_questions unconfirmed",
        "task",
    )
    question = _text(row["question"], "task.question")
    scope = _texts(row["scope"], "task.scope", required=True)
    key = _question_key(question, scope)
    if len(set(key[1])) != len(scope):
        raise LedgerError("task.scope: duplicate normalized scope")
    for other in state["tasks"].values():
        if other["id"] != row["id"] and key == _question_key(
            other["question"], other["scope"]
        ):
            raise LedgerError(f"duplicate question/scope: resume task {other['id']!r}")
    status = _text(row["status"], "task.status")
    if status not in _TASK_TRANSITIONS:
        raise LedgerError(f"task: unknown status {status!r}")
    if old:
        if (old["question"], old["scope"]) != (question, scope):
            raise LedgerError(
                "task: question/scope are immutable; create a distinct task"
            )
        if status not in _TASK_TRANSITIONS[old["status"]]:
            raise LedgerError(f"task: invalid transition {old['status']} -> {status}")
        if old["status"] != "planned" and row["origin"] != old["origin"]:
            raise LedgerError("task: origin is immutable after launch")
    elif status != "planned":
        raise LedgerError("task: initial status must be planned")
    start, end = row["started_at"], row["finished_at"]
    if start is None:
        if row["agent_id"] is not None or status in {"running", "completed"}:
            raise LedgerError(
                "task: started_at and native agent_id required for execution"
            )
    else:
        _text(row["agent_id"], "task.agent_id")
        started = _time(start, "task.started_at")
        if not _time(state["run"]["started_at"], "run.started_at") <= started <= at:
            raise LedgerError("task: start outside the recorded run interval")
    if status == "planned" and (start is not None or end is not None):
        raise LedgerError("task: planned task has execution timestamps")
    if status in {"planned", "running"}:
        if end is not None:
            raise LedgerError("task: unfinished task has finished_at")
    else:
        finished = _time(end, "task.finished_at")
        earliest = _time(start or state["run"]["started_at"], "task.start")
        if not earliest <= finished <= at:
            raise LedgerError("task: invalid finish interval")
    if old:
        if status == "running" and old["status"] in {
            "planned",
            "failed",
            "interrupted",
        }:
            if old["finished_at"] and _time(start, "task.start") < _time(
                old["finished_at"], "task.previous_finish"
            ):
                raise LedgerError("task: retry starts before the previous finish")
        elif (row["agent_id"], start) != (old["agent_id"], old["started_at"]):
            raise LedgerError("task: native agent_id/start changed within an attempt")
        if (
            old["finished_at"]
            and end
            and _time(end, "task.finish")
            < _time(old["finished_at"], "task.previous_finish")
        ):
            raise LedgerError("task: finish moved backwards")
    if status == "running" and any(
        other["id"] != row["id"]
        and other["status"] == "running"
        and other["agent_id"] == row["agent_id"]
        for other in state["tasks"].values()
    ):
        raise LedgerError("task: native agent_id already owns a running task")
    _text(row["conclusion"], "task.conclusion", empty=status in {"planned", "running"})
    _evidence_refs(
        row["evidence"], state, "task.evidence", required=status == "completed"
    )
    gaps = _gaps(row["unconfirmed"], "task.unconfirmed")
    if status == "completed" and gaps:
        raise LedgerError("task: completed task has unconfirmed scope")
    if status in {"failed", "interrupted", "cancelled"} and not gaps:
        raise LedgerError("task: unfinished outcome requires scope, reason and resume")
    if not isinstance(row["next_questions"], list):
        raise LedgerError("task.next_questions: expected question/scope entries")
    questions = set()
    for candidate in row["next_questions"]:
        _fields(candidate, "question scope", "next_question")
        candidate_key = _question_key(
            _text(candidate["question"], "next_question.question"),
            _texts(candidate["scope"], "next_question.scope", required=True),
        )
        if candidate_key in questions:
            raise LedgerError("task.next_questions: duplicate question/scope")
        questions.add(candidate_key)


def _usage_paths(value: Any, state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise LedgerError("usage_paths: expected a list")
    result = {}
    for path in value:
        _fields(path, "kind outcome evidence reason", "usage_path")
        kind = _text(path["kind"], "usage_path.kind")
        outcome = _text(path["outcome"], "usage_path.outcome")
        if kind not in USAGE_PATHS or kind in result:
            raise LedgerError(f"usage_paths: unknown or duplicate kind {kind!r}")
        if outcome not in {"used", "unused", "not-applicable", "unverified"}:
            raise LedgerError("usage_path: invalid outcome")
        _text(path["reason"], "usage_path.reason")
        _evidence_refs(
            path["evidence"],
            state,
            "usage_path.evidence",
            required=outcome != "unverified",
        )
        result[kind] = path
    return result


def _finding(
    row: dict[str, Any], old: dict[str, Any] | None, state: dict[str, Any]
) -> None:
    _fields(
        row,
        "id task_id kind status conclusion evidence counterevidence details",
        "finding",
    )
    _text(row["task_id"], "finding.task_id")
    kind = _text(row["kind"], "finding.kind")
    status = _text(row["status"], "finding.status")
    _text(row["conclusion"], "finding.conclusion")
    if status not in _FINDING_TRANSITIONS:
        raise LedgerError(f"finding: unknown status {status!r}")
    if old:
        if (kind, row["task_id"]) != (old["kind"], old["task_id"]):
            raise LedgerError("finding: kind/task_id are immutable")
        if status not in _FINDING_TRANSITIONS[old["status"]]:
            raise LedgerError(
                f"finding: invalid transition {old['status']} -> {status}"
            )
    elif status not in {"suspected", "unverified"}:
        raise LedgerError("finding: initial status must be suspected or unverified")
    evidence = _evidence_refs(
        row["evidence"], state, "finding.evidence", required=status == "confirmed"
    )
    _evidence_refs(
        row["counterevidence"],
        state,
        "finding.counterevidence",
        required=status == "refuted",
    )
    details = row["details"]
    if kind == "unused-code":
        _fields(details, "decision usage_paths", "unused-code.details")
        decision = _text(details["decision"], "unused-code.decision")
        if decision not in {"delete-candidate", "keep", "unverified"}:
            raise LedgerError("unused-code: invalid decision")
        paths = _usage_paths(details["usage_paths"], state)
        if status == "confirmed":
            if decision == "unverified":
                raise LedgerError(
                    "unused-code: confirmed requires a verified decision and usage paths"
                )
            if decision == "keep" and not any(
                p["outcome"] == "used" for p in paths.values()
            ):
                raise LedgerError("unused-code: keep requires a confirmed used path")
            if decision == "delete-candidate" and (
                set(paths) != USAGE_PATHS
                or any(
                    p["outcome"] not in {"unused", "not-applicable"}
                    for p in paths.values()
                )
            ):
                raise LedgerError(
                    "unused-code: deletion requires all usage paths checked without a used/unverified path"
                )
    elif kind in {"bug", "vulnerability"}:
        _fields(
            details,
            "target_version conditions reachability impact remediation verification usage_paths",
            f"{kind}.details",
        )
        for name in (
            "target_version",
            "reachability",
            "impact",
            "remediation",
            "verification",
        ):
            _text(details[name], f"{kind}.{name}", empty=status != "confirmed")
        _texts(
            details["conditions"],
            f"{kind}.conditions",
            required=status == "confirmed",
        )
        paths = _usage_paths(details["usage_paths"], state)
        if status == "confirmed":
            used = [p for p in paths.values() if p["outcome"] == "used"]
            if not used:
                raise LedgerError(
                    f"{kind}: confirmed requires a checked reachable usage path"
                )
            refs = set(evidence).union(*(p["evidence"] for p in used))
            if not any(
                details["target_version"]
                in (
                    state["evidence"][ref]["target_version"],
                    state["evidence"][ref]["pr_head"],
                )
                for ref in refs
            ):
                raise LedgerError(
                    f"{kind}: target_version has no matching evidence version/PR head"
                )
    else:
        _fields(details, "", "finding.details")


def _evidence(
    row: dict[str, Any],
    old: dict[str, Any] | None,
    at: datetime,
    base_sha: str,
) -> None:
    _fields(
        row, "id source_kind reference acquired_at target_version pr_head", "evidence"
    )
    if old:
        raise LedgerError(
            "evidence: immutable ID; record a new acquisition with a new ID"
        )
    kind = _text(row["source_kind"], "evidence.source_kind")
    _text(row["reference"], "evidence.reference")
    if _time(row["acquired_at"], "evidence.acquired_at") > at:
        raise LedgerError("evidence: acquired_at is after recording time")
    version, head = row["target_version"], row["pr_head"]
    if version is not None:
        _text(version, "evidence.target_version")
    if head is not None:
        _sha(head, "evidence.pr_head")
    if version is None and head is None:
        raise LedgerError("evidence: target_version or PR head required")
    if kind == "code" and (version != base_sha or head is not None):
        raise LedgerError(
            "evidence: code base_sha mismatch; PR code must use a separate source_kind and PR head"
        )


def _handoff(
    row: dict[str, Any], old: dict[str, Any] | None, state: dict[str, Any]
) -> None:
    _fields(row, "id finding_id kind reference related_parents reason", "handoff")
    finding_id = _text(row["finding_id"], "handoff.finding_id")
    if finding_id not in state["findings"]:
        raise LedgerError(f"handoff: unknown finding {finding_id!r}")
    if old and finding_id != old["finding_id"]:
        raise LedgerError("handoff: finding_id is immutable")
    kind = _text(row["kind"], "handoff.kind")
    if kind not in {"issue", "pr", "proposal", "unassigned"}:
        raise LedgerError("handoff: invalid kind")
    if kind == "unassigned":
        if row["reference"] is not None:
            raise LedgerError("handoff: unassigned must have a null reference")
    else:
        _text(row["reference"], "handoff.reference")
    _text(row["reason"], "handoff.reason")
    parents = row["related_parents"]
    if not isinstance(parents, list):
        raise LedgerError("handoff.related_parents: expected issue numbers")
    for parent in parents:
        _issue(parent)
    if len(set(parents)) != len(parents):
        raise LedgerError("handoff.related_parents: duplicate issue")


def _end(row: dict[str, Any], state: dict[str, Any]) -> None:
    _fields(
        row,
        "status reason checked_scope excluded_scope unverified_risks further_investigation",
        "run.end",
    )
    status = _text(row["status"], "run.end.status")
    if status not in {"completed", "incomplete"}:
        raise LedgerError("run.end: invalid status")
    _text(row["reason"], "run.end.reason")
    _texts(
        row["checked_scope"], "run.end.checked_scope", required=status == "completed"
    )
    excluded = _texts(row["excluded_scope"], "run.end.excluded_scope")
    risks = _gaps(row["unverified_risks"], "run.end.unverified_risks")
    _texts(row["further_investigation"], "run.end.further_investigation")
    if status == "completed" and (
        risks
        or any(
            t["status"] not in {"completed", "cancelled"}
            for t in state["tasks"].values()
        )
        or any(
            f["status"] in {"suspected", "unverified"}
            for f in state["findings"].values()
        )
    ):
        raise LedgerError(
            "run.end: unfinished tasks or unverified hypotheses/risks require incomplete"
        )
    if status == "completed" and any(
        t["status"] == "cancelled"
        and not {_normalized(s) for s in t["scope"]}
        <= {_normalized(s) for s in excluded}
        for t in state["tasks"].values()
    ):
        raise LedgerError("run.end: cancelled task scope must be explicitly excluded")


def _replay(document: Any) -> dict[str, Any]:
    doc = _fields(document, "version slot base_sha events", "ledger")
    if type(doc["version"]) is not int or doc["version"] != VERSION:
        raise LedgerError(f"unknown ledger version {doc['version']!r}")
    _contract(doc["slot"], doc["base_sha"])
    events = doc["events"]
    if not isinstance(events, list) or not events:
        raise LedgerError("ledger: non-empty event history required")
    state: dict[str, Any] = {
        "run": {},
        "tasks": {},
        "findings": {},
        "evidence": {},
        "handoffs": {},
    }
    previous_time = None
    for sequence, event in enumerate(events, 1):
        _fields(event, "sequence slot base_sha at kind data", "event")
        if type(event["sequence"]) is not int or event["sequence"] != sequence:
            raise LedgerError("event: non-contiguous sequence")
        if (event["slot"], event["base_sha"]) != (doc["slot"], doc["base_sha"]):
            raise LedgerError("event: slot/base_sha mismatch")
        at = _time(event["at"], "event.at")
        if previous_time and at < previous_time:
            raise LedgerError("event: recording time moved backwards")
        previous_time = at
        kind, data = _text(event["kind"], "event.kind"), event["data"]
        run = state["run"]
        if sequence == 1:
            if kind != "run.started":
                raise LedgerError("ledger: first event must be run.started")
            _fields(data, "goal report_issue", "run.started")
            _text(data["goal"], "run.goal")
            if data["report_issue"] is not None:
                _issue(data["report_issue"])
            run.update(data, started_at=event["at"], status="running", endings=[])
        elif kind == "run.report":
            _fields(data, "report_issue", kind)
            _issue(data["report_issue"])
            if run["report_issue"] is not None:
                raise LedgerError(
                    "run.report: report issue is already bound and immutable"
                )
            run["report_issue"] = data["report_issue"]
        elif kind == "run.resumed":
            _fields(data, "reason", kind)
            _text(data["reason"], "run.resume.reason")
            if run["status"] != "incomplete":
                raise LedgerError("run.resume: only an incomplete run can resume")
            run["status"] = "running"
        else:
            if run["status"] != "running":
                raise LedgerError(
                    "run is closed; explicitly resume an incomplete run before recording"
                )
            if kind == "run.ended":
                _end(data, state)
                run["status"] = data["status"]
                run["endings"].append({**data, "finished_at": event["at"]})
                continue
            entity, separator, action = kind.partition(".")
            if entity not in _TABLES or separator != "." or action != "recorded":
                raise LedgerError(f"unknown event kind {kind!r}")
            if not isinstance(data, dict):
                raise LedgerError(f"{kind}: expected a record")
            key = _text(data.get("id"), f"{entity}.id")
            table = state[_TABLES[entity]]
            old = table.get(key)
            if entity == "task":
                _task(data, old, state, at)
            elif entity == "finding":
                _finding(data, old, state)
            elif entity == "evidence":
                _evidence(data, old, at, doc["base_sha"])
            else:
                _handoff(data, old, state)
            table[key] = copy.deepcopy(data)
            if entity in {"task", "finding"}:
                _acyclic(state, doc["slot"])
    return state


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise LedgerError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise LedgerError(f"invalid JSON constant {value}")


def _read(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        if path.is_symlink():
            raise LedgerError("ledger path must not be a symlink")
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_keys,
            parse_constant=_invalid_constant,
        )
        state = _replay(document)
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        raise LedgerCorruptError(
            f"cannot replay ledger at {path}: {exc}; {_RECOVERY}"
        ) from exc
    return document, state


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(
        path.with_name(path.name + ".lock"),
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _atomic_write(path: Path, document: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as fh:
        tmp = Path(fh.name)
        try:
            json.dump(document, fh, ensure_ascii=False, allow_nan=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
            tmp.replace(path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            tmp.unlink(missing_ok=True)


def _event_entity(event: dict[str, Any]) -> tuple[str, str | None]:
    entity = event["kind"].partition(".")[0]
    return entity, event["slot"] if entity == "run" else event["data"]["id"]


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


class Ledger:
    """One slot and base SHA; explicit creation, fail-closed loading and writes.

    Record methods take complete snapshots, not patches. Evidence is immutable;
    task retries retain their ID and append another attempt. All returned objects
    are copies. A stale writer must reload and reconsider its update.
    """

    def __init__(
        self, path: Path, document: dict[str, Any], state: dict[str, Any]
    ) -> None:
        self.path = path.absolute()
        self._document = document
        self._state = state

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        slot: str,
        base_sha: str,
        goal: str,
        at: str,
        report_issue: int | None = None,
    ) -> Ledger:
        _contract(slot, base_sha)
        document = {
            "version": VERSION,
            "slot": slot,
            "base_sha": base_sha,
            "events": [
                {
                    "sequence": 1,
                    "slot": slot,
                    "base_sha": base_sha,
                    "at": at,
                    "kind": "run.started",
                    "data": {"goal": goal, "report_issue": report_issue},
                }
            ],
        }
        state = _replay(document)
        with _lock(path):
            if path.exists() or path.is_symlink():
                raise LedgerError("ledger already exists; load it, never reinitialize")
            _atomic_write(path, document)
        return cls(path, document, state)

    @classmethod
    def load(cls, path: Path, *, slot: str, base_sha: str) -> Ledger:
        _contract(slot, base_sha)
        document, state = _read(path)
        if (slot, base_sha) != (document["slot"], document["base_sha"]):
            raise LedgerError(
                "ledger slot/base_sha mismatch; reload with the original contract"
            )
        return cls(path, document, state)

    @property
    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "slot": self._document["slot"],
                "base_sha": self._document["base_sha"],
                **self._state,
            }
        )

    def _append(self, kind: str, data: dict[str, Any], at: str) -> None:
        with _lock(self.path):
            current, _ = _read(self.path)
            if current != self._document:
                raise LedgerConflictError(
                    "ledger changed since load; reload and reconsider the update"
                )
            document = copy.deepcopy(current)
            document["events"].append(
                {
                    "sequence": len(document["events"]) + 1,
                    "slot": document["slot"],
                    "base_sha": document["base_sha"],
                    "at": at,
                    "kind": kind,
                    "data": copy.deepcopy(data),
                }
            )
            state = _replay(document)
            _atomic_write(self.path, document)
            self._document, self._state = document, state

    def record_task(self, record: dict[str, Any], *, at: str) -> None:
        self._append("task.recorded", record, at)

    def record_finding(self, record: dict[str, Any], *, at: str) -> None:
        self._append("finding.recorded", record, at)

    def record_evidence(self, record: dict[str, Any], *, at: str) -> None:
        self._append("evidence.recorded", record, at)

    def record_handoff(self, record: dict[str, Any], *, at: str) -> None:
        self._append("handoff.recorded", record, at)

    def bind_report(self, issue: int, *, at: str) -> None:
        self._append("run.report", {"report_issue": issue}, at)

    def end(self, record: dict[str, Any], *, at: str) -> None:
        self._append("run.ended", record, at)

    def resume(self, reason: str, *, at: str) -> None:
        self._append("run.resumed", {"reason": reason}, at)

    def reference(self, entity: str, key: str) -> dict[str, Any]:
        """Return a versioned reference to the latest snapshot of one record."""
        for event in reversed(self._document["events"]):
            if _event_entity(event) == (entity, key):
                return {
                    "path": str(self.path),
                    "entity": entity,
                    "id": key,
                    "sequence": event["sequence"],
                    "sha256": self._prefix_digest(event["sequence"]),
                }
        raise LedgerError(f"unknown {entity} {key!r}")

    def _prefix_digest(self, sequence: int) -> str:
        return _digest(
            {**self._document, "events": self._document["events"][:sequence]}
        )

    def original(self, reference: dict[str, Any]) -> dict[str, Any]:
        """Resolve a reference at its recorded version, even after later writes.

        This returns the ledger record and provenance, not the external source
        body. Source acquisition and verification are separate caller duties.
        """
        ref = _fields(reference, "path entity id sequence sha256", "reference")
        sequence = ref["sequence"]
        events = self._document["events"]
        if (
            ref["path"] != str(self.path)
            or type(sequence) is not int
            or not 1 <= sequence <= len(events)
        ):
            raise LedgerError(
                "reference: wrong ledger or unavailable sequence; reload the original ledger"
            )
        event = events[sequence - 1]
        if ref["sha256"] != self._prefix_digest(sequence) or _event_entity(event) != (
            ref["entity"],
            ref["id"],
        ):
            raise LedgerError(
                "reference: content/entity mismatch; inspect the original history"
            )
        document = {**self._document, "events": events[:sequence]}
        state = _replay(document)
        record = (
            state["run"]
            if ref["entity"] == "run"
            else state[_TABLES[ref["entity"]]][ref["id"]]
        )
        return copy.deepcopy(record)

    def summary(self, *, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        """Page compact decisions and references; never inject full result history."""
        if type(limit) is not int or limit < 1 or type(offset) is not int or offset < 0:
            raise LedgerError(
                "summary: positive limit and non-negative offset required"
            )
        slot = self._document["slot"]
        tasks, findings = self._state["tasks"], self._state["findings"]
        unfinished = [
            t for t in tasks.values() if t["status"] not in {"completed", "cancelled"}
        ]
        hypotheses = [
            f for f in findings.values() if f["status"] in {"suspected", "unverified"}
        ]
        known = {_question_key(t["question"], t["scope"]) for t in tasks.values()}
        candidates = []
        seen = set()
        for task in tasks.values():
            questions = []
            if task["status"] in {"planned", "failed", "interrupted"}:
                questions.append(
                    {
                        "question": task["question"],
                        "scope": task["scope"],
                        "kind": "planned-task"
                        if task["status"] == "planned"
                        else "resume-task",
                    }
                )
            questions.extend(
                {**q, "kind": "followup"}
                for q in task["next_questions"]
                if _question_key(q["question"], q["scope"]) not in known
            )
            for question in questions:
                key = _question_key(question["question"], question["scope"])
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "kind": question["kind"],
                        "source_task_id": task["id"],
                        "question_excerpt": question["question"][:240],
                        "scope_excerpt": ", ".join(question["scope"])[:240],
                    }
                )
        counts = {
            "unfinished_tasks": len(unfinished),
            "unverified_hypotheses": len(hypotheses),
            "next_candidates": len(candidates),
        }
        return {
            "run": {
                "slot": slot,
                "base_sha": self._document["base_sha"],
                "status": self._state["run"]["status"],
                "report_issue": self._state["run"]["report_issue"],
                "ref": self.reference("run", slot),
            },
            "unfinished_tasks": [
                {
                    "id": t["id"],
                    "status": t["status"],
                    "question_excerpt": t["question"][:240],
                    "ref": self.reference("task", t["id"]),
                }
                for t in unfinished[offset : offset + limit]
            ],
            "unverified_hypotheses": [
                {
                    "id": f["id"],
                    "status": f["status"],
                    "conclusion_excerpt": f["conclusion"][:240],
                    "ref": self.reference("finding", f["id"]),
                }
                for f in hypotheses[offset : offset + limit]
            ],
            "next_candidates": [
                {**c, "ref": self.reference("task", c["source_task_id"])}
                for c in candidates[offset : offset + limit]
            ],
            "offset": offset,
            "totals": counts,
            "remaining": {
                name: max(0, count - offset - limit) for name, count in counts.items()
            },
        }
