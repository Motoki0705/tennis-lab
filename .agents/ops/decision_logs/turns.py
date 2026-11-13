"""Pure inventory and distinct turn facts over a fixed public file snapshot."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from worker_logs import schema as worker_schema
from worker_logs.journal import require

ROOT = ".agents/logs/parents"
WORKER = re.compile(
    rf"{re.escape(ROOT)}/parent-([1-9][0-9]*)/child-([1-9][0-9]*)"
    r"/(d-[a-f0-9]{32})/([0-9]{4,}-(?:initial|resume))"
    r"/(instruction\.md|final\.md|meta\.json)"
)
TurnKey = tuple[str, int, int, str, int]


def turn_key(scope: dict[str, Any]) -> TurnKey | None:
    delegation = scope["delegation"]
    if delegation is None:
        return None
    return (
        scope["repo"].casefold(),
        scope["parent"],
        scope["issue"],
        delegation["key"],
        delegation["turn"],
    )


@dataclass
class Turn:
    key: TurnKey
    directory: str | None = None
    meta: dict[str, Any] | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    execution: str = "unknown"
    final: str = "unknown"
    publication: str = "unknown"
    publication_detail: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    conflicting: bool = False

    @property
    def logical_directory(self) -> str:
        _, parent, issue, key, sequence = self.key
        return f"{ROOT}/parent-{parent}/child-{issue}/{key}/turn-{sequence:04d}"


def inventory(
    files: dict[str, bytes], rows: list[dict[str, Any]]
) -> dict[TurnKey, Turn]:
    """Validate all worker triples, including parents with no new events."""
    directories = {name.rsplit("/", 1)[0] for name in files if WORKER.fullmatch(name)}
    result: dict[TurnKey, Turn] = {}
    repos = {r["scope"]["repo"].casefold() for r in rows}
    for directory in sorted(directories):
        meta = worker_schema.validate(
            {n: files.get(directory + "/" + n, b"") for n in worker_schema.FILENAMES}
        )
        require(directory == worker_schema.path(meta), "record_source")
        key = (
            meta["repo"].casefold(),
            meta["parent"],
            meta["issue"],
            meta["delegation_key"],
            meta["turn"]["sequence"],
        )
        require(key not in result, "record_turn")
        result[key] = Turn(key, directory=directory, meta=meta)
        repos.add(key[0])
    require(len(repos) <= 1, "record_repository")
    for row in rows:
        event_key = turn_key(row["scope"])
        if event_key is not None:
            result.setdefault(event_key, Turn(event_key)).records.append(row)
    for item in result.values():
        _facts(item)
    return dict(sorted(result.items()))


def _facts(item: Turn) -> None:
    executions = set()
    reports: list[tuple[dict[str, Any], str]] = []
    previous = {r["id"]: r["data"].get("previous_result_id") for r in item.records}
    for row in item.records:
        data = row["data"]
        if row["kind"] == "action" and data["operation"] in {"start_ai", "continue_ai"}:
            state = data["state"]
            if state in {"running", "ended"}:
                executions.add(state)
            elif state == "rejected_before_start":
                executions.add("not_started")
        if row["kind"] != "result":
            continue
        observation = data.get("observation")
        if observation and observation["kind"] == "worker_turn":
            execution = observation["execution"]
            executions.add(execution["state"])
            item.evidence.extend(execution["evidence"])
            item.evidence.extend(observation["final"]["evidence"])
            if observation["final"]["state"] != "unknown":
                reports.append((row, observation["final"]["state"]))
            public = observation["publication"]
            if public["state"] != "unknown":
                item.publication = public["state"]
                item.publication_detail = public
                item.evidence.extend(public["evidence"])
        elif data["final_report"]["status"] == "observed":
            reports.append((row, "present"))
            item.evidence.append(data["final_report"]["value"])
        elif data["final_report"]["reason"] == "missing_final":
            reports.append((row, "absent"))
            item.evidence.extend(data["evidence"])
    item.execution = (
        "ended"
        if "ended" in executions
        else "running"
        if "running" in executions
        else "not_started"
        if "not_started" in executions
        else "unknown"
    )
    if reports:
        last, state = reports[-1]
        ancestors = set()
        prior = previous.get(last["id"])
        while prior and prior not in ancestors:
            ancestors.add(prior)
            prior = previous.get(prior)
        item.conflicting = any(
            s != state and r["id"] not in ancestors for r, s in reports[:-1]
        )
        if not item.conflicting and (state != "absent" or item.execution == "ended"):
            item.final = state
    if item.meta is not None:
        # Validated bytes in this tree are stronger evidence than a stale PR
        # observation. They do not certify parent acceptance or this PR's merge.
        item.execution, item.final, item.publication = "ended", "present", "main"
        item.conflicting = False
    seen: set[bytes] = set()
    unique = []
    for proof in item.evidence:
        token = worker_schema.json_bytes(proof)
        if token not in seen:
            unique.append(proof)
            seen.add(token)
    item.evidence = unique


def statistics(items: list[Turn]) -> dict[str, int]:
    published = sum(t.directory is not None for t in items)
    matched = sum(t.directory is not None and bool(t.records) for t in items)
    return {
        "published": published,
        "matched": matched,
        "unmatched": published - matched,
        "event_only": sum(t.directory is None for t in items),
        "total": len(items),
        "missing_final": sum(t.final == "absent" for t in items),
        "running": sum(t.execution == "running" for t in items),
        "end_unknown": sum(t.execution == "unknown" for t in items),
        "held_final": sum(
            t.final == "present" and t.publication in {"pending", "public_pr"}
            for t in items
        ),
    }


def by_prefix(items: dict[TurnKey, Turn]) -> dict[str, list[Turn]]:
    """Each ancestor contains each logical turn once, even with two view paths."""
    groups: dict[str, list[Turn]] = defaultdict(list)
    for item in items.values():
        _, parent, issue, key, _ = item.key
        parent_path = f"{ROOT}/parent-{parent}"
        for directory in (
            ROOT,
            parent_path,
            parent_path + f"/child-{issue}",
            parent_path + f"/child-{issue}/{key}",
            item.logical_directory,
            item.directory,
        ):
            if directory is not None:
                groups[directory + "/README.md"].append(item)
    return dict(groups)
