"""Pinned ci_watch reads and ledger-compatible evidence; no source mutation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ci_watch.store import Store as ObservationStore
from issues import ci_failure_marker

from .store import Store, digest

CATEGORIES = {"code_regression", "flaky", "runner_infrastructure", "unclassified"}


def source_reference(source: Path, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(source.resolve() / "observations.sqlite3"),
        "entity": "ci_observation",
        "id": record["sha"],
        "sequence": record["sequence"],
        "sha256": digest(record),
    }


def evidence(reference: dict[str, Any], at: str, sha: str, kind: str) -> dict[str, Any]:
    return {
        "id": kind + ":" + digest(reference),
        "source_kind": kind,
        "reference": json.dumps(reference, sort_keys=True),
        "acquired_at": at,
        "target_version": sha,
        "pr_head": None,
    }


def read_observations(
    source: ObservationStore, state: Store
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, tuple[int, str]]:
    latest: dict[str, dict[str, Any]] = {}
    collection = None
    after, last = 0, ""
    checkpoint = state.checkpoint()
    verified = checkpoint is None
    while True:
        page = source.records(after, 100, None)
        for record in page["records"]:
            if record["sequence"] != after + 1:
                raise ValueError("CI observation sequence has a gap or is nonmonotonic")
            after, last = record["sequence"], digest(record)
            if checkpoint and after == checkpoint[0]:
                if last != checkpoint[1]:
                    raise ValueError(
                        "CI observation history changed at saved checkpoint"
                    )
                verified = True
            if record["kind"] == "observation":
                latest[record["sha"]] = record
            elif record["kind"] == "collection":
                collection = record
            else:
                raise ValueError("unknown CI observation record kind")
        if not page["has_more"]:
            break
    if not verified or not latest:
        raise ValueError("CI observation history missing; preserve and reconcile state")
    return list(latest.values()), collection, (after, last)


def group_failures(
    records: list[dict[str, Any]], source: ObservationStore
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    checked: set[str] = set()

    def check_objects(value: Any) -> None:
        if isinstance(value, dict):
            if "object" in value and value["object"] not in checked:
                source.object(value["object"])
                checked.add(value["object"])
            for item in value.values():
                check_objects(item)
        elif isinstance(value, list):
            for item in value:
                check_objects(item)

    for record in records:
        observation = record["observation"]
        if observation["sha"] != record["sha"]:
            raise ValueError("CI observation SHA mismatch")
        check_objects(observation)
        ref = source_reference(source.directory, record)
        if datetime.fromisoformat(record["observed_at"]).tzinfo is None:
            raise ValueError("CI observation timestamp needs timezone")
        for failure in observation["failures"]:
            signature = failure["signature"]
            ci_failure_marker(signature)
            classification = failure["classification"]
            if (
                classification["category"] not in CATEGORIES
                or not classification["reasons"]
                or classification["assessment"] != "hypothesis"
                or classification["confidence"] not in {"low", "medium", "high"}
            ):
                raise ValueError("unknown or unsupported CI failure classification")
            if failure["sha"] != record["sha"] or not failure["evidence"]:
                raise ValueError("CI failure has mismatched SHA or no evidence")
            identity = [
                failure[k] for k in ("sha", "run_id", "attempt", "job_id", "signature")
            ]
            groups.setdefault(signature, []).append(
                {
                    "id": digest(identity),
                    "failure": failure,
                    "source": ref,
                    "evidence": evidence(
                        ref, record["observed_at"], record["sha"], "main-ci-failure"
                    ),
                }
            )
    for items in groups.values():
        items.sort(key=lambda x: x["id"])
        if len({x["id"] for x in items}) != len(items):
            raise ValueError("duplicate CI failure occurrence")
    return groups
