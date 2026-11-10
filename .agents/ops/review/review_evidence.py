"""Shared, read-only validation of parent follow-up receipts and public copies."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merge_gate_policy import REQUIRED_CHECK
from parent.policy import digest, object_value
from parent.session import ParentSession
from parent_review_policy import load_json, records
from review_workspace import changed_paths


def followups(
    directory: Path,
    result: Mapping[str, Any],
    parent_comments: Sequence[Mapping[str, Any]],
    pr_comments: Mapping[int, Sequence[Mapping[str, Any]]],
    login: str,
    current: Mapping[int, Mapping[str, Any]],
    fix_hashes: Mapping[int, str],
    *,
    selected: set[int] | None = None,
) -> list[dict[str, Any]]:
    statuses = []
    for record in records(parent_comments, login):
        if record.get("review_run_id") != result["review_run_id"] or record[
            "kind"
        ] not in {
            "resolution",
            "evidence",
        }:
            continue
        if selected is not None and record["pr"] not in selected:
            continue
        errors = []
        key = digest(record)
        prefix = "resolution" if record["kind"] == "resolution" else "evidence"
        path = directory / f"{prefix}-{key}.json"
        if not path.exists():
            errors.append("local_receipt_missing")
        else:
            saved = object_value(load_json(path.read_text(encoding="utf-8")))
            if saved.get("published") is not True or saved.get("record") != record:
                errors.append("local_receipt_unpublished_or_inconsistent")
        if errors:
            statuses.append({"record": record, "errors": errors})
            continue
        if record not in records(pr_comments.get(record["pr"], []), login):
            errors.append("pr_copy_missing")
            statuses.append({"record": record, "errors": errors})
            continue
        if record["kind"] == "evidence":
            snapshot = record["snapshot"]
            if record["evidence"]["reviewed_paths"] != changed_paths(
                snapshot["merge_base_sha"], snapshot["head_sha"]
            ):
                errors.append("evidence_paths_mismatch")
        else:
            diff = directory / f"resolution-{key}.diff"
            if not diff.exists():
                errors.append("resolution_diff_missing")
            elif (
                hashlib.sha256(diff.read_bytes()).hexdigest()
                != record["fix_diff_sha256"]
            ):
                errors.append("resolution_diff_mismatch")
            latest = current.get(record["pr"])
            if (
                latest
                and record["fixed_snapshot"]["head_sha"] == latest["head_sha"]
                and (record["fix_diff_sha256"] != fix_hashes.get(record["pr"]))
            ):
                errors.append("resolution_fix_diff_mismatch")
        statuses.append({"record": record, "errors": errors})
    return statuses


def python_check_runs(session: ParentSession, head: str) -> list[dict[str, Any]]:
    """Fetch the complete latest check-run evidence for the pinned commit."""
    runs: list[dict[str, Any]] = []
    ids: set[int] = set()
    total = None
    page = 1
    while True:
        data = object_value(
            session.gh.get(
                f"repos/{session.gh.repo}/commits/{head}/check-runs"
                f"?check_name={REQUIRED_CHECK}&filter=latest&per_page=100&page={page}"
            )
        )
        count, rows = data.get("total_count"), data.get("check_runs")
        if (
            type(count) is not int
            or count < 0
            or not isinstance(rows, list)
            or len(rows) > 100
            or (total is not None and count != total)
        ):
            raise ValueError("incomplete or inconsistent Python check-runs response")
        total = count
        for row in rows:
            run = object_value(row)
            identity = run.get("id")
            if type(identity) is not int or identity < 1 or identity in ids:
                raise ValueError("invalid or duplicate Python check-run identity")
            ids.add(identity)
            runs.append(run)
        if len(runs) > total:
            raise ValueError("Python check-runs count exceeds total_count")
        if len(rows) < 100:
            if len(runs) != total:
                raise ValueError("Python check-runs page is missing entries")
            return runs
        page += 1
