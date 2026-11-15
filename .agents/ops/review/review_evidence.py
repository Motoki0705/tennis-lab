"""Shared, read-only validation of parent follow-up receipts and public copies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merge_gate_policy import REQUIRED_CHECK
from parent.policy import digest, object_value
from parent.session import ParentSession
from parent_review_policy import exact, load_json, path_manifest, records
from review_workspace import changed_paths, full_diff, resolution_scope


def path_summary(paths: list[str]) -> str:
    """An explicitly bounded display; the complete list is in the local receipt."""
    manifest = path_manifest(paths)
    preview: list[str] = []
    size = 0
    for path in paths[:20]:
        escaped = json.dumps(path, ensure_ascii=True)
        addition = len(escaped) + (2 if preview else 0)
        if size + addition > 2000:
            break
        preview.append(escaped)
        size += addition
    return (
        f"{manifest['count']}件 / SHA-256: {manifest['sha256']}\n"
        f"表示 {len(preview)}/{len(paths)}件（最大20件・2000文字、全量はローカル記録）: "
        + ", ".join(preview)
    )


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
            if record["schema_version"] == 2:
                keys = (
                    {"reviewed_paths"}
                    if record["kind"] == "evidence"
                    else {"changed_paths", "head_changed_paths", "base_only_paths"}
                )
                try:
                    lists = exact(saved.get("paths"), keys)
                    manifests = {key: path_manifest(lists[key]) for key in keys}
                    expected = (
                        {"reviewed_paths": record["evidence"]["reviewed_paths"]}
                        if record["kind"] == "evidence"
                        else {key: record[key] for key in keys}
                    )
                    if manifests != expected:
                        errors.append("local_paths_mismatch")
                except ValueError:
                    errors.append("local_paths_invalid")
        if errors:
            statuses.append({"record": record, "errors": errors})
            continue
        if record not in records(pr_comments.get(record["pr"], []), login):
            errors.append("pr_copy_missing")
            statuses.append({"record": record, "errors": errors})
            continue
        if record["kind"] == "evidence":
            snapshot = record["snapshot"]
            paths = changed_paths(snapshot["merge_base_sha"], snapshot["head_sha"])
            expected_paths = (
                path_manifest(paths) if record["schema_version"] == 2 else paths
            )
            if record["evidence"]["reviewed_paths"] != expected_paths:
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
            original = next(
                (p for p in result["request"]["prs"] if p["number"] == record["pr"]),
                None,
            )
            if original is None or record["reviewed_head"] != original["head_sha"]:
                errors.append("resolution_reviewed_head_mismatch")
            else:
                fixed = record["fixed_snapshot"]
                if record["schema_version"] == 2:
                    scope = resolution_scope(original, fixed)
                    if any(
                        record[key] != path_manifest(paths)
                        for key, paths in scope.paths().items()
                    ):
                        errors.append("resolution_paths_mismatch")
                elif record["changed_paths"] != changed_paths(
                    original["head_sha"], fixed["head_sha"]
                ):
                    errors.append("resolution_paths_mismatch")
                if (
                    hashlib.sha256(
                        full_diff(original["head_sha"], fixed["head_sha"])
                    ).hexdigest()
                    != record["fix_diff_sha256"]
                ):
                    errors.append("resolution_fix_diff_mismatch")
            latest = current.get(record["pr"])
            if (
                latest
                and record["fixed_snapshot"]["head_sha"] == latest["head_sha"]
                and (record["fix_diff_sha256"] != fix_hashes.get(record["pr"]))
            ) and "resolution_fix_diff_mismatch" not in errors:
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
