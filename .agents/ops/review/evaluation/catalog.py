"""Frozen historical inputs; the answer key is never an input dependency."""

from __future__ import annotations

import hashlib
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from lib.review_tools.retrieval import path_arg, strict_json
from worker_logs.journal import atomic, private_directory
from worker_logs.schema import json_bytes

DATA = Path(__file__).with_name("cases")
APPROVED_CASES = (1699, 1766, 1867, 1872, 1888, 1686, 1744)
MODELS = ("gpt-6-astra", "gpt-6.1-sol")
DIFF_FLAGS = (
    "--no-ext-diff",
    "--no-textconv",
    "--binary",
    "--full-index",
    "--no-renames",
    "--no-color",
)


def digest(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = strict_json(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("historical timestamps require a timezone")
    return parsed


def catalog(path: Path = DATA / "manifest.json") -> dict[str, Any]:
    value = read(path)
    if (
        value.get("schema_version") != 1
        or value.get("repository") != "Motoki0705/tennis-lab"
    ):
        raise ValueError("unsupported review evaluation catalog")
    if tuple(c["pr"] for c in value["cases"]) != APPROVED_CASES:
        raise ValueError("comparison cases differ from the approved minimum")
    return value


def case_record(number: int, path: Path = DATA / "manifest.json") -> dict[str, Any]:
    return next(c for c in catalog(path)["cases"] if c["pr"] == number)


def frozen_input(case: dict[str, Any], directory: Path = DATA) -> dict[str, Any]:
    path_arg(case["input_file"])
    value = read(directory / case["input_file"])
    if digest(value) != case["input_sha256"]:
        raise ValueError("frozen input digest changed")
    validate_input(value, case)
    return value


def validate_input(value: dict[str, Any], case: dict[str, Any]) -> None:
    if set(value) != {
        "schema_version",
        "case_id",
        "cutoff",
        "context",
        "github",
        "provenance",
    }:
        raise ValueError("unknown or answer-key field in reviewer input")
    if (
        value["schema_version"] != 1
        or value["case_id"] != case["case_id"]
        or value["cutoff"] != case["cutoff"]
    ):
        raise ValueError("frozen input identity differs from its case")
    timestamp(value["cutoff"])
    context = value["context"]
    if set(context) != {"parent", "children", "prs", "instruction", "parent_comments"}:
        raise ValueError("unsupported reviewer context fields")
    for row in context["parent_comments"]:
        if max(timestamp(row["created_at"]), timestamp(row["updated_at"])) > timestamp(
            value["cutoff"]
        ):
            raise ValueError("comment or edit is newer than the cutoff")
    for endpoint, response in value["github"].items():
        if not re.fullmatch(
            r"repos/Motoki0705/tennis-lab/[A-Za-z0-9_./?=&-]+", endpoint
        ):
            raise ValueError("invalid frozen GitHub endpoint")
        if set(response) != {"as_of", "value"} or response["as_of"] != value["cutoff"]:
            raise ValueError("GitHub response is not pinned to the cutoff")

    # No truth/verification metadata is serialized into the reviewer context.
    forbidden = {
        "gold",
        "gold_findings",
        "expected_findings",
        "adjudications",
        "negative_review",
    }

    def inspect(node: Any) -> None:
        if isinstance(node, dict):
            if forbidden.intersection(node):
                raise ValueError("answer-key metadata in reviewer input")
            for key, item in node.items():
                if key in {"created_at", "updated_at", "submitted_at"} and (
                    not isinstance(item, str)
                    or timestamp(item) > timestamp(value["cutoff"])
                ):
                    raise ValueError("reviewer input contains a later comment/edit")
                inspect(item)
        elif isinstance(node, list):
            for item in node:
                inspect(item)

    inspect(value)


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        timeout=45,
    ).stdout


def verify_snapshot(repo: Path, case: dict[str, Any]) -> bytes:
    snapshot = case["snapshot"]
    for key in ("head_sha", "base_sha", "merge_base_sha"):
        if not re.fullmatch(r"[0-9a-f]{40}", snapshot[key]):
            raise ValueError("snapshot needs full immutable revisions")
        if git(repo, "cat-file", "-t", snapshot[key]).strip() != b"commit":
            raise ValueError("frozen revision is unavailable; no automatic fetch")
    actual = (
        git(repo, "merge-base", snapshot["base_sha"], snapshot["head_sha"])
        .decode()
        .strip()
    )
    if actual != snapshot["merge_base_sha"]:
        raise ValueError("frozen merge base changed")
    diff = git(repo, "diff", *DIFF_FLAGS, actual, snapshot["head_sha"], "--")
    if hashlib.sha256(diff).hexdigest() != snapshot["diff_sha256"]:
        raise ValueError("frozen diff digest changed")
    return diff


def prepare(
    repo: Path, state: Path, number: int, *, data: Path = DATA
) -> dict[str, Any]:
    case = case_record(number, data / "manifest.json")
    value = frozen_input(case, data)
    diff = verify_snapshot(repo, case)
    directory = state / "bundles" / case["case_id"]
    if directory.exists():
        saved = read(directory / "bundle.json")
        if (
            saved != {"schema_version": 1, "case": case, "input": value}
            or (directory / "review.diff").read_bytes() != diff
        ):
            raise ValueError("existing evaluation bundle differs")
        return saved
    private_directory(directory)
    saved = {"schema_version": 1, "case": case, "input": value}
    atomic(directory / "bundle.json", saved)
    (directory / "review.diff").write_bytes(diff)
    (directory / "review.diff").chmod(0o600)
    return saved
