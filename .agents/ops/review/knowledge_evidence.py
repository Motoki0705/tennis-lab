"""Bind the parent's verified acceptance to a PR HEAD and its execution evidence."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

from issues.format import raw_section
from parent.policy import digest, nonempty, number, object_value
from parent.snapshot import git
from parent.state import unique_object

from .knowledge_records import Comparison, commit_sha

_PR_FIELDS = {"pr_number", "head_sha", "acceptance", "summary", "execution"}
_EXECUTION_FIELDS = {"head_sha", "status", "command", "exit_code", "result"}
_UNVERIFIED = re.compile(
    r"未着手|未確認|未完了|未実施|未検証|未実行|"
    r"\b(?:unverified|incomplete|not (?:yet )?(?:started|verified|run|executed|completed))\b",
    re.I,
)
_SCRIPT_SUFFIXES = {
    ".py",
    ".pyw",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ksh",
    ".csh",
    ".tcsh",
    ".ps1",
    ".bat",
    ".cmd",
    ".r",
    ".jl",
    ".ipynb",
    ".rb",
    ".pl",
    ".awk",
    ".js",
    ".mjs",
    ".ts",
}


def _json_object(text: str, description: str) -> dict[str, Any]:
    try:
        return object_value(json.loads(text, object_pairs_hook=unique_object))
    except ValueError as exc:
        raise ValueError(f"{description} requires a JSON object: {exc}") from exc


def _verified_text(value: object, field: str) -> str:
    text = nonempty(value, field)
    if _UNVERIFIED.search(text):
        raise ValueError(f"{field}: evidence reports unverified or incomplete work")
    return text


def _parent_proof(value: object) -> dict[str, Any]:
    row = object_value(value)
    if set(row) != _PR_FIELDS:
        raise ValueError(
            "knowledge evidence requires pr_number, head_sha, acceptance, summary and execution"
        )
    n = number(row["pr_number"])
    commit_sha(row["head_sha"], f"PR #{n} evidence HEAD")
    if row["acceptance"] != "verified" or row["execution"] not in (
        "verified",
        "not_required",
    ):
        raise ValueError(f"PR #{n}: acceptance and required execution must be verified")
    return {
        **row,
        "summary": _verified_text(row["summary"], f"PR #{n} acceptance summary"),
    }


def read_evidence(text: str, prs: list[int]) -> dict[int, dict[str, Any]]:
    spec = _json_object(text, "knowledge acceptance evidence")
    if (
        set(spec) != {"schema_version", "prs"}
        or type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or not isinstance(spec["prs"], list)
    ):
        raise ValueError("expected version 1 knowledge evidence with a prs list")
    rows = [_parent_proof(row) for row in spec["prs"]]
    proof = {row["pr_number"]: row for row in rows}
    if len(proof) != len(rows) or set(proof) != set(prs):
        raise ValueError(
            "knowledge evidence must identify each selected PR exactly once"
        )
    return proof


def _execution_body(body: str, head: str) -> dict[str, Any]:
    section = raw_section(body.replace("\r\n", "\n"), "実行証拠")
    match = re.fullmatch(r"```json[ \t]*\n(.*?)\n```", section, re.DOTALL)
    if match is None:
        raise ValueError("PR body requires a JSON block under '## 実行証拠'")
    proof = _json_object(match[1], "PR execution evidence")
    if proof.get("head_sha") != head:
        raise ValueError("PR execution evidence does not match the current HEAD")
    if proof.get("status") == "not_required" and set(proof) == {"head_sha", "status"}:
        return proof
    if (
        set(proof) != _EXECUTION_FIELDS
        or proof["status"] != "passed"
        or type(proof["exit_code"]) is not int
        or proof["exit_code"] != 0
    ):
        raise ValueError(
            "PR body requires a completed successful execution, command and result"
        )
    nonempty(proof["command"], "execution command")
    _verified_text(proof["result"], "execution result")
    return proof


def _includes_script(root: Path, comparison: Comparison) -> bool:
    fork = git(
        "--no-replace-objects", "merge-base", comparison.base, comparison.head, cwd=root
    )
    ranges = (
        (fork.decode().strip(), comparison.head),
        (comparison.base, comparison.merge),
    )
    for start, end in ranges:
        fields = git(
            "--no-replace-objects",
            "diff",
            "--no-ext-diff",
            "--no-renames",
            "--no-relative",
            "--raw",
            "--diff-filter=AMT",
            "-z",
            start,
            end,
            "--",
            "knowledge/runs/",
            cwd=root,
        ).split(b"\0")
        if fields.pop() or len(fields) % 2:
            raise ValueError("incomplete Git diff for bundled run scripts")
        for metadata, path in zip(fields[::2], fields[1::2], strict=True):
            if (
                metadata.split()[1] == b"100755"
                or PurePosixPath(os.fsdecode(path)).suffix.casefold()
                in _SCRIPT_SUFFIXES
            ):
                return True
    return False


def bind(
    root: Path, pr: dict[str, Any], comparison: Comparison, parent_proof: object
) -> dict[str, Any]:
    proof = _parent_proof(parent_proof)
    n = pr["number"]
    if proof["pr_number"] != n or proof["head_sha"] != comparison.head:
        raise ValueError(
            f"PR #{n}: acceptance evidence does not match the PR and its current HEAD"
        )
    body = pr.get("body") or ""
    execution = _execution_body(body, comparison.head)
    required = execution["status"] == "passed"
    if not required and _includes_script(root, comparison):
        raise ValueError(
            f"PR #{n}: bundled run scripts require execution evidence in the PR body"
        )
    if proof["execution"] != ("verified" if required else "not_required"):
        raise ValueError(
            f"PR #{n}: parent execution verification disagrees with PR execution evidence"
        )
    return {"parent": proof, "execution": execution, "body_sha256": digest(body)}


def require_current(
    root: Path, pr: dict[str, Any], comparison: Comparison, saved: object
) -> None:
    if saved is None:
        raise ValueError(
            "current knowledge PR evidence is required; use merge-continue"
        )
    receipt = object_value(saved)
    if bind(root, pr, comparison, receipt.get("parent")) != receipt:
        raise ValueError(
            "PR body or execution evidence changed; use merge-continue with verified evidence"
        )
