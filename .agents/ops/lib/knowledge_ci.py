"""Classify pinned PR diffs and verify the Knowledge library CI result."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

WORKFLOW_PATH = ".github/workflows/knowledge.yml"
RESULT_FILENAME = "knowledge-records.json"
# Kept in sync with the pull_request paths in knowledge.yml by test_knowledge_ci.py.
KNOWLEDGE_CI_PATHS = (
    "knowledge/**",
    ".agents/skills/knowledge-control/**",
    "tests/e2e/skills/test_knowledge*.py",
    "tests/unit/skills/test_knowledge_citation_schema.py",
    "tests/unit/skills/test_citation_literature.py",
    "tests/unit/skills/test_citation_openalex.py",
    "tests/integration/skills/test_citation_explore.py",
    ".agents/ops/lib/agy_headless.py",
    ".agents/ops/lib/agents.conf",
    ".agents/ops/lib/knowledge_ci.py",
    "tests/fixtures/knowledge/**",
    WORKFLOW_PATH,
    ".github/workflows/ci.yml",
    "src/tennis_scene/configs/**",
    "LICENSES.md",
)
# The workflow uses * within a path component and /** for recursive paths.
_KNOWLEDGE_PATH_PATTERNS = tuple(
    re.compile(re.escape(path).replace(r"\*\*", ".*").replace(r"\*", "[^/]*"), re.S)
    for path in KNOWLEDGE_CI_PATHS
)
RECORD_DIRECTORIES = (
    "knowledge/nodes/",
    "knowledge/runs/",
    "knowledge/Papers/",
    "knowledge/citation_map/",
    "knowledge/reports/",
)
_RECORD_SCOPE = "records-only"
_FILE_MODES = {b"000000", b"100644", b"100755"}
_CHECKS = ("kg_validate", "kg_repro_paths")


@dataclass(frozen=True)
class Scope:
    records_only: bool
    reason: str


def _sha(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[0-9a-f]{40}", value)
        or value == "0" * 40
    ):
        raise ValueError(f"{field} must be a full commit SHA")
    return value


def _positive_int(value: object, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "--no-replace-objects", *args],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout


def ensure_commits(root: Path, *shas: str) -> None:
    """Obtain the exact commit objects without updating refs or FETCH_HEAD."""
    missing = []
    for value in dict.fromkeys(shas):
        sha = _sha(value, "pinned commit")
        try:
            kind = _git(root, "cat-file", "-t", sha)
        except subprocess.CalledProcessError:
            missing.append(sha)
        else:
            if kind.strip() != b"commit":
                raise ValueError(f"{sha} is not a commit")
    if missing:
        _git(
            root,
            "fetch",
            "--no-tags",
            "--no-write-fetch-head",
            "--refmap=",
            "--no-prune",
            "--no-prune-tags",
            "--no-recurse-submodules",
            "--no-auto-maintenance",
            "origin",
            *missing,
        )
        for sha in missing:
            if _git(root, "cat-file", "-t", sha).strip() != b"commit":
                raise ValueError(f"{sha} is not a commit")


def require_unchanged_base(root: Path, old_base: str, new_base: str) -> None:
    """Raise ValueError unless the base advanced only outside Knowledge CI paths."""
    old = _sha(old_base, "previous base")
    new = _sha(new_base, "current base")
    if old == new:
        return
    try:
        ensure_commits(root, old, new)
        try:
            _git(root, "merge-base", "--is-ancestor", old, new)
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 1:
                raise ValueError(f"{old} is not an ancestor of {new}") from exc
            raise
        diff = _git(
            root,
            "diff",
            "--name-only",
            "-z",
            "--no-renames",
            "--no-relative",
            "--ignore-submodules=none",
            old,
            new,
            "--",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(
            f"cannot verify knowledge paths from {old} to {new}: {exc}"
        ) from exc
    if diff and not diff.endswith(b"\0"):
        raise ValueError("incomplete Git diff for knowledge paths")
    paths = [os.fsdecode(name) for name in diff.split(b"\0")[:-1]]
    changed = [
        path
        for path in paths
        if any(pattern.fullmatch(path) for pattern in _KNOWLEDGE_PATH_PATTERNS)
    ]
    if changed:
        summary = ", ".join(repr(path) for path in changed[:3])
        if len(changed) > 3:
            summary += f", and {len(changed) - 3} more"
        raise ValueError(f"main changed knowledge paths since verified CI ({summary})")


def _require_record_diff(root: Path, start: str, end: str, comparison: str) -> None:
    # Disable rename detection so every move appears as a deletion and an addition.
    diff = _git(
        root,
        "diff",
        "--raw",
        "-z",
        "--no-renames",
        "--no-relative",
        "--no-ext-diff",
        "--ignore-submodules=none",
        start,
        end,
        "--",
    )
    if not diff:
        raise ValueError(f"{comparison} has no changed paths")
    fields = diff.split(b"\0")
    if fields.pop() or len(fields) % 2:
        raise ValueError(f"{comparison} has an incomplete Git diff")
    for metadata, name in zip(fields[::2], fields[1::2], strict=True):
        path = name.decode("utf-8")
        if path != "knowledge/summary.md" and not path.startswith(RECORD_DIRECTORIES):
            raise ValueError(f"{comparison} changes non-record path {path!r}")
        old_mode, new_mode, _, _, _ = metadata.removeprefix(b":").split()
        if old_mode not in _FILE_MODES or new_mode not in _FILE_MODES:
            raise ValueError(f"{comparison} changes a non-regular file at {path!r}")


def classify_pr(root: Path, *, head_sha: str, base_sha: str, merge_sha: str) -> Scope:
    """Classify a PR from its pinned HEAD, target tip and GitHub test merge commit.

    Require the full Git history. Check both the PR's proposed changes and the
    changes as merged onto the target. Unavailable evidence selects ordinary CI.
    """
    try:
        head = _sha(head_sha, "PR HEAD")
        base = _sha(base_sha, "PR base")
        merge = _sha(merge_sha, "PR test merge")
        parents = _git(root, "rev-list", "--parents", "-1", merge).split()
        if parents != [merge.encode(), base.encode(), head.encode()]:
            raise ValueError(
                "PR test merge parents do not match the pinned base and HEAD"
            )
        if _git(root, "rev-parse", "--is-shallow-repository").strip() != b"false":
            raise ValueError("full Git history is required to establish the PR diff")
        fork_point = _sha(
            _git(root, "merge-base", "--all", base, head).decode("ascii").strip(),
            "unique PR merge base",
        )
        _require_record_diff(root, fork_point, head, "PR diff")
        _require_record_diff(root, base, merge, "test merge diff")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return Scope(False, str(exc))
    return Scope(True, "the PR diff and test merge diff contain only knowledge records")


def artifact_name(run_id: int, run_attempt: int) -> str:
    return f"knowledge-records-{run_id}-{run_attempt}"


def verify_result(
    data: bytes | None,
    run: Mapping[str, object],
    *,
    pr_number: int,
    head_sha: str,
    base_sha: str,
    merge_sha: str,
) -> None:
    """Require a successful artifact from the exact Knowledge library PR run attempt.

    The caller supplies REST run metadata and the result file from that run's
    artifact, then compares against the PR, target branch and test merge used
    for scope classification. Raises ValueError with the failed check.
    """
    _positive_int(pr_number, "PR number")
    _sha(head_sha, "expected HEAD")
    _sha(base_sha, "expected base")
    _sha(merge_sha, "expected test merge")
    if run.get("event") != "pull_request" or run.get("path") != WORKFLOW_PATH:
        raise ValueError("expected a Knowledge library pull_request run")
    if run.get("head_sha") != head_sha:
        raise ValueError("workflow run HEAD does not match the expected HEAD")
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise ValueError("Knowledge library workflow run has not succeeded")
    run_id = _positive_int(run.get("id"), "workflow run ID")
    attempt = _positive_int(run.get("run_attempt"), "workflow run attempt")
    if not data:
        raise ValueError("missing knowledge records result")
    try:
        result = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid knowledge records result JSON") from exc
    if (
        not isinstance(result, dict)
        or type(result.get("schema_version")) is not int
        or result["schema_version"] != 1
        or result.get("scope") != _RECORD_SCOPE
    ):
        raise ValueError("expected a schema version 1 records-only result")
    for field, expected in (
        ("pr_number", pr_number),
        ("head_sha", head_sha),
        ("base_sha", base_sha),
        ("merge_sha", merge_sha),
        ("run_id", run_id),
        ("run_attempt", attempt),
    ):
        value = result.get(field)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(
                f"knowledge records result {field} does not match {expected}"
            )
    if result.get("conclusion") != "success" or result.get("checks") != dict.fromkeys(
        _CHECKS, "success"
    ):
        raise ValueError("both knowledge records checks must have succeeded")


def _actions_output(**fields: str) -> None:
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        for name, value in fields.items():
            print(f"{name}={value}", file=output)


def _actions_scope() -> None:
    event = os.environ["GITHUB_EVENT_NAME"]
    identity = {}
    decision = Scope(False, f"{event} uses ordinary CI")
    if event == "pull_request":
        try:
            root = Path.cwd()
            payload = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes())
            pr = payload["pull_request"]
            number = _positive_int(pr["number"], "PR number")
            head = _sha(pr["head"]["sha"], "PR HEAD")
            base = _sha(pr["base"]["sha"], "PR base")
            merge = _sha(os.environ["GITHUB_SHA"], "GITHUB_SHA")
            if _git(root, "rev-parse", "HEAD").strip() != merge.encode():
                raise ValueError("checkout differs from GITHUB_SHA")
            decision = classify_pr(root, head_sha=head, base_sha=base, merge_sha=merge)
            if decision.records_only:
                identity = {
                    "pr_number": str(number),
                    "head_sha": head,
                    "base_sha": base,
                    "merge_sha": merge,
                }
        except (
            KeyError,
            TypeError,
            OSError,
            subprocess.SubprocessError,
            ValueError,
        ) as exc:
            decision = Scope(False, f"cannot establish the PR comparison: {exc}")
    print(
        f"Knowledge CI records_only={str(decision.records_only).lower()}: {decision.reason}"
    )
    _actions_output(records_only=str(decision.records_only).lower(), **identity)


def _write_result(path: Path) -> None:
    env = os.environ
    checks = {
        "kg_validate": env["KG_VALIDATE_OUTCOME"],
        "kg_repro_paths": env["KG_REPRO_PATHS_OUTCOME"],
    }
    run_id = _positive_int(int(env["GITHUB_RUN_ID"]), "workflow run ID")
    attempt = _positive_int(int(env["GITHUB_RUN_ATTEMPT"]), "workflow run attempt")
    result = {
        "schema_version": 1,
        "scope": _RECORD_SCOPE,
        "pr_number": _positive_int(int(env["KG_PR_NUMBER"]), "PR number"),
        "head_sha": _sha(env["KG_HEAD_SHA"], "PR HEAD"),
        "base_sha": _sha(env["KG_BASE_SHA"], "PR base"),
        "merge_sha": _sha(env["KG_MERGE_SHA"], "PR test merge"),
        "run_id": run_id,
        "run_attempt": attempt,
        "checks": checks,
        "conclusion": "success"
        if all(v == "success" for v in checks.values())
        else "failure",
    }
    serialized = json.dumps(result, indent=2) + "\n"
    path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    _actions_output(artifact_name=artifact_name(run_id, attempt))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scope")
    writer = commands.add_parser("write-result")
    writer.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "scope":
        _actions_scope()
    else:
        _write_result(args.directory / RESULT_FILENAME)


if __name__ == "__main__":
    main()
