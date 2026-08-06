#!/usr/bin/env python3
"""Initialize the deterministic artifact tree for one tennis-lab GitHub issue."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from issue_task_candidate import initial_base_revision
from issue_task_issue import (
    AcceptanceItem,
    escape_table_cell,
    write_issue_snapshot,
)
from issue_task_state import CURRENT_SCHEMA_VERSION

DEFAULT_REPO = "Motoki0705/tennis-lab"
ISSUE_NUMBER_RE = re.compile(r"(?:^|/issues/)(\d+)(?:$|[/?#])")

TEMPLATES = {
    "00-feasibility/feasibility.md": """# Feasibility

- Issue: #{number}
- Attempt: {attempt}
- Status: PENDING
- Frozen issue SHA-256: `{issue_hash}`
- Frozen acceptance checklist SHA-256: `{checklist_hash}`

## Allowed and prohibited changes

## Required checks and baseline

## Breaking-change and compatibility impact

## Acceptance checklist feasibility

| ID | Issue checklist item | Verdict | Required change and evidence |
|---|---|---|---|
{feasibility_rows}

## Constraint conflicts

None

## Final feasibility verdict

PENDING

## Blocker resolution required

None
""",
    "01-exploration/exploration.md": """# Exploration

- Issue: #{number}
- Attempt: {attempt}
- Status: PENDING

## Scope and Issue interpretation

## Relevant files and symbols

## Entry points and execution paths

## Data, configuration, and interface contracts

## Existing tests and fixtures

## Invariants and compatibility constraints

## Risks and likely impact radius

## Unresolved questions

None

## Evidence table

| Kind | Claim | Evidence |
|---|---|---|
| PENDING | Replace this row | Replace this row |
""",
    "02-planning/plan.md": """# Plan

- Issue: #{number}
- Attempt: {attempt}
- Status: PENDING
- Frozen issue SHA-256: `{issue_hash}`
- Frozen acceptance checklist SHA-256: `{checklist_hash}`

## Acceptance checklist mapping

| ID | Issue checklist item | Planned implementation | Validation method |
|---|---|---|---|
{plan_rows}

## Planned files and symbols

## Implementation topology and ownership

## Independent test work unit

## Canonical verification commands

Define the exact commands in `02-planning/checks.json` and summarize their IDs here.

## Ordered execution plan

## Validation strategy

## Non-goals and prohibited changes

## Risks, rollback, and open decisions

None
""",
    "03-implementation/implementation.md": """# Implementation

- Issue: #{number}
- Attempt: {attempt}
- Test cycle: 1
- Status: PENDING

## Assigned ownership

## Files and symbols changed

## Behavior implemented

## Plan deviations and rationale

None

## Commands and results

## Known limitations and remaining risks

None

## Handoff
""",
    "03-implementation/preflight.md": """# Production preflight

- Issue: #{number}
- Attempt: {attempt}
- Test cycle: 1
- Status: PENDING
- Candidate SHA-256: `PENDING`

## Candidate identity

## Changed scope

## Deterministic policy checks

## Focused checks

## Canonical command results

## Baseline comparison

## Commands and exact outcomes

## Final production preflight verdict

PENDING

## RETURN implementation findings

None
""",
    "03-implementation/tests.md": """# Tests

- Issue: #{number}
- Attempt: {attempt}
- Test cycle: 1
- Status: PENDING
- Frozen acceptance checklist SHA-256: `{checklist_hash}`
- Candidate SHA-256: `PENDING`

## Candidate identity

## Acceptance-checklist-to-test mapping

| ID | Issue checklist item | Test or authoritative evidence | Result |
|---|---|---|---|
{test_rows}

## Tests added or changed

None

## Normal, boundary, invalid, and regression cases

## Canonical command results

## Commands and exact outcomes

## Failures encountered

None

## Untested risks and reasons

None

## Final test verdict

PENDING

## RETURN implementation findings

None
""",
    "03-implementation/seal.md": """# Final candidate seal

- Issue: #{number}
- Attempt: {attempt}
- Test cycle: 1
- Status: PENDING
- Candidate SHA-256: `PENDING`

## Candidate identity

## Changed-since-test inspection

## Canonical command results

## Complete scope inspection

## Commands and exact outcomes

## Final candidate seal verdict

PENDING

## RETURN implementation findings

None
""",
    "04-validation/validation.md": """# Validation

- Issue: #{number}
- Attempt: {attempt}
- Status: PENDING
- Frozen issue SHA-256: `{issue_hash}`
- Frozen acceptance checklist SHA-256: `{checklist_hash}`
- Candidate SHA-256: `PENDING`

## Inspection scope and revision

## Acceptance checklist verification

| ID | Issue checklist item | Verdict | Evidence |
|---|---|---|---|
{validation_rows}

## Code evidence

## Runtime and test evidence

## Regression and repository-rule checks

## Final verdict

PENDING

## RETURN exploration questions

None
""",
    "05-packaging/packaging.md": """# Packaging

- Issue: #{number}
- Attempt: {attempt}
- Status: PENDING
- Candidate SHA-256: `PENDING`
- PR number: 0
- PR head SHA: `PENDING`
- Remote checks: PENDING
- PR evidence SHA-256: `PENDING`

## Final candidate binding

## Pull request identity

## Complete paginated diff scope

## Remote required checks

## Packaging evidence

## Final packaging verdict

PENDING
""",
}

DEFAULT_CHECKS = {"schema_version": 1, "checks": []}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("issue", help="GitHub issue number or URL")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--root", type=Path, default=Path(".codex/tasks"))
    parser.add_argument(
        "--refresh-issue",
        action="store_true",
        help="Refresh the frozen Issue and restart at feasibility.",
    )
    return parser.parse_args()


def issue_number(value: str) -> int:
    if value.isdigit():
        return int(value)
    match = ISSUE_NUMBER_RE.search(value)
    if match is None:
        raise ValueError(f"cannot parse issue number from {value!r}")
    return int(match.group(1))


def run_gh(number: int, repo: str) -> dict[str, Any]:
    if shutil.which("gh") is None:
        raise RuntimeError("gh CLI is required and was not found on PATH")
    command = [
        "gh",
        "issue",
        "view",
        str(number),
        "--repo",
        repo,
        "--json",
        "number,title,body,url,state,labels,updatedAt",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "gh issue view failed")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("gh returned an unexpected issue payload")
    return payload


def render_feasibility_rows(items: list[AcceptanceItem]) -> str:
    return "\n".join(
        f"| {item.item_id} | {escape_table_cell(item.text)} | UNKNOWN | Replace this evidence |"
        for item in items
    )


def render_plan_rows(items: list[AcceptanceItem]) -> str:
    return "\n".join(
        f"| {item.item_id} | {escape_table_cell(item.text)} | PENDING | PENDING |"
        for item in items
    )


def render_test_rows(items: list[AcceptanceItem]) -> str:
    return "\n".join(
        f"| {item.item_id} | {escape_table_cell(item.text)} | PENDING | PENDING |"
        for item in items
    )


def render_validation_rows(items: list[AcceptanceItem]) -> str:
    return "\n".join(
        f"| {item.item_id} | {escape_table_cell(item.text)} | NOT VERIFIED | Replace this evidence |"
        for item in items
    )


def render_state(
    payload: dict[str, Any],
    digest: str,
    checklist_digest: str,
    checklist_count: int,
    attempt: int = 1,
    *,
    issue_snapshot_digest: str = "",
    base_revision: str = "",
    schema_version: int = CURRENT_SCHEMA_VERSION,
) -> str:
    """Render state; schema_version is explicit for migration tests."""
    now = datetime.now(UTC).isoformat()
    if schema_version == 4:
        return (
            "schema_version = 4\n"
            f"issue_number = {payload['number']}\n"
            f"issue_url = {json.dumps(payload['url'])}\n"
            f"issue_sha256 = {json.dumps(digest)}\n"
            f"acceptance_checklist_sha256 = {json.dumps(checklist_digest)}\n"
            f"acceptance_checklist_count = {checklist_count}\n"
            f"attempt = {attempt}\n"
            'feasibility_verdict = ""\n'
            "preflight_cycle = 0\n"
            'preflight_verdict = ""\n'
            "test_cycle = 0\n"
            'test_verdict = ""\n'
            "test_return_count = 0\n"
            "return_review_required = false\n"
            'return_review_action = ""\n'
            'return_review_reason = ""\n'
            'phase = "feasibility"\n'
            'status = "in_progress"\n'
            'verdict = ""\n'
            'block_kind = ""\n'
            'block_reason = ""\n'
            f"updated_at = {json.dumps(now)}\n"
        )
    return (
        f"schema_version = {CURRENT_SCHEMA_VERSION}\n"
        f"issue_number = {payload['number']}\n"
        f"issue_url = {json.dumps(payload['url'])}\n"
        f"issue_sha256 = {json.dumps(digest)}\n"
        f"issue_snapshot_sha256 = {json.dumps(issue_snapshot_digest)}\n"
        f"acceptance_checklist_sha256 = {json.dumps(checklist_digest)}\n"
        f"acceptance_checklist_count = {checklist_count}\n"
        f"base_revision = {json.dumps(base_revision)}\n"
        'candidate_binding_mode = "ENFORCED"\n'
        f"attempt = {attempt}\n"
        'feasibility_verdict = ""\n'
        "preflight_cycle = 0\n"
        'preflight_verdict = ""\n'
        'preflight_candidate_sha256 = ""\n'
        "test_cycle = 0\n"
        'test_verdict = ""\n'
        'test_candidate_sha256 = ""\n'
        "seal_cycle = 0\n"
        'seal_verdict = ""\n'
        'sealed_candidate_sha256 = ""\n'
        "test_return_count = 0\n"
        "return_review_required = false\n"
        'return_review_action = ""\n'
        'return_review_reason = ""\n'
        'validation_candidate_sha256 = ""\n'
        'packaging_candidate_sha256 = ""\n'
        "pr_number = 0\n"
        'pr_head_sha = ""\n'
        'remote_checks_verdict = ""\n'
        'pr_evidence_sha256 = ""\n'
        'phase = "feasibility"\n'
        'status = "in_progress"\n'
        'verdict = ""\n'
        'block_kind = ""\n'
        'block_reason = ""\n'
        f"updated_at = {json.dumps(now)}\n"
    )


def existing_attempt(state_path: Path) -> int:
    if not state_path.exists():
        return 0
    import tomllib

    with state_path.open("rb") as handle:
        state = tomllib.load(handle)
    value = state.get("attempt", 0)
    return int(value) if isinstance(value, int) else 0


def main() -> int:
    args = parse_args()
    try:
        number = issue_number(args.issue)
        payload = run_gh(number, args.repo)
        task_dir = args.root / f"issue-{number}"
        state_path = task_dir / "state.toml"
        if task_dir.exists() and not args.refresh_issue:
            raise RuntimeError(
                f"{task_dir} already exists; use --refresh-issue only when the upstream issue changed"
            )
        task_dir.mkdir(parents=True, exist_ok=True)
        issue_hash, issue_snapshot_hash, checklist_hash, items = write_issue_snapshot(
            task_dir,
            payload,
        )
        attempt = max(existing_attempt(state_path) + 1, 1) if args.refresh_issue else 1
        state_path.write_text(
            render_state(
                payload,
                issue_hash,
                checklist_hash,
                len(items),
                attempt,
                issue_snapshot_digest=issue_snapshot_hash,
                base_revision=initial_base_revision(task_dir),
            ),
            encoding="utf-8",
        )
        format_values = {
            "number": number,
            "attempt": attempt,
            "issue_hash": issue_hash,
            "checklist_hash": checklist_hash,
            "feasibility_rows": render_feasibility_rows(items),
            "plan_rows": render_plan_rows(items),
            "test_rows": render_test_rows(items),
            "validation_rows": render_validation_rows(items),
        }
        for relative_path, template in TEMPLATES.items():
            path = task_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or args.refresh_issue or relative_path == "00-feasibility/feasibility.md":
                path.write_text(template.format(**format_values), encoding="utf-8")
        checks_path = task_dir / "02-planning/checks.json"
        if not checks_path.exists() or args.refresh_issue:
            checks_path.write_text(
                json.dumps(DEFAULT_CHECKS, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(task_dir)
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
