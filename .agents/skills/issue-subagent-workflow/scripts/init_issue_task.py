#!/usr/bin/env python3
"""Initialize the deterministic artifact tree for one tennis-lab GitHub issue."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

DEFAULT_REPO = "Motoki0705/tennis-lab"
ISSUE_NUMBER_RE = re.compile(r"(?:^|/issues/)(\d+)(?:$|[/?#])")
TASK_LIST_RE = re.compile(r"(?m)^\s*[-*+]\s+\[([ xX])\]\s+(.+?)\s*$")


class AcceptanceItem(NamedTuple):
    item_id: str
    text: str
    source_checked: bool


TEMPLATES = {
    "01-exploration/exploration.md": """# Exploration\n\n- Issue: #{number}\n- Attempt: {attempt}\n- Status: PENDING\n\n## Scope and issue interpretation\n\n## Relevant files and symbols\n\n## Entry points and execution paths\n\n## Data, configuration, and interface contracts\n\n## Existing tests and fixtures\n\n## Invariants and compatibility constraints\n\n## Risks and likely impact radius\n\n## Unresolved questions\n\n## Evidence table\n\n| Kind | Claim | Evidence |\n|---|---|---|\n| PENDING | Replace this row | Replace this row |\n""",
    "02-planning/plan.md": """# Plan\n\n- Issue: #{number}\n- Attempt: {attempt}\n- Status: PENDING\n- Frozen issue SHA-256: `{issue_hash}`\n- Frozen acceptance checklist SHA-256: `{checklist_hash}`\n\n## Acceptance checklist mapping\n\n| ID | Issue checklist item | Planned implementation | Validation method |\n|---|---|---|---|\n{plan_rows}\n\n## Planned files and symbols\n\n## Implementation work units and ownership\n\n## Independent test work unit\n\n## Ordered execution plan\n\n## Validation strategy\n\n## Non-goals and prohibited changes\n\n## Risks, rollback, and open decisions\n""",
    "03-implementation/implementation.md": """# Implementation\n\n- Issue: #{number}\n- Attempt: {attempt}\n- Status: PENDING\n\n## Assigned ownership\n\n## Files and symbols changed\n\n## Behavior implemented\n\n## Plan deviations and rationale\n\n## Commands and results\n\n## Known limitations and remaining risks\n\n## Handoff\n""",
    "03-implementation/tests.md": """# Tests\n\n- Issue: #{number}\n- Attempt: {attempt}\n- Status: PENDING\n\n## Acceptance-checklist-to-test mapping\n\n## Tests added or changed\n\n## Normal, boundary, invalid, and regression cases\n\n## Commands and exact outcomes\n\n## Failures encountered\n\n## Untested risks and reasons\n""",
    "04-validation/validation.md": """# Validation\n\n- Issue: #{number}\n- Attempt: {attempt}\n- Status: PENDING\n- Frozen issue SHA-256: `{issue_hash}`\n- Frozen acceptance checklist SHA-256: `{checklist_hash}`\n\n## Inspection scope and revision\n\n## Acceptance checklist verification\n\n| ID | Issue checklist item | Verdict | Evidence |\n|---|---|---|---|\n{validation_rows}\n\n## Code evidence\n\n## Runtime and test evidence\n\n## Regression and repository-rule checks\n\n## Final verdict\n\nPENDING\n\n## RETURN exploration questions\n""",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("issue", help="GitHub issue number or URL")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--root", type=Path, default=Path(".codex/tasks"))
    parser.add_argument(
        "--refresh-issue",
        action="store_true",
        help="Refresh issue.md and restart at exploration without replacing phase files.",
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


def canonical_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def extract_acceptance_items(body: str) -> list[AcceptanceItem]:
    raw_items = TASK_LIST_RE.findall(body)
    if not raw_items:
        raise ValueError(
            "issue body must contain at least one Markdown task-list acceptance item, "
            "for example `- [ ] observable requirement`"
        )

    items: list[AcceptanceItem] = []
    seen_texts: set[str] = set()
    for index, (mark, raw_text) in enumerate(raw_items, start=1):
        text = " ".join(raw_text.split())
        if not text:
            raise ValueError("issue acceptance checklist cannot contain a blank item")
        if text in seen_texts:
            raise ValueError(f"issue acceptance checklist contains duplicate item: {text!r}")
        seen_texts.add(text)
        items.append(AcceptanceItem(f"AC-{index:03d}", text, mark.lower() == "x"))
    return items


def acceptance_hash(items: list[AcceptanceItem]) -> str:
    canonical = json.dumps(
        [{"id": item.item_id, "text": item.text} for item in items],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def escape_table_cell(text: str) -> str:
    return text.replace("|", "\\|")


def render_acceptance_list(items: list[AcceptanceItem]) -> str:
    return "\n".join(
        f"- {item.item_id}: {item.text} (source checkbox: "
        f"{'checked' if item.source_checked else 'unchecked'})"
        for item in items
    )


def render_plan_rows(items: list[AcceptanceItem]) -> str:
    return "\n".join(
        f"| {item.item_id} | {escape_table_cell(item.text)} | PENDING | PENDING |"
        for item in items
    )


def render_validation_rows(items: list[AcceptanceItem]) -> str:
    return "\n".join(
        f"| {item.item_id} | {escape_table_cell(item.text)} | NOT VERIFIED | Replace this evidence |"
        for item in items
    )


def render_issue(
    payload: dict[str, Any], digest: str, checklist_digest: str, items: list[AcceptanceItem]
) -> str:
    labels = payload.get("labels") or []
    label_names = [item.get("name", "") for item in labels if isinstance(item, dict)]
    body = payload.get("body") or ""
    return (
        f"# GitHub Issue #{payload['number']}\n\n"
        f"- URL: {payload['url']}\n"
        f"- State: {payload['state']}\n"
        f"- Upstream updated at: {payload['updatedAt']}\n"
        f"- Snapshot SHA-256: `{digest}`\n"
        f"- Acceptance checklist SHA-256: `{checklist_digest}`\n"
        f"- Acceptance checklist item count: {len(items)}\n"
        f"- Labels: {', '.join(label_names) if label_names else '(none)'}\n\n"
        "## Acceptance checklist\n\n"
        f"{render_acceptance_list(items)}\n\n"
        "The source checkbox state is metadata only, not proof of implementation. "
        "The validator must independently verify every item.\n\n"
        f"## Title\n\n{payload['title']}\n\n"
        f"## Body\n\n{body}\n"
    )


def render_state(
    payload: dict[str, Any],
    digest: str,
    checklist_digest: str,
    checklist_count: int,
    attempt: int = 1,
) -> str:
    now = datetime.now(UTC).isoformat()
    return (
        "schema_version = 2\n"
        f"issue_number = {payload['number']}\n"
        f"issue_url = {json.dumps(payload['url'])}\n"
        f"issue_sha256 = {json.dumps(digest)}\n"
        f"acceptance_checklist_sha256 = {json.dumps(checklist_digest)}\n"
        f"acceptance_checklist_count = {checklist_count}\n"
        f"attempt = {attempt}\n"
        'phase = "exploration"\n'
        'status = "in_progress"\n'
        'verdict = ""\n'
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
        body = payload.get("body") or ""
        items = extract_acceptance_items(body)
        digest = canonical_hash(payload)
        checklist_digest = acceptance_hash(items)
        task_dir = args.root / f"issue-{number}"
        state_path = task_dir / "state.toml"

        if task_dir.exists() and not args.refresh_issue:
            raise RuntimeError(
                f"{task_dir} already exists; use --refresh-issue only when the upstream issue changed"
            )

        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "issue.md").write_text(
            render_issue(payload, digest, checklist_digest, items), encoding="utf-8"
        )

        if args.refresh_issue:
            attempt = max(existing_attempt(state_path) + 1, 1)
        else:
            attempt = 1
        state_path.write_text(
            render_state(payload, digest, checklist_digest, len(items), attempt),
            encoding="utf-8",
        )

        format_values = {
            "number": number,
            "attempt": attempt,
            "issue_hash": digest,
            "checklist_hash": checklist_digest,
            "plan_rows": render_plan_rows(items),
            "validation_rows": render_validation_rows(items),
        }
        for relative_path, template in TEMPLATES.items():
            path = task_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(template.format(**format_values), encoding="utf-8")

        print(task_dir)
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
