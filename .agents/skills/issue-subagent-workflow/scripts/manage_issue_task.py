#!/usr/bin/env python3
"""Transition and validate issue-subagent workflow state."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PHASES = ("exploration", "planning", "implementation", "validation")
NEXT_PHASE = {
    "exploration": "planning",
    "planning": "implementation",
    "implementation": "validation",
}
REQUIRED_FILES = (
    "issue.md",
    "state.toml",
    "01-exploration/exploration.md",
    "02-planning/plan.md",
    "03-implementation/implementation.md",
    "03-implementation/tests.md",
    "04-validation/validation.md",
)
REQUIRED_HEADINGS = {
    "01-exploration/exploration.md": (
        "## Relevant files and symbols",
        "## Entry points and execution paths",
        "## Existing tests and fixtures",
        "## Unresolved questions",
        "## Evidence table",
    ),
    "02-planning/plan.md": (
        "## Acceptance criteria",
        "## Acceptance-to-change mapping",
        "## Implementation work units and ownership",
        "## Independent test work unit",
        "## Validation strategy",
    ),
    "03-implementation/implementation.md": (
        "## Files and symbols changed",
        "## Behavior implemented",
        "## Commands and results",
    ),
    "03-implementation/tests.md": (
        "## Acceptance-to-test mapping",
        "## Tests added or changed",
        "## Commands and exact outcomes",
    ),
    "04-validation/validation.md": (
        "## Requirement matrix",
        "## Code evidence",
        "## Runtime and test evidence",
        "## Final verdict",
    ),
}
VERSIONED_ARTIFACT_RE = re.compile(
    r"(?:exploration|plan|implementation|tests|validation)(?:[-_.](?:v?\d+|final|revised|attempt[-_]?\d+))\.md$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    transition = subparsers.add_parser("transition")
    transition.add_argument("task_dir", type=Path)
    transition.add_argument("phase", choices=PHASES[1:])

    verdict = subparsers.add_parser("verdict")
    verdict.add_argument("task_dir", type=Path)
    verdict.add_argument("verdict", choices=("PASS", "RETURN"))

    check = subparsers.add_parser("check")
    check.add_argument("task_dir", type=Path)

    return parser.parse_args()


def load_state(task_dir: Path) -> dict[str, Any]:
    with (task_dir / "state.toml").open("rb") as handle:
        state = tomllib.load(handle)
    return state


def write_state(task_dir: Path, state: dict[str, Any]) -> None:
    ordered_keys = (
        "schema_version",
        "issue_number",
        "issue_url",
        "issue_sha256",
        "attempt",
        "phase",
        "status",
        "verdict",
        "updated_at",
    )
    state["updated_at"] = datetime.now(UTC).isoformat()
    lines: list[str] = []
    for key in ordered_keys:
        value = state[key]
        if isinstance(value, str):
            rendered = json.dumps(value)
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        lines.append(f"{key} = {rendered}")
    (task_dir / "state.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def transition(task_dir: Path, requested: str) -> None:
    state = load_state(task_dir)
    if state.get("status") != "in_progress":
        raise ValueError("cannot transition a task that is not in_progress")
    current = state.get("phase")
    expected = NEXT_PHASE.get(str(current))
    if requested != expected:
        raise ValueError(f"invalid transition: {current!r} -> {requested!r}; expected {expected!r}")
    state["phase"] = requested
    state["verdict"] = ""
    write_state(task_dir, state)


def apply_verdict(task_dir: Path, verdict: str) -> None:
    state = load_state(task_dir)
    if state.get("phase") != "validation" or state.get("status") != "in_progress":
        raise ValueError("a verdict is valid only during in-progress validation")
    if verdict == "PASS":
        state["status"] = "complete"
        state["verdict"] = "PASS"
    else:
        state["attempt"] = int(state.get("attempt", 0)) + 1
        state["phase"] = "exploration"
        state["status"] = "in_progress"
        state["verdict"] = "RETURN"
    write_state(task_dir, state)


def check(task_dir: Path) -> list[str]:
    errors: list[str] = []
    for relative in REQUIRED_FILES:
        path = task_dir / relative
        if not path.is_file():
            errors.append(f"missing required file: {relative}")
        elif not path.read_text(encoding="utf-8").strip():
            errors.append(f"empty required file: {relative}")

    for path in task_dir.rglob("*.md"):
        if VERSIONED_ARTIFACT_RE.search(path.name):
            errors.append(f"versioned workflow artifact is forbidden: {path.relative_to(task_dir)}")

    if errors:
        return errors

    state = load_state(task_dir)
    if state.get("phase") not in PHASES:
        errors.append(f"invalid phase: {state.get('phase')!r}")
    if state.get("status") not in ("in_progress", "complete"):
        errors.append(f"invalid status: {state.get('status')!r}")

    for relative, headings in REQUIRED_HEADINGS.items():
        text = (task_dir / relative).read_text(encoding="utf-8")
        for heading in headings:
            if heading not in text:
                errors.append(f"{relative} is missing heading: {heading}")

    if state.get("status") == "complete":
        for relative in REQUIRED_HEADINGS:
            text = (task_dir / relative).read_text(encoding="utf-8")
            if "PENDING" in text or "Replace this" in text:
                errors.append(f"complete task contains placeholders: {relative}")
        validation = (task_dir / "04-validation/validation.md").read_text(encoding="utf-8")
        verdict_section = validation.split("## Final verdict", maxsplit=1)
        if len(verdict_section) != 2 or not re.search(
            r"(?m)^PASS\s*$", verdict_section[1]
        ):
            errors.append("complete task requires a standalone PASS under Final verdict")
        if state.get("verdict") != "PASS":
            errors.append("complete task state verdict must be PASS")

    return errors


def main() -> int:
    args = parse_args()
    try:
        if args.command == "transition":
            transition(args.task_dir, args.phase)
        elif args.command == "verdict":
            apply_verdict(args.task_dir, args.verdict)
        else:
            errors = check(args.task_dir)
            if errors:
                for error in errors:
                    print(f"error: {error}", file=sys.stderr)
                return 1
            print("ok")
        return 0
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
