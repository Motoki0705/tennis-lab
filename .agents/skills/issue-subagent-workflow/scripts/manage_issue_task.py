#!/usr/bin/env python3
"""Transition and validate issue-subagent workflow state."""

from __future__ import annotations

import argparse
import hashlib
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
TRANSITION_INPUTS = {
    "planning": ("01-exploration/exploration.md",),
    "implementation": ("02-planning/plan.md",),
    "validation": (
        "03-implementation/implementation.md",
        "03-implementation/tests.md",
    ),
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
        "## Acceptance checklist mapping",
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
        "## Acceptance-checklist-to-test mapping",
        "## Tests added or changed",
        "## Commands and exact outcomes",
    ),
    "04-validation/validation.md": (
        "## Acceptance checklist verification",
        "## Code evidence",
        "## Runtime and test evidence",
        "## Final verdict",
    ),
}
VERSIONED_ARTIFACT_RE = re.compile(
    r"(?:exploration|plan|implementation|tests|validation)(?:[-_.](?:v?\d+|final|revised|attempt[-_]?\d+))\.md$",
    re.IGNORECASE,
)
ACCEPTANCE_SECTION_RE = re.compile(
    r"## Acceptance checklist\s*\n(.*?)(?=\n## Title\s*$)", re.DOTALL | re.MULTILINE
)
ACCEPTANCE_ITEM_RE = re.compile(
    r"(?m)^- (AC-\d{3}): (.+?) \(source checkbox: (?:checked|unchecked)\)$"
)
VALIDATION_ROW_RE = re.compile(
    r"(?m)^\|\s*(AC-\d{3})\s*\|\s*((?:\\\||[^|])*)\|\s*"
    r"(PASS|FAIL|NOT VERIFIED)\s*\|"
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
        "acceptance_checklist_sha256",
        "acceptance_checklist_count",
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


def acceptance_items(task_dir: Path) -> list[tuple[str, str]]:
    issue_text = (task_dir / "issue.md").read_text(encoding="utf-8")
    section = ACCEPTANCE_SECTION_RE.search(issue_text)
    if section is None:
        raise ValueError("issue.md is missing the normalized Acceptance checklist section")
    items = ACCEPTANCE_ITEM_RE.findall(section.group(1))
    if not items:
        raise ValueError("issue.md acceptance checklist is empty or malformed")
    ids = [item_id for item_id, _ in items]
    if len(ids) != len(set(ids)):
        raise ValueError("issue.md acceptance checklist contains duplicate IDs")
    expected_ids = [f"AC-{index:03d}" for index in range(1, len(items) + 1)]
    if ids != expected_ids:
        raise ValueError("issue.md acceptance checklist IDs must be contiguous and ordered")
    return items


def acceptance_digest(items: list[tuple[str, str]]) -> str:
    canonical = json.dumps(
        [{"id": item_id, "text": text} for item_id, text in items],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def assert_acceptance_ids_present(path: Path, expected_ids: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [item_id for item_id in expected_ids if item_id not in text]
    if missing:
        raise ValueError(
            f"{path.name} does not map every issue checklist item; missing: {', '.join(missing)}"
        )


def assert_checklist_hash_present(path: Path, checklist_hash: str) -> None:
    if checklist_hash not in path.read_text(encoding="utf-8"):
        raise ValueError(f"{path.name} does not record the frozen acceptance checklist hash")


def assert_artifacts_ready(
    task_dir: Path, relative_paths: tuple[str, ...], attempt: int
) -> None:
    for relative in relative_paths:
        path = task_dir / relative
        if not path.is_file():
            raise ValueError(f"missing required artifact: {relative}")
        text = path.read_text(encoding="utf-8")
        if "PENDING" in text or "Replace this" in text:
            raise ValueError(f"artifact still contains placeholders: {relative}")
        if f"- Attempt: {attempt}" not in text:
            raise ValueError(
                f"artifact does not record current attempt {attempt}: {relative}"
            )


def validation_has_pass(task_dir: Path) -> bool:
    validation = (task_dir / "04-validation/validation.md").read_text(encoding="utf-8")
    verdict_section = validation.split("## Final verdict", maxsplit=1)
    return len(verdict_section) == 2 and bool(
        re.search(r"(?m)^PASS\s*$", verdict_section[1])
    )


def unescape_table_cell(text: str) -> str:
    return text.strip().replace("\\|", "|")


def validation_checklist_errors(task_dir: Path) -> list[str]:
    expected_items = acceptance_items(task_dir)
    expected_ids = [item_id for item_id, _ in expected_items]
    expected_text = dict(expected_items)
    validation = (task_dir / "04-validation/validation.md").read_text(encoding="utf-8")
    rows = VALIDATION_ROW_RE.findall(validation)
    row_ids = [item_id for item_id, _, _ in rows]
    errors: list[str] = []
    duplicates = sorted({item_id for item_id in row_ids if row_ids.count(item_id) > 1})
    if duplicates:
        errors.append(f"validation checklist has duplicate rows: {', '.join(duplicates)}")
    unknown = sorted(set(row_ids) - set(expected_ids))
    if unknown:
        errors.append(f"validation checklist has unknown IDs: {', '.join(unknown)}")
    missing = [item_id for item_id in expected_ids if item_id not in row_ids]
    if missing:
        errors.append(f"validation checklist is missing IDs: {', '.join(missing)}")
    text_by_id = {item_id: unescape_table_cell(text) for item_id, text, _ in rows}
    mismatched_text = [
        item_id
        for item_id in expected_ids
        if item_id in text_by_id and text_by_id[item_id] != expected_text[item_id]
    ]
    if mismatched_text:
        errors.append(
            "validation checklist item text differs from issue.md: "
            + ", ".join(mismatched_text)
        )
    verdict_by_id = {item_id: verdict for item_id, _, verdict in rows}
    not_passed = [
        item_id for item_id in expected_ids if verdict_by_id.get(item_id) != "PASS"
    ]
    if not_passed:
        errors.append(
            "every issue checklist item must have verdict PASS; not passed: "
            + ", ".join(not_passed)
        )
    return errors


def validate_state_checklist(task_dir: Path, state: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        items = acceptance_items(task_dir)
    except ValueError as exc:
        return [str(exc)]
    digest = acceptance_digest(items)
    if state.get("schema_version") != 2:
        errors.append("state.toml schema_version must be 2")
    if state.get("acceptance_checklist_count") != len(items):
        errors.append("state.toml acceptance_checklist_count does not match issue.md")
    if state.get("acceptance_checklist_sha256") != digest:
        errors.append("state.toml acceptance_checklist_sha256 does not match issue.md")
    return errors


def transition(task_dir: Path, requested: str) -> None:
    state = load_state(task_dir)
    if state.get("status") != "in_progress":
        raise ValueError("cannot transition a task that is not in_progress")
    checklist_errors = validate_state_checklist(task_dir, state)
    if checklist_errors:
        raise ValueError("; ".join(checklist_errors))
    current = state.get("phase")
    expected = NEXT_PHASE.get(str(current))
    if requested != expected:
        raise ValueError(f"invalid transition: {current!r} -> {requested!r}; expected {expected!r}")
    assert_artifacts_ready(
        task_dir, TRANSITION_INPUTS[requested], int(state.get("attempt", 0))
    )
    ids = [item_id for item_id, _ in acceptance_items(task_dir)]
    checklist_hash = str(state["acceptance_checklist_sha256"])
    if requested == "implementation":
        plan_path = task_dir / "02-planning/plan.md"
        assert_acceptance_ids_present(plan_path, ids)
        assert_checklist_hash_present(plan_path, checklist_hash)
    elif requested == "validation":
        assert_acceptance_ids_present(task_dir / "03-implementation/tests.md", ids)
    state["phase"] = requested
    state["verdict"] = ""
    write_state(task_dir, state)


def apply_verdict(task_dir: Path, verdict: str) -> None:
    state = load_state(task_dir)
    if state.get("phase") != "validation" or state.get("status") != "in_progress":
        raise ValueError("a verdict is valid only during in-progress validation")
    checklist_errors = validate_state_checklist(task_dir, state)
    if checklist_errors:
        raise ValueError("; ".join(checklist_errors))
    if verdict == "PASS":
        assert_artifacts_ready(
            task_dir, tuple(REQUIRED_HEADINGS), int(state.get("attempt", 0))
        )
        validation_path = task_dir / "04-validation/validation.md"
        assert_checklist_hash_present(
            validation_path, str(state["acceptance_checklist_sha256"])
        )
        validation_errors = validation_checklist_errors(task_dir)
        if validation_errors:
            raise ValueError("; ".join(validation_errors))
        if not validation_has_pass(task_dir):
            raise ValueError("validation.md must contain a standalone PASS verdict")
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
    errors.extend(validate_state_checklist(task_dir, state))
    if state.get("phase") not in PHASES:
        errors.append(f"invalid phase: {state.get('phase')!r}")
    if state.get("status") not in ("in_progress", "complete"):
        errors.append(f"invalid status: {state.get('status')!r}")

    for relative, headings in REQUIRED_HEADINGS.items():
        text = (task_dir / relative).read_text(encoding="utf-8")
        for heading in headings:
            if heading not in text:
                errors.append(f"{relative} is missing heading: {heading}")

    try:
        ids = [item_id for item_id, _ in acceptance_items(task_dir)]
    except ValueError:
        ids = []
    if ids and state.get("phase") in ("implementation", "validation"):
        plan_path = task_dir / "02-planning/plan.md"
        try:
            assert_acceptance_ids_present(plan_path, ids)
            assert_checklist_hash_present(
                plan_path, str(state.get("acceptance_checklist_sha256", ""))
            )
        except ValueError as exc:
            errors.append(str(exc))
    if ids and state.get("phase") == "validation":
        try:
            assert_acceptance_ids_present(task_dir / "03-implementation/tests.md", ids)
        except ValueError as exc:
            errors.append(str(exc))

    if state.get("status") == "complete":
        for relative in REQUIRED_HEADINGS:
            text = (task_dir / relative).read_text(encoding="utf-8")
            if "PENDING" in text or "Replace this" in text:
                errors.append(f"complete task contains placeholders: {relative}")
        attempt = int(state.get("attempt", 0))
        for relative in REQUIRED_HEADINGS:
            text = (task_dir / relative).read_text(encoding="utf-8")
            if f"- Attempt: {attempt}" not in text:
                errors.append(
                    f"complete task artifact does not record attempt {attempt}: {relative}"
                )
        validation_path = task_dir / "04-validation/validation.md"
        try:
            assert_checklist_hash_present(
                validation_path, str(state.get("acceptance_checklist_sha256", ""))
            )
        except ValueError as exc:
            errors.append(str(exc))
        errors.extend(validation_checklist_errors(task_dir))
        if not validation_has_pass(task_dir):
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
