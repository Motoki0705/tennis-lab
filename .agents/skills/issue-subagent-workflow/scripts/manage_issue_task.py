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
        "## Handoff",
    ),
    "03-implementation/tests.md": (
        "## Acceptance-checklist-to-test mapping",
        "## Tests added or changed",
        "## Commands and exact outcomes",
        "## Final test verdict",
        "## RETURN implementation findings",
    ),
    "04-validation/validation.md": (
        "## Acceptance checklist verification",
        "## Code evidence",
        "## Runtime and test evidence",
        "## Final verdict",
    ),
}
VERSIONED_ARTIFACT_RE = re.compile(
    r"(?:exploration|plan|implementation|tests|validation)"
    r"(?:[-_.](?:v?\d+|final|revised|attempt[-_]?\d+))\.md$",
    re.IGNORECASE,
)
ACCEPTANCE_SECTION_RE = re.compile(
    r"(?ms)^## Acceptance checklist\s*\n(.*?)(?=^## Title\s*$)"
)
ACCEPTANCE_ITEM_RE = re.compile(
    r"(?m)^- (AC-\d{3}): (.+?) \(source checkbox: (?:checked|unchecked)\)$"
)
TABLE_ROW_RE = re.compile(
    r"(?m)^\|\s*(AC-\d{3})\s*\|\s*((?:\\\||[^|])*)\|"
)
TEST_CYCLE_RE = re.compile(r"(?m)^- Test cycle: (\d+)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    transition_parser = subparsers.add_parser("transition")
    transition_parser.add_argument("task_dir", type=Path)
    transition_parser.add_argument("phase", choices=PHASES[1:])

    test_verdict_parser = subparsers.add_parser("test-verdict")
    test_verdict_parser.add_argument("task_dir", type=Path)
    test_verdict_parser.add_argument("verdict", choices=("PASS", "RETURN"))

    verdict_parser = subparsers.add_parser("verdict")
    verdict_parser.add_argument("task_dir", type=Path)
    verdict_parser.add_argument("verdict", choices=("PASS", "RETURN"))

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("task_dir", type=Path)

    return parser.parse_args()


def load_state(task_dir: Path) -> dict[str, Any]:
    with (task_dir / "state.toml").open("rb") as handle:
        return tomllib.load(handle)


def write_state(task_dir: Path, state: dict[str, Any]) -> None:
    ordered_keys = (
        "schema_version",
        "issue_number",
        "issue_url",
        "issue_sha256",
        "acceptance_checklist_sha256",
        "acceptance_checklist_count",
        "attempt",
        "test_cycle",
        "test_verdict",
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


def extract_section(text: str, heading: str) -> str:
    pattern = re.compile(rf"(?ms)^{re.escape(heading)}\s*\n(.*?)(?=^##\s+|\Z)")
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"missing required section: {heading}")
    return match.group(1).strip()


def unescape_table_cell(text: str) -> str:
    return text.strip().replace(r"\|", "|")


def mapping_table_errors(
    path: Path,
    heading: str,
    expected_items: list[tuple[str, str]],
) -> list[str]:
    section = extract_section(path.read_text(encoding="utf-8"), heading)
    rows = TABLE_ROW_RE.findall(section)
    expected_ids = [item_id for item_id, _ in expected_items]
    expected_text = dict(expected_items)
    row_ids = [item_id for item_id, _ in rows]
    errors: list[str] = []

    duplicates = sorted({item_id for item_id in row_ids if row_ids.count(item_id) > 1})
    if duplicates:
        errors.append(f"{path.name} has duplicate checklist rows: {', '.join(duplicates)}")

    unknown = sorted(set(row_ids) - set(expected_ids))
    if unknown:
        errors.append(f"{path.name} has unknown checklist IDs: {', '.join(unknown)}")

    missing = [item_id for item_id in expected_ids if item_id not in row_ids]
    if missing:
        errors.append(f"{path.name} is missing checklist IDs: {', '.join(missing)}")

    if row_ids and row_ids != expected_ids:
        errors.append(f"{path.name} checklist rows must preserve Issue order")

    text_by_id = {item_id: unescape_table_cell(item_text) for item_id, item_text in rows}
    mismatched = [
        item_id
        for item_id in expected_ids
        if item_id in text_by_id and text_by_id[item_id] != expected_text[item_id]
    ]
    if mismatched:
        errors.append(
            f"{path.name} checklist item text differs from issue.md: "
            + ", ".join(mismatched)
        )
    return errors


def assert_mapping_table(
    path: Path,
    heading: str,
    expected_items: list[tuple[str, str]],
) -> None:
    errors = mapping_table_errors(path, heading, expected_items)
    if errors:
        raise ValueError("; ".join(errors))


def assert_checklist_hash_present(path: Path, checklist_hash: str) -> None:
    if checklist_hash not in path.read_text(encoding="utf-8"):
        raise ValueError(f"{path.name} does not record the frozen acceptance checklist hash")


def assert_artifacts_ready(
    task_dir: Path,
    relative_paths: tuple[str, ...],
    attempt: int,
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


def artifact_test_cycle(path: Path) -> int:
    match = TEST_CYCLE_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"{path.name} does not record a test cycle")
    return int(match.group(1))


def assert_artifact_test_cycle(path: Path, expected_cycle: int) -> None:
    actual = artifact_test_cycle(path)
    if actual != expected_cycle:
        raise ValueError(
            f"{path.name} records test cycle {actual}; expected {expected_cycle}"
        )


def standalone_verdict(path: Path, heading: str) -> str:
    section = extract_section(path.read_text(encoding="utf-8"), heading)
    lines = [line.strip() for line in section.splitlines() if line.strip()]
    if len(lines) != 1 or lines[0] not in {"PASS", "RETURN"}:
        raise ValueError(
            f"{path.name} must contain exactly one standalone PASS or RETURN under {heading}"
        )
    return lines[0]


def assert_standalone_verdict(path: Path, heading: str, expected: str) -> None:
    actual = standalone_verdict(path, heading)
    if actual != expected:
        raise ValueError(
            f"{path.name} records {actual} under {heading}; expected {expected}"
        )


def assert_return_findings(path: Path) -> None:
    findings = extract_section(
        path.read_text(encoding="utf-8"),
        "## RETURN implementation findings",
    )
    if not findings or findings in {"None", "N/A", "なし"}:
        raise ValueError("tests.md must include concrete RETURN implementation findings")


def validation_matrix_errors(
    task_dir: Path,
    *,
    require_all_pass: bool,
) -> list[str]:
    path = task_dir / "04-validation/validation.md"
    expected_items = acceptance_items(task_dir)
    errors = mapping_table_errors(
        path,
        "## Acceptance checklist verification",
        expected_items,
    )
    if errors:
        return errors

    section = extract_section(
        path.read_text(encoding="utf-8"),
        "## Acceptance checklist verification",
    )
    verdict_row_re = re.compile(
        r"(?m)^\|\s*(AC-\d{3})\s*\|\s*((?:\\\||[^|])*)\|\s*"
        r"(PASS|FAIL|NOT VERIFIED)\s*\|"
    )
    rows = verdict_row_re.findall(section)
    expected_ids = [item_id for item_id, _ in expected_items]
    row_ids = [item_id for item_id, _, _ in rows]
    if row_ids != expected_ids:
        errors.append(
            "validation checklist must contain one ordered PASS/FAIL/NOT VERIFIED "
            "verdict for every AC ID"
        )
        return errors

    verdict_by_id = {item_id: verdict for item_id, _, verdict in rows}
    not_passed = [
        item_id for item_id in expected_ids if verdict_by_id[item_id] != "PASS"
    ]
    if require_all_pass and not_passed:
        errors.append(
            "every issue checklist item must have verdict PASS; not passed: "
            + ", ".join(not_passed)
        )
    if not require_all_pass and not not_passed:
        errors.append(
            "validation RETURN requires at least one FAIL or NOT VERIFIED checklist item"
        )
    return errors


def validate_state(task_dir: Path, state: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        items = acceptance_items(task_dir)
    except ValueError as exc:
        return [str(exc)]

    if state.get("schema_version") != 3:
        errors.append("state.toml schema_version must be 3")
    if state.get("acceptance_checklist_count") != len(items):
        errors.append("state.toml acceptance_checklist_count does not match issue.md")
    if state.get("acceptance_checklist_sha256") != acceptance_digest(items):
        errors.append("state.toml acceptance_checklist_sha256 does not match issue.md")

    test_cycle = state.get("test_cycle")
    if not isinstance(test_cycle, int) or test_cycle < 0:
        errors.append("state.toml test_cycle must be a non-negative integer")
    test_verdict = state.get("test_verdict")
    if test_verdict not in {"", "PASS", "RETURN"}:
        errors.append("state.toml test_verdict must be empty, PASS, or RETURN")

    phase = state.get("phase")
    status = state.get("status")
    if phase not in PHASES:
        errors.append(f"invalid phase: {phase!r}")
    if status not in {"in_progress", "complete"}:
        errors.append(f"invalid status: {status!r}")

    if phase in {"exploration", "planning"}:
        if test_cycle != 0 or test_verdict != "":
            errors.append("test_cycle and test_verdict must reset before implementation")
    if phase == "validation" or status == "complete":
        if not isinstance(test_cycle, int) or test_cycle < 1:
            errors.append("validation requires at least one completed test cycle")
        if test_verdict != "PASS":
            errors.append("validation requires test_verdict = PASS")
    return errors


def transition(task_dir: Path, requested: str) -> None:
    state = load_state(task_dir)
    if state.get("status") != "in_progress":
        raise ValueError("cannot transition a task that is not in_progress")
    state_errors = validate_state(task_dir, state)
    if state_errors:
        raise ValueError("; ".join(state_errors))

    current = str(state.get("phase"))
    expected = NEXT_PHASE.get(current)
    if requested != expected:
        raise ValueError(
            f"invalid transition: {current!r} -> {requested!r}; expected {expected!r}"
        )

    attempt = int(state["attempt"])
    items = acceptance_items(task_dir)
    checklist_hash = str(state["acceptance_checklist_sha256"])

    if requested == "planning":
        assert_artifacts_ready(
            task_dir,
            ("01-exploration/exploration.md",),
            attempt,
        )
    elif requested == "implementation":
        plan_path = task_dir / "02-planning/plan.md"
        assert_artifacts_ready(task_dir, ("02-planning/plan.md",), attempt)
        assert_checklist_hash_present(plan_path, checklist_hash)
        assert_mapping_table(plan_path, "## Acceptance checklist mapping", items)
        state["test_cycle"] = 0
        state["test_verdict"] = ""
    else:
        if state.get("test_verdict") != "PASS":
            raise ValueError("cannot enter validation before test-verdict PASS")
        cycle = int(state["test_cycle"])
        implementation_path = task_dir / "03-implementation/implementation.md"
        tests_path = task_dir / "03-implementation/tests.md"
        assert_artifacts_ready(
            task_dir,
            (
                "03-implementation/implementation.md",
                "03-implementation/tests.md",
            ),
            attempt,
        )
        assert_artifact_test_cycle(implementation_path, cycle)
        assert_artifact_test_cycle(tests_path, cycle)
        assert_checklist_hash_present(tests_path, checklist_hash)
        assert_mapping_table(
            tests_path,
            "## Acceptance-checklist-to-test mapping",
            items,
        )
        assert_standalone_verdict(tests_path, "## Final test verdict", "PASS")

    state["phase"] = requested
    state["verdict"] = ""
    write_state(task_dir, state)


def apply_test_verdict(task_dir: Path, verdict: str) -> None:
    state = load_state(task_dir)
    if state.get("phase") != "implementation" or state.get("status") != "in_progress":
        raise ValueError(
            "a test verdict is valid only during in-progress implementation"
        )
    state_errors = validate_state(task_dir, state)
    if state_errors:
        raise ValueError("; ".join(state_errors))

    attempt = int(state["attempt"])
    cycle = int(state["test_cycle"]) + 1
    items = acceptance_items(task_dir)
    implementation_path = task_dir / "03-implementation/implementation.md"
    tests_path = task_dir / "03-implementation/tests.md"

    assert_artifacts_ready(
        task_dir,
        (
            "03-implementation/implementation.md",
            "03-implementation/tests.md",
        ),
        attempt,
    )
    assert_artifact_test_cycle(implementation_path, cycle)
    assert_artifact_test_cycle(tests_path, cycle)
    assert_checklist_hash_present(
        tests_path,
        str(state["acceptance_checklist_sha256"]),
    )
    assert_mapping_table(
        tests_path,
        "## Acceptance-checklist-to-test mapping",
        items,
    )
    assert_standalone_verdict(tests_path, "## Final test verdict", verdict)
    if verdict == "RETURN":
        assert_return_findings(tests_path)

    state["test_cycle"] = cycle
    state["test_verdict"] = verdict
    state["verdict"] = ""
    write_state(task_dir, state)


def apply_validation_verdict(task_dir: Path, verdict: str) -> None:
    state = load_state(task_dir)
    if state.get("phase") != "validation" or state.get("status") != "in_progress":
        raise ValueError("a validator verdict is valid only during in-progress validation")
    state_errors = validate_state(task_dir, state)
    if state_errors:
        raise ValueError("; ".join(state_errors))

    attempt = int(state["attempt"])
    validation_path = task_dir / "04-validation/validation.md"
    if verdict == "PASS":
        assert_artifacts_ready(task_dir, tuple(REQUIRED_HEADINGS), attempt)
        assert_checklist_hash_present(
            validation_path,
            str(state["acceptance_checklist_sha256"]),
        )
        errors = validation_matrix_errors(task_dir, require_all_pass=True)
        if errors:
            raise ValueError("; ".join(errors))
        assert_standalone_verdict(validation_path, "## Final verdict", "PASS")
        state["status"] = "complete"
        state["verdict"] = "PASS"
    else:
        assert_artifacts_ready(
            task_dir,
            ("04-validation/validation.md",),
            attempt,
        )
        assert_checklist_hash_present(
            validation_path,
            str(state["acceptance_checklist_sha256"]),
        )
        errors = validation_matrix_errors(task_dir, require_all_pass=False)
        if errors:
            raise ValueError("; ".join(errors))
        assert_standalone_verdict(validation_path, "## Final verdict", "RETURN")
        state["attempt"] = attempt + 1
        state["test_cycle"] = 0
        state["test_verdict"] = ""
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
            errors.append(
                f"versioned workflow artifact is forbidden: {path.relative_to(task_dir)}"
            )

    if errors:
        return errors

    state = load_state(task_dir)
    errors.extend(validate_state(task_dir, state))

    for relative, headings in REQUIRED_HEADINGS.items():
        text = (task_dir / relative).read_text(encoding="utf-8")
        for heading in headings:
            if heading not in text:
                errors.append(f"{relative} is missing heading: {heading}")

    try:
        items = acceptance_items(task_dir)
    except ValueError:
        items = []

    phase = state.get("phase")
    status = state.get("status")
    checklist_hash = str(state.get("acceptance_checklist_sha256", ""))

    if items and (phase in {"implementation", "validation"} or status == "complete"):
        plan_path = task_dir / "02-planning/plan.md"
        try:
            assert_checklist_hash_present(plan_path, checklist_hash)
            assert_mapping_table(plan_path, "## Acceptance checklist mapping", items)
        except ValueError as exc:
            errors.append(str(exc))

    if items and (phase == "validation" or status == "complete"):
        implementation_path = task_dir / "03-implementation/implementation.md"
        tests_path = task_dir / "03-implementation/tests.md"
        cycle = int(state.get("test_cycle", 0))
        try:
            assert_artifact_test_cycle(implementation_path, cycle)
            assert_artifact_test_cycle(tests_path, cycle)
            assert_checklist_hash_present(tests_path, checklist_hash)
            assert_mapping_table(
                tests_path,
                "## Acceptance-checklist-to-test mapping",
                items,
            )
            assert_standalone_verdict(tests_path, "## Final test verdict", "PASS")
        except ValueError as exc:
            errors.append(str(exc))

    if status == "complete":
        attempt = int(state.get("attempt", 0))
        for relative in REQUIRED_HEADINGS:
            text = (task_dir / relative).read_text(encoding="utf-8")
            if "PENDING" in text or "Replace this" in text:
                errors.append(f"complete task contains placeholders: {relative}")
            if f"- Attempt: {attempt}" not in text:
                errors.append(
                    f"complete task artifact does not record attempt {attempt}: {relative}"
                )

        validation_path = task_dir / "04-validation/validation.md"
        try:
            assert_checklist_hash_present(validation_path, checklist_hash)
            errors.extend(
                validation_matrix_errors(task_dir, require_all_pass=True)
            )
            assert_standalone_verdict(validation_path, "## Final verdict", "PASS")
        except ValueError as exc:
            errors.append(str(exc))
        if state.get("verdict") != "PASS":
            errors.append("complete task state verdict must be PASS")

    return errors


def main() -> int:
    args = parse_args()
    try:
        if args.command == "transition":
            transition(args.task_dir, args.phase)
        elif args.command == "test-verdict":
            apply_test_verdict(args.task_dir, args.verdict)
        elif args.command == "verdict":
            apply_validation_verdict(args.task_dir, args.verdict)
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
