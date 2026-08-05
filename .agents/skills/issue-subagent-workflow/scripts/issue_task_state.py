"""State schema and artifact validators for issue-subagent-workflow."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CURRENT_SCHEMA_VERSION = 4
LEGACY_SCHEMA_VERSION = 3
PHASES = ("feasibility", "exploration", "planning", "implementation", "validation")
NEXT_PHASE = {
    "exploration": "planning",
    "planning": "implementation",
    "implementation": "validation",
}
CORE_REQUIRED_FILES = (
    "issue.md",
    "state.toml",
    "01-exploration/exploration.md",
    "02-planning/plan.md",
    "03-implementation/implementation.md",
    "03-implementation/tests.md",
    "04-validation/validation.md",
)
EFFICIENCY_REQUIRED_FILES = (
    "00-feasibility/feasibility.md",
    "03-implementation/preflight.md",
)
REQUIRED_HEADINGS = {
    "00-feasibility/feasibility.md": (
        "## Allowed and prohibited changes",
        "## Required checks and baseline",
        "## Breaking-change and compatibility impact",
        "## Acceptance checklist feasibility",
        "## Constraint conflicts",
        "## Final feasibility verdict",
        "## Blocker resolution required",
    ),
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
    "03-implementation/preflight.md": (
        "## Changed scope",
        "## Deterministic policy checks",
        "## Focused checks",
        "## Canonical required checks",
        "## Baseline comparison",
        "## Commands and exact outcomes",
        "## Final preflight verdict",
        "## RETURN implementation findings",
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
    r"(?:feasibility|exploration|plan|implementation|preflight|tests|validation)"
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
BLOCK_KINDS = (
    "constraint_conflict",
    "external_dependency",
    "missing_authority",
    "environment",
)


def normalize_state(state: dict[str, Any]) -> dict[str, Any]:
    version = state.get("schema_version")
    if version == LEGACY_SCHEMA_VERSION:
        phase = state.get("phase")
        status = state.get("status")
        test_cycle = state.get("test_cycle", 0)
        test_verdict = state.get("test_verdict", "")
        state["schema_version"] = CURRENT_SCHEMA_VERSION
        state.setdefault("feasibility_verdict", "LEGACY")
        state.setdefault(
            "preflight_cycle",
            test_cycle if phase == "validation" or status == "complete" else 0,
        )
        state.setdefault(
            "preflight_verdict",
            "PASS" if phase == "validation" or status == "complete" else "",
        )
        state.setdefault("test_return_count", 1 if test_verdict == "RETURN" else 0)
        state.setdefault("return_review_required", False)
        state.setdefault("return_review_action", "")
        state.setdefault("return_review_reason", "")
        state.setdefault("block_kind", "")
        state.setdefault("block_reason", "")
    return state


def load_state(task_dir: Path) -> dict[str, Any]:
    with (task_dir / "state.toml").open("rb") as handle:
        return normalize_state(tomllib.load(handle))


def write_state(task_dir: Path, state: dict[str, Any]) -> None:
    state = normalize_state(state)
    ordered_keys = (
        "schema_version",
        "issue_number",
        "issue_url",
        "issue_sha256",
        "acceptance_checklist_sha256",
        "acceptance_checklist_count",
        "attempt",
        "feasibility_verdict",
        "preflight_cycle",
        "preflight_verdict",
        "test_cycle",
        "test_verdict",
        "test_return_count",
        "return_review_required",
        "return_review_action",
        "return_review_reason",
        "phase",
        "status",
        "verdict",
        "block_kind",
        "block_reason",
        "updated_at",
    )
    state["schema_version"] = CURRENT_SCHEMA_VERSION
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


def standalone_value(path: Path, heading: str, allowed: set[str]) -> str:
    section = extract_section(path.read_text(encoding="utf-8"), heading)
    lines = [line.strip() for line in section.splitlines() if line.strip()]
    if len(lines) != 1 or lines[0] not in allowed:
        expected = " or ".join(sorted(allowed))
        raise ValueError(
            f"{path.name} must contain exactly one standalone {expected} under {heading}"
        )
    return lines[0]


def assert_standalone_value(
    path: Path,
    heading: str,
    expected: str,
    allowed: set[str],
) -> None:
    actual = standalone_value(path, heading, allowed)
    if actual != expected:
        raise ValueError(
            f"{path.name} records {actual} under {heading}; expected {expected}"
        )


def assert_nonempty_section(path: Path, heading: str, label: str) -> None:
    value = extract_section(path.read_text(encoding="utf-8"), heading)
    if not value or value in {"None", "N/A", "なし"}:
        raise ValueError(f"{path.name} must include concrete {label}")


def feasibility_matrix_errors(
    task_dir: Path,
    *,
    require_all_feasible: bool,
) -> list[str]:
    path = task_dir / "00-feasibility/feasibility.md"
    expected_items = acceptance_items(task_dir)
    errors = mapping_table_errors(
        path,
        "## Acceptance checklist feasibility",
        expected_items,
    )
    if errors:
        return errors

    section = extract_section(
        path.read_text(encoding="utf-8"),
        "## Acceptance checklist feasibility",
    )
    row_re = re.compile(
        r"(?m)^\|\s*(AC-\d{3})\s*\|\s*((?:\\\||[^|])*)\|\s*"
        r"(FEASIBLE|BLOCKED|UNKNOWN)\s*\|"
    )
    rows = row_re.findall(section)
    expected_ids = [item_id for item_id, _ in expected_items]
    row_ids = [item_id for item_id, _, _ in rows]
    if row_ids != expected_ids:
        errors.append(
            "feasibility checklist must contain one ordered FEASIBLE/BLOCKED/UNKNOWN "
            "verdict for every AC ID"
        )
        return errors

    verdict_by_id = {item_id: verdict for item_id, _, verdict in rows}
    not_feasible = [
        item_id for item_id in expected_ids if verdict_by_id[item_id] != "FEASIBLE"
    ]
    if require_all_feasible and not_feasible:
        errors.append(
            "every issue checklist item must be FEASIBLE; not feasible: "
            + ", ".join(not_feasible)
        )
    if not require_all_feasible and not not_feasible:
        errors.append(
            "feasibility BLOCKED requires at least one BLOCKED or UNKNOWN checklist item"
        )
    return errors


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
    state = normalize_state(state)
    errors: list[str] = []
    try:
        items = acceptance_items(task_dir)
    except ValueError as exc:
        return [str(exc)]

    if state.get("schema_version") != CURRENT_SCHEMA_VERSION:
        errors.append(f"state.toml schema_version must be {CURRENT_SCHEMA_VERSION}")
    if state.get("acceptance_checklist_count") != len(items):
        errors.append("state.toml acceptance_checklist_count does not match issue.md")
    if state.get("acceptance_checklist_sha256") != acceptance_digest(items):
        errors.append("state.toml acceptance_checklist_sha256 does not match issue.md")

    feasibility_verdict = state.get("feasibility_verdict")
    if feasibility_verdict not in {"", "PASS", "BLOCKED", "LEGACY"}:
        errors.append("state.toml feasibility_verdict is invalid")

    for field in ("preflight_cycle", "test_cycle", "test_return_count"):
        value = state.get(field)
        if not isinstance(value, int) or value < 0:
            errors.append(f"state.toml {field} must be a non-negative integer")

    preflight_verdict = state.get("preflight_verdict")
    if preflight_verdict not in {"", "PASS", "RETURN"}:
        errors.append("state.toml preflight_verdict must be empty, PASS, or RETURN")
    test_verdict = state.get("test_verdict")
    if test_verdict not in {"", "PASS", "RETURN"}:
        errors.append("state.toml test_verdict must be empty, PASS, or RETURN")

    review_required = state.get("return_review_required")
    if not isinstance(review_required, bool):
        errors.append("state.toml return_review_required must be a boolean")
    review_action = state.get("return_review_action")
    if review_action not in {"", "implementation", "exploration"}:
        errors.append("state.toml return_review_action is invalid")
    review_reason = state.get("return_review_reason")
    if not isinstance(review_reason, str):
        errors.append("state.toml return_review_reason must be a string")
    if review_action and not review_reason:
        errors.append("return_review_action requires return_review_reason")

    phase = state.get("phase")
    status = state.get("status")
    if phase not in PHASES:
        errors.append(f"invalid phase: {phase!r}")
    if status not in {"in_progress", "blocked", "complete"}:
        errors.append(f"invalid status: {status!r}")

    block_kind = state.get("block_kind")
    block_reason = state.get("block_reason")
    if not isinstance(block_kind, str) or not isinstance(block_reason, str):
        errors.append("block_kind and block_reason must be strings")
    if status == "blocked":
        if block_kind not in BLOCK_KINDS or not block_reason:
            errors.append("blocked task requires a valid block_kind and block_reason")
        if state.get("verdict") != "BLOCKED":
            errors.append("blocked task state verdict must be BLOCKED")
    elif block_kind or block_reason:
        errors.append("non-blocked task must not retain block_kind or block_reason")

    if phase == "feasibility":
        if status == "in_progress" and feasibility_verdict != "":
            errors.append("in-progress feasibility requires an empty feasibility_verdict")
        if status == "blocked" and feasibility_verdict not in {"", "BLOCKED"}:
            errors.append("blocked feasibility has an invalid feasibility_verdict")
    elif feasibility_verdict not in {"PASS", "LEGACY"} and status != "blocked":
        errors.append("phases after feasibility require feasibility_verdict PASS")

    test_cycle = state.get("test_cycle")
    preflight_cycle = state.get("preflight_cycle")
    if (
        isinstance(test_cycle, int)
        and isinstance(preflight_cycle, int)
        and preflight_cycle > test_cycle + 1
    ):
        errors.append("preflight_cycle cannot be more than one ahead of test_cycle")
    if isinstance(preflight_cycle, int) and preflight_verdict and preflight_cycle == 0:
        errors.append("a preflight verdict requires preflight_cycle >= 1")

    if phase in {"feasibility", "exploration", "planning"}:
        if test_cycle != 0 or test_verdict != "":
            errors.append("test_cycle and test_verdict must reset before implementation")
        if preflight_cycle != 0 or preflight_verdict != "":
            errors.append("preflight state must reset before implementation")

    if review_required:
        if phase != "implementation" or test_verdict != "RETURN":
            errors.append(
                "return_review_required is valid only after tester RETURN in implementation"
            )
        if state.get("test_return_count", 0) < 2:
            errors.append("return_review_required requires at least two tester RETURNs")

    if phase == "validation" or status == "complete":
        if not isinstance(test_cycle, int) or test_cycle < 1:
            errors.append("validation requires at least one completed test cycle")
        if test_verdict != "PASS":
            errors.append("validation requires test_verdict = PASS")
        if feasibility_verdict != "LEGACY":
            if preflight_verdict != "PASS" or preflight_cycle != test_cycle:
                errors.append(
                    "validation requires a matching preflight PASS for the test cycle"
                )
    return errors
