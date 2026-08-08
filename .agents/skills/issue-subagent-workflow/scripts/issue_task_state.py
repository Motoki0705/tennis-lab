"""State schema and shared validators for issue-subagent-workflow."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from issue_task_issue import validate_issue_snapshot
from issue_task_schema import ARTIFACT_CONTRACTS

CURRENT_SCHEMA_VERSION = 5
LEGACY_SCHEMA_VERSIONS = (3, 4)
PHASES = (
    "feasibility",
    "exploration",
    "planning",
    "implementation",
    "validation",
    "packaging",
)
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
ENFORCED_REQUIRED_FILES = (
    "issue.json",
    "02-planning/checks.json",
    "03-implementation/seal.md",
)
VERSIONED_ARTIFACT_RE = re.compile(
    r"(?:feasibility|exploration|plan|implementation|preflight|tests|seal|validation|packaging)"
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
FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
BLOCK_KINDS = (
    "constraint_conflict",
    "external_dependency",
    "missing_authority",
    "environment",
)
PACKAGING_PENDING_AC_ID = "AC-079"
PACKAGING_PENDING_EVIDENCE_TOKENS = (
    "post-Validator packaging",
    "capture-pr",
    "captured evidence",
    "packaging.md",
    "finalize-pr",
    "final workflow check",
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _normalize_fresh_test_cycle(state: dict[str, Any]) -> None:
    """Drop a prior Tester verdict once renewed preflight invalidates its candidate."""
    preflight_cycle = state.get("preflight_cycle")
    test_cycle = state.get("test_cycle")
    if (
        state.get("candidate_binding_mode") == "ENFORCED"
        and isinstance(preflight_cycle, int)
        and isinstance(test_cycle, int)
        and preflight_cycle > test_cycle
        and not state.get("test_candidate_sha256")
        and state.get("test_verdict") in {"PASS", "RETURN"}
    ):
        state["test_verdict"] = ""


def normalize_state(state: dict[str, Any], task_dir: Path | None = None) -> dict[str, Any]:
    version = state.get("schema_version")
    if version == 3:
        phase = state.get("phase")
        status = state.get("status")
        test_cycle = state.get("test_cycle", 0)
        test_verdict = state.get("test_verdict", "")
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
    if version in LEGACY_SCHEMA_VERSIONS:
        issue_text = ""
        if task_dir is not None and (task_dir / "issue.md").is_file():
            issue_text = (task_dir / "issue.md").read_text(encoding="utf-8")
        state["schema_version"] = CURRENT_SCHEMA_VERSION
        state.setdefault("issue_snapshot_sha256", _sha256_text(issue_text))
        state.setdefault("base_revision", "")
        state.setdefault("candidate_binding_mode", "LEGACY")
        state.setdefault("preflight_candidate_sha256", "")
        state.setdefault("test_candidate_sha256", "")
        state.setdefault("seal_cycle", 0)
        state.setdefault("seal_verdict", "")
        state.setdefault("sealed_candidate_sha256", "")
        state.setdefault("validation_candidate_sha256", "")
        state.setdefault("packaging_candidate_sha256", "")
        state.setdefault("pr_number", 0)
        state.setdefault("pr_head_sha", "")
        state.setdefault("remote_checks_verdict", "")
        state.setdefault("pr_evidence_sha256", "")
    _normalize_fresh_test_cycle(state)
    return state


def load_state(task_dir: Path) -> dict[str, Any]:
    with (task_dir / "state.toml").open("rb") as handle:
        return normalize_state(tomllib.load(handle), task_dir)


def _render_state(state: dict[str, Any]) -> str:
    ordered_keys = (
        "schema_version",
        "issue_number",
        "issue_url",
        "issue_sha256",
        "issue_snapshot_sha256",
        "acceptance_checklist_sha256",
        "acceptance_checklist_count",
        "base_revision",
        "candidate_binding_mode",
        "attempt",
        "feasibility_verdict",
        "preflight_cycle",
        "preflight_verdict",
        "preflight_candidate_sha256",
        "test_cycle",
        "test_verdict",
        "test_candidate_sha256",
        "seal_cycle",
        "seal_verdict",
        "sealed_candidate_sha256",
        "test_return_count",
        "return_review_required",
        "return_review_action",
        "return_review_reason",
        "validation_candidate_sha256",
        "packaging_candidate_sha256",
        "pr_number",
        "pr_head_sha",
        "remote_checks_verdict",
        "pr_evidence_sha256",
        "phase",
        "status",
        "verdict",
        "block_kind",
        "block_reason",
        "updated_at",
    )
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
    return "\n".join(lines) + "\n"


def write_state(task_dir: Path, state: dict[str, Any]) -> None:
    state = normalize_state(dict(state), task_dir)
    state["schema_version"] = CURRENT_SCHEMA_VERSION
    state["updated_at"] = datetime.now(UTC).isoformat()
    path = task_dir / "state.toml"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(_render_state(state), encoding="utf-8")
    os.replace(temporary, path)


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


def heading_count(text: str, heading: str) -> int:
    return len(re.findall(rf"(?m)^{re.escape(heading)}\s*$", text))


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


def assert_hash_present(path: Path, label: str, expected_hash: str) -> None:
    pattern = re.compile(rf"(?m)^- {re.escape(label)}: `{re.escape(expected_hash)}`\s*$")
    if pattern.search(path.read_text(encoding="utf-8")) is None:
        raise ValueError(f"{path.name} does not record {label}")


def assert_checklist_hash_present(path: Path, checklist_hash: str) -> None:
    assert_hash_present(path, "Frozen acceptance checklist SHA-256", checklist_hash)


def assert_issue_hash_present(path: Path, issue_hash: str) -> None:
    assert_hash_present(path, "Frozen issue SHA-256", issue_hash)


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


def assert_nonempty_section(
    path: Path,
    heading: str,
    label: str,
    *,
    allow_none: bool = False,
) -> None:
    value = extract_section(path.read_text(encoding="utf-8"), heading)
    if not value:
        raise ValueError(f"{path.name} must include concrete {label}")
    if not allow_none and value.strip() in {"None", "N/A", "なし"}:
        raise ValueError(f"{path.name} must include concrete {label}")


def feasibility_matrix_errors(
    task_dir: Path,
    *,
    require_all_feasible: bool,
) -> list[str]:
    path = task_dir / ARTIFACT_CONTRACTS["feasibility"].path
    expected_items = acceptance_items(task_dir)
    errors = mapping_table_errors(
        path,
        "## Acceptance checklist feasibility",
        expected_items,
    )
    if errors:
        return errors
    section = extract_section(path.read_text(encoding="utf-8"), "## Acceptance checklist feasibility")
    row_re = re.compile(
        r"(?m)^\|\s*(AC-\d{3})\s*\|\s*((?:\\\||[^|])*)\|\s*"
        r"(FEASIBLE|BLOCKED|UNKNOWN)\s*\|\s*((?:\\\||[^|])*)\|\s*$"
    )
    rows = row_re.findall(section)
    expected_ids = [item_id for item_id, _ in expected_items]
    row_ids = [item_id for item_id, _, _, _ in rows]
    if row_ids != expected_ids:
        errors.append(
            "feasibility checklist must contain one ordered FEASIBLE/BLOCKED/UNKNOWN verdict for every AC ID"
        )
        return errors
    for item_id, _, _, evidence in rows:
        if not unescape_table_cell(evidence) or "Replace" in evidence:
            errors.append(f"feasibility evidence is empty for {item_id}")
    verdict_by_id = {item_id: verdict for item_id, _, verdict, _ in rows}
    not_feasible = [item_id for item_id in expected_ids if verdict_by_id[item_id] != "FEASIBLE"]
    if require_all_feasible and not_feasible:
        errors.append("every issue checklist item must be FEASIBLE; not feasible: " + ", ".join(not_feasible))
    if not require_all_feasible and not not_feasible:
        errors.append("feasibility BLOCKED requires at least one BLOCKED or UNKNOWN checklist item")
    return errors



def test_matrix_errors(
    task_dir: Path,
    *,
    require_all_pass: bool,
) -> list[str]:
    path = task_dir / ARTIFACT_CONTRACTS["tests"].path
    expected_items = acceptance_items(task_dir)
    errors = mapping_table_errors(
        path,
        "## Acceptance-checklist-to-test mapping",
        expected_items,
    )
    if errors:
        return errors
    section = extract_section(
        path.read_text(encoding="utf-8"),
        "## Acceptance-checklist-to-test mapping",
    )
    row_re = re.compile(
        r"(?m)^\|\s*(AC-\d{3})\s*\|\s*((?:\\\||[^|])*)\|\s*"
        r"((?:\\\||[^|])*)\|\s*(PASS|FAIL|NOT RUN)\s*\|\s*$"
    )
    rows = row_re.findall(section)
    expected_ids = [item_id for item_id, _ in expected_items]
    row_ids = [item_id for item_id, _, _, _ in rows]
    if row_ids != expected_ids:
        errors.append(
            "test checklist must contain one ordered PASS/FAIL/NOT RUN result for every AC ID"
        )
        return errors
    result_by_id = {item_id: result for item_id, _, _, result in rows}
    for item_id, _, evidence, _ in rows:
        normalized = unescape_table_cell(evidence)
        if not normalized or normalized in {"None", "N/A", "なし"} or "PENDING" in normalized:
            errors.append(f"test evidence is not substantive for {item_id}")
    not_passed = [item_id for item_id in expected_ids if result_by_id[item_id] != "PASS"]
    if require_all_pass and not_passed:
        errors.append(
            "Tester PASS requires every AC row PASS; not passed: "
            + ", ".join(not_passed)
        )
    if not require_all_pass and not not_passed:
        errors.append("Tester RETURN requires at least one FAIL or NOT RUN row")
    return errors

def validation_matrix_errors(
    task_dir: Path,
    *,
    require_all_pass: bool,
) -> list[str]:
    path = task_dir / ARTIFACT_CONTRACTS["validation"].path
    expected_items = acceptance_items(task_dir)
    errors = mapping_table_errors(path, "## Acceptance checklist verification", expected_items)
    if errors:
        return errors
    section = extract_section(path.read_text(encoding="utf-8"), "## Acceptance checklist verification")
    row_re = re.compile(
        r"(?m)^\|\s*(AC-\d{3})\s*\|\s*((?:\\\||[^|])*)\|\s*"
        r"(PASS|FAIL|NOT VERIFIED)\s*\|\s*((?:\\\||[^|])*)\|\s*$"
    )
    rows = row_re.findall(section)
    expected_ids = [item_id for item_id, _ in expected_items]
    row_ids = [item_id for item_id, _, _, _ in rows]
    if row_ids != expected_ids:
        errors.append(
            "validation checklist must contain one ordered PASS/FAIL/NOT VERIFIED "
            "verdict for every AC ID"
        )
        return errors
    verdict_by_id = {item_id: verdict for item_id, _, verdict, _ in rows}
    evidence_by_id: dict[str, str] = {}
    for item_id, _, _, evidence in rows:
        normalized = unescape_table_cell(evidence)
        evidence_by_id[item_id] = normalized
        if not normalized or normalized in {"None", "N/A", "なし"} or "Replace" in normalized:
            errors.append(f"validation evidence is not substantive for {item_id}")

    enforced = load_state(task_dir).get("candidate_binding_mode") == "ENFORCED"
    has_packaging_ac = PACKAGING_PENDING_AC_ID in expected_ids
    if require_all_pass and enforced and has_packaging_ac:
        not_passed = [
            item_id
            for item_id in expected_ids
            if item_id != PACKAGING_PENDING_AC_ID
            and verdict_by_id[item_id] != "PASS"
        ]
        if not_passed:
            errors.append(
                "Validator PASS requires every pre-packaging AC row PASS; not passed: "
                + ", ".join(not_passed)
            )
        if verdict_by_id[PACKAGING_PENDING_AC_ID] != "NOT VERIFIED":
            errors.append(
                f"Validator PASS requires {PACKAGING_PENDING_AC_ID} to be exactly "
                "NOT VERIFIED until post-Validator packaging"
            )
        else:
            missing_tokens = [
                token
                for token in PACKAGING_PENDING_EVIDENCE_TOKENS
                if token not in evidence_by_id[PACKAGING_PENDING_AC_ID]
            ]
            if missing_tokens:
                errors.append(
                    f"{PACKAGING_PENDING_AC_ID} NOT VERIFIED evidence must name mandatory "
                    "post-validation packaging: " + ", ".join(missing_tokens)
                )
    elif require_all_pass:
        not_passed = [
            item_id for item_id in expected_ids if verdict_by_id[item_id] != "PASS"
        ]
        if not_passed:
            errors.append(
                "every issue checklist item must have verdict PASS; not passed: "
                + ", ".join(not_passed)
            )

    if not require_all_pass:
        failed = [
            item_id
            for item_id in expected_ids
            if verdict_by_id[item_id] in {"FAIL", "NOT VERIFIED"}
        ]
        if not failed:
            errors.append(
                "validation RETURN requires at least one FAIL or NOT VERIFIED checklist item"
            )
    return errors


def _valid_fingerprint(value: object, *, allow_empty: bool = True) -> bool:
    return isinstance(value, str) and ((allow_empty and not value) or bool(FINGERPRINT_RE.fullmatch(value)))


def validate_state(task_dir: Path, state: dict[str, Any]) -> list[str]:
    state = normalize_state(state, task_dir)
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
    errors.extend(validate_issue_snapshot(task_dir, state))

    mode = state.get("candidate_binding_mode")
    if mode not in {"ENFORCED", "LEGACY"}:
        errors.append("state.toml candidate_binding_mode must be ENFORCED or LEGACY")
    feasibility_verdict = state.get("feasibility_verdict")
    if feasibility_verdict not in {"", "PASS", "BLOCKED", "LEGACY"}:
        errors.append("state.toml feasibility_verdict is invalid")

    for field in ("preflight_cycle", "test_cycle", "seal_cycle", "test_return_count"):
        value = state.get(field)
        if not isinstance(value, int) or value < 0:
            errors.append(f"state.toml {field} must be a non-negative integer")
    if not isinstance(state.get("pr_number"), int) or int(state.get("pr_number", 0)) < 0:
        errors.append("state.toml pr_number must be a non-negative integer")

    for field in (
        "preflight_candidate_sha256",
        "test_candidate_sha256",
        "sealed_candidate_sha256",
        "validation_candidate_sha256",
        "packaging_candidate_sha256",
        "pr_evidence_sha256",
    ):
        if not _valid_fingerprint(state.get(field)):
            errors.append(f"state.toml {field} must be empty or a sha256 fingerprint")

    preflight_verdict = state.get("preflight_verdict")
    test_verdict = state.get("test_verdict")
    seal_verdict = state.get("seal_verdict")
    if preflight_verdict not in {"", "PASS", "RETURN"}:
        errors.append("state.toml preflight_verdict must be empty, PASS, or RETURN")
    if test_verdict not in {"", "PASS", "RETURN"}:
        errors.append("state.toml test_verdict must be empty, PASS, or RETURN")
    if seal_verdict not in {"", "PASS", "RETURN"}:
        errors.append("state.toml seal_verdict must be empty, PASS, or RETURN")

    review_required = state.get("return_review_required")
    if not isinstance(review_required, bool):
        errors.append("state.toml return_review_required must be a boolean")
    if state.get("return_review_action") not in {"", "implementation", "exploration"}:
        errors.append("state.toml return_review_action is invalid")
    if not isinstance(state.get("return_review_reason"), str):
        errors.append("state.toml return_review_reason must be a string")
    if state.get("return_review_action") and not state.get("return_review_reason"):
        errors.append("return_review_action requires return_review_reason")

    phase = state.get("phase")
    status = state.get("status")
    if phase not in PHASES:
        errors.append(f"invalid phase: {phase!r}")
    if status not in {"in_progress", "blocked", "validated", "complete"}:
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
    elif feasibility_verdict not in {"PASS", "LEGACY"} and status != "blocked":
        errors.append("phases after feasibility require feasibility_verdict PASS")

    if phase in {"feasibility", "exploration", "planning"}:
        for field in (
            "preflight_cycle",
            "test_cycle",
            "seal_cycle",
            "test_return_count",
        ):
            if state.get(field) != 0:
                errors.append(f"{field} must reset before implementation")
        for field in (
            "preflight_verdict",
            "test_verdict",
            "seal_verdict",
            "preflight_candidate_sha256",
            "test_candidate_sha256",
            "sealed_candidate_sha256",
            "validation_candidate_sha256",
            "packaging_candidate_sha256",
        ):
            if state.get(field) != "":
                errors.append(f"{field} must reset before implementation")

    if review_required:
        if phase != "implementation" or test_verdict != "RETURN":
            errors.append("return_review_required is valid only after Tester RETURN")
        if int(state.get("test_return_count", 0)) < 2:
            errors.append("return_review_required requires at least two Tester RETURNs")

    if mode == "ENFORCED":
        if preflight_verdict and not state.get("preflight_candidate_sha256"):
            errors.append("a preflight verdict requires preflight_candidate_sha256")
        if test_verdict and not state.get("test_candidate_sha256"):
            errors.append("a test verdict requires test_candidate_sha256")
        if seal_verdict and not state.get("sealed_candidate_sha256"):
            errors.append("a seal verdict requires sealed_candidate_sha256")
        if phase in {"validation", "packaging"} or status in {"validated", "complete"}:
            if test_verdict != "PASS" or seal_verdict != "PASS":
                errors.append("validation requires Tester PASS and candidate seal PASS")
            if state.get("test_candidate_sha256") != state.get("sealed_candidate_sha256"):
                errors.append("Tester and sealed candidate fingerprints must match")
        if (
            phase == "packaging" or status in {"validated", "complete"}
        ) and state.get("validation_candidate_sha256") != state.get(
            "sealed_candidate_sha256"
        ):
            errors.append("Validator and sealed candidate fingerprints must match")
        if status == "validated" and (
            phase != "packaging" or state.get("verdict") != "VALIDATED"
        ):
            errors.append("validated status requires packaging phase and VALIDATED verdict")
        if status == "complete":
            if phase != "packaging" or state.get("verdict") != "PASS":
                errors.append("complete status requires packaging phase and PASS verdict")
            if state.get("packaging_candidate_sha256") != state.get("validation_candidate_sha256"):
                errors.append("packaging and Validator candidate fingerprints must match")
            if int(state.get("pr_number", 0)) < 1 or not state.get("pr_head_sha"):
                errors.append("complete task requires PR number and head SHA")
            if state.get("remote_checks_verdict") != "PASS":
                errors.append("complete task requires remote_checks_verdict PASS")
            if not state.get("pr_evidence_sha256"):
                errors.append("complete task requires PR evidence binding")
    elif phase == "validation" or status == "complete":
        if test_verdict != "PASS":
            errors.append("legacy validation requires test_verdict PASS")
        if preflight_verdict != "PASS":
            errors.append("legacy validation requires preflight_verdict PASS")
    return errors
