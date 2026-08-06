"""Frozen GitHub Issue snapshot helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, NamedTuple

SOURCE_ACCEPTANCE_SECTION_RE = re.compile(
    r"(?ms)^## Acceptance checklist\s*\n(.*?)(?=^##\s+|\Z)"
)
TASK_LIST_RE = re.compile(r"(?m)^\s*[-*+]\s+\[([ xX])\]\s+(.+?)\s*$")


class AcceptanceItem(NamedTuple):
    item_id: str
    text: str
    source_checked: bool


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode())


def canonical_issue_hash(payload: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def extract_acceptance_items(body: str) -> list[AcceptanceItem]:
    section = SOURCE_ACCEPTANCE_SECTION_RE.search(body)
    if section is None:
        raise ValueError(
            "issue body must contain a `## Acceptance checklist` section with at least "
            "one Markdown task-list item"
        )
    raw_items = TASK_LIST_RE.findall(section.group(1))
    if not raw_items:
        raise ValueError(
            "the `## Acceptance checklist` section must contain at least one item"
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
    return sha256_bytes(
        canonical_json_bytes(
            [{"id": item.item_id, "text": item.text} for item in items]
        )
    )


def escape_table_cell(text: str) -> str:
    return text.replace("|", r"\|")


def render_acceptance_list(items: list[AcceptanceItem]) -> str:
    return "\n".join(
        f"- {item.item_id}: {item.text} (source checkbox: "
        f"{'checked' if item.source_checked else 'unchecked'})"
        for item in items
    )


def render_issue(
    payload: dict[str, Any],
    digest: str,
    checklist_digest: str,
    items: list[AcceptanceItem],
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


def write_issue_snapshot(
    task_dir: Path,
    payload: dict[str, Any],
) -> tuple[str, str, str, list[AcceptanceItem]]:
    body = payload.get("body") or ""
    items = extract_acceptance_items(body)
    issue_hash = canonical_issue_hash(payload)
    checklist_hash = acceptance_hash(items)
    issue_text = render_issue(payload, issue_hash, checklist_hash, items)
    (task_dir / "issue.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (task_dir / "issue.md").write_text(issue_text, encoding="utf-8")
    return issue_hash, sha256_text(issue_text), checklist_hash, items


def validate_issue_snapshot(task_dir: Path, state: dict[str, Any]) -> list[str]:
    """Verify the entire frozen Issue, not only its acceptance checklist."""
    if state.get("candidate_binding_mode") == "LEGACY":
        return []

    errors: list[str] = []
    json_path = task_dir / "issue.json"
    markdown_path = task_dir / "issue.md"
    if not json_path.is_file():
        return ["missing required file: issue.json"]
    if not markdown_path.is_file():
        return ["missing required file: issue.md"]

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"issue.json is invalid: {exc}"]
    if not isinstance(payload, dict):
        return ["issue.json must contain one JSON object"]

    try:
        items = extract_acceptance_items(str(payload.get("body") or ""))
        issue_hash = canonical_issue_hash(payload)
        checklist_hash = acceptance_hash(items)
        expected_markdown = render_issue(payload, issue_hash, checklist_hash, items)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"frozen Issue payload is invalid: {exc}"]

    if issue_hash != state.get("issue_sha256"):
        errors.append("state.toml issue_sha256 does not match issue.json")
    if sha256_text(expected_markdown) != state.get("issue_snapshot_sha256"):
        errors.append("state.toml issue_snapshot_sha256 does not match frozen Issue")
    if markdown_path.read_text(encoding="utf-8") != expected_markdown:
        errors.append("issue.md does not exactly match the frozen issue.json payload")
    if payload.get("number") != state.get("issue_number"):
        errors.append("state.toml issue_number does not match issue.json")
    if payload.get("url") != state.get("issue_url"):
        errors.append("state.toml issue_url does not match issue.json")
    if checklist_hash != state.get("acceptance_checklist_sha256"):
        errors.append("state.toml acceptance checklist hash does not match issue.json")
    if len(items) != state.get("acceptance_checklist_count"):
        errors.append("state.toml acceptance checklist count does not match issue.json")
    return errors
