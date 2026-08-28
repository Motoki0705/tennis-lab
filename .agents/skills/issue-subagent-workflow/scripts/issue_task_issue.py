"""Frozen GitHub Issue snapshot helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, NamedTuple

SOURCE_ACCEPTANCE_SECTION_RE = re.compile(
    r"(?ms)^## Acceptance checklist[ \t]*\r?\n(.*?)(?=^##[ \t]+|\Z)"
)
TASK_LIST_RE = re.compile(
    r"^(?P<indent>[ \t]*)[-*+][ \t]+\[(?P<mark>[ xX])\]"
    r"(?:[ \t]+(?P<text>.*?))?[ \t]*$"
)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


class AcceptanceItem(NamedTuple):
    item_id: str
    text: str
    source_checked: bool


class _SourceAcceptanceItem(NamedTuple):
    indent: str
    mark: str
    summary: str
    details: list[str]


def _mask_html_comment(match: re.Match[str]) -> str:
    """Remove template guidance without changing its physical line structure."""
    return re.sub(r"[^\r\n]", " ", match.group(0))


def _normalize_acceptance_text(value: str) -> str:
    return " ".join(value.split())


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
    section = SOURCE_ACCEPTANCE_SECTION_RE.search(
        HTML_COMMENT_RE.sub(_mask_html_comment, body)
    )
    if section is None:
        raise ValueError(
            "issue body must contain a `## Acceptance checklist` section with at least "
            "one Markdown task-list item"
        )

    raw_items: list[tuple[str, str, list[str]]] = []
    current: _SourceAcceptanceItem | None = None
    continuation_open = False
    for raw_line in section.group(1).splitlines():
        task_match = TASK_LIST_RE.fullmatch(raw_line)
        if task_match is not None:
            if current is not None:
                raw_items.append((current.mark, current.summary, current.details))
            summary = _normalize_acceptance_text(task_match.group("text") or "")
            current = None
            continuation_open = False
            if summary:
                current = _SourceAcceptanceItem(
                    task_match.group("indent"),
                    task_match.group("mark"),
                    summary,
                    [],
                )
                continuation_open = True
            continue

        if not raw_line.strip():
            if current is not None:
                continuation_open = False
            continue

        if current is None:
            if raw_items:
                raise ValueError(
                    "acceptance checklist prose after an item must be another task-list "
                    "item; detailed context belongs on consecutive indented lines"
                )
            # Preserve support for introductory prose before the first checklist item.
            continue

        if not continuation_open:
            raise ValueError(
                "acceptance checklist detail must immediately follow its task-list item "
                "without a blank line"
            )
        detail_prefix = current.indent + "  "
        if not raw_line.startswith(detail_prefix):
            raise ValueError(
                "acceptance checklist detail must be indented by at least two ASCII "
                "spaces more than its task-list item"
            )
        current.details.append(
            _normalize_acceptance_text(raw_line[len(detail_prefix) :])
        )

    if current is not None:
        raw_items.append((current.mark, current.summary, current.details))
    if not raw_items:
        raise ValueError(
            "the `## Acceptance checklist` section must contain at least one item"
        )

    items: list[AcceptanceItem] = []
    seen_texts: set[str] = set()
    for index, (mark, summary, details) in enumerate(raw_items, start=1):
        text = " ".join((summary, *details))
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
