"""Agent selection, prompt assembly, headless execution and strict output parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from . import shell
from .proposals import CATEGORIES, PRIORITIES, Proposal

AGENT_CHOICES = ("claude", "codex", "alternate")
PROMPT_PLACEHOLDERS = (
    "{{WEEK}}",
    "{{CATEGORIES}}",
    "{{STALE_PRS}}",
    "{{COLLECTED_JSON}}",
)
_FENCE_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


class AgentOutputError(ValueError):
    """The agent's final message does not follow the JSON contract in the prompt."""


def resolve_agent(setting: str, day: date) -> str:
    """``claude`` / ``codex`` as-is; ``alternate`` = claude on even ISO weeks, codex on odd."""
    if setting not in AGENT_CHOICES:
        raise ValueError(f"agent must be one of {AGENT_CHOICES}, got {setting!r}")
    if setting != "alternate":
        return setting
    return "claude" if day.isocalendar().week % 2 == 0 else "codex"


def build_prompt(
    template: str,
    *,
    week: str,
    stale_prs: list[dict[str, Any]],
    collected: dict[str, Any],
) -> str:
    for placeholder in PROMPT_PLACEHOLDERS:
        if placeholder not in template:
            raise ValueError(f"prompt template is missing placeholder {placeholder}")
    categories = "\n".join(f"- `{key}`: {label}" for key, label in CATEGORIES.items())
    stale = (
        "\n".join(
            f"- #{pr['number']} {pr['title']}（{pr['idle_days']}日更新なし）"
            for pr in stale_prs
        )
        or "（なし）"
    )
    prompt = (
        template.replace("{{WEEK}}", week)
        .replace("{{CATEGORIES}}", categories)
        .replace("{{STALE_PRS}}", stale)
        .replace(
            "{{COLLECTED_JSON}}", json.dumps(collected, ensure_ascii=False, indent=1)
        )
    )
    return prompt


@dataclass(frozen=True)
class AgentRun:
    agent: str
    agent_exec: Path
    repo_root: Path
    prompt_file: Path
    out_file: Path
    log_dir: Path
    model: str | None
    timeout: float


def execute(run: AgentRun) -> str:
    """Run agent_exec.sh read-only; non-zero status / timeout raises CommandError."""
    cmd = [
        "bash", str(run.agent_exec),
        "--agent", run.agent,
        "--cwd", str(run.repo_root),
        "--prompt-file", str(run.prompt_file),
        "--out", str(run.out_file),
        "--mode", "read-only",
        "--log-dir", str(run.log_dir),
    ]  # fmt: skip
    if run.model:
        cmd += ["--model", run.model]
    shell.run(cmd, cwd=run.repo_root, timeout=run.timeout)
    return run.out_file.read_text(encoding="utf-8")


@dataclass(frozen=True)
class AgentReport:
    summary: str
    proposals: list[Proposal]
    stale_pr_notes: dict[int, str]


def _extract_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("{"):
        candidate = stripped
    else:
        fences = _FENCE_RE.findall(text)
        if len(fences) != 1:
            raise AgentOutputError(
                f"expected one JSON object (raw or a single ```json fence), found {len(fences)} fences"
            )
        candidate = fences[0]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise AgentOutputError(f"agent output is not valid JSON: {exc}") from exc


def _req_str(
    obj: dict[str, Any], key: str, where: str, *, allow_empty: bool = False
) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise AgentOutputError(f"{where}.{key} must be a non-empty string")
    return value


def parse_agent_output(text: str, stale_pr_numbers: set[int]) -> AgentReport:
    data = _extract_json(text)
    if not isinstance(data, dict):
        raise AgentOutputError("top-level JSON must be an object")
    unknown_keys = set(data) - {"summary", "proposals", "stale_pr_notes"}
    if unknown_keys:
        raise AgentOutputError(f"unexpected top-level keys: {sorted(unknown_keys)}")
    summary = _req_str(data, "summary", "$")
    raw_props = data.get("proposals")
    if not isinstance(raw_props, list):
        raise AgentOutputError("$.proposals must be a list")
    props: list[Proposal] = []
    for i, raw in enumerate(raw_props):
        where = f"$.proposals[{i}]"
        if not isinstance(raw, dict):
            raise AgentOutputError(f"{where} must be an object")
        category = _req_str(raw, "category", where)
        if category not in CATEGORIES:
            raise AgentOutputError(
                f"{where}.category {category!r} not in {list(CATEGORIES)}"
            )
        priority = _req_str(raw, "priority", where)
        if priority not in PRIORITIES:
            raise AgentOutputError(
                f"{where}.priority {priority!r} not in {list(PRIORITIES)}"
            )
        options = raw.get("options", [])
        if not isinstance(options, list) or not all(
            isinstance(o, str) and o.strip() for o in options
        ):
            raise AgentOutputError(
                f"{where}.options must be a list of non-empty strings"
            )
        if len(options) == 1:
            raise AgentOutputError(f"{where}.options needs 0 or >=2 choices, got 1")
        props.append(
            Proposal(
                section=CATEGORIES[category],
                title=_req_str(raw, "title", where),
                detail=_req_str(raw, "detail", where),
                evidence=_req_str(raw, "evidence", where),
                priority=priority,
                options=tuple(options),
            )
        )
    raw_notes = data.get("stale_pr_notes", {})
    if not isinstance(raw_notes, dict):
        raise AgentOutputError("$.stale_pr_notes must be an object")
    notes: dict[int, str] = {}
    for key, note in raw_notes.items():
        if not str(key).isdigit() or int(key) not in stale_pr_numbers:
            raise AgentOutputError(f"$.stale_pr_notes has non-stale PR key {key!r}")
        if not isinstance(note, str) or not note.strip():
            raise AgentOutputError(
                f"$.stale_pr_notes[{key!r}] must be a non-empty string"
            )
        notes[int(key)] = note
    return AgentReport(summary=summary, proposals=props, stale_pr_notes=notes)
