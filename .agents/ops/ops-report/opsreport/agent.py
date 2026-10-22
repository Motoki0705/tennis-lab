"""Agent selection, prompt assembly, headless execution and strict output parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import memory, shell
from .proposals import CATEGORIES, PRIORITIES, STALE_PR_OPTIONS, Proposal, _one_line

AGENT_CHOICES = ("codex",)
PROMPT_PLACEHOLDERS = (
    "{{SLOT}}",
    "{{CATEGORIES}}",
    "{{STALE_PRS}}",
    "{{COLLECTED_JSON}}",
)
_FENCE_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


class AgentOutputError(ValueError):
    """The agent's final message does not follow the JSON contract in the prompt."""


def resolve_agent(setting: str) -> str:
    """Only the Codex path has the native scout contract required by this job."""
    if setting not in AGENT_CHOICES:
        raise ValueError(f"agent must be one of {AGENT_CHOICES}, got {setting!r}")
    return setting


def build_prompt(
    template: str,
    *,
    slot: str,
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
        template.replace("{{SLOT}}", slot)
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


def execute(run: AgentRun) -> str:
    """Run agent_exec.sh read-only, without a whole-audit wall-clock limit.

    Model and effort come from ``.agents/ops/lib/agents.conf`` (no override here).
    """
    cmd = [
        "bash", str(run.agent_exec),
        "--agent", run.agent,
        "--cwd", str(run.repo_root),
        "--prompt-file", str(run.prompt_file),
        "--out", str(run.out_file),
        "--mode", "read-only",
        "--profile", "ops_report",
        "--log-dir", str(run.log_dir),
        "--run-dir", str(run.log_dir / "agent-runtime"),
    ]  # fmt: skip
    shell.run_process_group(cmd, cwd=run.repo_root, timeout=None)
    return run.out_file.read_text(encoding="utf-8")


@dataclass(frozen=True)
class StalePrNote:
    recommendation: str  # one of STALE_PR_OPTIONS
    reason: str


@dataclass(frozen=True)
class AgentReport:
    summary: str
    proposals: list[Proposal]
    stale_pr_notes: dict[int, StalePrNote]
    memory_update: dict[str, Any] | None


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
        return json.loads(candidate, object_pairs_hook=_unique_keys)
    except json.JSONDecodeError as exc:
        raise AgentOutputError(f"agent output is not valid JSON: {exc}") from exc


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise AgentOutputError(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


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
    unknown_keys = set(data) - {
        "summary",
        "proposals",
        "stale_pr_notes",
        "memory_update",
    }
    if unknown_keys:
        raise AgentOutputError(f"unexpected top-level keys: {sorted(unknown_keys)}")
    if "memory_update" not in data:
        raise AgentOutputError("$.memory_update is required (null for no changes)")
    try:
        memory_update = memory.parse_update(data["memory_update"])
    except ValueError as exc:
        raise AgentOutputError(str(exc)) from exc
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
        if len({_one_line(o) for o in options}) != len(options):
            raise AgentOutputError(f"{where}.options must be unique after rendering")
        acceptance = raw.get("acceptance")
        if (
            not isinstance(acceptance, list)
            or not acceptance
            or not all(isinstance(a, str) and a.strip() for a in acceptance)
        ):
            raise AgentOutputError(f"{where}.acceptance must be non-empty text items")
        related_parent = raw.get("related_parent")
        if "related_parent" not in raw or (
            related_parent is not None
            and (type(related_parent) is not int or related_parent < 1)
        ):
            raise AgentOutputError(
                f"{where}.related_parent must be an issue number or null"
            )
        related_issues = raw.get("related_issues")
        if (
            not isinstance(related_issues, list)
            or any(type(n) is not int or n < 1 for n in related_issues)
            or len(related_issues) != len(set(related_issues))
        ):
            raise AgentOutputError(
                f"{where}.related_issues must be unique issue numbers"
            )
        props.append(
            Proposal(
                section=CATEGORIES[category],
                title=_req_str(raw, "title", where),
                detail=_req_str(raw, "detail", where),
                evidence=_req_str(raw, "evidence", where),
                priority=priority,
                options=tuple(options),
                purpose=_req_str(raw, "purpose", where),
                scope=_req_str(raw, "scope", where),
                expected_effect=_req_str(raw, "expected_effect", where),
                acceptance=tuple(acceptance),
                related_parent=related_parent,
                related_issues=tuple(related_issues),
            )
        )
    return AgentReport(
        summary=summary,
        proposals=props,
        stale_pr_notes=_parse_stale_pr_notes(
            data.get("stale_pr_notes"), stale_pr_numbers
        ),
        memory_update=memory_update,
    )


def _parse_stale_pr_notes(
    raw: Any, stale_pr_numbers: set[int]
) -> dict[int, StalePrNote]:
    """Exactly one note per stale PR, each with a recommendation and a reason."""
    if not isinstance(raw, dict):
        raise AgentOutputError("$.stale_pr_notes must be an object")
    bad_keys = [k for k in raw if not (isinstance(k, str) and k.isdigit())]
    if bad_keys:
        raise AgentOutputError(f"$.stale_pr_notes keys must be PR numbers: {bad_keys}")
    keys = {int(k) for k in raw}
    if len(keys) != len(raw) or any(str(int(k)) != k for k in raw):
        raise AgentOutputError(
            "$.stale_pr_notes keys must be unique canonical PR numbers"
        )
    if keys != stale_pr_numbers:
        raise AgentOutputError(
            "$.stale_pr_notes keys must be exactly the stale PRs "
            f"{sorted(stale_pr_numbers)}: missing {sorted(stale_pr_numbers - keys)}, "
            f"unexpected {sorted(keys - stale_pr_numbers)}"
        )
    notes: dict[int, StalePrNote] = {}
    for key, note in raw.items():
        where = f"$.stale_pr_notes[{key!r}]"
        if not isinstance(note, dict) or set(note) != {"recommendation", "reason"}:
            raise AgentOutputError(
                f"{where} must be an object with exactly recommendation and reason"
            )
        recommendation = _req_str(note, "recommendation", where)
        if recommendation not in STALE_PR_OPTIONS:
            raise AgentOutputError(
                f"{where}.recommendation {recommendation!r} not in {list(STALE_PR_OPTIONS)}"
            )
        notes[int(key)] = StalePrNote(
            recommendation=recommendation, reason=_req_str(note, "reason", where)
        )
    return notes
