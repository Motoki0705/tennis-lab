"""Purpose-only launch input and the read-only audit parent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from . import memory, shell
from .retrieval import strict_json

AGENT_CHOICES = ("codex",)
METADATA_FIELDS = {
    "slot",
    "base_sha",
    "repository",
    "period_start",
    "period_end",
    "report_issue",
    "input_sha256",
    "output",
    "max_threads",
}


class AgentOutputError(ValueError):
    """The parent's final memory patch is invalid or missing."""


def resolve_agent(setting: str) -> str:
    if setting not in AGENT_CHOICES:
        raise ValueError(f"agent must be one of {AGENT_CHOICES}, got {setting!r}")
    return setting


def build_prompt(template: str, *, metadata: dict[str, Any]) -> str:
    if set(metadata) != METADATA_FIELDS:
        raise ValueError("audit launch metadata fields do not match the contract")
    if "{{RUN_METADATA}}" not in template:
        raise ValueError("prompt template is missing {{RUN_METADATA}}")
    return template.replace(
        "{{RUN_METADATA}}", json.dumps(metadata, ensure_ascii=False, indent=2)
    )


@dataclass(frozen=True)
class AgentRun:
    agent: str
    agent_exec: Path
    repo_root: Path
    prompt_file: Path
    out_file: Path
    log_dir: Path
    mcp_config: Path


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
        "--mcp-config", str(run.mcp_config),
        "--log-dir", str(run.log_dir),
        "--run-dir", str(run.log_dir / "agent-runtime"),
    ]  # fmt: skip
    shell.run_process_group(
        cmd,
        cwd=run.repo_root,
        timeout=None,
    )
    return run.out_file.read_text(encoding="utf-8")


def parse_completion(text: str) -> dict[str, Any] | None:
    try:
        value = strict_json(text)
        if not isinstance(value, dict) or set(value) != {"memory_update"}:
            raise ValueError(
                "final JSON requires exactly memory_update (null for no changes)"
            )
        return cast(dict[str, Any] | None, memory.parse_update(value["memory_update"]))
    except ValueError as exc:
        raise AgentOutputError(f"invalid final JSON: {exc}") from exc
