"""Execute one training-queue sandbox spec through validated settings."""

from __future__ import annotations

from pathlib import Path

from src.automation.chatgpt_mcp.jobs import execute_sandbox_spec
from src.automation.chatgpt_mcp.settings import GatewaySettings


def run_from_spec(spec_path: Path) -> int:
    """Load one private job spec and return the sandboxed command exit code."""

    settings = GatewaySettings.from_env(require_public_base_url=False)
    settings.ensure_state()
    return execute_sandbox_spec(settings, spec_path)
