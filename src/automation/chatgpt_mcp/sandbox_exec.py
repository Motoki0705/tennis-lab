"""Execute one training-queue sandbox spec through validated settings."""

from __future__ import annotations

import os
from pathlib import Path

from src.automation.chatgpt_mcp.jobs import execute_sandbox_spec
from src.automation.chatgpt_mcp.settings import GatewaySettings


def run_from_spec(spec_path: Path) -> int:
    """Load one private job spec and return the sandboxed command exit code."""

    settings = GatewaySettings.from_env(require_public_base_url=False)
    settings.ensure_state()
    ack_value = os.environ.get("TRAINING_QUEUE_EXTERNAL_TEARDOWN_ACK")
    ack_path = Path(ack_value) if ack_value else None
    return execute_sandbox_spec(settings, spec_path, teardown_ack_path=ack_path)
