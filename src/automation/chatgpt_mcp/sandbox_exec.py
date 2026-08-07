"""Queue-only entry point that executes a validated Docker sandbox spec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.automation.chatgpt_mcp.jobs import execute_sandbox_spec
from src.automation.chatgpt_mcp.settings import GatewaySettings


def main() -> int:
    """Load one private job spec and return the sandboxed command exit code."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    arguments = parser.parse_args()
    settings = GatewaySettings.from_env(require_public_base_url=False)
    settings.ensure_state()
    return execute_sandbox_spec(settings, arguments.spec)


if __name__ == "__main__":
    sys.exit(main())
