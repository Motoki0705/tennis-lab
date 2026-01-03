"""Codex (OpenAI) CLI provider."""

import os
import shutil
from pathlib import Path

from src.agents.providers.base import Provider, ProviderRequest, ProviderRunner


class CodexRunner(ProviderRunner):
    """Runner for OpenAI Codex CLI."""

    provider = Provider.CODEX

    def get_command(self, request: ProviderRequest) -> list[str]:
        """Build the codex CLI command."""
        # Use codex exec for non-interactive execution
        cmd = ["codex", "exec"]

        if request.model:
            cmd.extend(["--model", request.model])

        # Use sandbox mode for file modifications
        cmd.extend(["--sandbox", "danger-full-access"])

        # Add the prompt
        cmd.append(request.prompt)
        return cmd

    def check_binary(self) -> bool:
        """Check if codex CLI is available."""
        return shutil.which("codex") is not None

    def check_auth(self) -> tuple[bool, str]:
        """Check Codex authentication status."""
        # Check for API key in environment
        if os.environ.get("OPENAI_API_KEY"):
            return True, "env:OPENAI_API_KEY"

        # Check for netrc
        netrc_path = Path.home() / ".netrc"
        if netrc_path.exists():
            content = netrc_path.read_text()
            if "api.openai.com" in content:
                return True, f"netrc:{netrc_path}"

        # Check for codex auth
        codex_auth_path = Path.home() / ".codex" / "auth.json"
        if codex_auth_path.exists():
            return True, f"codex-auth:{codex_auth_path}"

        return False, "No OpenAI API key or Codex auth found"
