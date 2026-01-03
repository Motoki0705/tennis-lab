"""Claude Code CLI provider."""

import shutil
from pathlib import Path

from src.agents.providers.base import Provider, ProviderRequest, ProviderRunner


class ClaudeRunner(ProviderRunner):
    """Runner for Claude Code CLI."""

    provider = Provider.CLAUDE

    def get_command(self, request: ProviderRequest) -> list[str]:
        """Build the claude CLI command."""
        # -p: print mode (non-interactive)
        # --dangerously-skip-permissions: bypass all permission checks
        cmd = ["claude", "-p", "--dangerously-skip-permissions"]

        if request.model:
            cmd.extend(["--model", request.model])

        if request.system_prompt:
            cmd.extend(["--system-prompt", request.system_prompt])

        cmd.append(request.prompt)
        return cmd

    def check_binary(self) -> bool:
        """Check if claude CLI is available."""
        return shutil.which("claude") is not None

    def check_auth(self) -> tuple[bool, str]:
        """Check Claude authentication status."""
        # Check for settings file
        settings_path = Path.home() / ".claude" / "settings.json"
        if settings_path.exists():
            return True, f"config:{settings_path}"

        # Check for credentials
        creds_path = Path.home() / ".claude" / "credentials.json"
        if creds_path.exists():
            return True, f"credentials:{creds_path}"

        return False, "No Claude configuration found"
