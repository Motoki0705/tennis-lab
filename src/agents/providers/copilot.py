"""GitHub Copilot CLI provider."""

import shutil
from pathlib import Path

from src.agents.providers.base import Provider, ProviderRequest, ProviderRunner


class CopilotRunner(ProviderRunner):
    """Runner for GitHub Copilot CLI (copilot command, not gh copilot)."""

    provider = Provider.COPILOT

    def get_command(self, request: ProviderRequest) -> list[str]:
        """Build the copilot CLI command."""
        # -p: non-interactive mode (print response and exit)
        # --allow-all-tools: skip all tool permission prompts
        cmd = ["copilot", "-p", "--allow-all-tools"]

        if request.model:
            cmd.extend(["--model", request.model])

        if request.system_prompt:
            cmd.extend(["--system-prompt", request.system_prompt])

        cmd.append(request.prompt)
        return cmd

    def check_binary(self) -> bool:
        """Check if copilot CLI is available."""
        return shutil.which("copilot") is not None

    def check_auth(self) -> tuple[bool, str]:
        """Check GitHub Copilot authentication status."""
        # Check for copilot CLI config
        copilot_path = Path.home() / ".copilot"
        if copilot_path.exists():
            return True, f"copilot-config:{copilot_path}"

        # Check for gh auth (copilot may use GitHub auth)
        hosts_path = Path.home() / ".config" / "gh" / "hosts.yml"
        if hosts_path.exists():
            return True, f"gh-auth:{hosts_path}"

        return False, "No Copilot configuration found (run: copilot auth)"
