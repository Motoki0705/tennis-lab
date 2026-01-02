"""GitHub Copilot CLI provider."""

import shutil
import subprocess
from pathlib import Path

from src.agents.providers.base import Provider, ProviderRequest, ProviderRunner


class CopilotRunner(ProviderRunner):
    """Runner for GitHub Copilot CLI."""

    provider = Provider.COPILOT

    def get_command(self, request: ProviderRequest) -> list[str]:
        """Build the GitHub Copilot CLI command."""
        cmd = ["gh", "copilot", "explain"]

        if request.model:
            cmd.extend(["--model", request.model])

        cmd.append(request.prompt)
        return cmd

    def check_binary(self) -> bool:
        """Check if gh CLI with copilot extension is available."""
        if not shutil.which("gh"):
            return False

        # Check if copilot extension is installed
        try:
            result = subprocess.run(
                ["gh", "extension", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return "copilot" in result.stdout.lower()
        except Exception:
            return False

    def check_auth(self) -> tuple[bool, str]:
        """Check GitHub Copilot authentication status."""
        # Check for gh auth
        hosts_path = Path.home() / ".config" / "gh" / "hosts.yml"
        if hosts_path.exists():
            return True, f"gh-auth:{hosts_path}"

        # Alternative location
        alt_hosts_path = Path.home() / ".gh" / "hosts.yml"
        if alt_hosts_path.exists():
            return True, f"gh-auth:{alt_hosts_path}"

        return False, "No GitHub authentication found (run: gh auth login)"
