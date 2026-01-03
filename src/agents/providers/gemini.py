"""Gemini CLI provider."""

import shutil
from pathlib import Path

from src.agents.providers.base import Provider, ProviderRequest, ProviderRunner


class GeminiRunner(ProviderRunner):
    """Runner for Gemini CLI."""

    provider = Provider.GEMINI

    def get_command(self, request: ProviderRequest) -> list[str]:
        """Build the gemini CLI command."""
        # -y (yolo mode): automatically accept all actions
        # NOTE: gemini does not support --system-prompt, use prompt prefix instead
        cmd = ["gemini", "-y"]

        if request.model:
            cmd.extend(["--model", request.model])

        # Build prompt with system prompt prefix if provided
        prompt = request.prompt
        if request.system_prompt:
            prompt = f"{request.system_prompt}\n\n{request.prompt}"

        cmd.append(prompt)
        return cmd

    def check_binary(self) -> bool:
        """Check if gemini CLI is available."""
        return shutil.which("gemini") is not None

    def check_auth(self) -> tuple[bool, str]:
        """Check Gemini authentication status."""
        # Check for settings file
        settings_path = Path.home() / ".gemini" / "settings.json"
        if settings_path.exists():
            return True, f"config:{settings_path}"

        # Check for credentials
        creds_path = Path.home() / ".gemini" / "oauth_creds.json"
        if creds_path.exists():
            return True, f"oauth:{creds_path}"

        return False, "No Gemini configuration found"
