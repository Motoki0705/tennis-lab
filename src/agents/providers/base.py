"""Base classes for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class Provider(str, Enum):
    """Supported LLM providers."""

    CLAUDE = "claude"
    GEMINI = "gemini"
    CODEX = "codex"
    COPILOT = "copilot"


@dataclass
class ProviderRequest:
    """Request to an LLM provider."""

    prompt: str
    system_prompt: str = ""
    timeout_s: float = 120.0
    cwd: str | None = None
    model: str | None = None


@dataclass
class ProviderResult:
    """Result from an LLM provider."""

    provider: Provider
    success: bool
    output: str = ""
    error: str = ""
    command: list[str] = field(default_factory=list)
    duration_s: float = 0.0


class ProviderRunner(ABC):
    """Abstract base class for LLM provider runners."""

    provider: Provider

    @abstractmethod
    def get_command(self, request: ProviderRequest) -> list[str]:
        """Build the CLI command for this provider."""
        ...

    @abstractmethod
    def check_binary(self) -> bool:
        """Check if the CLI binary is available."""
        ...

    @abstractmethod
    def check_auth(self) -> tuple[bool, str]:
        """Check if the provider is authenticated.

        Returns:
            Tuple of (is_authenticated, auth_method_or_error)
        """
        ...

    def run(self, request: ProviderRequest) -> ProviderResult:
        """Run the provider with the given request."""
        import subprocess
        import time

        cmd = self.get_command(request)
        start = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=request.timeout_s,
                cwd=request.cwd,
            )
            duration = time.time() - start

            # Combine stdout and stderr for full output
            # Some CLIs output to stderr even on success
            output = result.stdout
            if result.stderr:
                # Append stderr if there's content
                if output and not output.endswith("\n"):
                    output += "\n"
                output += result.stderr

            if result.returncode == 0:
                return ProviderResult(
                    provider=self.provider,
                    success=True,
                    output=output,
                    command=cmd,
                    duration_s=duration,
                )
            else:
                return ProviderResult(
                    provider=self.provider,
                    success=False,
                    error=result.stderr or f"Exit code: {result.returncode}",
                    output=output,  # Include output even on failure
                    command=cmd,
                    duration_s=duration,
                )
        except subprocess.TimeoutExpired:
            return ProviderResult(
                provider=self.provider,
                success=False,
                error=f"Timeout after {request.timeout_s}s",
                command=cmd,
                duration_s=request.timeout_s,
            )
        except Exception as e:
            return ProviderResult(
                provider=self.provider,
                success=False,
                error=str(e),
                command=cmd,
                duration_s=time.time() - start,
            )

