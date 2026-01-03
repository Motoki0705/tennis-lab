"""Provider registry for getting and checking providers."""

from src.agents.providers.base import Provider, ProviderRunner
from src.agents.providers.claude import ClaudeRunner
from src.agents.providers.codex import CodexRunner
from src.agents.providers.copilot import CopilotRunner
from src.agents.providers.gemini import GeminiRunner

_RUNNERS: dict[Provider, type[ProviderRunner]] = {
    Provider.CLAUDE: ClaudeRunner,
    Provider.GEMINI: GeminiRunner,
    Provider.CODEX: CodexRunner,
    Provider.COPILOT: CopilotRunner,
}


def get_provider(provider: Provider) -> ProviderRunner:
    """Get a runner instance for the specified provider."""
    runner_class = _RUNNERS.get(provider)
    if runner_class is None:
        raise ValueError(f"Unknown provider: {provider}")
    return runner_class()


def get_available_providers() -> list[Provider]:
    """Get list of providers with available binaries."""
    available = []
    for provider in Provider:
        runner = get_provider(provider)
        if runner.check_binary():
            available.append(provider)
    return available


def check_auth(provider: Provider) -> tuple[bool, str]:
    """Check authentication status for a provider."""
    runner = get_provider(provider)
    return runner.check_auth()
