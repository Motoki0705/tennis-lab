"""Consult sub-agents (multiple LLM providers) using Hydra configuration.

Example commands:
    `uv run python -m src.agents.scripts.consult 'task.prompt=Summarize the README'`
    `uv run python -m src.agents.scripts.consult claude.enable=false 'task.prompt=...'`
    `uv run python -m src.agents.scripts.consult system_prompt=codebase 'task.prompt=...'`

Config entry point: `src/agents/configs/consult.yaml`
"""

import logging

import hydra
from omegaconf import DictConfig

from src.agents.ensemble import Ensemble
from src.agents.providers import Provider, ProviderRequest

log = logging.getLogger(__name__)


def get_enabled_providers(cfg: DictConfig) -> list[Provider]:
    """Get list of enabled providers from config."""
    providers = []
    provider_configs = {
        "claude": Provider.CLAUDE,
        "gemini": Provider.GEMINI,
        "codex": Provider.CODEX,
        "copilot": Provider.COPILOT,
    }

    for name, provider in provider_configs.items():
        if cfg.get(name, {}).get("enable", True):
            providers.append(provider)

    return providers


def get_provider_model(cfg: DictConfig, provider: Provider) -> str | None:
    """Get model for a specific provider from config."""
    provider_name = provider.value
    return cfg.get(provider_name, {}).get("model", None)


def consult_subagents(cfg: DictConfig) -> None:
    """Consult enabled sub-agents with the configured prompt."""
    providers = get_enabled_providers(cfg)

    if not providers:
        log.warning("No providers enabled")
        return

    log.info(f"Consulting providers: {[p.value for p in providers]}")

    # Build request
    request = ProviderRequest(
        prompt=cfg.task.prompt,
        system_prompt=cfg.system_prompt.content,
        timeout_s=cfg.execution.timeout_s,
        cwd=cfg.task.get("cwd"),
    )

    # Create ensemble and consult
    ensemble = Ensemble(providers)

    if cfg.execution.parallel:
        result = ensemble.consult_parallel(
            request, max_workers=cfg.execution.max_workers
        )
    else:
        result = ensemble.consult_sequential(request)

    # Output results
    print(result.format_for_context())

    # Summary
    success_count = len(result.successful_results)
    total_count = len(result.results)
    log.info(f"Consultation complete: {success_count}/{total_count} successful")


@hydra.main(
    config_path="../configs",
    config_name="consult",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for sub-agent consultation."""
    if not cfg.task.prompt:
        log.error("No prompt provided. Use: task.prompt='Your question here'")
        return

    consult_subagents(cfg)


if __name__ == "__main__":
    main()
