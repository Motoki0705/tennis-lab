"""Pre-commit sub-agent script for analyzing and fixing pre-commit failures.

Example commands:
    `uv run python -m src.agents.scripts.pre_commit`
    `uv run python -m src.agents.scripts.pre_commit copilot.enable=false`
    `uv run python -m src.agents.scripts.pre_commit output=json`

Config entry point: `src/agents/configs/pre_commit.yaml`
"""

import logging
import subprocess

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


def get_changed_files() -> list[str]:
    """Get list of changed files from git."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return [f for f in result.stdout.strip().split("\n") if f]
        return []
    except Exception:
        return []


def run_pre_commit(files: list[str]) -> tuple[bool, str]:
    """Run pre-commit on specified files."""
    if not files:
        return True, "No files to check"

    try:
        cmd = ["uv", "run", "--no-sync", "pre-commit", "run", "--files"] + files
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Pre-commit timed out after 300s"
    except Exception as e:
        return False, f"Error running pre-commit: {e}"


def consult_for_fixes(cfg: DictConfig, pre_commit_output: str) -> None:
    """Consult sub-agents for fix suggestions."""
    providers = get_enabled_providers(cfg)

    if not providers:
        log.warning("No providers enabled for consultation")
        return

    prompt = f"""Pre-commit failed with the following output:

```
{pre_commit_output}
```

Please analyze the failures and suggest specific fixes for each issue.
Focus on actionable fixes that can be applied directly."""

    request = ProviderRequest(
        prompt=prompt,
        system_prompt=cfg.system_prompt.content,
        timeout_s=cfg.execution.timeout_s,
    )

    ensemble = Ensemble(providers)

    if cfg.execution.parallel:
        result = ensemble.consult_parallel(
            request, max_workers=cfg.execution.max_workers
        )
    else:
        result = ensemble.consult_sequential(request)

    print("\n" + "=" * 60)
    print("Sub-agent Fix Suggestions")
    print("=" * 60)
    print(result.format_for_context())


@hydra.main(
    config_path="../configs",
    config_name="pre_commit",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Main entry point for pre-commit sub-agent."""
    # Get changed files
    files = get_changed_files()
    log.info(f"Checking {len(files)} files: {files}")

    # Run pre-commit
    passed, output = run_pre_commit(files)

    print("=" * 60)
    print(f"Pre-commit Status: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    if not passed:
        print("\nPre-commit Output:")
        print(output)

        # Consult sub-agents for fixes
        consult_for_fixes(cfg, output)
    else:
        print("\n" + "=" * 60)
        print("Summary: Pre-commit passed")
        print("=" * 60)


if __name__ == "__main__":
    main()
