"""Test sub-agent script for analyzing and fixing test failures.

Example commands:
    `uv run python -m src.agents.scripts.test`
    `uv run python -m src.agents.scripts.test 'task.test_cmd=uv run pytest tests/unit/ -q'`
    `uv run python -m src.agents.scripts.test execution=long`

Config entry point: `src/agents/configs/test.yaml`
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


def run_tests(test_cmd: str, timeout_s: float = 300.0) -> tuple[bool, str]:
    """Run tests with the specified command."""
    try:
        # Split command string into list
        cmd_parts = test_cmd.split()

        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"Tests timed out after {timeout_s}s"
    except Exception as e:
        return False, f"Error running tests: {e}"


def consult_for_fixes(cfg: DictConfig, test_output: str) -> None:
    """Consult sub-agents for fix suggestions."""
    providers = get_enabled_providers(cfg)

    if not providers:
        log.warning("No providers enabled for consultation")
        return

    prompt = f"""Tests failed with the following output:

```
{test_output}
```

Please analyze the test failures and suggest specific fixes.
Focus on:
1. The root cause of each failure
2. Specific code changes to fix the issues
3. Any potential related issues that should be addressed"""

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
    config_name="test",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Main entry point for test sub-agent."""
    test_cmd = cfg.task.get("test_cmd", "uv run --no-sync pytest -q")
    log.info(f"Running tests: {test_cmd}")

    # Run tests
    passed, output = run_tests(test_cmd, timeout_s=cfg.execution.timeout_s)

    print("=" * 60)
    print(f"Test Status: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    if not passed:
        print("\nTest Output:")
        print(output)

        # Consult sub-agents for fixes
        consult_for_fixes(cfg, output)
    else:
        print("\n" + "=" * 60)
        print("Summary: All tests passed")
        print("=" * 60)


if __name__ == "__main__":
    main()
