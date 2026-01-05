"""Review code changes with multiple LLM providers.

This script reviews code changes (git diff) using multiple LLM providers
to identify issues, suggest improvements, and discover new tasks.

Example commands:
    `uv run python -m src.agents.scripts.review`
    `uv run python -m src.agents.scripts.review claude.enable=false`
    `uv run python -m src.agents.scripts.review review.scope=staged`

Config entry point: `src/agents/configs/review.yaml`
"""

import logging
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.agents.ensemble import Ensemble
from src.agents.providers import Provider, ProviderRequest

log = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """Result of code review."""

    provider: str
    success: bool
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    new_tasks: list[str] = field(default_factory=list)
    summary: str = ""
    raw_output: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


def get_repo_root() -> Path:
    """Get the repository root directory."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass
    return Path.cwd()


def get_git_diff(scope: str = "all") -> str:
    """Get git diff based on scope.

    Args:
        scope: One of "all", "staged", "unstaged", "head"
            - all: All changes (staged + unstaged) vs HEAD
            - staged: Only staged changes
            - unstaged: Only unstaged changes
            - head: Changes between HEAD~1 and HEAD (last commit)

    Returns:
        Git diff output as string.
    """
    try:
        if scope == "staged":
            cmd = ["git", "diff", "--cached"]
        elif scope == "unstaged":
            cmd = ["git", "diff"]
        elif scope == "head":
            cmd = ["git", "diff", "HEAD~1", "HEAD"]
        else:  # all
            cmd = ["git", "diff", "HEAD"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=get_repo_root(),
        )
        return result.stdout
    except Exception as e:
        log.error(f"Failed to get git diff: {e}")
        return ""


def get_changed_files(scope: str = "all") -> list[str]:
    """Get list of changed files based on scope."""
    try:
        if scope == "staged":
            cmd = ["git", "diff", "--cached", "--name-only"]
        elif scope == "unstaged":
            cmd = ["git", "diff", "--name-only"]
        elif scope == "head":
            cmd = ["git", "diff", "--name-only", "HEAD~1", "HEAD"]
        else:  # all
            cmd = ["git", "diff", "--name-only", "HEAD"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=get_repo_root(),
        )
        if result.returncode == 0:
            return [f for f in result.stdout.strip().split("\n") if f]
        return []
    except Exception as e:
        log.error(f"Failed to get changed files: {e}")
        return []


def build_review_prompt(
    diff: str,
    changed_files: list[str],
    focus: str | None = None,
) -> str:
    """Build the review prompt for LLM providers."""
    files_list = "\n".join(f"- {f}" for f in changed_files) if changed_files else "No files changed"

    focus_section = ""
    if focus:
        focus_section = f"""
### Review Focus
{focus}
"""

    return f"""## Code Review Request

Please review the following code changes and provide:

1. **Issues**: Any bugs, errors, or problems found in the changes
2. **Suggestions**: Improvements for code quality, performance, or maintainability
3. **New Tasks**: Follow-up tasks or TODOs discovered from reviewing the code

### Changed Files
{files_list}
{focus_section}
### Git Diff
```diff
{diff}
```

Please structure your response clearly with sections for Issues, Suggestions, and New Tasks.
If there are no issues, suggestions, or new tasks, explicitly state that.
"""


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


def review_with_providers(cfg: DictConfig) -> list[ReviewResult]:
    """Review code changes with enabled providers."""
    providers = get_enabled_providers(cfg)

    if not providers:
        log.warning("No providers enabled")
        return []

    # Get diff based on scope
    scope = cfg.review.get("scope", "all")
    diff = get_git_diff(scope)
    changed_files = get_changed_files(scope)

    if not diff:
        log.info("No changes to review")
        return []

    log.info(f"Reviewing {len(changed_files)} changed files with {len(providers)} providers")
    log.info(f"Providers: {[p.value for p in providers]}")

    # Build prompt
    focus = cfg.review.get("focus", None)
    prompt = build_review_prompt(diff, changed_files, focus)

    # Build request
    request = ProviderRequest(
        prompt=prompt,
        system_prompt=cfg.system_prompt.content,
        timeout_s=cfg.execution.timeout_s,
        cwd=str(get_repo_root()),
    )

    # Create ensemble and review
    ensemble = Ensemble(providers)

    if cfg.execution.parallel:
        consultation = ensemble.consult_parallel(
            request, max_workers=cfg.execution.max_workers
        )
    else:
        consultation = ensemble.consult_sequential(request)

    # Convert to ReviewResults
    results = []
    for provider_result in consultation.results:
        review = ReviewResult(
            provider=provider_result.provider.value,
            success=provider_result.success,
            raw_output=provider_result.output if provider_result.success else provider_result.error,
            summary=f"Review by {provider_result.provider.value}",
        )

        # Parse output for structured sections (best effort)
        if provider_result.success:
            output = provider_result.output.lower()
            # Simple heuristic: check if sections are mentioned
            if "issue" in output or "bug" in output or "error" in output:
                review.issues = ["See raw output for details"]
            if "suggest" in output or "improve" in output:
                review.suggestions = ["See raw output for details"]
            if "task" in output or "todo" in output or "follow" in output:
                review.new_tasks = ["See raw output for details"]

        results.append(review)

    return results


def format_results(results: list[ReviewResult]) -> str:
    """Format review results for output."""
    lines = ["=" * 60, "Code Review Results", "=" * 60, ""]

    for result in results:
        status = "✓" if result.success else "✗"
        lines.append(f"[{result.provider.upper()}] {status}")
        lines.append("-" * 40)

        if result.success:
            lines.append(result.raw_output.strip())
        else:
            lines.append(f"Error: {result.raw_output}")

        lines.append("")

    lines.append("=" * 60)

    # Summary
    success_count = sum(1 for r in results if r.success)
    lines.append(f"Review complete: {success_count}/{len(results)} successful")

    return "\n".join(lines)


@hydra.main(  # type: ignore[misc]
    config_path="../configs",
    config_name="review",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Main entry point for code review."""
    results = review_with_providers(cfg)

    if not results:
        log.info("No review results")

        return

    # Output results
    print(format_results(results))

    # Log summary
    success_count = sum(1 for r in results if r.success)
    log.info(f"Review complete: {success_count}/{len(results)} providers succeeded")


if __name__ == "__main__":
    main()
