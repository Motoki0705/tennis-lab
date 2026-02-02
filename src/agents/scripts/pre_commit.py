"""Pre-commit sub-agent script that delegates fixes to configurable LLM provider.

This script runs pre-commit on changed files and delegates failures to
the configured LLM provider for automatic fixing.

Example commands:
    `uv run python -m src.agents.scripts.pre_commit`
    `uv run python -m src.agents.scripts.pre_commit provider=codex`
    `uv run python -m src.agents.scripts.pre_commit provider=claude`

Config entry point: `src/agents/configs/pre_commit.yaml`
"""

import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.agents.providers import Provider, ProviderRequest, get_provider

# Patterns for paths to exclude from editable files
EXCLUDE_PATTERNS = (
    ".venv/",
    "venv/",
    ".env/",
    "__pycache__/",
    ".git/",
    "node_modules/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    "site-packages/",
    "dist/",
    "build/",
    ".tox/",
    ".nox/",
    ".eggs/",
    "*.egg-info/",
)


@dataclass
class PreCommitResult:
    """Result of pre-commit sub-agent execution."""

    status: str = "fail"  # "pass" or "fail"
    fixed: bool = False
    files_touched: list[str] = field(default_factory=list)
    remaining_errors: list[str] = field(default_factory=list)
    summary: str = ""
    needs_main: bool = False
    message_for_main: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def format_output(self) -> str:
        """Format as JSON output (single line for easy parsing)."""
        import json

        return json.dumps(self.to_dict())



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


def get_changed_files() -> list[str]:
    """Get list of changed files from git (staged + unstaged)."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "-z", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            files = [f for f in result.stdout.split("\0") if f.strip()]
            return files
        return []
    except Exception:
        return []


def is_excluded_path(path: str) -> bool:
    """Check if path should be excluded from editable files."""
    import fnmatch
    for pattern in EXCLUDE_PATTERNS:
        if pattern.endswith("/"):
            # Directory pattern
            bare_pattern = pattern.rstrip("/")
            # Check if path starts with pattern or contains /pattern/
            if path.startswith(bare_pattern + "/"):
                return True
            if f"/{bare_pattern}/" in path:
                return True
            # Also handle patterns with wildcards like *.egg-info/
            if "*" in bare_pattern:
                # Split path and check each component
                parts = path.replace("\\", "/").split("/")
                for part in parts:
                    if fnmatch.fnmatch(part, bare_pattern):
                        return True
        elif "*" in pattern:
            # Glob pattern matching anywhere in path
            parts = path.replace("\\", "/").split("/")
            for part in parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
        elif pattern in path:
            return True
    return False


def filter_editable_files(files: list[str]) -> list[str]:
    """Filter out files that should not be editable (e.g., .venv/)."""
    return [f for f in files if not is_excluded_path(f)]


def run_pre_commit(files: list[str]) -> tuple[int, str]:
    """Run pre-commit on specified files.

    Returns:
        Tuple of (exit_code, output)
    """
    if not files:
        return 0, "No files to check"

    try:
        cmd = [
            "uv",
            "run",
            "--no-sync",
            "pre-commit",
            "run",
            "--show-diff-on-failure",
            "--files",
            *files,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout + result.stderr
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return 1, "Pre-commit timed out after 300s"
    except Exception as e:
        return 1, f"Error running pre-commit: {e}"


def extract_error_files(log_content: str) -> list[str]:
    """Extract files involved in errors from pre-commit log."""
    files = set()

    # Match Python files mentioned in errors
    py_pattern = re.compile(r"([A-Za-z0-9_./-]+\.(?:py|pyi))")
    for match in py_pattern.findall(log_content):
        # Remove leading ./ if present
        file_path = match.lstrip("./")
        files.add(file_path)

    # Always include config files that might need modification
    files.add(".pre-commit-config.yaml")
    files.add("pyproject.toml")

    # Filter out excluded paths
    return filter_editable_files(sorted(files))


def build_prompt(
    error_files: list[str],
    check_cmd: str,
    log_content: str,
) -> str:
    """Build the prompt for the LLM provider."""
    error_files_list = "\n".join(f"- {f}" for f in error_files)
    return f"""## Task: Fix Pre-commit Failures

### Target Files (you may ONLY modify these):
{error_files_list}

### Command to pass:
```
{check_cmd}
```

### Pre-commit Output:
```
{log_content}
```

Analyze the errors and fix all issues so the command passes with exit code 0.
"""


def run_provider_fix(
    cfg: DictConfig,
    error_files: list[str],
    check_cmd: str,
    log_content: str,
) -> PreCommitResult:
    """Delegate fix to configured LLM provider."""
    provider_name = cfg.get("provider", "codex")

    try:
        provider = Provider(provider_name)
    except ValueError:
        return PreCommitResult(
            status="fail",
            fixed=False,
            summary=f"Unknown provider: {provider_name}",
            needs_main=True,
            message_for_main=f"Configure a valid provider: {[p.value for p in Provider]}",
        )

    # Get provider config
    provider_cfg = cfg.get(provider_name, {})
    model = provider_cfg.get("model")

    # Build request
    prompt = build_prompt(error_files, check_cmd, log_content)
    request = ProviderRequest(
        prompt=prompt,
        system_prompt=cfg.system_prompt.content,
        timeout_s=cfg.execution.timeout_s,
        cwd=str(get_repo_root()),
        model=model,
    )

    # Run provider
    runner = get_provider(provider)
    result = runner.run(request)

    if not result.success:
        return PreCommitResult(
            status="fail",
            fixed=False,
            summary=f"Provider {provider_name} failed: {result.error}",
            needs_main=True,
            message_for_main=f"Error from {provider_name}: {result.error}",
        )

    # Parse output to extract result
    return parse_provider_output(result.output, provider_name)


def parse_provider_output(output: str, provider_name: str) -> PreCommitResult:
    """Parse structured output from provider response."""
    # Look for structured output format
    lines = output.strip().split("\n")
    parsed = {}

    for line in lines:
        if ": " in line:
            key, _, value = line.partition(": ")
            key = key.strip().lower()
            parsed[key] = value.strip()

    # Map parsed values
    status = parsed.get("status", "fail")
    fixed = parsed.get("fixed", "false").lower() == "true"
    files_touched_str = parsed.get("files_touched", "")
    files_touched = [f.strip() for f in files_touched_str.split(",") if f.strip()]
    remaining_errors_str = parsed.get("remaining_errors", "")
    remaining_errors = [e.strip() for e in remaining_errors_str.split(",") if e.strip()]
    summary = parsed.get("summary", f"Processed by {provider_name}")
    needs_main = parsed.get("needs_main", "false").lower() == "true"
    message_for_main = parsed.get("message_for_main", "")

    return PreCommitResult(
        status=status,
        fixed=fixed,
        files_touched=files_touched,
        remaining_errors=remaining_errors,
        summary=summary,
        needs_main=needs_main,
        message_for_main=message_for_main,
    )


@hydra.main(  # type: ignore[misc]
    config_path="../configs",
    config_name="pre_commit",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for pre-commit sub-agent."""
    # Get changed files
    files = get_changed_files()

    if not files:
        result = PreCommitResult(
            status="pass",
            fixed=False,
            summary="No changed files to check",
        )
        print(result.format_output())
        return

    # Build check command for display
    check_cmd = (
        "uv run --no-sync pre-commit run --show-diff-on-failure --files "
        + " ".join(files)
    )

    # Run pre-commit (Pass 1)
    exit_code, output = run_pre_commit(files)

    if exit_code == 0:
        result = PreCommitResult(
            status="pass",
            fixed=False,
            summary="pre-commit passed",
        )
        print(result.format_output())
        return

    # Check if auto-fix modified files and retry (Pass 2)
    auto_fixed = False
    if "files were modified by this hook" in output:
        auto_fixed = True
        exit_code, output2 = run_pre_commit(files)

        if exit_code == 0:
            # Get files that were modified
            try:
                diff_result = subprocess.run(
                    ["git", "diff", "--name-only"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                files_touched = [
                    f for f in diff_result.stdout.strip().split("\n") if f
                ]
            except Exception:
                files_touched = []

            result = PreCommitResult(
                status="pass",
                fixed=True,
                files_touched=files_touched,
                summary="pre-commit passed after auto-fix",
            )
            print(result.format_output())
            return

        output = output2

    # Delegate to configured provider
    error_files = extract_error_files(output)
    result = run_provider_fix(cfg, error_files, check_cmd, output)
    result.fixed = auto_fixed or result.fixed

    print(result.format_output())


if __name__ == "__main__":
    main()
