"""Test sub-agent script that delegates fixes to configurable LLM provider.

This script runs tests and delegates failures to the configured LLM provider
for automatic fixing.

Example commands:
    `uv run python -m src.agents.scripts.test`
    `uv run python -m src.agents.scripts.test 'task.test_cmd=uv run pytest tests/unit/ -q'`
    `uv run python -m src.agents.scripts.test provider=claude`

Config entry point: `src/agents/configs/test.yaml`
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
class TestResult:
    """Result of test sub-agent execution."""

    status: str = "fail"  # "pass" or "fail"
    fixed: bool = False
    files_touched: list[str] = field(default_factory=list)
    remaining_failures: list[str] = field(default_factory=list)
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


def run_tests(test_cmd: str, timeout_s: float = 300.0) -> tuple[int, str]:
    """Run tests with the specified command.

    Returns:
        Tuple of (exit_code, output)
    """
    try:
        result = subprocess.run(
            ["bash", "-lc", test_cmd],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        output = result.stdout + result.stderr
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return 1, f"Tests timed out after {timeout_s}s"
    except Exception as e:
        return 1, f"Error running tests: {e}"


def extract_error_files(log_content: str, repo_root: Path) -> list[str]:
    """Extract files involved in test failures from log."""
    files = set()

    # Match FAILED test lines
    failed_pattern = re.compile(r"^FAILED\s+([^:\s]+)", re.MULTILINE)
    for match in failed_pattern.findall(log_content):
        files.add(match)

    # Match File "..." in tracebacks
    file_pattern = re.compile(r'File\s+"([^"]+\.py)"')
    for match in file_pattern.findall(log_content):
        # Convert absolute paths to relative
        try:
            path = Path(match)
            if path.is_absolute() and str(path).startswith(str(repo_root)):
                match = str(path.relative_to(repo_root))
        except Exception:
            pass
        files.add(match)

    # Match file:line: patterns
    line_pattern = re.compile(r"^([^\s:]+\.py):\d+:", re.MULTILINE)
    for match in line_pattern.findall(log_content):
        files.add(match)

    # Filter to only .py files and clean up
    result_files = set()
    for f in files:
        f = f.strip()
        if f.endswith(".py") or f.endswith(".pyi"):
            # Remove repo root prefix if present
            try:
                if f.startswith(str(repo_root)):
                    f = str(Path(f).relative_to(repo_root))
            except Exception:
                pass
            if f:
                result_files.add(f)

    # Filter out excluded paths
    return filter_editable_files(sorted(result_files))


def build_prompt(
    error_files: list[str],
    test_cmd: str,
    log_content: str,
) -> str:
    """Build the prompt for the LLM provider."""
    error_files_list = "\n".join(f"- {f}" for f in error_files)
    return f"""## Task: Fix Test Failures

### Target Files (you may ONLY modify these):
{error_files_list}

### Command to pass:
```
{test_cmd}
```

### Test Output:
```
{log_content}
```

Analyze the failures and fix all issues so the command passes with exit code 0.
"""


def run_provider_fix(
    cfg: DictConfig,
    error_files: list[str],
    test_cmd: str,
    log_content: str,
) -> TestResult:
    """Delegate fix to configured LLM provider."""
    provider_name = cfg.get("provider", "codex")

    try:
        provider = Provider(provider_name)
    except ValueError:
        return TestResult(
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
    prompt = build_prompt(error_files, test_cmd, log_content)
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
        return TestResult(
            status="fail",
            fixed=False,
            summary=f"Provider {provider_name} failed: {result.error}",
            needs_main=True,
            message_for_main=f"Error from {provider_name}: {result.error}",
        )

    # Parse output to extract result
    return parse_provider_output(result.output, provider_name)


def parse_provider_output(output: str, provider_name: str) -> TestResult:
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
    remaining_failures_str = parsed.get("remaining_failures", "")
    remaining_failures = [e.strip() for e in remaining_failures_str.split(",") if e.strip()]
    summary = parsed.get("summary", f"Processed by {provider_name}")
    needs_main = parsed.get("needs_main", "false").lower() == "true"
    message_for_main = parsed.get("message_for_main", "")

    return TestResult(
        status=status,
        fixed=fixed,
        files_touched=files_touched,
        remaining_failures=remaining_failures,
        summary=summary,
        needs_main=needs_main,
        message_for_main=message_for_main,
    )


@hydra.main(  # type: ignore[misc]
    config_path="../configs",
    config_name="test",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Main entry point for test sub-agent."""
    repo_root = get_repo_root()

    # Get test command from config
    test_cmd = cfg.task.get("test_cmd", "uv run --no-sync pytest -q -n auto")

    # Run tests
    exit_code, output = run_tests(test_cmd, timeout_s=cfg.execution.timeout_s)

    if exit_code == 0:
        result = TestResult(
            status="pass",
            fixed=False,
            summary="tests passed",
        )
        print(result.format_output())
        return

    # Delegate to configured provider
    error_files = extract_error_files(output, repo_root)
    result = run_provider_fix(cfg, error_files, test_cmd, output)

    print(result.format_output())


if __name__ == "__main__":
    main()
