"""Test sub-agent script that delegates fixes to Codex.

This script runs tests and delegates failures to the Codex sub-agent
for automatic fixing (similar to test_subagent.sh).

Example commands:
    `uv run python -m src.agents.scripts.test`
    `uv run python -m src.agents.scripts.test 'task.test_cmd=uv run pytest tests/unit/ -q'`

Config entry point: `src/agents/configs/test.yaml`
"""

import contextlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig


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

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), ensure_ascii=False)


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
    result = set()
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
                result.add(f)

    return sorted(result)


def run_codex_fix(
    error_files: list[str],
    test_cmd: str,
    test_log_path: Path,
    codex_log_path: Path,
) -> TestResult:
    """Delegate fix to Codex sub-agent."""
    # Check if network is disabled
    if os.environ.get("CODEX_SANDBOX_NETWORK_DISABLED") == "1":
        return TestResult(
            status="fail",
            fixed=False,
            summary="tests failed (network disabled; cannot run codex exec)",
            needs_main=True,
            message_for_main=f"Re-run in a network-enabled environment. Logs: {test_log_path}",
        )

    # Create output schema
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["pass", "fail"]},
            "fixed": {"type": "boolean"},
            "files_touched": {"type": "array", "items": {"type": "string"}},
            "remaining_failures": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
            "needs_main": {"type": "boolean"},
            "message_for_main": {"type": "string"},
        },
        "required": [
            "status",
            "fixed",
            "files_touched",
            "remaining_failures",
            "summary",
            "needs_main",
            "message_for_main",
        ],
        "additionalProperties": False,
    }

    # Create prompt for Codex
    error_files_csv = ",".join(error_files)
    prompt = f"""You are a specialized sub-agent for triaging and fixing failing Python tests.

Target Files (Modify these):
{error_files_csv}

Constraints & Permissions:
1. **Modification**: You may ONLY modify the "Target Files" listed above.
2. **Read Access**: You are ALLOWED to read any file in the repository if it helps fix the test.
3. **Command**: Do not run any commands other than the provided test command.

Task:
- Fix the failures so that this command passes with exit code 0:
  {test_cmd}
- You may re-run the command multiple times.
- If the fix is straightforward and can be done within the allowed file set, implement it.
- If you determine the root cause requires modifying files outside the allowed list, set needs_main=true and explain.
- Prefer permanent, root-cause fixes; avoid temporary suppression unless there is no reasonable alternative.

Return JSON that matches the provided output schema.
"""

    schema_file = None
    prompt_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as sf:
            json.dump(schema, sf)
            schema_file = sf.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as pf:
            pf.write(prompt)
            prompt_file = pf.name

        # Run codex exec
        with open(prompt_file) as stdin_file:
            result = subprocess.run(
                [
                    "codex",
                    "exec",
                    "--sandbox",
                    "danger-full-access",
                    "--output-schema",
                    schema_file,
                    "-",
                ],
                stdin=stdin_file,
                capture_output=True,
                text=True,
                timeout=600,
            )

        # Save codex log
        codex_log_path.write_text(result.stderr)

        if result.returncode != 0 or not result.stdout.strip():
            return TestResult(
                status="fail",
                fixed=False,
                summary="tests failed (codex exec failed)",
                needs_main=True,
                message_for_main=f"Inspect logs: {test_log_path} and {codex_log_path}",
            )

        # Parse Codex output
        codex_result = json.loads(result.stdout.strip())
        return TestResult(**codex_result)

    except json.JSONDecodeError as e:
        return TestResult(
            status="fail",
            fixed=False,
            summary=f"Failed to parse codex output: {e}",
            needs_main=True,
            message_for_main=f"Inspect logs: {test_log_path} and {codex_log_path}",
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            status="fail",
            fixed=False,
            summary="codex exec timed out",
            needs_main=True,
            message_for_main=f"Inspect logs: {test_log_path}",
        )
    except FileNotFoundError:
        return TestResult(
            status="fail",
            fixed=False,
            summary="codex CLI not found",
            needs_main=True,
            message_for_main="Install codex CLI or fix manually",
        )
    except Exception as e:
        return TestResult(
            status="fail",
            fixed=False,
            summary=f"Error running codex: {e}",
            needs_main=True,
            message_for_main=f"Inspect logs: {test_log_path}",
        )
    finally:
        # Cleanup temp files
        if schema_file:
            with contextlib.suppress(Exception):
                os.unlink(schema_file)
        if prompt_file:
            with contextlib.suppress(Exception):
                os.unlink(prompt_file)


@hydra.main(  # type: ignore[misc]
    config_path="../configs",
    config_name="test",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Main entry point for test sub-agent."""
    repo_root = get_repo_root()

    # Setup log directory
    log_dir = repo_root / "agents_workspace" / "sub_agents" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_log_path = log_dir / f"pytest_{run_id}.log"
    codex_log_path = log_dir / f"pytest_codex_{run_id}.log"

    # Get test command from config
    test_cmd = cfg.task.get("test_cmd", "uv run --no-sync pytest -q -n auto")

    # Run tests
    exit_code, output = run_tests(test_cmd, timeout_s=cfg.execution.timeout_s)
    test_log_path.write_text(output)

    if exit_code == 0:
        result = TestResult(
            status="pass",
            fixed=False,
            summary="tests passed",
        )
        print(result.to_json())
        return

    # Delegate to Codex
    error_files = extract_error_files(output, repo_root)
    result = run_codex_fix(error_files, test_cmd, test_log_path, codex_log_path)

    print(result.to_json())


if __name__ == "__main__":
    main()
