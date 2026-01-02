"""Pre-commit sub-agent script that delegates fixes to Codex.

This script runs pre-commit on changed files and delegates failures to
the Codex sub-agent for automatic fixing (similar to pre_commit_subagent.sh).

Example commands:
    `uv run python -m src.agents.scripts.pre_commit`
    `uv run python -m src.agents.scripts.pre_commit codex.enable=false`

Config entry point: `src/agents/configs/pre_commit.yaml`
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
class PreCommitResult:
    """Result of pre-commit sub-agent execution."""

    status: str = "fail"  # "pass" or "fail"
    fixed: bool = False
    files_touched: list[str] = field(default_factory=list)
    remaining_errors: list[str] = field(default_factory=list)
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
        ] + files
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

    return sorted(files)


def run_codex_fix(
    error_files: list[str],
    check_cmd: str,
    check_log_path: Path,
    codex_log_path: Path,
) -> PreCommitResult:
    """Delegate fix to Codex sub-agent."""
    # Check if network is disabled
    if os.environ.get("CODEX_SANDBOX_NETWORK_DISABLED") == "1":
        return PreCommitResult(
            status="fail",
            fixed=False,
            summary="pre-commit failed (network disabled; cannot run codex exec)",
            needs_main=True,
            message_for_main=f"Re-run in a network-enabled environment. Logs: {check_log_path}",
        )

    # Create output schema
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["pass", "fail"]},
            "fixed": {"type": "boolean"},
            "files_touched": {"type": "array", "items": {"type": "string"}},
            "remaining_errors": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
            "needs_main": {"type": "boolean"},
            "message_for_main": {"type": "string"},
        },
        "required": [
            "status",
            "fixed",
            "files_touched",
            "remaining_errors",
            "summary",
            "needs_main",
            "message_for_main",
        ],
        "additionalProperties": False,
    }

    # Create prompt for Codex
    error_files_csv = ",".join(error_files)
    prompt = f"""You are a specialized sub-agent for fixing pre-commit failures (ruff, mypy, etc.) in a Python repo.

Target Files (Modify these):
{error_files_csv}

Constraints & Permissions:
1. **Modification**: You may ONLY modify the "Target Files" listed above.
2. **Read Access**: You are ALLOWED to read any file in the repository if it helps resolve the error.
3. **Command**: Do not run any commands other than the provided pre-commit command.

Task:
- Analyze the logs and fix the issues so that the command passes:
  {check_cmd}
- Iterate until clean.
- Prefer permanent, root-cause fixes; avoid temporary suppression unless there is no reasonable alternative.

Return JSON matching the schema.
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
            return PreCommitResult(
                status="fail",
                fixed=False,
                summary="pre-commit failed (codex exec failed)",
                needs_main=True,
                message_for_main=f"Inspect logs: {check_log_path} and {codex_log_path}",
            )

        # Parse Codex output
        codex_result = json.loads(result.stdout.strip())
        return PreCommitResult(**codex_result)

    except json.JSONDecodeError as e:
        return PreCommitResult(
            status="fail",
            fixed=False,
            summary=f"Failed to parse codex output: {e}",
            needs_main=True,
            message_for_main=f"Inspect logs: {check_log_path} and {codex_log_path}",
        )
    except subprocess.TimeoutExpired:
        return PreCommitResult(
            status="fail",
            fixed=False,
            summary="codex exec timed out",
            needs_main=True,
            message_for_main=f"Inspect logs: {check_log_path}",
        )
    except FileNotFoundError:
        return PreCommitResult(
            status="fail",
            fixed=False,
            summary="codex CLI not found",
            needs_main=True,
            message_for_main="Install codex CLI or fix manually",
        )
    except Exception as e:
        return PreCommitResult(
            status="fail",
            fixed=False,
            summary=f"Error running codex: {e}",
            needs_main=True,
            message_for_main=f"Inspect logs: {check_log_path}",
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
    config_name="pre_commit",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Main entry point for pre-commit sub-agent."""
    repo_root = get_repo_root()

    # Setup log directory
    log_dir = repo_root / "agents_workspace" / "sub_agents" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    check_log_path = log_dir / f"pre_commit_{run_id}.log"
    codex_log_path = log_dir / f"pre_commit_codex_{run_id}.log"

    # Get changed files
    files = get_changed_files()

    if not files:
        result = PreCommitResult(
            status="pass",
            fixed=False,
            summary="No changed files to check",
        )
        print(result.to_json())
        return

    # Build check command for display
    check_cmd = (
        "uv run --no-sync pre-commit run --show-diff-on-failure --files "
        + " ".join(files)
    )

    # Run pre-commit (Pass 1)
    exit_code, output = run_pre_commit(files)
    check_log_path.write_text(output)

    if exit_code == 0:
        result = PreCommitResult(
            status="pass",
            fixed=False,
            summary="pre-commit passed",
        )
        print(result.to_json())
        return

    # Check if auto-fix modified files and retry (Pass 2)
    fixed = False
    if "files were modified by this hook" in output:
        fixed = True
        exit_code, output2 = run_pre_commit(files)
        check_log_path.write_text(output + "\n\n--- RETRY ---\n\n" + output2)

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
            print(result.to_json())
            return

        output = output2

    # Delegate to Codex
    error_files = extract_error_files(output)
    result = run_codex_fix(error_files, check_cmd, check_log_path, codex_log_path)
    result.fixed = fixed or result.fixed

    print(result.to_json())


if __name__ == "__main__":
    main()
