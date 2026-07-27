#!/usr/bin/env python3
"""Delegate a prompt to Hermes and persist its conversation session.

The script keeps Hermes' final response on stdout and diagnostics on stderr so
the caller can relay the response without adding an intermediary wrapper.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows does not provide fcntl.
    fcntl = None


STATE_VERSION = 1
SESSION_ID_PATTERN = re.compile(r"^\s*session_id:\s*(\S+)\s*$", re.MULTILINE)
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
REASONING_TOP_PATTERN = re.compile(r"^\s*┌─\s*Reasoning\s*─*┐\s*$")
REASONING_BOTTOM_PATTERN = re.compile(r"^\s*└─+┘\s*$")


class DelegationError(RuntimeError):
    """Raised when a session cannot be safely delegated or persisted."""


def _state_root() -> Path:
    """Return the state directory without placing state in the repository."""

    configured = os.environ.get("HERMES_DELEGATION_STATE_DIR")
    if configured:
        return Path(configured).expanduser()

    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home).expanduser() / "hermes-delegation"

    return Path.home() / ".local" / "state" / "hermes-delegation"


def _safe_key(value: str) -> str:
    """Make a user- or environment-provided key safe for a filename."""

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    if not safe:
        raise DelegationError("session key must contain at least one safe character")
    return safe[:120]


def default_session_key(cwd: Path) -> str:
    """Return the stable key for the current Codex task or standalone CWD."""

    thread_id = os.environ.get("CODEX_THREAD_ID", "").strip()
    if thread_id:
        return f"thread-{_safe_key(thread_id)}"

    digest = hashlib.sha256(str(cwd.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"cwd-{digest}"


def session_path(
    *,
    cwd: Path,
    session_file: str | None,
    session_key: str | None,
) -> Path:
    """Resolve the state file used to remember the Hermes session ID."""

    if session_file:
        return Path(session_file).expanduser()

    key = session_key or default_session_key(cwd)
    return _state_root() / f"{_safe_key(key)}.json"


def load_session(path: Path) -> str | None:
    """Load and validate a session ID, failing on corrupt state."""

    if not path.exists():
        return None

    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DelegationError(f"cannot read session state {path}: {exc}") from exc

    if not isinstance(data, dict) or data.get("schema_version") != STATE_VERSION:
        raise DelegationError(f"unsupported session state in {path}")

    session_id = data.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise DelegationError(f"session state {path} has no valid session_id")
    return session_id.strip()


def save_session(path: Path, session_id: str, cwd: Path) -> None:
    """Atomically save a session ID with private permissions."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = {
        "schema_version": STATE_VERSION,
        "session_id": session_id,
        "cwd": str(cwd.resolve()),
    }

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as temporary_file:
            json.dump(payload, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_name, path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.unlink(temporary_name)
        raise DelegationError(f"cannot save session state {path}: {exc}") from exc


def extract_session_id(stderr: str) -> str:
    """Extract the latest session ID emitted by ``hermes chat``."""

    matches = SESSION_ID_PATTERN.findall(stderr)
    if not matches:
        raise DelegationError("Hermes did not emit a session_id on stderr")
    return matches[-1]


def strip_display_reasoning(stdout: str) -> str:
    """Remove Hermes' optional terminal reasoning box, preserving the answer."""

    lines = stdout.splitlines(keepends=True)
    cleaned: list[str] = []
    in_reasoning = False
    for line in lines:
        plain_line = ANSI_ESCAPE_PATTERN.sub("", line).strip()
        if not in_reasoning and REASONING_TOP_PATTERN.match(plain_line):
            in_reasoning = True
            continue
        if in_reasoning and REASONING_BOTTOM_PATTERN.match(plain_line):
            in_reasoning = False
            continue
        if not in_reasoning:
            cleaned.append(line)

    if in_reasoning:
        raise DelegationError("Hermes emitted an unterminated reasoning display block")
    return "".join(cleaned)


def build_command(
    *,
    hermes: str,
    prompt: str,
    session_id: str | None,
    model: str | None,
    provider: str | None,
    toolsets: str | None,
    max_turns: int | None,
    source: str,
    ignore_user_config: bool,
) -> list[str]:
    """Build an argument-list command without invoking a shell."""

    command = [
        hermes,
        "chat",
        "--query",
        prompt,
        "--quiet",
        "--pass-session-id",
        "--source",
        source,
    ]
    if ignore_user_config:
        command.append("--ignore-user-config")
    if session_id:
        command.extend(["--resume", session_id])
    if model:
        command.extend(["--model", model])
    if provider:
        command.extend(["--provider", provider])
    if toolsets:
        command.extend(["--toolsets", toolsets])
    if max_turns is not None:
        command.extend(["--max-turns", str(max_turns)])
    return command


def _read_prompt(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.prompt is not None and args.prompt_file is not None:
        parser.error("provide either PROMPT or --prompt-file, not both")

    if args.prompt_file:
        try:
            prompt = (
                sys.stdin.read()
                if args.prompt_file == "-"
                else Path(args.prompt_file).read_text(encoding="utf-8")
            )
        except OSError as exc:
            parser.error(f"cannot read --prompt-file: {exc}")
    elif args.prompt is not None:
        prompt = args.prompt
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read()
    else:
        parser.error("provide PROMPT, --prompt-file, or prompt text on stdin")

    if not prompt.strip():
        parser.error("prompt must not be empty")
    return prompt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Delegate one prompt to Hermes while preserving its session."
    )
    parser.add_argument("prompt", nargs="?", help="Prompt to send to Hermes")
    parser.add_argument(
        "--prompt-file", help="Read the prompt from a UTF-8 file, or '-' for stdin"
    )
    parser.add_argument(
        "--session-file", help="Explicit JSON state file for the Hermes session"
    )
    parser.add_argument(
        "--session-key", help="Filename key when --session-file is omitted"
    )
    parser.add_argument(
        "--resume-required",
        action="store_true",
        help="Fail if no saved session exists instead of starting a new one",
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Start a new session and replace the saved session ID",
    )
    parser.add_argument(
        "--hermes", default="hermes", help="Hermes executable (default: hermes)"
    )
    parser.add_argument("--model")
    parser.add_argument("--provider")
    parser.add_argument("--toolsets")
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--source", default="tool")
    parser.add_argument(
        "--use-user-config",
        action="store_true",
        help="Keep Hermes' personal config instead of using reproducible defaults",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one delegated query and print only Hermes' final response."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.max_turns is not None and args.max_turns < 1:
        parser.error("--max-turns must be positive")

    cwd = Path.cwd()
    state_file = session_path(
        cwd=cwd,
        session_file=args.session_file,
        session_key=args.session_key,
    )
    prompt = _read_prompt(args, parser)

    try:
        hermes_path = shutil.which(args.hermes) or args.hermes
        previous_session = None if args.new_session else load_session(state_file)
        if args.resume_required and previous_session is None:
            raise DelegationError(
                f"no saved Hermes session at {state_file}; refusing to start a new session"
            )

        lock_path = state_file.with_suffix(f"{state_file.suffix}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            completed = subprocess.run(
                build_command(
                    hermes=hermes_path,
                    prompt=prompt,
                    session_id=previous_session,
                    model=args.model,
                    provider=args.provider,
                    toolsets=args.toolsets,
                    max_turns=args.max_turns,
                    source=args.source,
                    ignore_user_config=not args.use_user_config,
                ),
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        if completed.returncode != 0:
            detail = completed.stderr.strip() or "no diagnostic output"
            raise DelegationError(
                f"Hermes exited with status {completed.returncode}: {detail}"
            )
        session_id = extract_session_id(completed.stderr)
        response = strip_display_reasoning(completed.stdout)
        save_session(state_file, session_id, cwd)
        sys.stdout.write(response)
        return 0
    except FileNotFoundError:
        print(f"Hermes executable not found: {args.hermes}", file=sys.stderr)
        return 1
    except DelegationError as exc:
        print(f"hermes-delegation: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"hermes-delegation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
