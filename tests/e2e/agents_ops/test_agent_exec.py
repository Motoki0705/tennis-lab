"""Exercise the headless agent wrapper with fake claude/codex executables."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
AGENT_EXEC = ROOT / ".agents/ops/lib/agent_exec.sh"

FAKE_CLAUDE = """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$FAKE_ARGS"
pwd > "$FAKE_CWD"
cat
"""

FAKE_CODEX = """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$FAKE_ARGS"
out=""
while (($#)); do
  if [[ "$1" == --output-last-message ]]; then out="$2"; shift 2; else shift; fi
done
cat > "$out"
"""


def _install_fake(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _install_fake(bin_dir, "claude", FAKE_CLAUDE)
    _install_fake(bin_dir, "codex", FAKE_CODEX)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_ARGS": str(tmp_path / "args.txt"),
        "FAKE_CWD": str(tmp_path / "cwd.txt"),
    }
    return subprocess.run(
        ["bash", str(AGENT_EXEC), *args],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("agent", ["claude", "codex"])
@pytest.mark.parametrize(
    ("mode", "claude_flag", "codex_flag"),
    [
        ("read-only", "--disallowedTools", "read-only"),
        ("write", "bypassPermissions", "--dangerously-bypass-approvals-and-sandbox"),
    ],
)
def test_prompt_round_trips_to_output(
    tmp_path: Path, agent: str, mode: str, claude_flag: str, codex_flag: str
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    prompt = tmp_path / "prompt.md"
    prompt.write_text("hello agent\n", encoding="utf-8")
    out = tmp_path / "out" / "result.md"

    result = _run(
        tmp_path,
        "--agent", agent,
        "--cwd", str(work),
        "--prompt-file", str(prompt),
        "--out", str(out),
        "--mode", mode,
        "--log-dir", str(tmp_path / "logs"),
    )  # fmt: skip

    assert result.returncode == 0, result.stderr
    assert out.read_text(encoding="utf-8") == "hello agent\n"
    args = (tmp_path / "args.txt").read_text(encoding="utf-8").splitlines()
    assert (claude_flag if agent == "claude" else codex_flag) in args
    if agent == "claude":
        assert (tmp_path / "cwd.txt").read_text(encoding="utf-8").strip() == str(work)
    else:
        assert args[args.index("--cd") + 1] == str(work)


def test_empty_output_fails_loudly(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    prompt = tmp_path / "prompt.md"
    prompt.write_text("", encoding="utf-8")

    result = _run(
        tmp_path,
        "--agent", "claude",
        "--cwd", str(work),
        "--prompt-file", str(prompt),
        "--out", str(tmp_path / "out.md"),
        "--log-dir", str(tmp_path / "logs"),
    )  # fmt: skip

    assert result.returncode != 0
    assert "empty final message" in result.stderr


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--agent", "gemini"],
        ["--mode", "admin"],
    ],
)
def test_invalid_arguments_are_rejected(tmp_path: Path, bad_args: list[str]) -> None:
    work = tmp_path / "work"
    work.mkdir()
    prompt = tmp_path / "prompt.md"
    prompt.write_text("x", encoding="utf-8")
    base = {
        "--agent": "claude",
        "--cwd": str(work),
        "--prompt-file": str(prompt),
        "--out": str(tmp_path / "out.md"),
    }
    base.update(dict(zip(bad_args[::2], bad_args[1::2], strict=True)))
    args = [item for pair in base.items() for item in pair]

    result = _run(tmp_path, *args)

    assert result.returncode == 2
