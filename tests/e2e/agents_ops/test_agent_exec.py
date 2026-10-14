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
        ("read-only", "--tools", "read-only"),
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
    if agent == "claude" and mode == "read-only":
        assert "bypassPermissions" not in args
        tools = args[args.index("--tools") + 1 : args.index("--permission-mode")]
        assert tools == ["Read", "Grep", "Glob"]
    model = args[args.index("--model") + 1]
    assert model == ("claude-opus-5-5" if agent == "claude" else "gpt-6-astra")
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


def test_relative_paths_resolve_against_caller_cwd(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "prompt.md").write_text("relative\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install_fake(bin_dir, "claude", FAKE_CLAUDE)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_ARGS": str(tmp_path / "args.txt"),
        "FAKE_CWD": str(tmp_path / "cwd.txt"),
    }

    result = subprocess.run(
        ["bash", str(AGENT_EXEC), "--agent", "claude", "--cwd", "work",
         "--prompt-file", "prompt.md", "--out", "out/result.md", "--log-dir", "logs"],
        cwd=tmp_path, env=env, text=True, capture_output=True, check=False,
    )  # fmt: skip

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out/result.md").read_text(encoding="utf-8") == "relative\n"
    assert not (work / "out").exists()


def test_stale_output_is_not_reported_as_success(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    prompt = tmp_path / "prompt.md"
    prompt.write_text("", encoding="utf-8")
    out = tmp_path / "out.md"
    out.write_text("stale result\n", encoding="utf-8")

    result = _run(
        tmp_path,
        "--agent", "codex",
        "--cwd", str(work),
        "--prompt-file", str(prompt),
        "--out", str(out),
        "--log-dir", str(tmp_path / "logs"),
    )  # fmt: skip

    assert result.returncode != 0
    assert "empty final message" in result.stderr


def _run_installer(
    tmp_path: Path, unit_names: list[str], *args: str, systemctl_status: int = 0,
    script_in_worktree: bool = False, cwd_in_worktree: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    repo = tmp_path / "repo"
    ops = repo / ".agents/ops"
    (ops / "demo/systemd").mkdir(parents=True)
    for name in unit_names:
        (ops / "demo/systemd" / name).write_text("[Unit]\n", encoding="utf-8")
    installer = ops / "install_units.sh"
    installer.write_text(
        (ROOT / ".agents/ops/install_units.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}

    def git(*git_args: str) -> None:
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=test",
             "-c", "user.email=test@example.com", *git_args],
            env=env, check=True, capture_output=True,
        )  # fmt: skip

    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "fixture")
    worktree = tmp_path / "linked"
    if script_in_worktree or cwd_in_worktree:
        git("worktree", "add", "--detach", str(worktree), "HEAD")
    if script_in_worktree:
        installer = worktree / ".agents/ops/install_units.sh"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "systemctl_calls.txt"
    _install_fake(
        bin_dir,
        "systemctl",
        f'#!/usr/bin/env bash\necho "$*" >> {calls}\n'
        f'[[ "$*" == *enable* ]] && exit {systemctl_status}\nexit 0\n',
    )
    result = subprocess.run(
        ["bash", str(installer), *args],
        cwd=worktree if cwd_in_worktree else repo,
        env={**env, "PATH": f"{bin_dir}:{os.environ['PATH']}",
             "XDG_CONFIG_HOME": str(tmp_path / "config")},
        text=True, capture_output=True, check=False,
    )  # fmt: skip
    return result, calls


@pytest.mark.parametrize(
    "unit_names", [["demo.service"], ["demo.service", "demo.timer", "zz.service"]]
)
def test_install_units_enable_succeeds_and_enables_only_timers(
    tmp_path: Path, unit_names: list[str]
) -> None:
    result, calls = _run_installer(tmp_path, unit_names, "--enable")

    assert result.returncode == 0, result.stderr
    enabled = [
        c for c in calls.read_text(encoding="utf-8").splitlines() if "enable" in c
    ]
    assert enabled == (
        ["--user enable --now demo.timer"] if "demo.timer" in unit_names else []
    )


def test_install_units_propagates_enable_failure(tmp_path: Path) -> None:
    result, _ = _run_installer(
        tmp_path, ["demo.service", "demo.timer"], "--enable", systemctl_status=5
    )

    assert result.returncode == 5


def test_install_units_creates_missing_unit_dir(tmp_path: Path) -> None:
    result, calls = _run_installer(tmp_path, ["demo.service"])

    assert result.returncode == 0, result.stderr
    unit = tmp_path / "config/systemd/user/demo.service"
    assert unit.is_symlink()
    assert unit.resolve() == tmp_path / "repo/.agents/ops/demo/systemd/demo.service"
    assert calls.read_text().splitlines() == ["--user daemon-reload"]


@pytest.mark.parametrize("cwd_in_worktree", [False, True])
@pytest.mark.parametrize("extra", [[], ["--enable"], ["--dry-run"]])
def test_install_units_rejects_linked_source_before_side_effects(
    tmp_path: Path, cwd_in_worktree: bool, extra: list[str]
) -> None:
    result, calls = _run_installer(
        tmp_path, ["demo.service", "demo.timer"], *extra,
        script_in_worktree=True, cwd_in_worktree=cwd_in_worktree,
    )  # fmt: skip

    assert result.returncode != 0
    assert "linked worktree" in result.stderr
    assert not (tmp_path / "config").exists()
    assert not calls.exists()
    assert "link demo" not in result.stdout


def test_install_units_uses_main_source_when_cwd_is_linked(tmp_path: Path) -> None:
    result, calls = _run_installer(tmp_path, ["demo.service"], cwd_in_worktree=True)

    assert result.returncode == 0, result.stderr
    unit = tmp_path / "config/systemd/user/demo.service"
    assert unit.resolve() == tmp_path / "repo/.agents/ops/demo/systemd/demo.service"
    assert calls.read_text().splitlines() == ["--user daemon-reload"]
