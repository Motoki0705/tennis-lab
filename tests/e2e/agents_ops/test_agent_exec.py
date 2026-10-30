"""Exercise the headless agent wrapper with fake claude/codex executables."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
AGENT_EXEC = ROOT / ".agents/ops/lib/agent_exec.sh"

FAKE_CLAUDE = Path(__file__).with_name("fake_agent_cli.py").read_text(encoding="utf-8")
FAKE_CODEX = FAKE_CLAUDE


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
        "CODEX_HOME": str(tmp_path / "codex-home"),
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
    assert model == ("claude-opus-5-5" if agent == "claude" else "gpt-6.1-sol")
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


def _codex_args(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    prompt = tmp_path / "prompt.md"
    prompt.write_text("profile\n", encoding="utf-8")
    return _run(
        tmp_path,
        "--agent", "codex",
        "--cwd", str(work),
        "--prompt-file", str(prompt),
        "--out", str(tmp_path / "out.md"),
        "--log-dir", str(tmp_path / "logs"),
        *extra,
    )  # fmt: skip


def test_profile_uses_its_own_pins(tmp_path: Path) -> None:
    result = _codex_args(tmp_path, "--mode", "read-only", "--profile", "review")

    assert result.returncode == 0, result.stderr
    args = (tmp_path / "args.txt").read_text(encoding="utf-8").splitlines()
    assert args[args.index("--model") + 1] == "gpt-6-astra"
    # xhigh, not the default CODEX_EFFORT (max): the profile's own pin is used.
    assert 'model_reasoning_effort="xhigh"' in args
    assert 'service_tier="default"' in args
    assert "profile=review" in result.stderr


def test_audit_profile_enables_native_scouts_and_retains_session_logs(
    tmp_path: Path,
) -> None:
    result = _codex_args(tmp_path, "--profile", "ops_report")
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert args[args.index("--enable") + 1] == "multi_agent"
    assert "--json" in args and "--ephemeral" not in args
    assert "agents.max_threads=3" in args and "agents.max_depth=1" in args
    assert "read-only" in args
    assert not any("agents.scout" in a for a in args)  # child definition owns its model
    rejected = _codex_args(tmp_path, "--profile", "ops_report", "--mode", "write")
    assert rejected.returncode == 2
    assert "requires codex in read-only mode" in rejected.stderr


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--profile", "nosuch"], "agents.conf must set NOSUCH_CODEX_MODEL"),
        (["--profile", "review", "--model", "x"], "cannot be combined"),
        (["--profile", "Bad-Name"], "invalid --profile"),
        (["--profile", ""], "invalid --profile"),
    ],
)
def test_profile_errors_never_fall_back(
    tmp_path: Path, extra: list[str], message: str
) -> None:
    result = _codex_args(tmp_path, *extra)

    assert result.returncode == 2
    assert message in result.stderr
    assert not (tmp_path / "args.txt").exists()


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
        "CODEX_HOME": str(tmp_path / "codex-home"),
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
    tmp_path: Path,
    unit_names: list[str],
    *args: str,
    systemctl_status: int = 0,
    script_in_worktree: bool = False,
    cwd_in_worktree: bool = False,
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


RETIRED_UNITS = [
    ("tennis-lab-cross-review.service", ".agents/ops/review/README.md"),
    ("tennis-lab-cross-review.timer", ".agents/ops/review/README.md"),
    ("tennis-lab-weekly-report.service", ".agents/ops/ops-report/README.md"),
    ("tennis-lab-weekly-report.timer", ".agents/ops/ops-report/README.md"),
]


@pytest.mark.parametrize(("name", "readme"), RETIRED_UNITS)
@pytest.mark.parametrize("where", ["direct", "dangling", "wants"])
@pytest.mark.parametrize("extra", [[], ["--enable"], ["--dry-run"]])
def test_installer_refuses_retired_units_before_side_effects(
    tmp_path: Path, name: str, readme: str, where: str, extra: list[str]
) -> None:
    directory = tmp_path / "config/systemd/user"
    if where == "wants":
        directory /= "timers.target.wants"
    directory.mkdir(parents=True)
    old = directory / name
    if where == "direct":
        old.write_text("[Unit]\n")
    else:
        old.symlink_to(tmp_path / "removed-source" / name)
    result, calls = _run_installer(tmp_path, ["demo.service", "demo.timer"], *extra)
    assert result.returncode == 2
    assert "retired unit remains" in result.stderr
    assert str(old) in result.stderr
    assert readme in result.stderr
    assert "daemon-reload" in result.stderr
    assert old.is_file() if where == "direct" else old.is_symlink()
    assert not calls.exists()
    assert not (tmp_path / "config/systemd/user/demo.service").exists()
    assert not (tmp_path / "config/systemd/user/demo.timer").exists()
    assert "link " not in result.stdout


@pytest.mark.parametrize(("name", "readme"), RETIRED_UNITS)
@pytest.mark.parametrize("extra", [[], ["--enable"], ["--dry-run"]])
def test_installer_never_reintroduces_retired_source(
    tmp_path: Path, name: str, readme: str, extra: list[str]
) -> None:
    result, calls = _run_installer(tmp_path, ["demo.timer", name], *extra)
    assert result.returncode == 2
    assert "retired source unit" in result.stderr
    assert name in result.stderr
    assert readme in result.stderr
    assert not calls.exists()
    assert not (tmp_path / "config").exists()
    assert "link " not in result.stdout


def test_installer_reinstall_keeps_report_and_cannot_restore_review(
    tmp_path: Path,
) -> None:
    result, calls = _run_installer(
        tmp_path, ["tennis-lab-ops-report.timer", "tennis-lab-ops-report.service"]
    )
    assert result.returncode == 0, result.stderr
    installer = tmp_path / "repo/.agents/ops/install_units.sh"
    env = {
        **os.environ,
        "PATH": str(tmp_path / "bin") + os.pathsep + os.environ["PATH"],
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
    }
    again = subprocess.run(
        ["bash", str(installer)], env=env, text=True, capture_output=True, check=False
    )
    assert again.returncode == 0, again.stderr
    units = list((tmp_path / "config/systemd/user").glob("*"))
    assert sorted(p.name for p in units) == [
        "tennis-lab-ops-report.service",
        "tennis-lab-ops-report.timer",
    ]
    assert calls.read_text().splitlines() == [
        "--user daemon-reload",
        "--user daemon-reload",
    ]


def test_installer_installs_only_current_report_units(tmp_path: Path) -> None:
    source = ROOT / ".agents/ops/ops-report/systemd"
    units = sorted(p.name for p in source.iterdir())
    assert set(units) == {
        "tennis-lab-ops-report.service",
        "tennis-lab-ops-report.timer",
        "tennis-lab-ops-triage.service",
        "tennis-lab-ops-triage.timer",
    }
    result, calls = _run_installer(tmp_path, units, "--enable")
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in (tmp_path / "config/systemd/user").iterdir()) == units
    assert calls.read_text().splitlines() == [
        "--user daemon-reload",
        "--user enable --now tennis-lab-ops-report.timer",
        "--user enable --now tennis-lab-ops-triage.timer",
    ]
    service = (source / "tennis-lab-ops-report.service").read_text()
    timer = (source / "tennis-lab-ops-report.timer").read_text()
    assert "ops_report.py report --if-due" in service
    assert "ops-report.env" in service
    assert "OnCalendar=*-*-* 08:00:00 Asia/Tokyo" in timer
    assert "Persistent=true" in timer


def test_review_mcp_is_read_only_and_records_frozen_configuration(
    tmp_path: Path,
) -> None:
    config = tmp_path / "mcp.json"
    config.write_text(
        '{"review":{"command":"python","args":["-m","review_server"],"env":{"PYTHONPATH":"/fixed/ops"},"tool_timeout_sec":960}}'
    )
    result = _codex_args(
        tmp_path,
        "--mode",
        "read-only",
        "--profile",
        "review",
        "--mcp-config",
        str(config),
    )
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert 'mcp_servers.review.command="python"' in args
    assert 'mcp_servers.review.env.PYTHONPATH="/fixed/ops"' in args
    assert "--dangerously-bypass-approvals-and-sandbox" not in args
    assert args[args.index("multi_agent") - 1] == "--disable"
    saved = list((tmp_path / "logs").rglob("mcp-config.json"))
    assert len(saved) == 1 and saved[0].read_bytes() == config.read_bytes()
    rejected = _codex_args(
        tmp_path, "--mode", "write", "--profile", "review", "--mcp-config", str(config)
    )
    assert rejected.returncode == 2


@pytest.mark.parametrize(
    "profile,model,effort",
    [(None, "gpt-6.1-sol", "max"), ("review", "gpt-6-astra", "xhigh")],
)
def test_audit_mcp_does_not_change_other_profile_or_implementation_argv(
    tmp_path: Path,
    profile: str | None,
    model: str,
    effort: str,
) -> None:
    result = _codex_args(tmp_path, *(["--profile", profile] if profile else []))
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert args == [
        "--disable",
        "daemon_auto_start",
        "exec",
        "--cd",
        str(tmp_path / "work"),
        "--sandbox",
        "read-only",
        "--json",
        "--output-last-message",
        args[args.index("--output-last-message") + 1],
        "--model",
        model,
        "-c",
        f'model_reasoning_effort="{effort}"',
        "-c",
        'service_tier="default"',
    ] + (["--disable", "multi_agent"] if profile == "review" else []) + ["-"]


def test_audit_mcp_arguments_are_after_exec_and_only_on_audit_profile(
    tmp_path: Path,
) -> None:
    config = tmp_path / "audit-mcp.json"
    config.write_text(
        json.dumps(
            {
                "ops_report": {
                    "command": sys.executable,
                    "args": ["-B", "-m", "opsreport.mcp_server"],
                    "env": {"PYTHONPATH": str(ROOT / ".agents/ops/ops-report")},
                }
            }
        )
    )
    result = _codex_args(
        tmp_path, "--profile", "ops_report", "--mcp-config", str(config)
    )
    assert result.returncode == 0, result.stderr
    args = (tmp_path / "args.txt").read_text().splitlines()
    overrides = [a for a in args if a.startswith("mcp_servers.")]
    assert len(overrides) == 3
    assert all(args.index(a) > args.index("exec") for a in overrides)
    assert any(a.startswith("mcp_servers.ops_report.command=") for a in overrides)
    assert any(
        a.startswith("mcp_servers.ops_report.env.PYTHONPATH=") for a in overrides
    )
    assert "agents.max_threads=3" in args and "agents.max_depth=1" in args
    assert (
        "--sandbox" in args and "--dangerously-bypass-approvals-and-sandbox" not in args
    )
    saved = list((tmp_path / "logs").rglob("mcp-config.json"))
    assert len(saved) == 1 and saved[0].read_bytes() == config.read_bytes()


@pytest.mark.parametrize("connection", [None, "{}", "{invalid"])
def test_audit_explicit_connection_missing_or_malformed_never_falls_back(
    tmp_path: Path, connection: str | None
) -> None:
    path = tmp_path / "connection.json"
    if connection is not None:
        path.write_text(connection)
    result = _codex_args(tmp_path, "--profile", "ops_report", "--mcp-config", str(path))
    assert result.returncode != 0, result.stderr
    assert not (tmp_path / "args.txt").exists()
