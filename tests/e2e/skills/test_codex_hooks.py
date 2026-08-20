from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
HOOK = ROOT / ".codex/hooks/pre_tool_use_large_read.py"
THRESHOLD = 20 * 1024


def run_hook(
    command: str,
    cwd: Path,
    *,
    payload_overrides: dict[str, Any] | None = None,
) -> subprocess.CompletedProcess[str]:
    payload: dict[str, Any] = {
        "session_id": "test-session",
        "turn_id": "test-turn",
        "cwd": str(cwd),
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_use_id": "test-tool",
        "tool_input": {"command": command},
    }
    if payload_overrides:
        payload.update(payload_overrides)
    environment = os.environ.copy()
    environment.pop("CODEX_LARGE_READ_THRESHOLD_BYTES", None)
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )


def deny_output(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert result.returncode == 0
    assert result.stderr == ""
    output: dict[str, Any] = json.loads(result.stdout)
    return output


def test_config_sets_poll_limit_and_synchronous_git_root_hook() -> None:
    config = tomllib.loads((ROOT / ".codex/config.toml").read_text(encoding="utf-8"))

    assert config["background_terminal_max_timeout"] == 18_000_000
    matcher = config["hooks"]["PreToolUse"][0]
    assert matcher["matcher"] == "^Bash$"
    hook = matcher["hooks"][0]
    assert hook["type"] == "command"
    assert hook["async"] is False
    assert "git rev-parse --show-toplevel" in hook["command"]
    assert ".codex/hooks/pre_tool_use_large_read.py" in hook["command"]


def test_cat_threshold_boundary(tmp_path: Path) -> None:
    below = tmp_path / "below.txt"
    exact = tmp_path / "exact.txt"
    below.write_bytes(b"x" * (THRESHOLD - 1))
    exact.write_bytes(b"x" * THRESHOLD)

    allowed = run_hook(f"cat {below.name}", tmp_path)
    denied = run_hook(f"cat {exact.name}", tmp_path)

    assert allowed.returncode == 0
    assert allowed.stdout == ""
    assert allowed.stderr == ""
    assert deny_output(denied)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_bounded_sed_allows_small_slice_and_denies_large_slice(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_bytes((b"x" * 7_000 + b"\n") * 4)

    small = run_hook("sed -n '1,2p' evidence.txt", tmp_path)
    large = run_hook("sed -n '1,3p' evidence.txt", tmp_path)

    assert small.returncode == 0
    assert small.stdout == ""
    output = deny_output(large)
    assert "sed -n read" in output["hookSpecificOutput"]["permissionDecisionReason"]


def test_head_and_tail_enforce_selected_output_size(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_bytes((b"x" * 7_000 + b"\n") * 4)

    assert run_hook("head -n 2 evidence.txt", tmp_path).stdout == ""
    assert deny_output(run_hook("head -n 3 evidence.txt", tmp_path))
    assert run_hook("tail -n 2 evidence.txt", tmp_path).stdout == ""
    assert deny_output(run_hook("tail -n 3 evidence.txt", tmp_path))


def test_broad_find_exec_and_unbounded_git_reads_are_denied(tmp_path: Path) -> None:
    (tmp_path / "small.txt").write_text("small\n", encoding="utf-8")

    find_result = run_hook(r"find . -type f -exec cat {} \;", tmp_path)
    diff_result = run_hook("git diff", tmp_path)
    show_result = run_hook("git show HEAD", tmp_path)

    assert "find -exec" in deny_output(find_result)["systemMessage"]
    assert "unbounded git diff" in deny_output(diff_result)["systemMessage"]
    assert "unbounded git show" in deny_output(show_result)["systemMessage"]


def test_patch_flags_reenable_unbounded_git_output(tmp_path: Path) -> None:
    assert run_hook("git diff --stat", tmp_path).stdout == ""
    assert run_hook("git show --stat", tmp_path).stdout == ""

    for command in (
        "git diff --stat --patch",
        "git diff --patch --stat",
        "git show --stat --patch",
        "git show --no-patch --patch",
    ):
        output = deny_output(run_hook(command, tmp_path))
        assert "patch" in output["hookSpecificOutput"]["permissionDecisionReason"]


def test_common_search_awk_interpreter_and_nested_shell_bypasses_are_denied(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "large.txt"
    evidence.write_bytes((b"large evidence\n") * 2_000)

    commands = (
        "rg '' large.txt",
        "awk '{print}' large.txt",
        "python3 -c 'print(open(\"large.txt\").read())'",
        "bash -c 'cat large.txt'",
    )
    for command in commands:
        output = deny_output(run_hook(command, tmp_path))
        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_recognized_reads_with_unknown_output_bounds_are_denied(tmp_path: Path) -> None:
    (tmp_path / "small.txt").write_text("small\n", encoding="utf-8")

    commands = (
        "rg evidence .",
        "awk '{for (;;) print}' small.txt",
        "python3 -c 'import sys; print(open(sys.argv[1]).read())' small.txt",
    )
    for command in commands:
        reason = deny_output(run_hook(command, tmp_path))["hookSpecificOutput"][
            "permissionDecisionReason"
        ]
        assert "no safely computable output upper bound" in reason


def test_xargs_nested_read_commands_fail_closed(tmp_path: Path) -> None:
    evidence = tmp_path / "large.txt"
    evidence.write_bytes(b"x" * THRESHOLD)

    commands = (
        """printf 'large.txt\\0' | xargs -0 sh -c 'cat "$0"'""",
        r"printf 'large.txt\0' | xargs -0 cat",
        (
            r"printf 'large.txt\0' | xargs -0 python3 -c "
            r"'import sys; print(open(sys.argv[1]).read())'"
        ),
    )
    for command in commands:
        reason = deny_output(run_hook(command, tmp_path))["hookSpecificOutput"][
            "permissionDecisionReason"
        ]
        assert "xargs" in reason
        assert "no safely computable output upper bound" in reason


def test_terminal_output_suppression_and_bounded_pipeline_sink_are_allowed(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "large.txt"
    evidence.write_bytes(b"x\n" * THRESHOLD)

    redirected = run_hook("cat large.txt > /dev/null", tmp_path)
    summarized = run_hook("cat large.txt | wc -l", tmp_path)
    byte_bounded = run_hook("cat large.txt | head -c 100", tmp_path)

    for result in (redirected, summarized, byte_bounded):
        assert result.returncode == 0
        assert result.stdout == ""
        assert result.stderr == ""


def test_sequential_read_output_is_aggregated_across_one_bash_command(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"a" * (12 * 1024))
    second.write_bytes(b"b" * (12 * 1024))

    output = deny_output(run_hook("cat first.txt; cat second.txt", tmp_path))

    reason = output["hookSpecificOutput"]["permissionDecisionReason"]
    assert "cat read + cat read" in reason
    assert "24576 bytes" in reason


def test_non_read_and_malformed_input_fail_open(tmp_path: Path) -> None:
    non_read = run_hook("printf 'hello\\n'", tmp_path)
    wrong_tool = run_hook(
        "cat missing.txt",
        tmp_path,
        payload_overrides={"tool_name": "apply_patch"},
    )
    malformed = subprocess.run(
        [sys.executable, str(HOOK)],
        input="{not-json",
        text=True,
        capture_output=True,
        check=False,
    )

    for result in (non_read, wrong_tool, malformed):
        assert result.returncode == 0
        assert result.stdout == ""
        assert result.stderr == ""


def test_deny_json_instructs_fresh_scout_delegation_and_wait(tmp_path: Path) -> None:
    evidence = tmp_path / "large.txt"
    evidence.write_bytes(b"x" * THRESHOLD)

    output = deny_output(run_hook("cat large.txt", tmp_path))
    hook_output = output["hookSpecificOutput"]
    instruction = hook_output["additionalContext"]

    assert output["systemMessage"].startswith("Large Bash read denied before execution")
    assert hook_output["hookEventName"] == "PreToolUse"
    assert hook_output["permissionDecision"] == "deny"
    assert hook_output["permissionDecisionReason"]
    assert 'agent_type="codebase_scout"' in instruction
    assert 'fork_turns="none"' in instruction
    assert "terminal-only summary" in instruction
    assert "wait for that summary" in instruction
    assert "cannot spawn built-in agents" in instruction
