"""The public agy adapter and smoke entry point, using executable fakes only."""

from __future__ import annotations

import importlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".agents/ops/lib"))
agy: Any = importlib.import_module("agy_headless")
opsproc: Any = importlib.import_module("opsproc")
hook: Any = importlib.import_module("agy_hook")


@pytest.fixture
def fake_agy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    profile_home = tmp_path / "user-home"
    profile_home.mkdir()
    monkeypatch.setenv("HOME", str(profile_home))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "smoke-state"))
    executable = tmp_path / "agy"
    executable.write_text(
        f"#!{sys.executable}\n"
        + Path(__file__).with_name("fake_agy.py").read_text().split("\n", 1)[1]
    )
    executable.chmod(0o755)
    (tmp_path / "fake_agy_tools.py").write_text(
        Path(__file__).with_name("fake_agy_tools.py").read_text()
    )
    monkeypatch.setenv("FAKE_AGY_ROOT", str(tmp_path))
    monkeypatch.setenv("FAKE_AGY_CASE", "success")
    (tmp_path / "payload.json").write_text(
        json.dumps(
            {
                "conversation_id": "private-session",
                "status": "SUCCESS",
                "response": "ignored prose",
                "structured_output": {"answer": "見出し"},
                "duration_seconds": 0.27,
                "usage": {
                    "input_tokens": 103,
                    "output_tokens": 17,
                    "thinking_tokens": 5,
                    "cache_read_tokens": 41,
                    "total_tokens": 125,
                },
            }
        )
    )
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Read the template; do not edit files.\n日本語も保つ。")
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            }
        )
    )
    config = tmp_path / "agents.conf"
    config.write_text(
        'AGY_VERSION="1.4.11"\nAGY_URL="https://example.invalid/1.4.11.tar.gz"\n'
        f'AGY_SHA256="{"0" * 64}"\nISSUE_REWRITE_AGY_MODEL="gemini-3.8-flash-high"\n'
    )
    (tmp_path / "work").mkdir()
    return {
        "agy": executable,
        "cwd": tmp_path / "work",
        "prompt_file": prompt,
        "json_schema": schema,
        "config_file": config,
        "log_file": tmp_path / "turn.log",
    }


def test_success_requires_evidence_and_passes_readonly_settings(
    fake_agy: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGY_VERSION", "9.0.0")
    monkeypatch.setenv("ISSUE_REWRITE_AGY_MODEL", "other-model")
    monkeypatch.setenv("AGY_CLI_DISABLE_AUTO_UPDATE", "false")
    fake_agy.pop("config_file")  # use the production pins through the default API
    payload_path = fake_agy["agy"].parent / "payload.json"
    payload = json.loads(payload_path.read_text())
    payload["denied_actions"] = [{"action": "command", "display_name": "RunCommand"}]
    payload_path.write_text(json.dumps(payload))
    result = agy.run_headless(**fake_agy, timeout=60)
    assert result.structured_output == {"answer": "見出し"}
    assert result.observed_version == "1.4.11"
    assert result.observed_model == "Gemini 3.8 Flash (High)"
    assert result.usage == {
        "input_tokens": 103,
        "output_tokens": 17,
        "thinking_tokens": 5,
        "cache_read_tokens": 41,
        "total_tokens": 125,
    }
    assert result.denied_actions == [
        {"action": "command", "display_name": "RunCommand"}
    ]
    assert result.duration_seconds == 0.27
    assert result.elapsed_seconds > 0
    calls = [
        json.loads(line)
        for line in (fake_agy["agy"].parent / "invocations.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(calls) == 3
    assert calls[0]["args"] == calls[2]["args"] == ["--version"]
    assert all(call["disable_auto_update"] == "true" for call in calls)
    assert all(
        call["read_environment"]
        == {
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "GH_PAGER": "cat",
            "RIPGREP_CONFIG_PATH": "/dev/null",
        }
        for call in calls
    )
    assert calls[1]["cwd"] == str(fake_agy["cwd"])
    args = calls[1]["args"]
    assert args[args.index("-p") + 1] == fake_agy["prompt_file"].read_text()
    assert args[args.index("--model") + 1] == "gemini-3.8-flash-high"
    assert args[args.index("--mode") + 1] == "plan"
    assert args[args.index("--output-format") + 1] == "stream-json"
    settings_arg = args[args.index("--settings") + 1]
    settings_file = Path(settings_arg)
    assert settings_file == fake_agy["log_file"].with_name("turn.log.settings.json")
    assert not settings_file.resolve().is_relative_to(fake_agy["cwd"].resolve())
    assert settings_file.stat().st_mode & 0o777 == 0o600
    fixed = calls[1]["session_settings"]
    assert json.loads(settings_file.read_text()) == fixed
    assert json.dumps(fixed) not in args
    assert len(settings_arg.encode()) < 1024
    assert fixed["toolPermission"] == "request-review"
    assert fixed["agentMode"] == args[args.index("--mode") + 1] == "plan"
    assert fixed["enableTerminalSandbox"] is False
    assert fixed["allowNonWorkspaceAccess"] is False
    settings = fixed["permissions"]
    assert all(
        rule.startswith("command(regex:^") and rule.endswith("$)")
        for rule in settings["allow"]
    )
    assert {
        "write_file(*)",
        "read_url(*)",
        "execute_url(*)",
        "unsandboxed(*)",
        "mcp(*)",
    } <= set(settings["deny"])
    assert fake_agy["log_file"].stat().st_mode & 0o777 == 0o600
    capture = json.loads(
        fake_agy["log_file"].with_name("turn.log.commands.json").read_text()
    )
    integrity = capture["session_settings_integrity"]
    assert integrity["unchanged"] is True
    assert integrity["expected_sha256"] == integrity["observed_sha256"]


@pytest.mark.parametrize(
    "fault", ["content", "delete", "symlink", "mode", "content_error"]
)
def test_session_settings_file_tampering_cannot_return_success(
    fake_agy: dict[str, Path], monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    monkeypatch.setenv("FAKE_AGY_CASE", "settings_" + fault)
    with pytest.raises(
        agy.AgyError, match="session settings file was modified"
    ) as caught:
        agy.run_headless(**fake_agy, timeout=60)
    assert caught.value.reason == "invalid_output"
    capture = json.loads(
        fake_agy["log_file"].with_name("turn.log.commands.json").read_text()
    )
    assert capture["session_settings_integrity"]["unchanged"] is False
    if fault.startswith("content"):
        integrity = capture["session_settings_integrity"]
        assert integrity["expected_sha256"] != integrity["observed_sha256"]
    assert not (fake_agy["cwd"] / ".agents").exists()


def test_settings_larger_than_a_single_exec_argument_are_passed_by_file(
    fake_agy: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = agy.readonly_settings()
    settings["permissions"]["deny"].append("read_file(" + "a" * (128 * 1024) + ")")
    monkeypatch.setattr(agy, "readonly_settings", lambda: settings)
    assert len(json.dumps(settings).encode()) > 128 * 1024
    result = agy.run_headless(**fake_agy, timeout=60)
    assert result.structured_output == {"answer": "見出し"}
    calls = [
        json.loads(line)
        for line in (fake_agy["agy"].parent / "invocations.jsonl")
        .read_text()
        .splitlines()
    ]
    assert calls[1]["session_settings"] == settings
    assert len(json.dumps(calls[1]["args"]).encode()) < 8192


def test_large_prompt_is_one_complete_stdin_event(fake_agy: dict[str, Path]) -> None:
    prompt = "全文を保つ日本語\n" * 20000
    fake_agy["prompt_file"].write_text(prompt)
    result = agy.run_headless(**fake_agy, timeout=60)
    assert result.structured_output == {"answer": "見出し"}
    calls = [
        json.loads(line)
        for line in (fake_agy["agy"].parent / "invocations.jsonl")
        .read_text()
        .splitlines()
    ]
    assert "-p" not in calls[1]["args"]
    assert calls[1]["stdin_event"] == {"event": "user", "message": {"content": prompt}}
    assert (
        calls[1]["args"][calls[1]["args"].index("--input-format") + 1] == "stream-json"
    )


@pytest.mark.parametrize(
    "settings",
    [
        None,
        {},
        {"toolPermission": "request-review"},
        {"toolPermission": "strict"},
        {"agentMode": ""},
        {"agentMode": "default"},
        {"agentMode": "plan"},
        {"allowNonWorkspaceAccess": False, "enableTerminalSandbox": False},
        {
            "permissions": {
                "allow": [],
                "ask": ["command(*)"],
                "deny": ["write_file(*)"],
            }
        },
        {"$schema": "https://antigravity.google/schemas/v1/settings.json"},
        {"model": ""},
        {"trustedWorkspaces": []},
    ],
)
def test_safe_user_settings_allow_startup_without_rewriting_the_profile(
    fake_agy: dict[str, Path],
    settings: dict[str, Any] | None,
) -> None:
    path = Path(os.environ["HOME"]) / ".gemini/antigravity-cli/settings.json"
    raw = None
    if settings is not None:
        path.parent.mkdir(parents=True)
        raw = json.dumps(settings).encode()
        path.write_bytes(raw)
    result = agy.run_headless(**fake_agy, timeout=60)
    assert result.structured_output == {"answer": "見出し"}
    assert (fake_agy["agy"].parent / "invocations.jsonl").exists()
    assert (path.read_bytes() if path.exists() else None) == raw


@pytest.mark.parametrize(
    "settings",
    [
        {"toolPermission": "always-proceed"},
        {"toolPermission": "proceed-in-sandbox"},
        {"toolPermission": "future-mode"},
        {"toolPermission": None},
        {"agentMode": "auto"},
        {"agentMode": "accept-edits"},
        {"agentMode": "future-mode"},
        {"permissions": {"allow": ["command(*)"]}},
        {"permissions": {"allow": None}},
        {"permissions": {"allow": False}},
        {"permissions": {"allow": {}}},
        {"permissions": {"unknown": []}},
        {"permissions": None},
        {"permissions": {"ask": "command(*)"}},
        {"permissions": {"deny": [False]}},
        {"permissions": {"deny": [""]}},
        {"enableTerminalSandbox": True},
        {"enableTerminalSandbox": 0},
        {"allowNonWorkspaceAccess": True},
        {"allowNonWorkspaceAccess": 0},
        {"unknownSetting": False},
        {"colorScheme": "dark"},
        {"$schema": {}},
        {"model": None},
        {"model": 17},
        {"model": []},
        {"trustedWorkspaces": None},
        {"trustedWorkspaces": "/other-workspace"},
        {"trustedWorkspaces": {}},
        {"trustedWorkspaces": [True]},
        {"trustedWorkspaces": ["/other-workspace", 17]},
        [],
        "broken",
        '{"toolPermission":"request-review","toolPermission":"always-proceed"}',
    ],
)
def test_unsafe_or_invalid_user_settings_stop_before_any_agy_process(
    fake_agy: dict[str, Path],
    settings: Any,
) -> None:
    path = Path(os.environ["HOME"]) / ".gemini/antigravity-cli/settings.json"
    path.parent.mkdir(parents=True)
    path.write_text(settings if isinstance(settings, str) else json.dumps(settings))
    with pytest.raises(agy.AgyError) as caught:
        agy.run_headless(**fake_agy, timeout=60)
    assert caught.value.reason == "unsafe_user_settings"
    assert not (fake_agy["agy"].parent / "invocations.jsonl").exists()
    assert not fake_agy["log_file"].exists()


@pytest.mark.parametrize("agent_mode", ["plan", "auto", "accept-edits"])
def test_login_settings_keep_the_model_pin_and_explain_unsafe_agent_mode(
    fake_agy: dict[str, Path], agent_mode: str
) -> None:
    path = Path(os.environ["HOME"]) / ".gemini/antigravity-cli/settings.json"
    path.parent.mkdir(parents=True)
    raw = json.dumps(
        {
            "agentMode": agent_mode,
            "model": "A different saved model label",
            "trustedWorkspaces": [str(Path(os.environ["HOME"]) / "login-checkout")],
        }
    )
    path.write_text(raw)
    calls_path = fake_agy["agy"].parent / "invocations.jsonl"
    if agent_mode == "plan":
        result = agy.run_headless(**fake_agy, timeout=60)
        assert result.observed_model == "Gemini 3.8 Flash (High)"
        calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
        args = calls[1]["args"]
        assert args[args.index("--model") + 1] == "gemini-3.8-flash-high"
    else:
        with pytest.raises(agy.AgyError) as caught:
            agy.run_headless(**fake_agy, timeout=60)
        assert caught.value.reason == "unsafe_user_settings"
        assert re.search(r"remove.*agentMode.*plan", str(caught.value))
        assert not calls_path.exists()
        assert not fake_agy["log_file"].exists()
    assert path.read_text() == raw


@pytest.mark.parametrize(
    "relationship",
    [
        "same",
        "ancestor",
        "prefix_sibling",
        "descendant",
        "cwd_symlink",
        "trusted_symlink",
        "relative_ancestor",
        "home_ancestor",
        "normalized_ancestor",
    ],
)
def test_trusted_workspace_boundaries_are_checked_before_startup(
    fake_agy: dict[str, Path], tmp_path: Path, relationship: str
) -> None:
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    work = trusted / "job"
    roots = [str(trusted)]
    if relationship == "same":
        work = trusted
    elif relationship == "prefix_sibling":
        work = tmp_path / "trusted-unrelated"
    elif relationship == "descendant":
        work = trusted
        roots = [str(trusted / "child")]
    elif relationship in ("cwd_symlink", "trusted_symlink"):
        alias = tmp_path / "alias"
        alias.symlink_to(trusted, target_is_directory=True)
        if relationship == "cwd_symlink":
            work = alias / "job"
        else:
            roots = [str(alias)]
    elif relationship == "relative_ancestor":
        roots = [".."]
    elif relationship == "home_ancestor":
        work = Path(os.environ["HOME"]) / "job"
        roots = ["~"]
    elif relationship == "normalized_ancestor":
        roots = [str(trusted / "extra" / "..")]
    work.mkdir(parents=True, exist_ok=True)
    fake_agy["cwd"] = work
    path = Path(os.environ["HOME"]) / ".gemini/antigravity-cli/settings.json"
    path.parent.mkdir(parents=True)
    raw = json.dumps({"trustedWorkspaces": roots})
    path.write_text(raw)
    calls_path = tmp_path / "invocations.jsonl"
    if relationship in ("prefix_sibling", "descendant"):
        result = agy.run_headless(**fake_agy, timeout=60)
        assert result.structured_output == {"answer": "見出し"}
        assert calls_path.exists()
    else:
        with pytest.raises(agy.AgyError) as caught:
            agy.run_headless(**fake_agy, timeout=60)
        assert caught.value.reason == "unsafe_user_settings"
        assert "trustedWorkspaces" in str(caught.value)
        assert not calls_path.exists()
        assert not fake_agy["log_file"].exists()
    assert path.read_text() == raw


@pytest.mark.parametrize(
    "failure", ["unreadable", "directory", "invalid_utf8", "dangling_symlink"]
)
def test_unreadable_user_settings_never_start_agy(
    fake_agy: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    path = Path(os.environ["HOME"]) / ".gemini/antigravity-cli/settings.json"
    path.parent.mkdir(parents=True)
    if failure == "directory":
        path.mkdir()
    elif failure == "dangling_symlink":
        path.symlink_to(path.with_name("missing.json"))
    else:
        path.write_bytes(b"\xff" if failure == "invalid_utf8" else b"{}")
    if failure == "unreadable":
        original = Path.read_text

        def read(target: Path, *args: Any, **kwargs: Any) -> str:
            if target == path:
                raise PermissionError("controlled settings read failure")
            return original(target, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(agy.AgyError) as caught:
        agy.run_headless(**fake_agy, timeout=60)
    assert caught.value.reason == "unsafe_user_settings"
    assert not (fake_agy["agy"].parent / "invocations.jsonl").exists()


@pytest.mark.parametrize("home", ["", "relative-home"])
def test_unresolved_user_home_never_starts_agy(
    fake_agy: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    home: str,
) -> None:
    monkeypatch.setenv("HOME", home)
    with pytest.raises(agy.AgyError) as caught:
        agy.run_headless(**fake_agy, timeout=60)
    assert caught.value.reason == "unsafe_user_settings"
    assert not (fake_agy["agy"].parent / "invocations.jsonl").exists()


def test_user_settings_are_rechecked_before_the_model_turn(
    fake_agy: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_AGY_CASE", "version_widens_settings")
    with pytest.raises(agy.AgyError) as caught:
        agy.run_headless(**fake_agy, timeout=60)
    assert caught.value.reason == "unsafe_user_settings"
    calls = [
        json.loads(line)
        for line in (fake_agy["agy"].parent / "invocations.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [call["args"] for call in calls] == [["--version"]]


@pytest.mark.parametrize(
    ("scenario", "reason"),
    [
        ("empty", "invalid_output"),
        ("no_structured", "invalid_output"),
        ("schema", "invalid_output"),
        ("status", "invalid_output"),
        ("finished_reason", "invalid_output"),
        ("malformed", "invalid_output"),
        ("duplicate", "invalid_output"),
        ("multiple_results", "invalid_output"),
        ("array_result", "invalid_output"),
        ("invalid_usage", "invalid_output"),
        ("model", "model_mismatch"),
        ("missing_model", "model_mismatch"),
        ("conflicting_model", "model_mismatch"),
        ("version", "version_mismatch"),
        ("updated", "version_mismatch"),
        ("cli_error", "cli_error"),
    ],
)
def test_unusable_turn_raises(
    fake_agy: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    reason: str,
) -> None:
    monkeypatch.setenv("FAKE_AGY_CASE", scenario)
    with pytest.raises(agy.AgyError) as caught:
        agy.run_headless(**fake_agy, timeout=60)
    assert caught.value.reason == reason
    if scenario == "no_structured":
        assert caught.value.denied_actions == [
            {"action": "command", "display_name": "RunCommand"}
        ]
    if scenario == "version":
        calls = (fake_agy["agy"].parent / "invocations.jsonl").read_text().splitlines()
        assert len(calls) == 1  # refuse before spending a model turn
    if scenario == "cli_error":
        assert "AGY_ERROR" in str(caught.value)


def test_timeout_reaps_the_fake_and_its_tool(
    fake_agy: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_AGY_CASE", "timeout")
    with pytest.raises(agy.AgyError) as caught:
        agy.run_headless(**fake_agy, timeout=5)
    assert caught.value.reason == "timeout"
    child = int((fake_agy["agy"].parent / "child.pid").read_text())
    assert opsproc.process_identity(child) is None


def test_missing_metrics_stay_unavailable(
    fake_agy: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_AGY_CASE", "missing_metadata")
    result = agy.run_headless(**fake_agy, timeout=60)
    assert result.usage is None
    assert result.duration_seconds is None


def test_stale_model_log_cannot_be_used_as_evidence(fake_agy: dict[str, Path]) -> None:
    fake_agy["log_file"].write_text(
        'Propagating selected model override to backend: label="Gemini 3.8 Flash (High)"\n'
    )
    with pytest.raises(agy.AgyError) as caught:
        agy.run_headless(**fake_agy, timeout=60)
    assert caught.value.reason == "invalid_input"
    assert not (fake_agy["agy"].parent / "invocations.jsonl").exists()


def test_model_override_cannot_bypass_the_pin(fake_agy: dict[str, Path]) -> None:
    with pytest.raises(agy.AgyError) as caught:
        agy.run_headless(**fake_agy, timeout=60, model="another-model")
    assert caught.value.reason == "invalid_input"
    assert not (fake_agy["agy"].parent / "invocations.jsonl").exists()


@pytest.mark.parametrize(
    "scenario", ["success", "no_structured", "wrong_headings", "cli_error"]
)
def test_smoke_publishes_only_validated_diagnostics(
    fake_agy: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    monkeypatch.setenv("FAKE_AGY_CASE", scenario)
    payload = fake_agy["agy"].parent / "payload.json"
    value = json.loads(payload.read_text())
    value["structured_output"] = {
        "headings": re.findall(
            r"^## (.+)$",
            (ROOT / ".github/ISSUE_TEMPLATE/06_parent.md").read_text(),
            re.MULTILINE,
        )
    }
    value["denied_actions"] = [
        {"action": "command", "display_name": "RunCommand"},
        {"action": "command", "display_name": "private-detail"},
    ]
    payload.write_text(json.dumps(value))
    summary = fake_agy["cwd"] / "summary.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/github_actions/agy_smoke.py"),
            "--agy",
            str(fake_agy["agy"]),
            "--config-file",
            str(fake_agy["config_file"]),
        ],
        cwd=ROOT,
        env={**os.environ, "GITHUB_STEP_SUMMARY": str(summary)},
        text=True,
        capture_output=True,
        timeout=180,
    )
    report = json.loads(proc.stdout)
    assert proc.returncode == (0 if scenario == "success" else 1), proc.stderr
    assert "private-session" not in summary.read_text()
    assert "private-detail" not in summary.read_text()
    runs = list(
        (fake_agy["agy"].parent / "smoke-state/tennis-lab-agents/agy-smoke").glob(
            "run-*"
        )
    )
    assert len(runs) == 1
    assert (runs[0] / "agy.log.stream.jsonl").is_file()
    assert (runs[0] / "agy.log.commands.json").stat().st_mode & 0o777 == 0o600
    assert not list(runs[0].glob("workspace-*"))
    if scenario != "success":
        assert (runs[0] / "failure.json").stat().st_mode & 0o777 == 0o600
    if scenario == "success":
        assert report["observed_version"] == "1.4.11"
        assert report["observed_model"] == "Gemini 3.8 Flash (High)"
        assert report["usage"]["input_tokens"] == 103
        assert report["duration_seconds"] == 0.27
        assert report["denied_action_count"] == 2
        assert "denied_actions" not in report
    else:
        assert report["status"] == "failed"


@pytest.mark.parametrize(
    ("scenario", "failure"),
    [
        ("probe_success", None),
        ("probe_outside_denied", None),
        ("probe_missing_denial", "command_hook_mismatch"),
        ("probe_read_missing_hook", "command_hook_mismatch"),
        ("probe_workspace_write", "workspace_modified"),
        ("probe_outside_write", "outside_modified"),
        ("probe_workspace_edit", "workspace_modified"),
        ("probe_agents_write", "workspace_modified"),
        ("probe_read_denied", "runtime_or_read_failure"),
        ("probe_bad_model", "runtime_or_read_failure"),
        ("probe_cli_error", "runtime_or_read_failure"),
    ],
)
def test_permission_smoke_checks_denial_reads_and_file_integrity(
    fake_agy: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    failure: str | None,
) -> None:
    monkeypatch.setenv("FAKE_AGY_CASE", scenario)
    profile = Path(os.environ["HOME"]) / ".gemini/antigravity-cli/settings.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        json.dumps(
            {
                "agentMode": "plan",
                "model": "A different saved model label",
                "trustedWorkspaces": [str(fake_agy["cwd"] / "login-checkout")],
            }
        )
    )
    summary = fake_agy["cwd"] / "summary.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/github_actions/agy_smoke.py"),
            "--agy",
            str(fake_agy["agy"]),
            "--config-file",
            str(fake_agy["config_file"]),
            "--permissions-probe",
            "--timeout",
            "60",
        ],
        cwd=ROOT,
        env={**os.environ, "GITHUB_STEP_SUMMARY": str(summary)},
        text=True,
        capture_output=True,
        timeout=300,
    )
    assert proc.returncode == (0 if failure is None else 1), proc.stderr
    report = json.loads(proc.stdout)
    for secret in (
        "private-detail",
        "private-session",
        "AGY_PROBE_TOKEN=",
        "execution-canary",
        str(fake_agy["cwd"]),
    ):
        assert secret not in summary.read_text()
    if failure is None:
        assert report["read_checks_passed"] is True
        assert report["read_commands_requested"] == 9
        assert report["read_continued_after_hook_denial"] is True
        assert report["commands"]["hook_denied_count"] == 1
        assert report["commands"]["read_file"]["denied_count"] == 1
        assert report["denial_commands_rejected"] == 27
        assert report["outside_read_denied"] is (scenario == "probe_outside_denied")
        assert all(
            case["command_denials"] > 0
            and case["workspace_unchanged"]
            and case["outside_unchanged"]
            for case in report["probes"]
        )
        calls = [
            json.loads(line)
            for line in (fake_agy["agy"].parent / "invocations.jsonl")
            .read_text()
            .splitlines()
        ]
        probes = [
            json.loads(call["args"][call["args"].index("-p") + 1].splitlines()[1])
            for call in calls
            if "-p" in call["args"]
        ]
        assert len(probes) == 29
        denial_probes = [probe for probe in probes if probe["stage"] == "deny"]
        assert len({probe["command"] for probe in denial_probes}) == 27
        assert any("\n" in probe["command"] for probe in denial_probes)
        assert {
            "touch b",
            "rm x",
            "mkdir new-directory",
            "cp a copied",
            "mv a moved",
            "python3 -c pass",
            "sh -c true",
        } <= {probe["command"] for probe in denial_probes}
        observer = next(probe for probe in probes if probe["stage"] == "observe")
        assert observer["command"].startswith("cat /")
    else:
        assert report["reason"] == failure


@pytest.mark.parametrize(
    "command,decision",
    [
        ("cat a", "allow"),
        ("head -n 1 a", "allow"),
        ("python3 -c pass", "deny"),
        ("cat a | head -n 1", "deny"),
        ("cat a\n", "deny"),
        (None, "deny"),
        (17, "deny"),
    ],
)
def test_command_hook_contract_uses_trusted_code(
    tmp_path: Path,
    command: Any,
    decision: str,
) -> None:
    payload = json.dumps(
        {
            "stepIdx": 2,
            "toolCall": {"name": "run_command", "args": {"CommandLine": command}},
        }
    )
    _check_hook(tmp_path, payload, decision)


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        "null",
        "{}",
        '{"stepIdx":2,"stepIdx":3}',
        '{"stepIdx":true,"toolCall":{"name":"run_command","args":{"CommandLine":"cat a"}}}',
    ],
)
def test_malformed_command_hook_input_is_explicitly_denied(
    tmp_path: Path, raw: str
) -> None:
    _check_hook(tmp_path, raw, "deny")


def _check_hook(tmp_path: Path, raw: str, decision: str) -> None:
    work, control = tmp_path / "work", tmp_path / "control"
    work.mkdir()
    control.mkdir(mode=0o700)
    record = control / "hooks.jsonl"
    record.touch(mode=0o600)
    owned = hook.install_command_hook(work, record)
    assert list(owned) == [work / ".agents/hooks.json"]
    config = json.loads(next(iter(owned.values())))
    command = config["readonly-command-guard"]["PreToolUse"][0]["hooks"][0]["command"]
    argv = shlex.split(command)
    assert "-I" in argv and "-B" in argv
    program = Path(argv[3])
    assert not program.is_relative_to(work)
    original_code = program.read_bytes()
    # Workspace code and PYTHONPATH must never replace the policy imported by the hook.
    for name in ("agy_permissions.py", "sitecustomize.py"):
        (work / name).write_text("raise RuntimeError('untrusted hook implementation')")
    proc = subprocess.run(
        argv,
        cwd=work,
        input=raw,
        env={**os.environ, "PYTHONPATH": str(work)},
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    response = json.loads(proc.stdout)
    assert response["decision"] == decision
    assert response["reason"]
    assert json.loads(record.read_text())["decision"] == decision
    assert program.read_bytes() == original_code


@pytest.mark.parametrize(
    "ending,reason",
    [
        ("success", None),
        ("error", "cli_error"),
        ("timeout", "timeout"),
        ("malformed", "invalid_output"),
    ],
)
def test_command_audit_survives_every_exit_without_public_raw_commands(
    fake_agy: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    ending: str,
    reason: str | None,
) -> None:
    monkeypatch.setenv("FAKE_AGY_CASE", "audit_" + ending)
    if reason is None:
        outcome = agy.run_headless(**fake_agy, timeout=60)
        assert outcome.structured_output == {"answer": "見出し"}
    else:
        with pytest.raises(agy.AgyError) as caught:
            agy.run_headless(**fake_agy, timeout=5 if ending == "timeout" else 60)
        outcome = caught.value
        assert outcome.reason == reason
    summary = outcome.command_summary
    assert summary == {
        "attempted_count": 2,
        "denied_count": 1,
        "hook_denied_count": 1,
        "forms": [
            {"form": "<unlisted command>", "count": 1},
            {"form": "cat <PATH>", "count": 1},
        ],
    }
    assert any(
        action.get("command") == "python3 -c private-command-text"
        for action in outcome.denied_actions
    )
    log = fake_agy["log_file"]
    capture = json.loads(log.with_name(log.name + ".commands.json").read_text())
    assert [entry["command"] for entry in capture["commands"]] == [
        "python3 -c private-command-text",
        "cat private-file.txt",
    ]
    assert [entry["decision"] for entry in capture["hook_decisions"]] == [
        "deny",
        "allow",
    ]
    for suffix in ("", ".commands.json", ".stream.jsonl", ".hooks.jsonl"):
        assert log.with_name(log.name + suffix).stat().st_mode & 0o777 == 0o600
    assert not (fake_agy["cwd"] / ".agents").exists()
    assert "private-command-text" not in json.dumps(summary)
    assert "private-file.txt" not in json.dumps(summary)


@pytest.mark.parametrize("existing", ["hook", "symlink", "controller"])
def test_hook_never_replaces_persistent_or_controller_configuration(
    fake_agy: dict[str, Path],
    existing: str,
) -> None:
    work = fake_agy["cwd"]
    if existing == "controller":
        fake_agy["cwd"] = ROOT
    elif existing == "symlink":
        (work / ".agents").symlink_to(ROOT / ".agents", target_is_directory=True)
    else:
        (work / ".agents").mkdir()
        (work / ".agents/hooks.json").write_text("preserve existing hook")
    with pytest.raises(agy.AgyError) as caught:
        agy.run_headless(**fake_agy, timeout=60)
    assert caught.value.reason == "invalid_input"
    if existing == "hook":
        assert (work / ".agents/hooks.json").read_text() == "preserve existing hook"
    assert not (fake_agy["agy"].parent / "invocations.jsonl").exists()


@pytest.mark.parametrize(
    "kind,denied",
    [
        ("inside", False),
        ("absolute_inside", False),
        ("outside", True),
        ("parent", True),
        ("sibling_prefix", True),
        ("symlink_outside", True),
        ("symlink_inside", False),
        ("symlink_loop", True),
    ],
)
def test_read_file_hook_resolves_paths_and_continues(
    fake_agy: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    denied: bool,
) -> None:
    work = fake_agy["cwd"]
    (work / "inside.txt").write_text("permitted input")
    outside = work.parent / "private-outside.txt"
    outside.write_text("outside input must not be read")
    path = "inside.txt"
    if kind == "absolute_inside":
        path = str(work / "inside.txt")
    elif kind == "outside":
        path = str(outside)
    elif kind == "parent":
        path = "../private-outside.txt"
    elif kind == "sibling_prefix":
        sibling = work.with_name(work.name + "-other")
        sibling.mkdir()
        path = str(sibling / "private-outside.txt")
    elif kind.startswith("symlink_"):
        target = (
            outside
            if kind == "symlink_outside"
            else work / ("alias" if kind == "symlink_loop" else "inside.txt")
        )
        (work / "alias").symlink_to(target)
        path = "alias"
    monkeypatch.setenv("FAKE_AGY_CASE", "read_recovery")
    monkeypatch.setenv("FAKE_READ_PATH", path)
    result = agy.run_headless(**fake_agy, timeout=60)
    assert result.structured_output == {"answer": "見出し"}
    assert result.command_summary["read_file"] == {
        "attempted_count": 2,
        "denied_count": int(denied),
        "forms": [{"form": "read_file <PATH>", "count": 2}],
    }
    capture = json.loads(
        fake_agy["log_file"].with_name("turn.log.commands.json").read_text()
    )
    assert [entry["decision"] for entry in capture["hook_decisions"]] == [
        "deny" if denied else "allow",
        "allow",
    ]
    assert capture["hook_decisions"][0]["path"] == path
    assert len(capture["reads"]) == 2
    assert [entry["hook_decision"] for entry in capture["reads"]] == [
        "deny" if denied else "allow",
        "allow",
    ]
    assert all(action["action"] == "read_file" for action in result.denied_actions)
    assert "private-outside" not in json.dumps(result.command_summary)


@pytest.mark.parametrize(
    "args", [{}, {"AbsolutePath": "a"}, {"Path": None}, {"Path": 17}, {"Path": ""}]
)
def test_read_file_hook_rejects_wrong_argument_shapes(
    tmp_path: Path, args: dict[str, Any]
) -> None:
    _check_hook(
        tmp_path,
        json.dumps({"stepIdx": 3, "toolCall": {"name": "read_file", "args": args}}),
        "deny",
    )


@pytest.mark.parametrize(
    "kind", ["missing", "step", "changed", "denied_execution", "truncated"]
)
def test_headless_reconciles_every_command_against_its_hook(
    fake_agy: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    monkeypatch.setenv("FAKE_AGY_CASE", "verification_" + kind)
    if kind == "truncated":
        result = agy.run_headless(**fake_agy, timeout=60)
        assert result.structured_output == {"answer": "見出し"}
    else:
        with pytest.raises(agy.AgyError) as caught:
            agy.run_headless(**fake_agy, timeout=60)
        assert caught.value.reason == "command_hook_mismatch"
    capture = json.loads(
        fake_agy["log_file"].with_name("turn.log.commands.json").read_text()
    )
    if kind == "truncated":
        assert capture["commands"][0]["command"] == "cat " + "p" * 600
        assert capture["commands"][0]["stream_command"].endswith("…")
        assert capture["commands"][0]["command_capture_source"] == "pre_tool_hook"
    else:
        assert capture["hook_verification_errors"]
    assert capture["commands"]


@pytest.mark.parametrize(
    "kind", ["missing", "denied_execution", "step", "path", "inside", "denied"]
)
def test_headless_reconciles_every_read_against_its_hook(
    fake_agy: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    monkeypatch.setenv("FAKE_AGY_CASE", "read_verification_" + kind)
    if kind in {"inside", "denied"}:
        result = agy.run_headless(**fake_agy, timeout=60)
        assert result.structured_output == {"answer": "見出し"}
    else:
        with pytest.raises(agy.AgyError) as caught:
            agy.run_headless(**fake_agy, timeout=60)
        assert caught.value.reason == "command_hook_mismatch"
        assert caught.value.command_summary["read_file"]["attempted_count"] >= 1
        assert "private-read-input.txt" not in json.dumps(caught.value.command_summary)
    capture = json.loads(
        fake_agy["log_file"].with_name("turn.log.commands.json").read_text()
    )
    assert len(capture["reads"]) == 1
    assert capture["reads"][0]["parameters"]["Path"]
    if kind in {"inside", "denied"}:
        assert capture["reads"][0]["hook_decision"] == (
            "allow" if kind == "inside" else "deny"
        )
        assert capture["reads"][0]["state"] == ("DONE" if kind == "inside" else "ERROR")
    else:
        assert capture["hook_verification_errors"]
