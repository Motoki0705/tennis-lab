"""The public agy adapter and smoke entry point, using executable fakes only."""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".agents/ops/lib"))
agy: Any = importlib.import_module("agy_headless")
opsproc: Any = importlib.import_module("opsproc")


@pytest.fixture
def fake_agy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    executable = tmp_path / "agy"
    executable.write_text(
        f"#!{sys.executable}\n"
        + Path(__file__).with_name("fake_agy.py").read_text().split("\n", 1)[1]
    )
    executable.chmod(0o755)
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
    return {
        "agy": executable,
        "cwd": tmp_path,
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
    payload_path = fake_agy["cwd"] / "payload.json"
    payload = json.loads(payload_path.read_text())
    payload["denied_actions"] = [{"action": "command", "display_name": "RunCommand"}]
    payload_path.write_text(json.dumps(payload))
    result = agy.run_headless(**fake_agy, timeout=3)
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
        for line in (fake_agy["cwd"] / "invocations.jsonl").read_text().splitlines()
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
    settings = json.loads(args[args.index("--settings") + 1])["permissions"]
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
        agy.run_headless(**fake_agy, timeout=3)
    assert caught.value.reason == reason
    if scenario == "no_structured":
        assert caught.value.denied_actions == [
            {"action": "command", "display_name": "RunCommand"}
        ]
    if scenario == "version":
        calls = (fake_agy["cwd"] / "invocations.jsonl").read_text().splitlines()
        assert len(calls) == 1  # refuse before spending a model turn
    if scenario == "cli_error":
        assert "AGY_ERROR" in str(caught.value)


def test_timeout_reaps_the_fake_and_its_tool(
    fake_agy: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_AGY_CASE", "timeout")
    with pytest.raises(agy.AgyError) as caught:
        agy.run_headless(**fake_agy, timeout=1)
    assert caught.value.reason == "timeout"
    child = int((fake_agy["cwd"] / "child.pid").read_text())
    assert opsproc.process_identity(child) is None


def test_missing_metrics_stay_unavailable(
    fake_agy: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_AGY_CASE", "missing_metadata")
    result = agy.run_headless(**fake_agy, timeout=3)
    assert result.usage is None
    assert result.duration_seconds is None


def test_stale_model_log_cannot_be_used_as_evidence(fake_agy: dict[str, Path]) -> None:
    fake_agy["log_file"].write_text(
        'Propagating selected model override to backend: label="Gemini 3.8 Flash (High)"\n'
    )
    with pytest.raises(agy.AgyError) as caught:
        agy.run_headless(**fake_agy, timeout=3)
    assert caught.value.reason == "invalid_input"
    assert not (fake_agy["cwd"] / "invocations.jsonl").exists()


def test_model_override_cannot_bypass_the_pin(fake_agy: dict[str, Path]) -> None:
    with pytest.raises(agy.AgyError) as caught:
        agy.run_headless(**fake_agy, timeout=3, model="another-model")
    assert caught.value.reason == "invalid_input"
    assert not (fake_agy["cwd"] / "invocations.jsonl").exists()


@pytest.mark.parametrize(
    "scenario", ["success", "no_structured", "wrong_headings", "cli_error"]
)
def test_smoke_publishes_only_validated_diagnostics(
    fake_agy: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    monkeypatch.setenv("FAKE_AGY_CASE", scenario)
    payload = fake_agy["cwd"] / "payload.json"
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
        {"action": "command", "display_name": "private-detail```"},
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
        timeout=5,
    )
    report = json.loads(proc.stdout)
    assert proc.returncode == (0 if scenario == "success" else 1), proc.stderr
    assert "private-session" not in summary.read_text()
    assert "private-detail" not in summary.read_text()
    if scenario == "success":
        assert report["observed_version"] == "1.4.11"
        assert report["observed_model"] == "Gemini 3.8 Flash (High)"
        assert report["usage"]["input_tokens"] == 103
        assert report["duration_seconds"] == 0.27
        assert report["denied_actions"] == [
            {"action": "command", "display_name": "RunCommand"},
            {"action": "command", "display_name": "[redacted]"},
        ]
    else:
        assert report["status"] == "failed"


@pytest.mark.parametrize(
    ("scenario", "failure"),
    [
        ("probe_success", None),
        ("probe_missing_denial", "missing_command_denial"),
        ("probe_workspace_write", "workspace_modified"),
        ("probe_outside_write", "outside_modified"),
        ("probe_workspace_edit", "workspace_modified"),
        ("probe_read_denied", "runtime_or_read_failure"),
        ("probe_bad_model", "unconfirmed_model"),
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
            "3",
        ],
        cwd=ROOT,
        env={**os.environ, "GITHUB_STEP_SUMMARY": str(summary)},
        text=True,
        capture_output=True,
        timeout=10,
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
        assert report["denial_commands_rejected"] == 20
        assert all(
            case["command_denials"] > 0
            and case["workspace_unchanged"]
            and case["outside_unchanged"]
            for case in report["probes"]
        )
        calls = [
            json.loads(line)
            for line in (fake_agy["cwd"] / "invocations.jsonl").read_text().splitlines()
        ]
        probes = [
            json.loads(call["args"][call["args"].index("-p") + 1].splitlines()[1])
            for call in calls
            if "-p" in call["args"]
        ]
        assert len(probes) == 21
        assert len({probe["command"] for probe in probes[1:]}) == 20
        assert any("\n" in probe["command"] for probe in probes[1:])
    else:
        assert report["reason"] == failure
