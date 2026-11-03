"""Manual validation budgets and evidence, independent of a real agy/account."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/github_actions"))
harness: Any = importlib.import_module("agy_allowlist_validation")


def test_ledger_refuses_turn_21_without_changing_the_record(tmp_path: Path) -> None:
    ledger = harness.Ledger(tmp_path)
    for _ in range(20):
        ledger.reserve(phase="test")
    before = ledger.path.read_bytes()
    with pytest.raises(harness.ValidationStop, match="turn_budget_exhausted"):
        ledger.reserve(phase="test")
    assert ledger.path.read_bytes() == before
    assert ledger.remaining == 0


def test_corrupt_ledger_is_not_a_new_budget(tmp_path: Path) -> None:
    ledger = harness.Ledger(tmp_path)
    ledger.path.write_text('{"ordinal":2}\n')
    with pytest.raises(harness.ValidationStop, match="invalid_ledger"):
        ledger.reserve(phase="test")


def test_effective_mode_needs_all_three_sources() -> None:
    capture: dict[str, Any] = {
        "init": {"permission_mode": "request-review"},
        "session_settings_integrity": {"unchanged": True},
    }
    assert harness.effective_policy(capture, "applying agent mode plan")
    assert not harness.effective_policy(capture, "selected model only")
    capture["init"]["permission_mode"] = "always-proceed"
    assert not harness.effective_policy(capture, "applying agent mode plan")
    capture["init"]["permission_mode"] = "request-review"
    capture["session_settings_integrity"]["unchanged"] = False
    assert not harness.effective_policy(capture, "applying agent mode plan")


def test_quota_detection_does_not_search_prompt_or_successful_model_prose() -> None:
    assert harness.quota_or_auth_error(
        "cli_error", "agy exited 3: AGY_ERROR quota exceeded", {}
    )
    assert harness.quota_or_auth_error(
        "invalid_output",
        "",
        {"result": {"status": "ERROR", "error": "RESOURCE_EXHAUSTED"}},
    )
    assert not harness.quota_or_auth_error(
        "cli_error", "agy -p 'quota in issue' exited 3: different failure", {}
    )
    assert not harness.quota_or_auth_error(
        "invalid_output",
        "",
        {"result": {"status": "SUCCESS", "response": "quota discussion"}},
    )


def test_diagnostic_requires_actual_success_and_both_denial_sources() -> None:
    commands = harness.diagnostic_commands()
    negatives = ["touch x"]
    capture = {
        "commands": [
            {
                "command": command,
                "outcome": "completed",
                "command_exit_code": 0,
                "hook_decision": "allow",
            }
            for command in commands
        ]
        + [{"command": "touch x", "outcome": "hook_denied", "hook_decision": "deny"}],
        "hook_decisions": [
            {"tool": "read_file", "path": "/outside/canary", "decision": "deny"}
        ],
    }
    assert (
        harness.verify_diagnostic(capture, commands, negatives, "/outside/canary")[
            "allowed_commands_completed"
        ]
        == 10
    )
    capture["commands"][0]["outcome"] = "soft_denied"
    with pytest.raises(harness.ValidationStop, match="not_completed"):
        harness.verify_diagnostic(capture, commands, negatives, "/outside/canary")
    capture["commands"][0]["outcome"] = "completed"
    capture["hook_decisions"] = []
    with pytest.raises(harness.ValidationStop, match="outside_read_denial_missing"):
        harness.verify_diagnostic(capture, commands, negatives, "/outside/canary")


def test_snapshot_adapter_cannot_publish() -> None:
    adapter = harness.SnapshotGitHub({"repo": "owner/repo", "issue_number": 1})
    with pytest.raises(harness.ValidationStop, match="github_requests_forbidden"):
        adapter.request("PATCH", "issues/1", {"body": "changed"})


def test_fourth_diagnostic_is_refused_before_cli_start(tmp_path: Path) -> None:
    runner = harness.Runner(tmp_path, tmp_path / "never-execute", "unused")
    for _ in range(3):
        runner.ledger.reserve(phase="diagnostic")
    with pytest.raises(harness.ValidationStop, match="diagnostic_budget_exhausted"):
        runner.turn(cwd=tmp_path)
    assert len(runner.ledger.rows()) == 3


def test_resume_never_retries_a_quota_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=harness.ROOT, text=True
    ).strip()
    harness.write(
        tmp_path / "plan.json",
        {
            "cap": 20,
            "manual_policy_authorized": True,
            "source_head": head,
            "persistent_settings": {"before_sha256": "unused"},
            "selections": [],
        },
    )
    harness.Ledger(tmp_path).reserve(phase="diagnostic")
    harness.write(tmp_path / "stop.json", {"reason": "quota_or_authentication_stop"})

    def forbidden(**kwargs: Any) -> None:
        pytest.fail("quota resume attempted another CLI call")

    monkeypatch.setattr(harness.agy, "run_headless", forbidden)
    harness.run_campaign(tmp_path, tmp_path / "never-execute", resume=True)
    assert len(harness.Ledger(tmp_path).rows()) == 1
    assert (tmp_path / "stop.json").read_text().find(
        "quota_or_authentication_stop"
    ) >= 0
