"""Real shared runner with offline Codex: choices cannot change its contract."""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_agent_runtime import CONFIG, Harness


def choice_files(
    h: Harness,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    mode: str = "select",
    human: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    monkeypatch.syspath_prepend(str(CONFIG.parents[1]))
    settings = importlib.import_module("lib.review_settings")
    config = CONFIG
    if mode != "select":
        config = h.root / "review.conf"
        config.write_text(
            re.sub(
                r'^REVIEW_SELECTION_MODE="[^"]*"$',
                f'REVIEW_SELECTION_MODE="{mode}"',
                CONFIG.read_text(),
                flags=re.MULTILINE,
            )
        )
    choice = settings.choose(
        settings.load_policy(config),
        model=model,
        reason="人間の指定"
        if human
        else "表示文言だけの修正で該当画面で確認できるためSolを選ぶ。"
        if model == "gpt-6.1-sol"
        else "復旧後に採用する永続状態が変わるためAstraを選ぶ。",
        specified_by="human_instruction" if human else None,
        human_reference="親が元指示を確認済み" if human else None,
    )
    path = h.root / "choice.json"
    path.write_text(json.dumps(choice))
    h.env["FAKE_INVOCATIONS"] = str(h.root / "invocations.jsonl")
    return config, path, choice


@pytest.mark.parametrize(
    "model,mode,human",
    [
        ("gpt-6.1-sol", "select", False),
        ("gpt-6-astra", "select", False),
        ("gpt-6-astra", "select", True),
        ("gpt-6-astra", "fixed", False),
    ],
)
def test_allowed_review_choice_keeps_tools_schema_and_observed_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, mode: str, human: bool
) -> None:
    h = Harness(tmp_path)
    config, path, choice = choice_files(h, monkeypatch, model, mode, human)
    schema, mcp = h.root / "schema.json", h.root / "mcp.json"
    schema.write_text(
        '{"type":"object","properties":{"message":{"type":"string"}},"required":["message"],"additionalProperties":false}'
    )
    mcp.write_text('{"review":{"command":"python","args":["-m","review_server"]}}')
    proc = h.run(
        extra=(
            "--profile",
            "review",
            "--config-file",
            str(config),
            "--review-selection-file",
            str(path),
            "--json-schema",
            str(schema),
            "--mcp-config",
            str(mcp),
        )
    )
    assert proc.returncode == 0, proc.stderr
    result, args = h.result(), h.args()
    assert args[args.index("--model") + 1] == model
    assert 'model_reasoning_effort="xhigh"' in args
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert "--output-schema" in args and 'mcp_servers.review.command="python"' in args
    assert "--dangerously-bypass-approvals-and-sandbox" not in args
    assert args[args.index("multi_agent") - 1] == "--disable"
    assert result["contract"]["review_selection"] == choice
    assert result["runtime"]["effective"]["model"]["value"] == model
    assert result["runtime"]["effective"]["effort"]["value"] == "xhigh"
    assert result["runtime"]["effective"]["service_tier"]["status"] == "unavailable"
    assert json.loads((h.root / "initial/review-selection.json").read_text()) == choice


@pytest.mark.parametrize(
    "scenario,reason",
    [
        ("wrong_model", "metadata_error"),
        ("wrong_effort", "metadata_error"),
        ("model_unavailable", "cli_error"),
    ],
)
def test_review_runtime_failure_never_retries_another_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str, reason: str
) -> None:
    h = Harness(tmp_path)
    config, path, _ = choice_files(h, monkeypatch, "gpt-6.1-sol")
    proc = h.run(
        scenario=scenario,
        extra=(
            "--profile",
            "review",
            "--config-file",
            str(config),
            "--review-selection-file",
            str(path),
        ),
    )
    assert proc.returncode != 0
    assert h.result()["turn_end_reason"] == reason
    if scenario == "model_unavailable":
        assert "not available" in h.result()["error"]
        assert h.result()["runtime"]["effective"]["model"]["status"] == "unavailable"
        assert not (h.root / "initial/final.md").exists()
    calls = [
        json.loads(line)
        for line in (h.root / "invocations.jsonl").read_text().splitlines()
    ]
    assert len(calls) == 1
    assert calls[0][calls[0].index("--model") + 1] == "gpt-6.1-sol"


def test_review_resume_preserves_choice_and_rejects_policy_change_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = Harness(tmp_path)
    config, path, choice = choice_files(h, monkeypatch, "gpt-6.1-sol")
    mutable_config = h.root / "mutable-review.conf"
    mutable_config.write_bytes(config.read_bytes())
    config = mutable_config
    initial = (
        "--profile",
        "review",
        "--config-file",
        str(config),
        "--review-selection-file",
        str(path),
    )
    assert h.run(extra=initial).returncode == 0
    resume = (
        "--profile",
        "review",
        "--config-file",
        str(config),
        "--resume-from",
        str(h.root / "initial"),
    )
    proc = h.run("continued", action="resume", extra=resume)
    assert proc.returncode == 0, proc.stderr
    assert h.result("continued")["contract"]["review_selection"] == choice
    assert (
        h.result("continued")["runtime"]["session_id"]
        == h.result()["runtime"]["session_id"]
    )
    alternate = h.root / "alternate.json"
    changed = json.loads(path.read_text())
    changed["selected"]["model"] = "gpt-6-astra"
    alternate.write_text(json.dumps(changed))
    changed_run = h.run(
        "different-choice",
        action="resume",
        extra=(
            "--profile",
            "review",
            "--config-file",
            str(config),
            "--resume-from",
            str(h.root / "continued"),
            "--review-selection-file",
            str(alternate),
        ),
    )
    assert changed_run.returncode == 2 and "saved contract" in changed_run.stderr
    assert not (h.root / "different-choice").exists()
    config.write_text(
        config.read_text().replace(
            'REVIEW_SELECTION_MODE="select"', 'REVIEW_SELECTION_MODE="fixed"'
        )
    )
    proc = h.run(
        "rejected",
        action="resume",
        extra=(
            "--profile",
            "review",
            "--config-file",
            str(config),
            "--resume-from",
            str(h.root / "continued"),
        ),
    )
    assert proc.returncode == 2
    assert "policy changed" in proc.stderr
    assert not (h.root / "rejected").exists()
    assert len((h.root / "invocations.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize(
    "agent,profile,mode",
    [
        ("codex", None, "write"),
        ("codex", "review", "write"),
        ("codex", "ops_scout", "read-only"),
    ],
)
def test_review_selection_cannot_expand_other_profiles_or_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent: str,
    profile: str | None,
    mode: str,
) -> None:
    h = Harness(tmp_path)
    config, path, _ = choice_files(h, monkeypatch, "gpt-6.1-sol")
    extra = (
        "--mode",
        mode,
        "--config-file",
        str(config),
        "--review-selection-file",
        str(path),
        *(["--profile", profile] if profile else []),
    )
    proc = h.run(agent=agent, extra=extra)
    assert proc.returncode == 2 and "read-only Codex review" in proc.stderr
    assert not (h.root / "invocations.jsonl").exists()
    assert not (h.root / "initial").exists()
