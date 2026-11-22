"""Real shared runner with offline Codex: choices cannot change its contract."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_agent_runtime import CONFIG, Harness
from tests.support.agents_ops.model_pins import (
    OLD_ASTRA,
    OLD_SOL,
    REVIEW_FIXED_MODEL,
    REVIEW_OTHER_MODEL,
    previous_config,
)


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
    if mode == "select":
        # Exercise persisted select-policy contracts using an explicit fixture.
        config = h.root / "select.conf"
        config.write_text(
            CONFIG.read_text()
            .replace('MODE="fixed"', 'MODE="select"')
            .replace(
                f'REVIEW_CODEX_ALLOWED_PAIRS="{REVIEW_FIXED_MODEL}:xhigh"',
                f'REVIEW_CODEX_ALLOWED_PAIRS="{REVIEW_FIXED_MODEL}:xhigh,{REVIEW_OTHER_MODEL}:xhigh"',
            )
        )
    choice = settings.choose(
        settings.load_policy(config),
        model=model,
        reason="人間の指定"
        if human
        else "表示文言だけの修正で該当画面で確認できるためSolを選ぶ。"
        if model == REVIEW_OTHER_MODEL
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
        (REVIEW_OTHER_MODEL, "select", False),
        (REVIEW_FIXED_MODEL, "select", False),
        (REVIEW_FIXED_MODEL, "select", True),
        (REVIEW_FIXED_MODEL, "fixed", False),
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
        ("wrong_model", "completed"),
        ("wrong_effort", "completed"),
        ("model_unavailable", "cli_error"),
    ],
)
def test_review_observes_effective_settings_without_switching_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str, reason: str
) -> None:
    h = Harness(tmp_path)
    config, path, _ = choice_files(h, monkeypatch, REVIEW_OTHER_MODEL)
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
    assert (proc.returncode != 0) == (scenario == "model_unavailable")
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
    assert calls[0][calls[0].index("--model") + 1] == REVIEW_OTHER_MODEL


def test_review_resume_preserves_choice_and_rejects_policy_change_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = Harness(tmp_path)
    config, path, choice = choice_files(h, monkeypatch, REVIEW_OTHER_MODEL)
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
    changed["selected"]["model"] = REVIEW_FIXED_MODEL
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
    "mode,model",
    [("select", OLD_SOL), ("select", OLD_ASTRA), ("fixed", OLD_ASTRA)],
    ids=["selected-sol", "selected-astra", "fixed-astra"],
)
def test_review_resume_keeps_saved_selection_after_pin_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, model: str
) -> None:
    h = Harness(tmp_path)
    monkeypatch.syspath_prepend(str(CONFIG.parents[1]))
    settings = importlib.import_module("lib.review_settings")
    old_config = previous_config(h.root / "previous.conf", review_mode=mode)
    profile = ("--profile", "review")
    old_choice: tuple[str, ...] = ()
    if mode == "select":
        choice = settings.choose(
            settings.load_policy(old_config), model=model, reason="Saved decision."
        )
        path = h.root / "previous-choice.json"
        path.write_text(json.dumps(choice))
        old_choice = ("--review-selection-file", str(path))
        rejected = h.run("fresh", extra=(*profile, *old_choice))
        assert rejected.returncode == 2 and "policy changed" in rejected.stderr
        assert not (h.root / "fresh").exists() and not (h.root / "args.txt").exists()
    initial = h.run(extra=(*profile, "--config-file", str(old_config), *old_choice))
    assert initial.returncode == 0, initial.stderr
    proc = h.run(
        "continued",
        action="resume",
        extra=(*profile, "--resume-from", str(h.root / "initial")),
    )
    if model == OLD_ASTRA:
        assert proc.returncode == 2 and not (h.root / "continued").exists()
        return
    assert proc.returncode == 0, proc.stderr
    before, after = h.result(), h.result("continued")
    assert after["runtime"]["session_id"] == before["runtime"]["session_id"]
    assert h.args()[h.args().index("--model") + 1] == model
    assert after["contract"].get("review_selection") == before["contract"].get(
        "review_selection"
    )
    saved = after["contract"]["settings"]
    assert saved["selected"] == {
        "model": model,
        "effort": "xhigh",
        "service_tier": "default",
    }
    assert (
        saved["selection_sources"]
        == before["contract"]["settings"]["selection_sources"]
    )
    assert saved["model_continuation"] == "REVIEW_CODEX_CONTINUATION_PAIRS"


def test_review_resume_rejects_unlisted_saved_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = Harness(tmp_path)
    monkeypatch.syspath_prepend(str(CONFIG.parents[1]))
    settings = importlib.import_module("lib.review_settings")
    config = previous_config(h.root / "previous.conf")
    config.write_text(config.read_text().replace(f"{OLD_SOL}:xhigh", f"{OLD_SOL}:high"))
    choice = settings.choose(
        settings.load_policy(config),
        model=OLD_SOL,
        effort="high",
        reason="Previous policy.",
    )
    path = h.root / "previous-choice.json"
    path.write_text(json.dumps(choice))
    profile = ("--profile", "review")
    assert (
        h.run(
            extra=(
                *profile,
                "--config-file",
                str(config),
                "--review-selection-file",
                str(path),
            )
        ).returncode
        == 0
    )
    proc = h.run(
        "rejected",
        action="resume",
        extra=(*profile, "--resume-from", str(h.root / "initial")),
    )
    assert proc.returncode == 2 and "policy changed" in proc.stderr
    assert not (h.root / "rejected").exists()


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
    config, path, _ = choice_files(h, monkeypatch, REVIEW_OTHER_MODEL)
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
