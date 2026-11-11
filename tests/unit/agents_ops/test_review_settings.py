"""Frozen review choices, historical evidence, and explicit policy boundaries."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any

import pytest

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(OPS))
    return importlib.import_module("lib.review_settings")


def pins(mode: str = "fixed") -> dict[str, str]:
    return {
        "REVIEW_SELECTION_MODE": mode,
        "REVIEW_SELECTION_POLICY_VERSION": "1",
        "REVIEW_CODEX_ALLOWED_PAIRS": "gpt-6.1-sol:xhigh,gpt-6-astra:xhigh",
        "REVIEW_CODEX_MODEL": "gpt-6-astra",
        "REVIEW_CODEX_EFFORT": "xhigh",
        "REVIEW_CODEX_SERVICE_TIER": "default",
    }


def test_fixed_default_and_matching_choice_preserve_astra(settings: Any) -> None:
    policy = settings.policy_from_pins(pins())
    implicit = settings.choose(policy)
    assert implicit["selected"] == {
        "model": "gpt-6-astra",
        "effort": "xhigh",
        "service_tier": "default",
    }
    assert implicit["specified_by"] == "settings"
    explicit = settings.choose(policy, model="gpt-6-astra", reason="固定値を明示")
    assert explicit["selected"] == implicit["selected"]
    assert explicit["specified_by"] == "parent_choice"
    with pytest.raises(ValueError, match="fixed"):
        settings.choose(policy, model="gpt-6.1-sol", reason="通常の確認")


def test_fixed_effort_stays_pinned_when_select_would_be_ambiguous(
    settings: Any,
) -> None:
    data = pins()
    data["REVIEW_CODEX_ALLOWED_PAIRS"] += ",gpt-6-astra:high"
    assert (
        settings.choose(settings.policy_from_pins(data))["selected"]["effort"]
        == "xhigh"
    )
    data["REVIEW_SELECTION_MODE"] = "select"
    with pytest.raises(ValueError, match="unambiguous"):
        settings.choose(
            settings.policy_from_pins(data), model="gpt-6-astra", reason="比較確認"
        )


@pytest.mark.parametrize("model", ["gpt-6.1-sol", "gpt-6-astra"])
def test_select_requires_a_reason_and_an_allowed_pair(
    settings: Any, model: str
) -> None:
    policy = settings.policy_from_pins(pins("select"))
    with pytest.raises(ValueError, match="reason"):
        settings.choose(policy, model=model)
    selected = settings.choose(policy, model=model, reason="対象の難しさを確認")
    assert selected["selected"]["model"] == model
    assert selected["selected"]["effort"] == "xhigh"
    with pytest.raises(ValueError, match="allowed"):
        settings.choose(policy, model=model, effort="max", reason="不許可の組")


def test_human_constraint_cannot_be_overridden_by_parent(settings: Any) -> None:
    policy = settings.policy_from_pins(pins("select"))
    args = {
        "human_model": "gpt-6-astra",
        "human_reference": "人間の元指示を親が確認済み",
        "reason": "人間の明示指定",
    }
    chosen = settings.choose(policy, **args)
    assert chosen["selected"]["model"] == "gpt-6-astra"
    assert chosen["specified_by"] == "human_instruction"
    with pytest.raises(ValueError, match="human"):
        settings.choose(policy, model="gpt-6.1-sol", **args)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_requirements_check_is_preserved_for_both_authorized_versions(
    monkeypatch: pytest.MonkeyPatch, version: int
) -> None:
    monkeypatch.syspath_prepend(str(OPS))
    execution = importlib.import_module("parent.execution")
    request = {
        "schema_version": version,
        "parent_issue": 1,
        "requirements": {"1": "a" * 64},
    }
    if version == 1:
        execution.require_requirement_targets(request, {2})
    else:
        with pytest.raises(ValueError, match="missing"):
            execution.require_requirement_targets(request, {2})


def test_unknown_request_version_cannot_skip_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(OPS))
    execution = importlib.import_module("parent.execution")
    with pytest.raises(ValueError, match="unknown"):
        execution.require_requirement_targets({"schema_version": 4}, {2})


@pytest.mark.parametrize("change", ["model", "effort", "missing", "tier"])
def test_runtime_must_match_the_frozen_choice(settings: Any, change: str) -> None:
    choice = settings.choose(
        settings.policy_from_pins(pins("select")),
        model="gpt-6.1-sol",
        reason="定型的な確認",
    )
    runtime = {
        "contract": {
            "mode": "read-only",
            "review_selection": choice,
            "settings": {
                "agent": "codex",
                "profile": "review",
                "selected": choice["selected"],
            },
        },
        "runtime": {
            "effective": {
                "model": {
                    "status": "observed",
                    "value": "gpt-6.1-sol",
                    "source": "transcript",
                },
                "effort": {
                    "status": "observed",
                    "value": "xhigh",
                    "source": "transcript",
                },
                "service_tier": {
                    "status": "unavailable",
                    "value": None,
                    "reason": "not exposed",
                },
            }
        },
    }
    request = {"schema_version": 3, "review_selection": choice}
    settings.verify_runtime(request, runtime)
    altered = copy.deepcopy(runtime)
    effective = altered["runtime"]["effective"]
    if change == "missing":
        effective["model"] = {
            "status": "unavailable",
            "value": None,
            "reason": "missing",
        }
    elif change == "tier":
        effective["service_tier"] = {
            "status": "observed",
            "value": "priority",
            "source": "transcript",
        }
    else:
        effective[change]["value"] = "gpt-6-astra" if change == "model" else "max"
    with pytest.raises(ValueError, match="effective"):
        settings.verify_runtime(request, altered)


@pytest.mark.parametrize(
    "mutation", ["mode", "version", "missing", "duplicate", "pair"]
)
def test_invalid_review_policy_is_never_replaced_with_defaults(
    settings: Any, mutation: str
) -> None:
    data = pins()
    if mutation == "mode":
        data["REVIEW_SELECTION_MODE"] = "automatic"
    elif mutation == "version":
        data["REVIEW_SELECTION_POLICY_VERSION"] = "2"
    elif mutation == "missing":
        del data["REVIEW_CODEX_ALLOWED_PAIRS"]
    elif mutation == "duplicate":
        data["REVIEW_CODEX_ALLOWED_PAIRS"] += ",gpt-6-astra:xhigh"
    else:
        data["REVIEW_CODEX_ALLOWED_PAIRS"] = "gpt-6-astra"
    with pytest.raises(ValueError):
        settings.policy_from_pins(data)


@pytest.mark.parametrize(
    "agent,profile,mode",
    [
        ("codex", None, "write"),
        ("claude", None, "write"),
        ("claude", "ops_report", "read-only"),
        ("codex", "ops_scout", "read-only"),
        ("codex", "issue_judgement", "read-only"),
    ],
)
def test_unrelated_profiles_ignore_review_policy_and_keep_argv_and_resume_contract(
    settings: Any, tmp_path: Path, agent: str, profile: str | None, mode: str
) -> None:
    runner = importlib.import_module("lib.agent_settings")
    base = runner.resolve(runner.DEFAULT_CONFIG, agent, profile, None, None, mode)
    config = tmp_path / "changed.conf"
    config.write_text(
        runner.DEFAULT_CONFIG.read_text().replace(
            'REVIEW_SELECTION_MODE="fixed"', 'REVIEW_SELECTION_MODE="invalid"'
        )
    )
    changed = runner.resolve(config, agent, profile, None, None, mode)
    for session in (None, "existing-session"):
        assert runner.command(
            base, mode, tmp_path, tmp_path / "out", session, []
        ) == runner.command(changed, mode, tmp_path, tmp_path / "out", session, [])
    assert runner.execution_contract({"settings": base}) == runner.execution_contract(
        {"settings": changed}
    )


def test_agy_and_explanation_pins_do_not_depend_on_review_policy(
    settings: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.syspath_prepend(str(OPS / "lib"))
    agy = importlib.import_module("agy_headless")
    runner = importlib.import_module("lib.agent_settings")
    config = tmp_path / "changed.conf"
    config.write_text(
        runner.DEFAULT_CONFIG.read_text().replace(
            'REVIEW_SELECTION_MODE="fixed"', 'REVIEW_SELECTION_MODE="invalid"'
        )
    )
    assert agy.load_config(config) == agy.load_config(runner.DEFAULT_CONFIG)
    assert (
        runner.read_pins(config)["AGY_VERSION"]
        == runner.read_pins(runner.DEFAULT_CONFIG)["AGY_VERSION"]
    )
