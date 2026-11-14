"""Frozen review choices, historical evidence, and explicit policy boundaries."""

from __future__ import annotations

import argparse
import copy
import importlib
import re
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


def test_fixed_rollback_and_matching_choice_preserve_astra(
    settings: Any, tmp_path: Path
) -> None:
    config = tmp_path / "fixed.conf"
    config.write_text(
        (OPS / "lib/agents.conf")
        .read_text()
        .replace('REVIEW_SELECTION_MODE="select"', 'REVIEW_SELECTION_MODE="fixed"')
    )
    policy = settings.load_policy(config)
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


def test_repository_defaults_require_explicit_model_and_reason(settings: Any) -> None:
    policy = settings.load_policy()
    assert policy["mode"] == "select"
    with pytest.raises(ValueError, match="--review-model.*--model-reason-file"):
        settings.choose(policy)
    for reason in (None, "", " \n"):
        with pytest.raises(ValueError, match="--model-reason-file"):
            settings.choose(policy, model="gpt-6.1-sol", reason=reason)


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


def test_human_source_preserves_effort_when_model_is_inferred(settings: Any) -> None:
    chosen = settings.choose(
        settings.policy_from_pins(pins()),
        model="gpt-6-astra",
        specified_by="human_instruction",
        human_effort="xhigh",
        human_reference="人間の明示effortを確認済み",
        reason="人間の指定",
    )
    # Matching the fixed choice must not erase the explicit human constraint.
    assert chosen["human"] == {
        "model": "gpt-6-astra",
        "effort": "xhigh",
        "reference": "人間の明示effortを確認済み",
    }
    assert chosen["selected"]["effort"] == "xhigh"


def test_human_source_rejects_parent_effort_conflict(settings: Any) -> None:
    with pytest.raises(ValueError, match="parent effort contradicts"):
        settings.choose(
            settings.policy_from_pins(pins()),
            model="gpt-6-astra",
            effort="xhigh",
            specified_by="human_instruction",
            human_effort="max",
            human_reference="人間の元指定はmax",
            reason="親による上書きは認めない",
        )


def test_human_source_rejects_disallowed_effort(settings: Any) -> None:
    with pytest.raises(ValueError, match="allowed pair"):
        settings.choose(
            settings.policy_from_pins(pins()),
            model="gpt-6-astra",
            specified_by="human_instruction",
            human_effort="max",
            human_reference="人間の元指定はmax",
            reason="不許可の指定は拒否する",
        )


def test_parent_arguments_preserve_human_review_effort(
    settings: Any, tmp_path: Path
) -> None:
    choice = importlib.import_module("parent.review_choice")
    reason = tmp_path / "reason.md"
    reason.write_text("人間の明示指定を優先する")
    parser = argparse.ArgumentParser()
    choice.add_arguments(parser)
    args = parser.parse_args(
        [
            "--review-model-source",
            "human_instruction",
            "--review-model",
            "gpt-6-astra",
            "--human-review-effort",
            "max",
            "--human-model-reference",
            "人間の元指示を確認済み",
            "--model-reason-file",
            str(reason),
        ]
    )
    # This path reads policy/reason and binds CLI arguments only; it creates no
    # parent request, runtime or GitHub operation.
    with pytest.raises(ValueError, match="allowed pair"):
        choice.selection(None, args)


@pytest.mark.parametrize("bad", ["missing", "coverage", "approval", "duplicate"])
def test_handoff_candidate_envelope_cannot_claim_coverage_or_approval(
    settings: Any, bad: str
) -> None:
    candidate = {
        "status": "candidate_only",
        "source_run": "parent-1-" + "a" * 32,
        "validation": {
            **{
                key: "a" * 64
                for key in (
                    "source_request_sha256",
                    "source_execution_sha256",
                    "source_result_sha256",
                    "runtime_sha256",
                    "events_sha256",
                    "final_sha256",
                    "output_sha256",
                )
            },
            "current_prs": [],
        },
        "cross_findings": [],
        "prs": [{"number": 100, "findings": [], "evidence_waits": []}],
    }
    handoff: dict[str, Any] = {
        "source_run": candidate["source_run"],
        "source_request_sha256": "a" * 64,
        "source_result_sha256": "a" * 64,
        "scope": "追加確認",
        "candidates": [candidate],
    }
    if bad == "missing":
        del handoff["candidates"]
    elif bad == "coverage":
        handoff["candidates"][0]["prs"][0]["coverage"] = "reviewed"
    elif bad == "approval":
        handoff["candidates"][0]["status"] = "approved"
    else:
        handoff["candidates"].append(copy.deepcopy(candidate))
    with pytest.raises(ValueError):
        settings.choose(settings.policy_from_pins(pins()), handoff=handoff)


@pytest.mark.parametrize("kind", ["finding", "wait", "cross"])
def test_candidate_issues_block_only_the_affected_pr(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.syspath_prepend(str(OPS))
    monkeypatch.syspath_prepend(str(OPS / "review"))
    policy = importlib.import_module("parent_review_policy")
    # This pure policy reads the issue projection, whose source/schema proof is
    # exercised by the parent CLI E2E cases.
    candidate = {
        "cross_findings": [{"prs": [100]}] if kind == "cross" else [],
        "prs": [
            {
                "number": 100,
                "findings": [{"id": "pr-100-1"}] if kind == "finding" else [],
                "evidence_waits": [
                    {"question": "必要な実証", "required_evidence": "テスト結果"}
                ]
                if kind == "wait"
                else [],
            },
            {"number": 101, "findings": [], "evidence_waits": []},
        ],
    }
    result = {"request": {"review_selection": {"handoff": {"candidates": [candidate]}}}}
    assert policy.pending_handoff_candidates(result, 100) is True
    assert policy.pending_handoff_candidates(result, 101) is False


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
        re.sub(
            r'^REVIEW_SELECTION_MODE="[^"]*"$',
            'REVIEW_SELECTION_MODE="invalid"',
            runner.DEFAULT_CONFIG.read_text(),
            flags=re.MULTILINE,
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
        re.sub(
            r'^REVIEW_SELECTION_MODE="[^"]*"$',
            'REVIEW_SELECTION_MODE="invalid"',
            runner.DEFAULT_CONFIG.read_text(),
            flags=re.MULTILINE,
        )
    )
    assert agy.load_config(config) == agy.load_config(runner.DEFAULT_CONFIG)
    assert (
        runner.read_pins(config)["AGY_VERSION"]
        == runner.read_pins(runner.DEFAULT_CONFIG)["AGY_VERSION"]
    )
