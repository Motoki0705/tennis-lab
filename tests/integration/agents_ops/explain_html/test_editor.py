"""Candidate/evidence/verifier boundaries through executable agy and a real store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.agents_ops.explain_editor import (
    HTML,
    SOURCE,
    candidate,
    document,
    gemini,
    schema,
    stores,
)
from tests.support.agents_ops.explain_editor import (
    editor_env as editor_env,
)

pytestmark = pytest.mark.integration


def test_initial_and_explicit_numeric_update_keep_unknowns_and_require_rendering(
    editor_env: Any,
) -> None:
    env = editor_env
    initial = candidate(document())
    env.outputs(initial)
    first = env.adapter.write_initial(initial.request, env.context)
    assert first.candidate == initial
    assert first.validation.semantic.status == "passed"
    assert not first.validation.passed
    assert first.writer.selected_effort == first.writer.observed_effort == "high"
    assert first.verifier.observed_model == "Gemini 3.8 Flash (High)"
    # A trusted service must still supply a real rendering check before commit.
    initial_store = stores.RevisionStore(env.root / "initial-store")
    staged = initial_store.stage(first.candidate)
    with pytest.raises(schema.ContractError, match="validation_failed"):
        initial_store.commit(staged, first.validation)

    data = env.current.document.model_dump(mode="json")
    data["evidence"][0]["values"]["baseline_m"] = 11
    data["evidence"][0]["statement"] = "The baseline is 11 m."
    data["html"] = HTML.replace("7 m", "11 m").replace(
        "</section>",
        "<p>Depth cannot be determined without pixel measurements.</p></section>",
    )
    after = schema.parse(schema.Document, json.dumps(data))
    request = env.request.model_copy(
        update={
            "instruction": "Correct baseline to 11 m using the supplied new calibration; "
            "explain which information is missing to calculate depth.",
        }
    )
    proposed = candidate(after, current=env.current, request=request)
    context = env.context.model_copy(
        update={
            "evidence": after.evidence,
            "authorized_evidence_changes": [
                schema.EvidenceChange(
                    evidence_id="baseline",
                    kind="updated",
                    reason="New supplied calibration.",
                )
            ],
        }
    )
    env.outputs(proposed)
    result = env.adapter.edit(request, env.current, context)
    assert result.candidate.document.evidence[0].values == {"baseline_m": 11}
    assert result.candidate.document.evidence[1] == env.current.document.evidence[1]
    assert "cannot be determined" in result.candidate.document.html
    calls = env.calls()
    turns = [call for call in calls if call["prompt"] is not None]
    assert len(turns) == 4
    assert len({call["cwd"] for call in turns}) == 4
    for call in calls:
        args = call["args"]
        if args != ["--version"]:
            assert args[args.index("--model") + 1] == "gemini-3.8-flash-high"
            assert args[args.index("--effort") + 1] == "high"
    assert env.store.current("geometry") == env.current


def test_long_local_edit_delivers_complete_context_to_both_calls(
    editor_env: Any,
) -> None:
    env = editor_env
    preferences = "Keep detailed explanations. 日本語。" * 6000
    context = env.context.model_copy(update={"preferences": preferences})
    result = env.adapter.edit(env.request, env.current, context)
    assert result.candidate == env.proposed
    turns = [call for call in env.calls() if call["prompt"] is not None]
    assert len(turns) == 2
    for call in turns:
        inputs = json.loads(call["prompt"].split("\nINPUT_JSON\n", 1)[1])
        assert inputs["context"]["preferences"] == preferences
        assert inputs["current_revision"] == env.current.model_dump(mode="json")
        assert inputs["request"] == env.request.model_dump(mode="json")
        assert inputs["context"]["change_summaries"] == ["Initial evidence."]
        assert "-p" not in call["args"]
    assert "candidate" not in json.loads(turns[0]["prompt"].split("\nINPUT_JSON\n")[1])
    assert "candidate" in json.loads(turns[1]["prompt"].split("\nINPUT_JSON\n")[1])
    for path in (env.root / "private").rglob("*"):
        assert path.stat().st_mode & 0o077 == 0
    assert all(not Path(call["cwd"]).exists() for call in turns)


@pytest.mark.parametrize("mismatch", ["model", "effort"])
def test_model_and_effort_mismatch_cannot_return_a_candidate(
    editor_env: Any,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    env = editor_env
    with pytest.raises(schema.ContractError, match="model_mismatch"):
        gemini.Gemini(env.root / "unused", **{mismatch: "other"})
    monkeypatch.setenv("FAKE_EXPLAIN_CASE", mismatch)
    monkeypatch.setenv("ISSUE_SUPPLEMENT_AGY_MODEL", "other")
    with pytest.raises(schema.ContractError, match="model_mismatch"):
        env.adapter.edit(env.request, env.current, env.context)
    assert env.store.current("geometry") == env.current


@pytest.mark.parametrize("damage", ["number", "source"])
def test_protected_number_and_source_changes_stop_before_verifier(
    editor_env: Any,
    damage: str,
) -> None:
    env = editor_env
    data = env.proposed.document.model_dump(mode="json")
    data["html"] = (
        data["html"].replace("7 m", "9 m")
        if damage == "number"
        else data["html"].replace(f'<a href="{SOURCE}">Calibration source</a>', "")
    )
    proposed = candidate(
        schema.parse(schema.Document, json.dumps(data)),
        current=env.current,
        request=env.request,
    )
    env.outputs(proposed)
    with pytest.raises(schema.ContractError, match="validation_failed"):
        env.adapter.edit(env.request, env.current, env.context)
    assert len([call for call in env.calls() if call["prompt"] is not None]) == 1
    assert env.store.current("geometry") == env.current


def test_incomplete_candidate_change_accounting_is_rejected(editor_env: Any) -> None:
    env = editor_env
    damaged = env.proposed.model_dump(mode="json")
    damaged["changes"] = []
    env.outputs(damaged, {})
    with pytest.raises(schema.ContractError, match="invalid_model_output"):
        env.adapter.edit(env.request, env.current, env.context)
    assert env.store.current("geometry") == env.current


def test_verifier_rejection_is_not_a_committable_result(editor_env: Any) -> None:
    env = editor_env
    env.outputs(
        env.proposed,
        {
            "candidate_digest": schema.candidate_digest(env.proposed),
            "check": {
                "status": "failed",
                "issues": [
                    {
                        "rule": "unsupported_claim",
                        "message": "Depth is not supported by these inputs.",
                    }
                ],
            },
            "reviewed_evidence_ids": ["baseline", "cameras"],
        },
    )
    with pytest.raises(schema.ContractError, match="validation_failed"):
        env.adapter.edit(env.request, env.current, env.context)
    assert env.store.current("geometry") == env.current


@pytest.mark.parametrize(
    "damage,code",
    [
        ("schema", "invalid_model_output"),
        ("digest", "validation_mismatch"),
    ],
)
def test_malformed_or_unbound_verdict_is_rejected(
    editor_env: Any,
    damage: str,
    code: str,
) -> None:
    env = editor_env
    verdict = json.loads((env.root / "verifier.json").read_text())
    if damage == "schema":
        verdict["check"]["status"] = "probably"
    else:
        verdict["candidate_digest"] = "0" * 64
    env.outputs(env.proposed, verdict)
    with pytest.raises(schema.ContractError, match=code):
        env.adapter.edit(env.request, env.current, env.context)
    assert env.store.current("geometry") == env.current
