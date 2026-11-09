"""Candidate/evidence/verifier boundaries through executable agy and a real store."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from tests.support.agents_ops.explain_editor import (
    HTML,
    SOURCE,
    candidate,
    document,
    editor,
    gemini,
    schema,
    stores,
)
from tests.support.agents_ops.explain_editor import (
    editor_env as editor_env,
)
from tests.support.agents_ops.explain_html_visuals import COMPARISON, visuals

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
    slot = '<figure data-figure-id="rays"></figure>'
    figure = schema.Figure(
        figure_id="rays",
        content_id="geometry",
        kind="svg",
        asset_ids=[],
        caption="Two calibrated rays.",
        alt="Two rays intersect at a point.",
        provenance=schema.Provenance(kind="derived", source="Supplied calibration."),
    )
    raw = candidate(
        env.proposed.document.model_copy(
            update={
                "html": env.proposed.document.html.replace(
                    "</section>", slot + "</section>"
                ),
                "figures": [figure],
            }
        ),
        current=env.current,
        request=env.request,
    )
    payload = b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L1 1"/></svg>'
    asset = schema.Asset(
        asset_id="rays-svg",
        path="assets/rays-svg.svg",
        media_type="image/svg+xml",
        sha256=sha256(payload).hexdigest(),
        kind="svg",
        caption=figure.caption,
        alt=figure.alt,
        provenance=figure.provenance,
    )
    reject = True

    def compile_document(value: Any) -> Any:
        assert value == raw.document
        if reject:
            raise schema.ContractError("invalid_document")
        return value.model_copy(
            update={
                "html": value.html.replace(
                    slot,
                    '<figure data-figure-id="rays"><div data-content-id="compiled-view">'
                    '<img data-asset-id="rays-svg" src="assets/rays-svg.svg" '
                    f'alt="{figure.alt}"></div></figure>',
                ),
                "assets": [asset],
                "figures": [figure.model_copy(update={"asset_ids": [asset.asset_id]})],
            }
        )

    env.adapter = editor.GeminiEditor(
        env.adapter.gemini, compile_document=compile_document
    )
    context = env.context.model_copy(
        update={"preferences": preferences, "figures": [figure]}
    )
    env.outputs(raw)
    with pytest.raises(schema.ContractError, match="invalid_document"):
        env.adapter.edit(env.request, env.current, context)
    assert len([call for call in env.calls() if call["prompt"] is not None]) == 1
    assert env.store.current("geometry") == env.current
    assert len(env.store.history("geometry")) == 1

    # Explicit retry with a working compiler; there is no adapter retry/fallback.
    reject = False
    final = candidate(
        compile_document(raw.document), current=env.current, request=env.request
    )
    verdict = json.loads((env.root / "verifier.json").read_text())
    verdict["candidate_digest"] = schema.candidate_digest(final)
    env.outputs(raw, verdict)
    result = env.adapter.edit(env.request, env.current, context)
    assert result.candidate == final
    assert result.candidate.changes != raw.changes
    assert {change.content_id for change in result.candidate.changes} == {
        "geometry",
        "compiled-view",
    }
    assert schema.candidate_digest(raw) != result.validation.candidate_digest
    turns = [call for call in env.calls() if call["prompt"] is not None]
    assert len(turns) == 3
    for call in turns:
        inputs = json.loads(call["prompt"].split("\nINPUT_JSON\n", 1)[1])
        assert inputs["context"]["preferences"] == preferences
        assert inputs["current_revision"] == env.current.model_dump(mode="json")
        assert inputs["request"] == env.request.model_dump(mode="json")
        assert inputs["context"]["change_summaries"] == ["Initial evidence."]
        assert "-p" not in call["args"]
    assert "candidate" not in json.loads(turns[0]["prompt"].split("\nINPUT_JSON\n")[1])
    verified = json.loads(turns[-1]["prompt"].split("\nINPUT_JSON\n")[1])
    assert verified["candidate"] == final.model_dump(mode="json")
    assert verified["candidate_digest"] == schema.candidate_digest(final)
    assert env.store.current("geometry") == env.current
    staged = env.store.stage(result.candidate, {asset.asset_id: payload})
    assert staged.candidate_digest == result.validation.candidate_digest
    assert (
        env.store.staged_asset_bytes("geometry", staged.stage_id, asset.asset_id)
        == payload
    )
    # The fake compile/render fixture supplies its rendering verdict; production
    # must independently validate visuals and render the exact staged bytes.
    checked = result.validation.model_copy(
        update={"rendering": schema.CheckResult(status="passed")}
    )
    revision = env.store.commit(staged, checked)
    assert revision.document == final.document
    assert env.store.current("geometry") == revision
    with pytest.raises(TypeError, match="compile_document"):
        editor.GeminiEditor(env.adapter.gemini)
    for path in (env.root / "private").rglob("*"):
        assert path.stat().st_mode & 0o077 == 0
    assert all(not Path(call["cwd"]).exists() for call in turns)


def test_visual_compiler_binds_verification_and_commit_to_the_final_document(
    editor_env: Any,
) -> None:
    env = editor_env
    renderer = visuals.managed_renderer()
    figure = schema.Figure(
        figure_id="comparison",
        content_id="limits",
        kind="graph",
        asset_ids=[renderer.asset.asset_id],
        caption="Compare two hypothetical camera tracks.",
        alt="Two asymmetric illustrative position series.",
        provenance=schema.Provenance(kind="hypothetical", source="Synthetic inputs."),
        renderer=schema.Renderer(asset_id=renderer.asset.asset_id, version="2"),
        definition=COMPARISON,
    )
    request = env.request.model_copy(
        update={
            "target": schema.ContentTarget(content_id="limits"),
            "instruction": "Add a hypothetical comparison graph; keep facts and citations.",
        }
    )
    html = env.current.document.html.replace(
        "<html><body>",
        "<!doctype html><html><!-- harmless literal <head> -->"
        "<head><title>Camera tracks</title></head><body>",
    ).replace("</aside>", '<figure data-figure-id="comparison"></figure></aside>')
    raw = candidate(
        env.current.document.model_copy(
            update={"html": html, "assets": [renderer.asset], "figures": [figure]}
        ),
        current=env.current,
        request=request,
    )
    payloads = {renderer.asset.asset_id: renderer.content}
    # A structurally valid empty slot cannot pass the real visual validator.
    with pytest.raises(schema.ContractError, match="invalid_document"):
        visuals.validate_visuals(raw.document, payloads)
    final = candidate(
        visuals.render_document(raw.document), current=env.current, request=request
    )
    env.outputs(
        raw,
        {
            "candidate_digest": schema.candidate_digest(final),
            "check": {"status": "passed", "issues": []},
            "reviewed_evidence_ids": ["baseline", "cameras"],
        },
    )
    adapter = editor.GeminiEditor(
        env.adapter.gemini, compile_document=visuals.render_document
    )
    context = env.context.model_copy(
        update={"assets": [renderer.asset], "figures": [figure]}
    )
    result = adapter.edit(request, env.current, context)
    assert result.candidate == final
    assert schema.candidate_digest(raw) != result.validation.candidate_digest
    assert result.validation.candidate_digest == schema.candidate_digest(final)
    assert 'http-equiv="Content-Security-Policy"' in result.candidate.document.html
    assert "<script " in result.candidate.document.html
    assert (
        f'data-asset-id="{renderer.asset.asset_id}"' in result.candidate.document.html
    )
    assert 'data-values="1"' in result.candidate.document.html
    turns = [call for call in env.calls() if call["prompt"] is not None]
    assert len(turns) == 2
    verified = json.loads(turns[-1]["prompt"].split("\nINPUT_JSON\n")[1])
    assert verified["candidate"] == final.model_dump(mode="json")
    assert verified["candidate_digest"] == result.validation.candidate_digest
    assert env.store.current("geometry") == env.current
    staged = env.store.stage(result.candidate, payloads)
    assert staged.candidate_digest == result.validation.candidate_digest
    staged_document = env.store.staged_candidate("geometry", staged.stage_id).document
    staged_payloads = {
        asset.asset_id: env.store.staged_asset_bytes(
            "geometry", staged.stage_id, asset.asset_id
        )
        for asset in staged_document.assets
    }
    visuals.validate_visuals(staged_document, staged_payloads)
    assert staged_payloads == payloads
    # Meaning and browser rendering verdicts are test fakes. The real visual
    # validator above checks the compiled staged bundle without forging a verdict.
    checked = result.validation.model_copy(
        update={"rendering": schema.CheckResult(status="passed")}
    )
    revision = env.store.commit(staged, checked)
    assert revision.document == final.document
    assert env.store.current("geometry") == revision


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
