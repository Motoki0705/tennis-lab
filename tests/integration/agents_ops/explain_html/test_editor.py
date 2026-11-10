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
    validation,
    writer_output,
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
    proposed = proposed.model_copy(
        update={"evidence_changes": context.authorized_evidence_changes}
    )
    env.outputs(proposed)
    result = env.adapter.edit(request, env.current, context)
    assert result.candidate.document.evidence[0].values == {"baseline_m": 11}
    assert result.candidate.document.evidence[1] == env.current.document.evidence[1]
    assert "cannot be determined" in result.candidate.document.html
    assert result.candidate.evidence_changes == context.authorized_evidence_changes
    calls = env.calls()
    turns = [call for call in calls if call["prompt"] is not None]
    assert len(turns) == 4
    assert len({call["cwd"] for call in turns}) == 4
    initial_template = json.loads(turns[0]["prompt"].split("\nINPUT_JSON\n")[1])[
        "candidate_template"
    ]
    assert initial_template["request"] == initial.request.model_dump(mode="json")
    assert set(initial_template) == {"request", "document", "change_summary"}
    writer_args = turns[0]["args"]
    writer_schema = json.loads(
        Path(writer_args[writer_args.index("--json-schema") + 1]).read_text()
    )
    assert set(writer_schema["properties"]) == {
        "request",
        "document",
        "change_summary",
    }
    edit_template = json.loads(turns[2]["prompt"].split("\nINPUT_JSON\n")[1])[
        "candidate_template"
    ]
    assert set(edit_template) == {"request", "document", "change_summary"}
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
    template = json.loads(turns[0]["prompt"].split("\nINPUT_JSON\n")[1])[
        "candidate_template"
    ]
    assert template["request"] == env.request.model_dump(mode="json")
    assert (
        template["document"]["evidence"] == context.model_dump(mode="json")["evidence"]
    )
    assert template["document"]["assets"] == context.model_dump(mode="json")["assets"]
    assert set(template) == {"request", "document", "change_summary"}
    assert "candidate_template" not in verified
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
    assert revision.changes == final.changes
    assert env.store.history("geometry")[0].changes == final.changes
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


@pytest.mark.parametrize("field", ["changes", "evidence_changes"])
def test_writer_change_declarations_are_rejected(editor_env: Any, field: str) -> None:
    env = editor_env
    damaged = writer_output(env.proposed)
    damaged[field] = []
    env.outputs(damaged, {})
    with pytest.raises(schema.ContractError, match="invalid_model_output"):
        env.adapter.edit(env.request, env.current, env.context)
    assert env.store.current("geometry") == env.current


@pytest.mark.parametrize("operation", ["question", "restructure"])
def test_compiled_changes_ignore_slot_normalization_and_unchanged_block_moves(
    editor_env: Any, operation: str
) -> None:
    env = editor_env
    renderer = visuals.managed_renderer()
    graph = schema.Figure(
        figure_id="comparison",
        content_id="limits",
        kind="graph",
        asset_ids=[renderer.asset.asset_id],
        caption="Hypothetical tracks.",
        alt="Two asymmetric position series.",
        provenance=schema.Provenance(kind="hypothetical", source="Synthetic inputs."),
        renderer=schema.Renderer(asset_id=renderer.asset.asset_id, version="2"),
        definition=COMPARISON,
    )
    slot_html = env.current.document.html.replace(
        "<html><body>", "<html><head></head><body>"
    ).replace("</aside>", '<figure data-figure-id="comparison"></figure></aside>')
    slotted = env.current.document.model_copy(
        update={"html": slot_html, "assets": [renderer.asset], "figures": [graph]}
    )
    initial = candidate(visuals.render_document(slotted))
    store = stores.RevisionStore(env.root / "compiled-store")
    current = store.import_initial(
        initial, {renderer.asset.asset_id: renderer.content}, validation(initial)
    )
    request = env.request.model_copy(
        update={
            "base_revision": current.revision_id,
            "target": schema.DocumentTarget()
            if operation == "restructure"
            else env.request.target,
            "instruction": "Move the comparison before the geometry."
            if operation == "restructure"
            else "Explain the rays near geometry; keep other content.",
        }
    )
    if operation == "question":
        raw_html = slot_html.replace(
            "</section>", "<p>Each calibrated image point defines a ray.</p></section>"
        )
        expected = [{"content_id": "geometry", "kind": "updated"}]
    else:
        geometry, limits = (
            slot_html.split("<body>")[1].split("</body>")[0].split("<aside", 1)
        )
        raw_html = (
            "<html><head></head><body><aside" + limits + geometry + "</body></html>"
        )
        expected = []
    raw = candidate(
        slotted.model_copy(update={"html": raw_html}), current=current, request=request
    )
    final = candidate(
        visuals.render_document(raw.document), current=current, request=request
    )
    env.outputs(
        raw,
        {
            "candidate_digest": schema.candidate_digest(final),
            "check": {"status": "passed"},
            "reviewed_evidence_ids": ["baseline", "cameras"],
        },
    )
    context = env.context.model_copy(
        update={"assets": slotted.assets, "figures": [graph]}
    )
    result = editor.GeminiEditor(
        env.adapter.gemini, compile_document=visuals.render_document
    ).edit(request, current, context)
    assert [
        item.model_dump(mode="json") for item in result.candidate.changes
    ] == expected
    assert {item.content_id for item in raw.changes} >= {"limits"}
    assert set(json.loads((env.root / "writer.json").read_text())) == {
        "request",
        "document",
        "change_summary",
    }
    verified = json.loads(
        [call for call in env.calls() if call["prompt"] is not None][-1][
            "prompt"
        ].split("\nINPUT_JSON\n")[1]
    )
    assert verified["candidate"]["changes"] == expected
    staged = store.stage(result.candidate)
    revision = store.commit(
        staged,
        result.validation.model_copy(
            update={"rendering": schema.CheckResult(status="passed")}
        ),
    )
    assert [item.model_dump(mode="json") for item in revision.changes] == expected
    assert store.history("geometry")[0].changes == revision.changes


@pytest.mark.parametrize("origin", ["writer", "context"])
def test_evidence_updates_need_trusted_authorization(
    editor_env: Any, origin: str
) -> None:
    env = editor_env
    fact = env.proposed.document.evidence[0].model_copy(
        update={"conditions": ["An unauthorized calibration condition."]}
    )
    evidence = [fact, env.proposed.document.evidence[1]]
    draft = writer_output(env.proposed)
    draft["document"]["evidence"] = [item.model_dump(mode="json") for item in evidence]
    env.outputs(draft, {})
    context = (
        env.context.model_copy(update={"evidence": evidence})
        if origin == "context"
        else env.context
    )
    with pytest.raises(
        schema.ContractError,
        match="invalid_request" if origin == "context" else "validation_failed",
    ):
        env.adapter.edit(env.request, env.current, context)
    turns = env.calls() if (env.root / "calls.jsonl").exists() else []
    assert len([call for call in turns if call["prompt"] is not None]) == (
        0 if origin == "context" else 1
    )
    assert env.store.current("geometry") == env.current


def test_local_scope_semantic_rejection_uses_computed_changes(editor_env: Any) -> None:
    env = editor_env
    proposed = candidate(
        env.proposed.document.model_copy(
            update={
                "html": env.proposed.document.html.replace(
                    "calibration is required.",
                    "calibration is required. Unrelated rewrite.",
                )
            }
        ),
        current=env.current,
        request=env.request,
    )
    env.outputs(
        proposed,
        {
            "candidate_digest": schema.candidate_digest(proposed),
            "check": {
                "status": "failed",
                "issues": [
                    {
                        "rule": "edit_scope",
                        "message": "Unrelated limits rewrite.",
                        "content_id": "limits",
                    }
                ],
            },
            "reviewed_evidence_ids": ["baseline", "cameras"],
        },
    )
    with pytest.raises(editor.SemanticValidationError) as rejected:
        env.adapter.edit(env.request, env.current, env.context)
    verified = json.loads(
        [call for call in env.calls() if call["prompt"] is not None][-1][
            "prompt"
        ].split("\nINPUT_JSON\n")[1]
    )
    assert verified["candidate"]["changes"] == [
        {"content_id": "geometry", "kind": "updated"},
        {"content_id": "limits", "kind": "updated"},
    ]
    assert rejected.value.validation.semantic.issues[0].rule == "edit_scope"
    assert env.store.current("geometry") == env.current
    assert len(env.store.history("geometry")) == 1


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
    with pytest.raises(
        editor.SemanticValidationError, match="validation_failed"
    ) as rejected:
        env.adapter.edit(env.request, env.current, env.context)
    validation = rejected.value.validation
    assert validation.candidate_digest == schema.candidate_digest(env.proposed)
    assert validation.semantic.status == "failed"
    assert validation.semantic.issues[0].rule == "unsupported_claim"
    assert not validation.passed
    staged = env.store.stage(env.proposed)
    # Even a rendering approval cannot clear the rejected semantic check.
    with pytest.raises(schema.ContractError, match="validation_failed"):
        env.store.commit(
            staged,
            validation.model_copy(
                update={"rendering": schema.CheckResult(status="passed")}
            ),
        )
    assert env.store.current("geometry") == env.current


@pytest.mark.parametrize("category", ["unsupported_claims", "technical_errors"])
def test_claim_findings_override_approval_and_survive_validation(
    editor_env: Any, category: str
) -> None:
    env = editor_env
    quote, reason = (
        ("300 fps以上", "No supplied product specification supports this rate.")
        if category == "unsupported_claims"
        else (
            "光学中心 (c_x, c_y)",
            "These image coordinates denote the principal point.",
        )
    )
    after = env.proposed.document.model_copy(
        update={
            "html": env.proposed.document.html.replace(
                "</section>", f"<p>{quote}</p></section>"
            )
        }
    )
    proposed = candidate(after, current=env.current, request=env.request)
    env.outputs(
        proposed,
        {
            "candidate_digest": schema.candidate_digest(proposed),
            "check": {"status": "passed", "issues": []},
            "reviewed_evidence_ids": ["baseline", "cameras"],
            category: [{"quote": quote, "reason": reason, "content_id": "geometry"}],
        },
    )
    with pytest.raises(
        editor.SemanticValidationError, match="validation_failed"
    ) as rejected:
        env.adapter.edit(env.request, env.current, env.context)
    validation = schema.parse(
        schema.ValidationResult, rejected.value.validation.model_dump_json()
    )
    assert validation.candidate_digest == schema.candidate_digest(proposed)
    assert validation.semantic.status == "failed"
    assert validation.rendering.status == "not_run"
    assert [
        (issue.rule, issue.content_id, issue.evidence_id)
        for issue in validation.semantic.issues
    ] == [
        (
            "unsupported_claim"
            if category == "unsupported_claims"
            else "technical_error",
            "geometry",
            None,
        )
    ]
    assert quote in validation.semantic.issues[0].message
    assert reason in validation.semantic.issues[0].message
    assert len([call for call in env.calls() if call["prompt"] is not None]) == 2
    staged = env.store.stage(proposed)
    with pytest.raises(schema.ContractError, match="validation_failed"):
        env.store.commit(
            staged,
            validation.model_copy(
                update={"rendering": schema.CheckResult(status="passed")}
            ),
        )
    assert env.store.current("geometry") == env.current
    assert len(env.store.history("geometry")) == 1


@pytest.mark.parametrize("missing", ["unsupported_claims", "technical_errors"])
def test_verdict_must_explicitly_report_both_claim_audits(
    editor_env: Any, missing: str
) -> None:
    env = editor_env
    path = env.root / "verifier.json"
    verdict = json.loads(path.read_text())
    del verdict[missing]
    path.write_text(json.dumps(verdict))
    with pytest.raises(schema.ContractError, match="invalid_model_output"):
        env.adapter.edit(env.request, env.current, env.context)
    assert env.store.current("geometry") == env.current


def test_observed_triangulation_findings_are_retained_with_the_original_candidate(
    editor_env: Any,
) -> None:
    env = editor_env
    fixture = (
        Path(__file__).resolve().parents[3]
        / "support/agents_ops/fixtures/explain_html/triangulation-initial.json"
    )
    inputs = json.loads(fixture.read_text())["input"]
    proposed = schema.parse(schema.Candidate, json.dumps(inputs["candidate"]))
    # Preserve the observed document; only initial change reasons now belong
    # to runtime rather than the historical writer response.
    proposed = proposed.model_copy(
        update={
            "evidence_changes": [
                item.model_copy(update={"reason": "Supplied initial evidence."})
                for item in proposed.evidence_changes
            ]
        }
    )
    context = editor.WritingContext.model_validate(inputs["context"])
    env.outputs(
        proposed,
        {
            "candidate_digest": schema.candidate_digest(proposed),
            "check": {"status": "failed", "issues": []},
            "reviewed_evidence_ids": ["projection", "trial"],
            "unsupported_claims": [
                {
                    "quote": "8〜12台以上の同期高速カメラ（300 fps以上）",
                    "reason": "No supplied product specifications.",
                    "content_id": "real-world-optimization",
                },
                {
                    "quote": "時速150〜200 km",
                    "reason": "No supplied speed measurements.",
                    "content_id": "calibration-sync",
                },
            ],
            "technical_errors": [
                {
                    "quote": "光学中心 $(c_x, c_y)$",
                    "reason": "Principal point is not the 3D camera centre.",
                    "content_id": "calibration-sync",
                    "evidence_id": "projection",
                },
                {
                    "quote": "視線が交差することは原理的にあり得ません",
                    "reason": "Asynchrony alone does not imply impossibility.",
                    "content_id": "calibration-sync",
                },
            ],
        },
    )
    with pytest.raises(editor.SemanticValidationError) as rejected:
        env.adapter.write_initial(proposed.request, context)
    validation = rejected.value.validation
    assert validation.candidate_digest == schema.candidate_digest(proposed)
    assert [issue.rule for issue in validation.semantic.issues] == [
        "unsupported_claim",
        "unsupported_claim",
        "technical_error",
        "technical_error",
    ]
    assert validation.semantic.issues[2].evidence_id == "projection"
    assert all(
        issue.content_id and issue.message for issue in validation.semantic.issues
    )
    assert not validation.passed
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


def _slot_repair_inputs(env: Any) -> tuple[Any, Any, Any]:
    renderer = visuals.managed_renderer()
    figure = schema.Figure(
        figure_id="comparison",
        content_id="limits",
        kind="graph",
        asset_ids=[renderer.asset.asset_id],
        caption="Hypothetical tracks.",
        alt="Two different position series.",
        provenance=schema.Provenance(kind="hypothetical", source="Synthetic inputs."),
        renderer=schema.Renderer(asset_id=renderer.asset.asset_id, version="2"),
        definition=COMPARISON,
    )
    raw_document = env.proposed.document.model_copy(
        update={
            "html": env.proposed.document.html.replace(
                "<html><body>", "<html><head></head><body>"
            ),
            "assets": [renderer.asset],
            "figures": [figure],
        }
    )
    fixed_document = raw_document.model_copy(
        update={
            "html": raw_document.html.replace(
                "</aside>", '<figure data-figure-id="comparison"></figure></aside>'
            )
        }
    )
    fixed = candidate(fixed_document, current=env.current, request=env.request)
    # A literal malformed output cannot be created through candidate(), whose
    # change calculator itself checks the broken figure reference.
    broken = writer_output(fixed)
    broken["document"]["html"] = raw_document.html
    context = env.context.model_copy(
        update={"assets": [renderer.asset], "figures": [figure]}
    )
    return broken, fixed, context


@pytest.mark.parametrize(
    "problem", ["missing_slot", "missing_metadata", "wrong_content_block"]
)
def test_figure_correspondence_gets_one_diagnostic_repair_and_final_verification(
    editor_env: Any,
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    env = editor_env
    broken, fixed, context = _slot_repair_inputs(env)
    if problem != "missing_slot":
        broken = writer_output(fixed)
        if problem == "missing_metadata":
            broken["document"]["figures"] = []
        else:
            slot = '<figure data-figure-id="comparison"></figure>'
            broken["document"]["html"] = (
                broken["document"]["html"]
                .replace(slot, "")
                .replace("</section>", slot + "</section>")
            )
    timeouts = []
    turn = env.adapter.gemini.turn

    def recorded_turn(*args: Any, **kwargs: Any) -> Any:
        timeouts.append(kwargs["timeout"])
        return turn(*args, **kwargs)

    monkeypatch.setattr(env.adapter.gemini, "turn", recorded_turn)
    final = candidate(
        visuals.render_document(fixed.document),
        current=env.current,
        request=env.request,
    )
    env.outputs(
        broken,
        {
            "candidate_digest": schema.candidate_digest(final),
            "check": {"status": "passed", "issues": []},
            "reviewed_evidence_ids": ["baseline", "cameras"],
        },
    )
    (env.root / "repair.json").write_text(json.dumps(writer_output(fixed)))
    adapter = editor.GeminiEditor(
        env.adapter.gemini, compile_document=visuals.render_document
    )
    result = adapter.edit(env.request, env.current, context)
    assert result.candidate == final
    assert len(result.repairs) == 1
    assert timeouts == [900, 600, 240]
    turns = [call for call in env.calls() if call["prompt"] is not None]
    assert len(turns) == 3
    repair = json.loads(turns[1]["prompt"].split("\nINPUT_JSON\n")[1])
    expected = {"figure_id": "comparison", "problem": problem}
    if problem != "missing_metadata":
        expected["content_id"] = "limits"
    assert repair["figure_repair"]["issues"] == [expected]
    assert repair["figure_repair"]["attempt"] == 1
    verified = json.loads(turns[2]["prompt"].split("\nINPUT_JSON\n")[1])
    assert verified["candidate"] == final.model_dump(mode="json")
    assert "figure_repair" not in verified
    visuals.validate_visuals(
        final.document, {"visuals_v2": visuals.managed_renderer().content}
    )
    assert env.store.current("geometry") == env.current


@pytest.mark.parametrize(
    "damage,code",
    [
        ("missing", "invalid_model_output"),
        ("evidence", "validation_failed"),
        ("script", "invalid_document"),
        ("cancelled", "cancelled"),
    ],
)
def test_slot_repair_cannot_retry_again_or_bypass_checks(
    editor_env: Any,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
    code: str,
) -> None:
    from threading import Event

    env = editor_env
    broken, fixed, context = _slot_repair_inputs(env)
    repaired = writer_output(fixed)
    if damage == "missing":
        repaired = broken
    elif damage == "evidence":
        repaired["document"]["purpose"] = "Changed protected purpose."
    elif damage == "script":
        repaired["document"]["html"] = repaired["document"]["html"].replace(
            "</body>", "<script>evil()</script></body>"
        )
    env.outputs(broken, {})
    (env.root / "repair.json").write_text(json.dumps(repaired))
    cancel = Event()
    turn = env.adapter.gemini.turn

    def checked_turn(*args: Any, **kwargs: Any) -> Any:
        if '"figure_repair"' in args[0] and damage == "cancelled":
            cancel.set()
        return turn(*args, **kwargs)

    monkeypatch.setattr(env.adapter.gemini, "turn", checked_turn)
    adapter = editor.GeminiEditor(
        env.adapter.gemini, compile_document=visuals.render_document
    )
    with pytest.raises(schema.ContractError, match=code):
        adapter.edit(env.request, env.current, context, cancel=cancel)
    turns = [call for call in env.calls() if call["prompt"] is not None]
    assert len(turns) == (1 if damage == "cancelled" else 2)
    assert env.store.current("geometry") == env.current


def test_writer_and_verifier_have_independent_time_budgets(
    editor_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = editor_env
    elapsed = [0.0]
    timeouts = []
    turn = env.adapter.gemini.turn

    def timed_turn(*args: Any, **kwargs: Any) -> Any:
        timeouts.append(kwargs["timeout"])
        result = turn(*args, **kwargs)
        elapsed[0] += 850 if len(timeouts) == 1 else 200
        return result

    import time

    monkeypatch.setattr(time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(env.adapter.gemini, "turn", timed_turn)
    result = env.adapter.edit(env.request, env.current, env.context)
    assert result.candidate == env.proposed
    assert timeouts == [900, 240]
