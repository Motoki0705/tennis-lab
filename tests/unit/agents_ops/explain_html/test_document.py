"""Structural/wire contracts; no store, subprocess, browser or external API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]
import pytest

from tests.support.agents_ops.explain_html import (
    HTML,
    candidate_for,
    changed_document,
    document_api,
    make_document,
    paths_api,
    revision_for,
    schema,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def document() -> Any:
    return make_document()


def test_initial_partial_and_restructured_layouts_share_identity_contract(
    document: Any,
) -> None:
    initial = candidate_for(document)
    document_api.validate_candidate(initial, None)
    previous = revision_for(document)
    updated = changed_document(document)
    edit = candidate_for(
        updated,
        before=previous,
        request_id="example",
        changes=[{"content_id": "geometry", "kind": "updated"}],
        evidence_changes=[],
        target={"kind": "content", "content_id": "geometry"},
    )
    document_api.validate_candidate(edit, previous)
    assert [
        (item.content_id, item.kind)
        for item in document_api.content_changes(document, updated)
    ] == [("geometry", "updated")]
    graph = updated.model_copy(deep=True)
    graph.figures[1].definition["data"] = [[0, 7], [1, 10], [3, 20]]
    assert [
        (item.content_id, item.kind)
        for item in document_api.content_changes(updated, graph)
    ] == [("limits", "updated")]
    # Move the aside before the derivation and change the surrounding layout.
    index = document_api.inspect_document(updated)
    overview, geometry, limits = [
        index.block_html(item) for item in ("overview", "geometry", "limits")
    ]
    html = f"<html><body><article>{overview}<div class='columns'>{limits}{geometry}</div></article></body></html>"
    reconstructed = updated.model_copy(update={"html": html})
    candidate = candidate_for(
        reconstructed,
        before=revision_for(updated),
        request_id="restructure",
        changes=[],
        evidence_changes=[],
    )
    document_api.validate_candidate(candidate, revision_for(updated))
    assert set(document_api.inspect_document(reconstructed).blocks) == {
        "overview",
        "geometry",
        "limits",
    }
    # Evidence deletion must be declared, independently of the HTML layout.
    missing = reconstructed.model_copy(update={"evidence": reconstructed.evidence[1:]})
    with pytest.raises(schema.ContractError, match="broken_reference"):
        document_api.inspect_document(missing)
    bad_accounting = edit.model_copy(update={"changes": []})
    with pytest.raises(schema.ContractError, match="invalid_candidate"):
        document_api.validate_candidate(bad_accounting, previous)


@pytest.mark.parametrize("namespace", ["content", "html", "asset", "evidence"])
def test_duplicate_identifiers_are_rejected(document: Any, namespace: str) -> None:
    duplicate = document.model_copy(deep=True)
    if namespace == "content":
        duplicate = duplicate.model_copy(
            update={
                "html": HTML.replace(
                    'data-content-id="limits"', 'data-content-id="geometry"'
                )
            }
        )
    elif namespace == "html":
        duplicate = duplicate.model_copy(
            update={
                "html": HTML.replace(
                    '<main class="grid">', '<main id="overview" class="grid">'
                )
            }
        )
    elif namespace == "asset":
        duplicate.assets.append(duplicate.assets[0])
    else:
        duplicate.evidence.append(duplicate.evidence[0])
    with pytest.raises(schema.ContractError, match="duplicate_id"):
        document_api.inspect_document(duplicate)


def test_selection_uses_dom_text_utf16_units_and_rejects_surrogate_splitting(
    document: Any,
) -> None:
    # A block containing an astral character and an HTML entity; offsets are
    # independently calculated from the browser string "😀 A & B A & B".
    changed = document.model_copy(
        update={
            "html": HTML.replace(
                "<p>😀 Intersect two rays. Then measure 7 m.</p>",
                "<p>😀 A &amp; B A &amp; B</p>",
            )
        }
    )
    request = schema.EditRequest(
        document_id="ray-geometry",
        base_revision="1" * 32,
        request_id="selection",
        instruction="Explain this expression.",
        target=schema.SelectionTarget(
            content_id="geometry",
            exact="A & B",
            prefix="😀 ",
            suffix=" A & B",
            start=12,
            end=17,
        ),
    )
    # geometry text also starts with "Geometry\n" (9 UTF-16 units).
    document_api.validate_request(request, revision_for(changed))
    split = request.model_copy(
        update={
            "target": schema.SelectionTarget(
                content_id="geometry", exact="😀", start=10, end=11
            )
        }
    )
    with pytest.raises(schema.ContractError, match="invalid_target"):
        document_api.validate_request(split, revision_for(changed))


@pytest.mark.parametrize("failure", ["missing", "quote", "position", "context"])
def test_missing_or_mismatched_targets_never_relocate(
    document: Any, failure: str
) -> None:
    target = {
        "kind": "selection",
        "content_id": "overview",
        "exact": "Two views",
        "prefix": "",
        "suffix": " locate",
        "start": 0,
        "end": 9,
    }
    if failure == "missing":
        target["content_id"] = "deleted"
    elif failure == "quote":
        target["exact"] = "two views"
    elif failure == "position":
        target["start"], target["end"] = 1, 10
    else:
        target["suffix"] = " were removed"
    request = schema.parse(
        schema.EditRequest,
        json.dumps(
            {
                "document_id": "ray-geometry",
                "base_revision": "1" * 32,
                "request_id": "selection",
                "instruction": "Explain this.",
                "target": target,
            }
        ),
    )
    with pytest.raises(schema.ContractError, match="invalid_target"):
        document_api.validate_request(request, revision_for(document))


@pytest.mark.parametrize("reference", ["fragment", "asset", "figure", "evidence"])
def test_broken_references_fail_before_publication(
    document: Any, reference: str
) -> None:
    data = document.model_dump(mode="json")
    if reference == "fragment":
        data["html"] = HTML.replace('href="#geometry"', 'href="#deleted"')
    elif reference == "asset":
        data["html"] = HTML.replace("assets/court.svg", "assets/missing.svg")
    elif reference == "figure":
        data["figures"][1]["renderer"]["asset_id"] = "missing"
    else:
        data["html"] = HTML.replace(
            'data-evidence-id="baseline"', 'data-evidence-id="missing"'
        )
    invalid = schema.parse(schema.Document, json.dumps(data))
    with pytest.raises(schema.ContractError, match="broken_reference"):
        document_api.inspect_document(invalid)


@pytest.mark.parametrize("escape", ["traversal", "absolute", "symlink"])
def test_artifact_paths_cannot_read_or_change_outside_files(
    tmp_path: Path, document: Any, escape: str
) -> None:
    root = tmp_path / "output"
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_bytes(b"must remain outside")
    with paths_api.SafeDirectory(root, create=True) as filesystem:
        if escape == "symlink":
            (root / "linked").symlink_to(outside, target_is_directory=True)
            path = "linked/secret.txt"
        else:
            path = "../outside/secret.txt" if escape == "traversal" else str(secret)
            data = document.model_dump(mode="json")
            data["assets"][0]["path"] = path
            with pytest.raises(schema.ContractError, match="unsafe_path"):
                document_api.inspect_document(
                    schema.parse(schema.Document, json.dumps(data))
                )
        with pytest.raises(schema.ContractError, match="unsafe_path"):
            filesystem.read(path)
        with pytest.raises(schema.ContractError, match="unsafe_path"):
            filesystem.write(path, b"overwrite")
        with pytest.raises(schema.ContractError, match="unsafe_path"):
            filesystem.atomic_write(path, b"overwrite")
    assert secret.read_bytes() == b"must remain outside"
    assert list(outside.iterdir()) == [secret]


def test_schema_examples_errors_and_job_states_are_shared_wire_contract() -> None:
    contract = schema.contract_schema()
    for model in (
        schema.Document,
        schema.Candidate,
        schema.ValidationResult,
        schema.EditRequest,
        schema.RestoreRequest,
        schema.ErrorResponse,
    ):
        example = schema.examples()[model.__name__]
        schema.parse(model, json.dumps(example))
        generated = {
            **contract["components"]["schemas"][model.__name__],
            "components": contract["components"],
        }
        jsonschema.Draft202012Validator.check_schema(generated)
        jsonschema.Draft202012Validator(generated).validate(example)
    assert contract["paths"]["/api/documents/{document_id}/requests"]["post"][
        "responses"
    ]["202"]
    assert "succeeded" not in schema.JOB_TRANSITIONS["queued"]
    assert schema.JOB_TRANSITIONS["validating"] >= {
        "succeeded",
        "failed",
        "cancelled",
        "interrupted",
    }
    assert all(
        not schema.JOB_TRANSITIONS[state]
        for state in ("succeeded", "failed", "cancelled", "interrupted")
    )
    error = schema.ContractError("stale_revision", current_revision="2" * 32)
    assert error.status_code == 409
    assert error.error.current_revision == "2" * 32
    assert "private" not in error.error.message
    invalid = {
        **schema.examples()["EditRequest"],
        "instruction": 123,
        "filesystem_path": "/private/secret",
    }
    with pytest.raises(schema.ContractError, match="invalid_request"):
        schema.parse(schema.EditRequest, json.dumps(invalid))
    job = {
        "document_id": "triangulation",
        "request_id": "example",
        "base_revision": "1" * 32,
        "state": "succeeded",
    }
    with pytest.raises(schema.ContractError, match="invalid_request"):
        schema.parse(schema.JobResponse, json.dumps(job))
    succeeded = schema.parse(
        schema.JobResponse,
        json.dumps(
            {**job, "revision_id": "2" * 32, "change_summary": "Added example."}
        ),
    )
    assert succeeded.revision_id != succeeded.base_revision
    schema.parse(
        schema.JobResponse,
        json.dumps(
            {
                **job,
                "state": "interrupted",
                "error": schema.ContractError("interrupted").error.model_dump(
                    mode="json"
                ),
            }
        ),
    )
