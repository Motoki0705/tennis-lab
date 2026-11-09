"""Real revision/filesystem boundaries; injected validation and save faults only."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any
from uuid import UUID

import pytest

from tests.support.agents_ops.explain_html import (
    HTML,
    PAYLOADS,
    candidate_for,
    changed_document,
    document_api,
    make_document,
    paths_api,
    publication_fault,
    schema,
    store_api,
    validation_for,
)
from tests.support.agents_ops.explain_html_visuals import (
    images,
    make_renderer_history,
    make_visual_bundle,
    png,
    visuals,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def initial(tmp_path: Path) -> Any:
    return initial_store(tmp_path, make_document(), PAYLOADS)


@pytest.fixture
def visual_initial(tmp_path: Path) -> Any:
    document, payloads = make_visual_bundle()
    return initial_store(tmp_path, document, payloads)


def initial_store(tmp_path: Path, document: Any, payloads: dict[str, bytes]) -> Any:
    root = tmp_path / "store"
    store = store_api.RevisionStore(root)
    candidate = candidate_for(document)
    revision = store.import_initial(candidate, payloads, validation_for(candidate))
    return root, store, revision


def edit_for(revision: Any, request_id: str = "add-example") -> Any:
    return candidate_for(
        changed_document(revision.document),
        before=revision,
        request_id=request_id,
        changes=[{"content_id": "geometry", "kind": "updated"}],
        evidence_changes=[],
    )


@pytest.mark.parametrize("phase", ["before", "after", "unreadable"])
def test_snapshot_publication_fault_reports_only_verified_exports(
    initial: Any, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    root, store, first = initial
    after = phase != "before"
    snapshot_id = "e" * 32
    monkeypatch.setattr(store_api, "uuid4", lambda: UUID(hex=snapshot_id))
    read_model = store._read_model

    def read_manifest(filesystem: Any, path: str, model: Any) -> Any:
        if phase == "unreadable" and path == f"exports/{snapshot_id}/manifest.json":
            raise OSError("Cannot verify the installed snapshot manifest.")
        return read_model(filesystem, path, model)

    with publication_fault(
        monkeypatch, operation="rename", destination=snapshot_id, after=after
    ) as fault:
        with monkeypatch.context() as patch:
            patch.setattr(store, "_read_model", read_manifest)
            if phase == "after":
                snapshot = store.snapshot("ray-geometry", first.revision_id)
                assert snapshot.snapshot_id == snapshot_id
                assert (
                    schema.parse(
                        schema.SnapshotResponse,
                        store.snapshot_file(snapshot_id, "manifest.json"),
                    )
                    == snapshot
                )
            else:
                with pytest.raises(schema.ContractError, match="storage_error"):
                    store.snapshot("ray-geometry", first.revision_id)
        if phase == "before":
            with pytest.raises(schema.ContractError, match="snapshot_not_found"):
                store.snapshot_file(snapshot_id, "manifest.json")
        assert fault.fired and fault.replaced == after
    assert store.current("ray-geometry") == first
    if after:
        assert store.snapshot_file(snapshot_id, "index.html") == HTML.encode()
        for asset_id, content in PAYLOADS.items():
            asset = next(
                item for item in first.document.assets if item.asset_id == asset_id
            )
            assert store.snapshot_file(snapshot_id, asset.path) == content


def test_partial_restructure_restore_keep_complete_bundles_and_invalidate_old_bases(
    visual_initial: Any,
) -> None:
    root, store, first = visual_initial
    with pytest.raises(schema.ContractError, match="image_generation_unavailable"):
        images.generate_image(
            "New image requested while reading.",
            asset_id="new_image",
            caption="New image.",
            alt="New concept.",
        )
    assert store.current("ray-geometry").revision_id == first.revision_id
    assert len(store.history("ray-geometry")) == 1
    candidate = edit_for(first)
    data = candidate.document.model_dump(mode="json")
    revised_svg = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 7 3"><circle cx="2" cy="1" r="1"/></svg>'
    data["assets"][0]["sha256"] = hashlib.sha256(revised_svg).hexdigest()
    data["evidence"][0]["conditions"].append("Rays must not be parallel.")
    data["figures"][3]["caption"] = "Read this image alongside the depth graph."
    document = visuals.render_document(schema.parse(schema.Document, json.dumps(data)))
    candidate = candidate.model_copy(
        update={
            "document": document,
            "changes": [
                schema.ContentChange(content_id="geometry", kind="updated"),
                schema.ContentChange(content_id="limits", kind="updated"),
            ],
            "evidence_changes": [
                schema.EvidenceChange(
                    evidence_id="baseline",
                    kind="updated",
                    reason="State the degeneracy condition.",
                )
            ],
        }
    )
    staged = store.stage(candidate, {"court": revised_svg})
    visuals.validate_visuals(
        store.staged_candidate("ray-geometry", staged.stage_id).document,
        {
            asset.asset_id: store.staged_asset_bytes(
                "ray-geometry", staged.stage_id, asset.asset_id
            )
            for asset in document.assets
        },
    )
    assert store.current("ray-geometry").revision_id == first.revision_id
    second = store.commit(staged, validation_for(candidate))
    # Save another result now; it must become stale after a later undo.
    delayed = candidate_for(
        second.document,
        before=second,
        request_id="delayed",
        changes=[],
        evidence_changes=[],
    )
    late_stage = store.stage(delayed)
    index = document_api.inspect_document(second.document)
    rearranged_html = (
        "<html><head><title>Rearranged geometry</title></head><body><div class='comparison'>"
        + index.block_html("limits")
        + index.block_html("overview")
        + index.block_html("geometry")
        + "</div></body></html>"
    )
    rearranged = visuals.render_document(
        second.document.model_copy(update={"html": rearranged_html})
    )
    full = candidate_for(
        rearranged,
        before=second,
        request_id="restructure",
        changes=[],
        evidence_changes=[],
    )
    third = store.commit(store.stage(full), validation_for(full))
    restore_request = schema.RestoreRequest(
        document_id="ray-geometry",
        base_revision=third.revision_id,
        request_id="undo",
        revision_id=first.revision_id,
    )
    restored = store.restore(restore_request)
    reopened = store_api.RevisionStore(root)
    current = reopened.current("ray-geometry")
    assert reopened.restore(restore_request).revision_id == restored.revision_id
    assert current.revision_id == restored.revision_id
    assert (
        len(
            {
                first.revision_id,
                second.revision_id,
                third.revision_id,
                restored.revision_id,
            }
        )
        == 4
    )
    assert (
        current.parent_revision == third.revision_id
        and current.restored_from == first.revision_id
    )
    assert current.document.html == first.document.html
    assert current.document.evidence[0].conditions == ["Synchronized views."]
    assert current.document.figures[1].definition["data"] == {
        "focal_px": 600,
        "baseline_m": 0.8,
        "disparity_px": 30,
        "sample_errors_px": [-10, 0, 10],
    }
    assert current.document.figures[3].caption == "Raster import fixture."
    assert (
        second.document.figures[3].caption
        == "Read this image alongside the depth graph."
    )
    assert reopened.asset_bytes("ray-geometry", current.revision_id, "concept") == png()
    assert (
        reopened.asset_bytes("ray-geometry", current.revision_id, "visuals_v2")
        == visuals.managed_renderer().content
    )
    assert (
        reopened.asset_bytes("ray-geometry", current.revision_id, "court")
        == PAYLOADS["court"]
    )
    assert (
        reopened.asset_bytes("ray-geometry", second.revision_id, "court") == revised_svg
    )
    assert [item.revision_id for item in reopened.history("ray-geometry")] == [
        restored.revision_id,
        third.revision_id,
        second.revision_id,
        first.revision_id,
    ]
    with pytest.raises(schema.ContractError, match="stale_revision"):
        reopened.commit(late_stage, validation_for(delayed))
    with pytest.raises(schema.ContractError, match="stale_revision"):
        reopened.stage(edit_for(first, "stale-after-undo"))
    new = edit_for(current, "after-undo")
    final = reopened.commit(reopened.stage(new), validation_for(new))
    assert final.parent_revision == restored.revision_id


def test_failed_incomplete_or_mismatched_validation_cannot_publish(
    initial: Any,
) -> None:
    root, store, first = initial
    candidate = edit_for(first)
    staged = store.stage(candidate)
    for validation, code in [
        (
            validation_for(
                candidate,
                semantic={
                    "status": "failed",
                    "issues": [
                        {"rule": "changed_fact", "message": "A number changed."}
                    ],
                },
            ),
            "validation_failed",
        ),
        (
            validation_for(candidate, rendering={"status": "not_run"}),
            "validation_failed",
        ),
        (
            validation_for(
                candidate,
                structural={
                    "status": "passed",
                    "issues": [
                        {"rule": "contradiction", "message": "Unresolved issue."}
                    ],
                },
            ),
            "validation_failed",
        ),
        (validation_for(candidate, candidate_digest="0" * 64), "validation_mismatch"),
    ]:
        with pytest.raises(schema.ContractError, match=code):
            store.commit(staged, validation)
        reopened = store_api.RevisionStore(root)
        assert reopened.current("ray-geometry").revision_id == first.revision_id
        assert reopened.current("ray-geometry").document.html == HTML
        assert (
            reopened.asset_bytes("ray-geometry", first.revision_id, "court")
            == PAYLOADS["court"]
        )
    assert len(store.history("ray-geometry")) == 1


@pytest.mark.parametrize(
    "phase", ["public_files_written", "revision_installed", "head_replace"]
)
def test_save_interruption_reopens_original_complete_revision(
    initial: Any, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    root, store, first = initial
    candidate = edit_for(first)
    staged = store.stage(candidate)

    def fault(checkpoint: str) -> None:
        if checkpoint == phase:
            raise OSError("injected persistence fault")

    store.fault = fault
    if phase == "head_replace":
        original = paths_api.os.replace

        def fail_replace(source: str, destination: str, **kwargs: Any) -> None:
            if destination == "current.json":
                raise OSError("injected atomic pointer failure")
            original(source, destination, **kwargs)

        monkeypatch.setattr(paths_api.os, "replace", fail_replace)
    with pytest.raises(schema.ContractError, match="storage_error"):
        store.commit(staged, validation_for(candidate))
    reopened = store_api.RevisionStore(root)
    current = reopened.current("ray-geometry")
    assert current.revision_id == first.revision_id
    assert current.document.html == HTML
    assert current.document.evidence[0].values == {"baseline_m": 7}
    assert current.document.figures[1].renderer.version == "1"
    assert (
        reopened.asset_bytes("ray-geometry", current.revision_id, "court")
        == PAYLOADS["court"]
    )
    assert (
        reopened.asset_bytes("ray-geometry", current.revision_id, "theme")
        == PAYLOADS["theme"]
    )
    assert len(reopened.history("ray-geometry")) == 1
    # An installed orphan is not a readable published revision.
    if phase in {"revision_installed", "head_replace"}:
        orphan = next(
            path.name
            for path in (root / "private/documents/ray-geometry/revisions").iterdir()
            if not path.name.startswith(".") and path.name != first.revision_id
        )
        with pytest.raises(schema.ContractError, match="revision_not_found"):
            reopened.revision("ray-geometry", orphan)


def test_snapshot_is_pinned_and_exports_only_declared_public_files(
    visual_initial: Any,
) -> None:
    root, store, first = visual_initial
    private = root / "private/documents/ray-geometry"
    (private / "raw-prompt.txt").write_text(
        "NEVER EXPORT THIS INSTRUCTION", encoding="utf-8"
    )
    (private / "credentials.txt").write_text(
        "NEVER EXPORT THIS TOKEN", encoding="utf-8"
    )
    candidate = edit_for(first)
    staged = store.stage(candidate)
    snapshot = store.snapshot("ray-geometry", first.revision_id)
    second = store.commit(staged, validation_for(candidate))
    assert second.revision_id != snapshot.revision_id
    export = root / "exports" / snapshot.snapshot_id
    actual = sorted(
        str(path.relative_to(export)) for path in export.rglob("*") if path.is_file()
    )
    expected = [
        "assets/concept.png",
        "assets/court.svg",
        "assets/theme.css",
        "assets/visuals_v2.js",
        "evidence.json",
        "figures.json",
        "index.html",
        "manifest.json",
    ]
    assert actual == expected == snapshot.files
    assert store.content_file(
        "ray-geometry", first.revision_id, "index.html"
    ) == first.document.html.encode("utf-8")
    assert (
        store.staged_asset_bytes("ray-geometry", staged.stage_id, "court")
        == PAYLOADS["court"]
    )
    assert (export / "index.html").read_text(encoding="utf-8") == first.document.html
    assert (export / "assets/court.svg").read_bytes() == PAYLOADS["court"]
    assert json.loads((export / "evidence.json").read_text())[0]["values"] == {
        "baseline_m": 7
    }
    assert json.loads((export / "figures.json").read_text())[1]["definition"][
        "data"
    ] == {
        "focal_px": 600,
        "baseline_m": 0.8,
        "disparity_px": 30,
        "sample_errors_px": [-10, 0, 10],
    }
    assert (export / "assets/concept.png").read_bytes() == png()
    assert (
        export / "assets/visuals_v2.js"
    ).read_bytes() == visuals.managed_renderer().content
    visuals.validate_visuals(
        first.document,
        {
            asset.asset_id: (export / asset.path).read_bytes()
            for asset in first.document.assets
        },
    )
    assert not any(
        b"NEVER EXPORT" in path.read_bytes()
        for path in export.rglob("*")
        if path.is_file()
    )
    with pytest.raises(schema.ContractError, match="snapshot_not_found"):
        store.snapshot_file(snapshot.snapshot_id, "raw-prompt.txt")


def test_v1_renderer_snapshot_and_restore_survive_v2_publication(
    tmp_path: Path,
) -> None:
    store, first, second, snapshot = make_renderer_history(tmp_path / "versions")
    old_bytes = store.asset_bytes("ray-geometry", first.revision_id, "visuals_v1")
    assert (
        hashlib.sha256(old_bytes).hexdigest()
        == "9dbb09d4501df75d8580ea9bbc3189631ce7ab5af927126338d781b987f16591"
    )
    assert first.document.figures[1].renderer.version == "1"
    assert second.document.figures[1].renderer.version == "2"
    assert (
        "ExplainVisualsV1" in first.document.html
        and "ExplainVisualsV2" not in first.document.html
    )
    assert "ExplainVisualsV2" in second.document.html
    assert (
        store.snapshot_file(snapshot.snapshot_id, "assets/visuals_v1.js") == old_bytes
    )
    assert (
        store.snapshot_file(snapshot.snapshot_id, "index.html")
        == first.document.html.encode()
    )
    restored = store.restore(
        schema.RestoreRequest(
            document_id="ray-geometry",
            base_revision=second.revision_id,
            request_id="restore-v1",
            revision_id=first.revision_id,
        )
    )
    assert restored.document == first.document
    assert (
        store.asset_bytes("ray-geometry", restored.revision_id, "visuals_v1")
        == old_bytes
    )
    visuals.validate_visuals(
        restored.document,
        {
            asset.asset_id: store.asset_bytes(
                "ray-geometry", restored.revision_id, asset.asset_id
            )
            for asset in restored.document.assets
        },
    )


def test_competing_commits_are_serialized_and_successful_commit_is_idempotent(
    initial: Any,
) -> None:
    root, store, first = initial
    one, two = edit_for(first, "first-writer"), edit_for(first, "second-writer")
    stage_one, stage_two = store.stage(one), store.stage(two)
    barrier = Barrier(2)

    def commit(staged: Any, candidate: Any) -> Any:
        barrier.wait(timeout=5)
        try:
            return store_api.RevisionStore(root).commit(
                staged, validation_for(candidate)
            )
        except schema.ContractError as exc:
            return exc.error.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_one = executor.submit(commit, stage_one, one)
        future_two = executor.submit(commit, stage_two, two)
        results = [future_one.result(timeout=5), future_two.result(timeout=5)]
    assert sum(result == "stale_revision" for result in results) == 1
    winner = next(result for result in results if result != "stale_revision")
    winning_stage, winning_candidate = (
        (stage_one, one) if winner.request_id == "first-writer" else (stage_two, two)
    )
    assert (
        store.commit(winning_stage, validation_for(winning_candidate)).revision_id
        == winner.revision_id
    )
    assert len(store.history("ray-geometry")) == 2
    conflict = edit_for(winner, winner.request_id)
    conflict_stage = store.stage(conflict)
    with pytest.raises(schema.ContractError, match="request_conflict"):
        store.commit(conflict_stage, validation_for(conflict))


def test_tampered_stage_and_symlink_revision_never_become_current(
    initial: Any, tmp_path: Path
) -> None:
    root, store, first = initial
    candidate = edit_for(first)
    staged = store.stage(candidate)
    public = root / "private/documents/ray-geometry/stages" / staged.stage_id / "public"
    (public / "assets/court.svg").write_bytes(b"corrupt image")
    with pytest.raises(schema.ContractError, match="corrupt_store"):
        store.commit(staged, validation_for(candidate))
    assert store.current("ray-geometry").revision_id == first.revision_id
    outside = tmp_path / "outside-assets"
    outside.mkdir()
    (outside / "court.svg").write_bytes(PAYLOADS["court"])
    # A symlink remains forbidden even when the outside bytes match the digest.
    (public / "assets/court.svg").unlink()
    (public / "assets/court.svg").symlink_to(outside / "court.svg")
    with pytest.raises(schema.ContractError, match="unsafe_path"):
        store.commit(staged, validation_for(candidate))
    reopened = store_api.RevisionStore(root)
    assert reopened.current("ray-geometry").document.html == HTML
    assert (
        reopened.asset_bytes("ray-geometry", first.revision_id, "court")
        == PAYLOADS["court"]
    )
    assert (outside / "court.svg").read_bytes() == PAYLOADS["court"]
