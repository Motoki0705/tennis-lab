"""Durable deduplication, publication fencing and recovery with real bundles.

Design: 13 cases, each <5s/setup <1s, Event synchronization only. Contract/store
input combinations already covered by test_document/test_revisions are reused,
not multiplied here. All writers and visual verdicts are explicit test doubles.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.agents_ops.explain_editor import candidate
from tests.support.agents_ops.explain_editor import editor_env as editor_env
from tests.support.agents_ops.explain_html import (
    PAYLOADS,
    candidate_for,
    schema,
    validation_for,
)
from tests.support.agents_ops.explain_html_visuals import COMPARISON, visuals
from tests.support.agents_ops.explain_jobs import Gate, jobs
from tests.support.agents_ops.explain_jobs import job_world as job_world

managed: Any = importlib.import_module("explain_html.managed")

pytestmark = pytest.mark.integration


def test_managed_service_commits_compiled_graph_and_rejects_new_ai_image(
    editor_env: Any,
) -> None:
    env = editor_env
    renderer = visuals.managed_renderer()
    graph = schema.Figure(
        figure_id="comparison",
        content_id="limits",
        kind="graph",
        asset_ids=[renderer.asset.asset_id],
        caption="Compare two hypothetical tracks.",
        alt="Two asymmetric illustrative position series.",
        provenance=schema.Provenance(kind="hypothetical", source="Synthetic inputs."),
        renderer=schema.Renderer(asset_id=renderer.asset.asset_id, version="2"),
        definition=COMPARISON,
    )
    request = env.request.model_copy(
        update={
            "instruction": "Add a hypothetical comparison graph; retain facts and citations."
        }
    )
    html = env.current.document.html.replace(
        "<html><body>", "<html><head><title>Tracks</title></head><body>"
    ).replace("</aside>", '<figure data-figure-id="comparison"></figure></aside>')
    raw = candidate(
        env.current.document.model_copy(
            update={"html": html, "assets": [renderer.asset], "figures": [graph]}
        ),
        current=env.current,
        request=request,
    )
    final = candidate(
        visuals.render_document(raw.document), current=env.current, request=request
    )
    env.outputs(
        raw,
        {
            "candidate_digest": schema.candidate_digest(final),
            "check": {"status": "passed"},
            "reviewed_evidence_ids": ["baseline", "cameras"],
        },
    )
    resources = managed.ManagedResources()
    seen: list[Any] = []

    def prepare(request: Any, current: Any, history: Any, *, cancel: Any) -> Any:
        prepared = resources.prepare(request, current, history, cancel=cancel)
        if request.request_id == env.request.request_id:
            return jobs.PreparedEdit(
                prepared.context.model_copy(
                    update={"assets": [renderer.asset], "figures": [graph]}
                ),
                {renderer.asset.asset_id: renderer.content},
            )
        return prepared

    def rendering(candidate: Any, assets: Any, *, cancel: Any) -> Any:
        # Only browser/meaning verdicts are fakes; service runs real visual safety
        # on these exact staged bytes before this test rendering boundary.
        seen.append((candidate, dict(assets)))
        return schema.CheckResult(status="passed")

    with jobs.JobController(
        env.store,
        resources.editor_factory(env.adapter.gemini),
        managed.with_visual_validation(rendering),
        prepare=prepare,
    ) as controller:
        controller.submit(request)
        done = controller.wait("geometry", request.request_id, timeout=5)
        assert done.state == "succeeded"
        revision = env.store.current("geometry")
        assert revision.document == final.document
        assert seen[0][0] == final and seen[0][1] == {
            renderer.asset.asset_id: renderer.content
        }
        assert (
            "<script " in revision.document.html
            and 'data-values="1"' in revision.document.html
        )
        assert '<meta data-visual-policy="1"' in revision.document.html
        verified = json.loads(
            [call for call in env.calls() if call["prompt"] is not None][-1][
                "prompt"
            ].split("\nINPUT_JSON\n")[1]
        )
        assert verified["candidate_digest"] == schema.candidate_digest(final)
        assert (
            resources.images.capability().reason in verified["context"]["preferences"]
        )
        assert (
            env.store.asset_bytes(
                "geometry", revision.revision_id, renderer.asset.asset_id
            )
            == renderer.content
        )
        image_request = request.model_copy(
            update={
                "request_id": "new-image",
                "base_revision": revision.revision_id,
                "instruction": "Add a new AI-generated conceptual image of two cameras.",
            }
        )
        image = schema.Figure(
            figure_id="new-concept",
            content_id="limits",
            kind="image",
            asset_ids=[],
            caption="Conceptual illustration of two camera rays.",
            alt="Two cameras looking at one point.",
            provenance=schema.Provenance(
                kind="generated_concept",
                source="Requested; unavailable during reading.",
            ),
        )
        image_raw = candidate(
            revision.document.model_copy(
                update={
                    "html": revision.document.html.replace(
                        "</aside>",
                        '<figure data-figure-id="new-concept"></figure></aside>',
                    ),
                    "figures": [*revision.document.figures, image],
                }
            ),
            current=revision,
            request=image_request,
        )
        env.outputs(image_raw)
        controller.submit(image_request)
        rejected = controller.wait("geometry", "new-image", timeout=5)
        assert (
            rejected.state == "failed"
            and rejected.error.code == "image_generation_unavailable"
        )
        assert (
            env.store.current("geometry") == revision
            and len(env.store.history("geometry")) == 2
        )
        assert len(seen) == 1
        assert len([call for call in env.calls() if call["prompt"] is not None]) == 3
        assert not resources.images.capability().available
        assert "initial" in resources.images.capability().reason


def test_retries_and_status_cannot_duplicate_generation_or_publication(
    job_world: Any,
) -> None:
    world = job_world
    gate = world.editor.gates["edit"] = Gate()
    compiled_svg = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 9 4"><path d="M1 1L4 3L8 1"/></svg>'

    def editor_factory(prepared: Any) -> Any:
        def edit(*args: Any, **kwargs: Any) -> Any:
            result = world.editor.edit(*args, **kwargs)
            document = result.candidate.document
            assets = [
                asset.model_copy(
                    update={"sha256": hashlib.sha256(compiled_svg).hexdigest()}
                )
                if asset.asset_id == "court"
                else asset
                for asset in document.assets
            ]
            compiled = result.candidate.model_copy(
                update={"document": document.model_copy(update={"assets": assets})}
            )
            prepared.assets["court"] = compiled_svg
            return SimpleNamespace(
                candidate=compiled,
                validation=validation_for(compiled, rendering={"status": "not_run"}),
            )

        return SimpleNamespace(edit=edit)

    controller = world.controller(editor_factory=editor_factory)
    request = world.request()
    with ThreadPoolExecutor(2) as pool:
        accepted = list(pool.map(controller.submit, [request, request]))
    assert all(job.state in {"queued", "running"} for job in accepted)
    assert gate.entered.wait(5)
    for _ in range(3):
        assert controller.status("ray-geometry", "edit").state == "running"
        assert controller.submit(request).state == "running"
    with pytest.raises(schema.ContractError, match="request_conflict"):
        controller.submit(
            request.model_copy(update={"instruction": "A different instruction."})
        )
    assert world.store.current("ray-geometry") == world.first
    gate.release.set()
    done = controller.wait("ray-geometry", "edit", timeout=5)
    assert done.state == "succeeded"
    assert controller.submit(request) == done
    assert len(world.editor.calls) == len(world.commits) == 1
    assert len(world.store.history("ray-geometry")) == 2
    assert done.changes[0].content_id == "geometry"
    assert world.validator.seen[0][1] == {**PAYLOADS, "court": compiled_svg}
    assert (
        world.store.asset_bytes("ray-geometry", done.revision_id, "court")
        == compiled_svg
    )
    assert (
        world.store.asset_bytes("ray-geometry", world.first.revision_id, "court")
        == PAYLOADS["court"]
    )
    assert world.editor.contexts[0].change_summaries == [world.first.change_summary]
    with pytest.raises(schema.ContractError, match="stale_revision"):
        controller.submit(world.request("stale"))
    with pytest.raises(schema.ContractError, match="request_conflict"):
        controller.submit(world.request("initial", base_revision=done.revision_id))


def test_document_queue_rechecks_base_and_other_document_can_finish(
    job_world: Any,
) -> None:
    world = job_world
    other = world.first.document.model_copy(update={"document_id": "other"})
    initial = candidate_for(other)
    initial = initial.model_copy(
        update={"request": initial.request.model_copy(update={"document_id": "other"})}
    )
    other_revision = world.store.import_initial(
        initial, PAYLOADS, validation_for(initial)
    )
    gate = world.editor.gates["first"] = Gate()
    controller = world.controller()
    controller.submit(world.request("first"))
    assert gate.entered.wait(5)
    assert controller.submit(world.request("second")).state == "queued"
    controller.submit(
        world.request(
            "other-edit", document_id="other", base_revision=other_revision.revision_id
        )
    )
    assert controller.wait("other", "other-edit", timeout=5).state == "succeeded"
    assert [request.request_id for request in world.editor.calls] == [
        "first",
        "other-edit",
    ]
    gate.release.set()
    assert controller.wait("ray-geometry", "first", timeout=5).state == "succeeded"
    rejected = controller.wait("ray-geometry", "second", timeout=5)
    assert rejected.state == "failed" and rejected.error.code == "stale_revision"
    assert len(world.editor.calls) == 2


def test_cancel_at_validation_fences_late_result_and_cancelled_queue(
    job_world: Any,
) -> None:
    world = job_world
    gate = world.validator.gates["edit"] = Gate()
    controller = world.controller()
    controller.submit(world.request())
    assert gate.entered.wait(5)
    controller.submit(world.request("queued"))
    assert controller.cancel("ray-geometry", "queued").state == "cancelled"
    cancelled = controller.cancel("ray-geometry", "edit")
    assert cancelled.state == "cancelled" and cancelled.error.code == "cancelled"
    assert gate.cancel is not None and gate.cancel.is_set()
    gate.release.set()
    controller.close()  # joins the late validator, so the assertion cannot race it
    assert controller.status("ray-geometry", "edit") == cancelled
    assert world.commits == [] and len(world.editor.calls) == 1
    assert world.store.current("ray-geometry") == world.first


def test_undo_at_validation_invalidates_late_result_and_allows_new_edit(
    job_world: Any,
) -> None:
    world = job_world
    gate = world.validator.gates["edit"] = Gate()
    controller = world.controller()
    controller.submit(world.request())
    assert gate.entered.wait(5)
    undo = schema.RestoreRequest(
        document_id="ray-geometry",
        request_id="undo",
        base_revision=world.first.revision_id,
        revision_id=world.first.revision_id,
    )
    restored = controller.restore(undo)
    assert restored.revision_id != world.first.revision_id
    assert restored.document == world.first.document
    assert controller.status("ray-geometry", "edit").error.code == "stale_revision"
    gate.release.set()
    controller.submit(world.request("after-undo", base_revision=restored.revision_id))
    done = controller.wait("ray-geometry", "after-undo", timeout=5)
    assert done.state == "succeeded"
    assert world.store.current("ray-geometry").parent_revision == restored.revision_id
    assert controller.restore(undo) == restored
    with pytest.raises(schema.ContractError, match="request_conflict"):
        controller.submit(world.request("undo", base_revision=done.revision_id))
    assert len(world.commits) == 2


def test_shutdown_fences_late_editor_result_and_does_not_resend_on_restart(
    job_world: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = job_world
    gate = world.editor.gates["edit"] = Gate()
    controller = world.controller()
    controller.submit(world.request())
    assert gate.entered.wait(5)
    controller.submit(world.request("queued"))
    released_early: list[bool] = []
    original_close = controller._resources.close

    def close_resources() -> None:
        released_early.append(not gate.release.is_set())
        original_close()

    monkeypatch.setattr(controller._resources, "close", close_resources)
    with ThreadPoolExecutor(2) as pool:
        closing = pool.submit(controller.close)
        assert gate.cancel is not None and gate.cancel.wait(5)
        also_closing = pool.submit(controller.close)
        assert controller.wait("ray-geometry", "edit", timeout=5).state == "interrupted"
        assert not closing.done()
        with pytest.raises(schema.ContractError, match="request_conflict"):
            world.controller()
        gate.release.set()
        closing.result(timeout=5)
        also_closing.result(timeout=5)
    assert released_early == [False]
    reopened = world.controller()
    assert reopened.submit(world.request()).state == "interrupted"
    assert reopened.status("ray-geometry", "queued").state == "interrupted"
    assert world.store.current("ray-geometry") == world.first
    assert len(world.editor.calls) == 1 and world.commits == []


@pytest.mark.parametrize("state", ["queued", "running", "validating"])
def test_recovery_from_real_active_journal_bytes_never_regenerates(
    job_world: Any, state: str
) -> None:
    world = job_world
    gate = Gate()
    if state == "validating":
        world.validator.gates["edit"] = gate
    else:
        world.editor.gates["blocker" if state == "queued" else "edit"] = gate
    controller = world.controller()
    if state == "queued":
        controller.submit(world.request("blocker"))
        assert gate.entered.wait(5)
    controller.submit(world.request())
    assert gate.entered.wait(5)
    path = world.root / "private/requests/ray-geometry/edit.json"
    before = path.read_bytes()
    assert controller.status("ray-geometry", "edit").state == state
    # Replay the exact bytes persisted at the crash point, after fencing/reaping
    # the test's old thread. No fabricated schema or wall-clock race is involved.
    with ThreadPoolExecutor(1) as pool:
        closing = pool.submit(controller.close)
        assert gate.cancel is not None and gate.cancel.wait(5)
        gate.release.set()
        closing.result(timeout=5)
    path.write_bytes(before)
    count = len(world.editor.calls)
    recovered = world.controller()
    result = recovered.status("ray-geometry", "edit")
    assert result.state == "interrupted" and result.error.code == "interrupted"
    assert recovered.submit(world.request()) == result
    assert len(world.editor.calls) == count and world.commits == []
    assert world.store.current("ray-geometry") == world.first


def test_recovery_reconciles_commit_before_job_acknowledgement(
    job_world: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = job_world
    gate = world.validator.gates["edit"] = Gate()
    controller = world.controller()
    controller.submit(world.request())
    assert gate.entered.wait(5)
    original_write = jobs.SafeDirectory.atomic_write

    def fail_ack(filesystem: Any, path: str, data: bytes) -> None:
        if (
            path.endswith("ray-geometry/edit.json")
            and json.loads(data)["job"]["state"] == "succeeded"
        ):
            raise OSError("Injected disk failure after current replacement.")
        original_write(filesystem, path, data)

    with monkeypatch.context() as patch:
        patch.setattr(jobs.SafeDirectory, "atomic_write", fail_ack)
        gate.release.set()
        with pytest.raises(schema.ContractError, match="storage_error"):
            controller.wait("ray-geometry", "edit", timeout=5)
        controller.close()
    published = world.store.current("ray-geometry")
    recovered = world.controller()
    done = recovered.status("ray-geometry", "edit")
    assert done.state == "succeeded" and done.revision_id == published.revision_id
    assert recovered.submit(world.request()) == done
    assert len(world.editor.calls) == len(world.commits) == 1
    assert len(world.store.history("ray-geometry")) == 2


@pytest.mark.parametrize(
    "damage,code",
    [("rendering", "validation_failed"), ("digest", "validation_mismatch")],
)
def test_rejected_or_unbound_validation_cannot_publish(
    job_world: Any, damage: str, code: str
) -> None:
    world = job_world
    world.validator.reject = damage == "rendering"
    world.editor.damage = damage
    controller = world.controller()
    controller.submit(world.request())
    done = controller.wait("ray-geometry", "edit", timeout=5)
    assert done.state == "failed" and done.error.code == code
    assert world.store.current("ray-geometry") == world.first
    assert world.commits == []


def test_unavailable_image_preparer_preserves_current_without_calling_editor(
    job_world: Any,
) -> None:
    world = job_world

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise schema.ContractError("image_generation_unavailable")

    controller = world.controller(prepare=unavailable)
    controller.submit(
        world.request(instruction="Add a new generated conceptual illustration.")
    )
    done = controller.wait("ray-geometry", "edit", timeout=5)
    assert done.state == "failed" and done.error.code == "image_generation_unavailable"
    assert world.editor.calls == [] and world.commits == []
    assert world.store.current("ray-geometry") == world.first
