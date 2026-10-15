from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from src.tasks.annotation.contracts import (
    OperationInput,
    OperationReceipt,
    ReviewOperation,
)
from src.tasks.annotation.operations import execute
from src.tasks.ball_detection.annotation.review.contracts import (
    BallEdits,
    BallReviewProof,
    ConfirmAbsence,
    ReplaceFrame,
)
from src.tasks.ball_detection.annotation.review.worker import BallRowPage
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    AdoptionReceipt,
    ArtifactRef,
    FrameRange,
    ReviewEvidence,
)
from src.tasks.base.annotation.editing import EditCommand, Revision
from src.tasks.base.annotation.handoff import read_handoff
from src.tasks.base.annotation.images import read_rendered
from src.tasks.base.annotation.images.contracts import Point
from src.tasks.base.annotation.storage import StorageCatalog
from src.tasks.base.annotation.workers.contracts import AttemptResult, Launch
from src.tasks.base.annotation.workers.pool import WorkerPool
from src.tasks.base.annotation.workers.review.contracts import (
    EditRequest,
    HandoffRequest,
    ReviewWorkerInput,
    RowsRequest,
)
from src.tasks.base.annotation.workers.review.service import (
    execute_service,
    service_input,
)
from src.tasks.base.annotation.workers.services import ServiceReply, ServiceRequest
from src.utils.paths import PROJECT_ROOT
from tests.support.tasks.ball_detection.review_worker import ReviewSession


@pytest.fixture
def session(tmp_path: Path) -> ReviewSession:
    return ReviewSession(tmp_path / "annotation")


def test_ball_input_pin_and_retry_inputs_include_media_prediction_and_edit_evidence(
    session: ReviewSession,
) -> None:
    env = session.env
    protected = {
        ref
        for pin in StorageCatalog(env.layout).snapshot().protections
        for ref in pin.references
    }
    assert env.manifest.media in protected
    assert env.manifest.source.media in protected
    assert env.context.prediction in protected
    image = session.image(1)
    edit = session.confirm(image, 1, 2)
    session.open(2)
    launch = read_document(env.layout, env.actor, Launch)
    revision = read_document(env.layout, edit.revision, Revision)
    assert edit.revision in launch.prepared.inputs
    assert revision.payload in launch.prepared.inputs
    assert edit.command in launch.prepared.inputs
    assert image.image in launch.prepared.inputs
    assert image.metadata in launch.prepared.inputs


def test_parent_renders_pinned_markers_and_serves_bounded_exact_rows(
    session: ReviewSession,
) -> None:
    env = session.env
    handoff = read_handoff(
        env.layout,
        session.call(
            "review_handoff", HandoffRequest(scope=FrameRange(start=1, stop=4))
        ),
    )
    assert handoff.revision == env.initial
    assert env.labels().frames[1].sha256 in handoff.content.rows[0].detail
    image = session.image(1, 2, drawing="prediction")
    rendering, receipt = read_rendered(env.layout, image)
    assert rendering.annotation == env.context.prediction
    assert rendering.tiles[0].drawings == (Point(x=16.0, y=12.0),)
    assert receipt.tiles[0].source_pts == env.manifest.frames[1].source_pts
    session.confirm(image, 1, 3)
    page = read_document(
        env.layout,
        session.call(
            "review_rows",
            RowsRequest(
                expected_revision=env.head(),
                scope=FrameRange(start=1, stop=3),
            ),
        ),
        BallRowPage,
    )
    assert page.rows == env.labels().frames[1:3]
    assert page.image_proof == ("pending_current_attempt", "pending_current_attempt")
    assert all(row.review_actor == env.actor for row in page.rows)
    with pytest.raises(ValueError, match="eight"):
        session.call(
            "review_rows",
            RowsRequest(
                expected_revision=env.head(), scope=FrameRange(start=1, stop=10)
            ),
        )
    with pytest.raises(ValueError, match="stale"):
        session.call(
            "review_rows",
            RowsRequest(
                expected_revision=env.initial, scope=FrameRange(start=1, stop=2)
            ),
        )


def test_worker_output_is_frozen_before_revision_references_it(
    session: ReviewSession,
) -> None:
    image = session.image(1)
    receipt = session.confirm(image, 1, 2)
    command = read_document(session.env.layout, receipt.command, EditCommand)
    assert (
        session.env.layout.artifact_path(command.changes).parent
        == session.directory / "edits"
    )
    for path in (session.directory / "work").glob("*.json"):
        path.write_text("worker changed its scratch file")
    report = session.evidence()
    outcome = session.finish(report)
    assert outcome.status == "complete" and outcome.adoption is not None


@pytest.mark.parametrize(
    "tamper", ["inbox", "input", "launch", "outside_edit", "ended"]
)
def test_services_reject_foreign_or_inactive_attempts(
    session: ReviewSession, tamper: str
) -> None:
    env = session.env
    ref = session.request_service(
        "review_handoff", HandoffRequest(scope=FrameRange(start=1, stop=2))
    )
    if tamper == "inbox":
        ref = env.save(read_document(env.layout, ref, ServiceRequest))
    elif tamper == "input":
        service = read_document(env.layout, ref, ServiceRequest)
        changed = service.model_copy(
            update={
                "task_request": env.save(
                    read_document(env.layout, session.input, ReviewWorkerInput)
                )
            }
        )
        ref = write_document(
            env.layout,
            session.directory / "work/service-requests" / (changed.sha256 + ".json"),
            changed,
        )
    elif tamper == "launch":
        with session.ledger.registry.transaction() as db:
            db.execute(
                "UPDATE worker_tasks SET current_attempt='other' WHERE id=?",
                (session.request.work_id,),
            )
    elif tamper == "ended":
        session.evidence()
    else:
        ref = session.request_service(
            "review_edit",
            EditRequest(
                expected_revision=env.head(), changes=env.blob, reason="outside"
            ),
        )
    with pytest.raises((ValueError, FileNotFoundError)):
        execute_service(env.layout, env.plan, ref, session.services)
    assert env.head() == env.initial


def test_edit_is_atomic_idempotent_and_does_not_accept_stale_head(
    session: ReviewSession,
) -> None:
    env = session.env
    image = session.image(1)
    bad = BallEdits(
        operations=(
            ConfirmAbsence(
                scope=FrameRange(start=1, stop=3),
                out_of_frame_tracks=(),
                evidence=(image.sha256,),
            ),
        ),
        images=(image,),
    )
    changes = write_document(env.layout, session.directory / "work/bad.json", bad)
    with pytest.raises(ValueError, match="image references"):
        session.call(
            "review_edit",
            EditRequest(
                expected_revision=env.initial, changes=changes, reason="invalid batch"
            ),
        )
    assert env.head() == env.initial
    receipt = session.confirm(image, 1, 2)
    services = [
        annotation_reference(env.layout, path)
        for path in (session.directory / "work/service-requests").glob("*.json")
    ]
    successful = next(
        ref
        for ref in services
        if (s := read_document(env.layout, ref, ServiceRequest)).operation
        == "review_edit"
        and s.parameters["reason"] == "synthetic image confirmation"
    )
    assert (
        read_document(
            env.layout,
            execute_service(env.layout, env.plan, successful, session.services),
            type(receipt),
        )
        == receipt
    )
    with pytest.raises(ValueError, match="revision changed"):
        session.call(
            "review_edit",
            EditRequest(
                expected_revision=env.initial, changes=changes, reason="stale batch"
            ),
        )


def test_rendering_without_actual_image_input_cannot_adopt(
    session: ReviewSession,
) -> None:
    session.confirm(session.image(1), 1, 2)
    report = session.evidence(shown=())
    with pytest.raises(ValueError, match="actually shown"):
        session.finish(report)
    assert (
        session.env.store.current(session.env.context_ref, session.env.adapter)[1]
        is None
    )


@pytest.mark.parametrize("previous", ["deferred", "cancelled", "missing"])
def test_retry_keeps_previous_evidence_or_requires_explicit_reconfirmation(
    session: ReviewSession, previous: str
) -> None:
    env = session.env
    session.confirm(session.image(1), 1, 2)
    old_actor = env.actor
    if previous == "deferred":
        report = session.evidence()
        session.finish(report, status="deferred")
    elif previous == "cancelled":
        report = session.evidence(status="cancelled")
        session.state("cancelled")
    else:
        report = None
        session.state("failed")
    session.open(2)
    assert session.ledger.task(session.request.work_id)["result"] is None
    history = session.ledger.history(env.plan, session.request)
    assert history[0].launch == old_actor and history[0].telemetry == report
    initial = read_document(env.layout, session.input, ReviewWorkerInput)
    handoff = read_handoff(env.layout, initial.initial_handoff)
    assert (
        "image_proof=missing" if previous == "missing" else "image_proof=verified"
    ) in handoff.content.rows[0].detail
    current = session.evidence(shown=())
    if previous == "missing":
        with pytest.raises(ValueError, match="actually shown"):
            session.finish(current)
        assert env.store.current(env.context_ref, env.adapter)[1] is None
    else:
        outcome = session.finish(current)
        adoption = read_document(env.layout, outcome.adoption, AdoptionReceipt)
        evidence = read_document(env.layout, adoption.review_evidence, ReviewEvidence)
        proof = read_document(env.layout, evidence.evidence[0], BallReviewProof)
        assert proof.telemetry == (report, current)
        assert env.labels().frames[1].review_actor == old_actor


def test_retry_reconfirms_missing_proof_with_new_actor_and_retained_image(
    session: ReviewSession,
) -> None:
    env = session.env
    old_image = session.image(1)
    session.confirm(old_image, 1, 2)
    old_actor = env.actor
    session.state("failed")
    session.open(2)
    page = read_document(
        env.layout,
        session.call(
            "review_rows",
            RowsRequest(
                expected_revision=env.head(),
                scope=FrameRange(start=1, stop=2),
            ),
        ),
        BallRowPage,
    )
    assert page.image_proof == ("missing",) and page.images == (old_image,)
    edits = BallEdits(
        operations=(
            ReplaceFrame(
                frame_index=1,
                expected_frame_digest=page.rows[0].sha256,
                decision=page.rows[0].decision,
            ),
        )
    )
    changes = write_document(
        env.layout, session.directory / "work/reconfirm.json", edits
    )
    session.call(
        "review_edit",
        EditRequest(
            expected_revision=env.head(),
            changes=changes,
            reason="reconfirmed from retained image",
        ),
    )
    assert env.labels().frames[1].review_actor == env.actor != old_actor
    # Re-viewing the retained image is captured as new-attempt evidence.
    report = session.evidence(shown=(old_image,))
    assert session.finish(report).adoption is not None


def test_history_rejects_cross_attempt_telemetry_and_final_receipts(
    session: ReviewSession,
) -> None:
    env = session.env
    report = session.evidence()
    session.finish(report, status="deferred")
    old = session.directory
    session.open(2)
    write_document(env.layout, session.directory / "telemetry.ref.json", report)
    with pytest.raises(ValueError, match="another launch/exit"):
        session.ledger.history(env.plan, session.request)
    (session.directory / "telemetry.ref.json").unlink()
    final = read_document(
        env.layout,
        annotation_reference(env.layout, old / "result.receipt.json"),
        AttemptResult,
    )
    write_document(env.layout, session.directory / "result.receipt.json", final)
    with pytest.raises(ValueError, match="another launch/exit"):
        session.ledger.history(env.plan, session.request)


def test_recovery_after_seal_reuses_original_adoption(session: ReviewSession) -> None:
    session.confirm(session.image(1), 1, 2)
    report = session.evidence()
    # Simulate a crash after seal but before writing result.receipt.json or updating SQLite.
    first = session.finish(report, record=False)
    assert not (session.directory / "result.receipt.json").exists()
    assert session.finish(report).adoption == first.adoption


def test_composition_stage_requires_supervisor_binding_and_owns_service_output(
    session: ReviewSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = session.env
    service = session.request_service(
        "review_handoff", HandoffRequest(scope=FrameRange(start=1, stop=4))
    )
    operation = OperationInput(
        plan=env.plan_ref, operation=ReviewOperation(service_request=service)
    )
    assert operation.stage == "render"
    ref = write_document(
        env.layout,
        env.layout.path(
            "campaigns", env.plan.campaign_id, "jobs", "review-1", "operation.json"
        ),
        operation,
    )
    with pytest.raises(ValueError, match="resource-accounted"):
        execute(env.layout, ref)
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", ref.digest)
    result = execute(env.layout, ref)
    receipt = read_document(env.layout, result, OperationReceipt)
    assert receipt.status == "complete" and len(receipt.results) == 1
    assert read_handoff(env.layout, receipt.results[0]).revision == env.initial
    assert execute(env.layout, ref) == result


def test_parameters_cannot_inject_overlay_or_unbounded_input(
    session: ReviewSession,
) -> None:
    image = session.image(1)
    original = next((session.directory / "work/service-requests").glob("*.json"))
    service = json.loads(original.read_bytes())
    service["parameters"]["tiles"][0]["drawings"] = [
        {"kind": "point", "x": 2.0, "y": 3.0}
    ]
    changed = ServiceRequest.model_validate_json(json.dumps(service))
    ref = write_document(
        session.env.layout,
        session.directory / "work/service-requests" / (changed.sha256 + ".json"),
        changed,
    )
    with pytest.raises(ValueError, match="Extra inputs"):
        execute_service(session.env.layout, session.env.plan, ref, session.services)
    oversized = ref.model_copy(update={"size_bytes": 32769})
    with pytest.raises(ValueError, match="byte limit"):
        service_input(session.env.layout, oversized)
    assert image in session.worker.telemetry_images(
        session.env.layout, session.env.plan, session.request, session.directory
    )


@pytest.mark.parametrize("show_image", [True, False])
def test_supervised_worker_services_rollout_and_adoption_protocol(
    session: ReviewSession,
    monkeypatch: pytest.MonkeyPatch,
    show_image: bool,
) -> None:
    env = session.env
    session.state("failed")  # First attempt has no claims; retry through the real pool.
    binary = env.layout.repository / "synthetic-codex"
    binary.write_text(
        f"#!{sys.executable}\nimport sys, runpy\nsys.path.insert(0, {str(PROJECT_ROOT)!r})\nmodule = runpy.run_path({str(PROJECT_ROOT / 'tests/support/tasks/ball_detection/review_worker.py')!r})\nmodule['run_synthetic_cli'](show_image={show_image!r})\n"
    )
    binary.chmod(0o700)
    codex_home = env.layout.repository / "synthetic-codex-home"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    pool = WorkerPool(
        env.layout,
        env.plan,
        (session.worker,),
        codex_binary=str(binary),
        codex_home=codex_home,
        timeout_seconds=20,
    )
    assert pool.start(session.request, retry=True)
    directory = session.directory.with_name("attempt-0002")
    deadline = time.monotonic() + 25
    try:
        while True:
            state = pool.reconcile(session.request)
            if state["state"] not in {"running", "exited"}:
                break
            for path in (directory / "work/service-requests").glob("*.json"):
                service_ref = annotation_reference(env.layout, path)
                service = read_document(env.layout, service_ref, ServiceRequest)
                reply = directory / "services" / service.sha256 / "reply.json"
                if not reply.exists():
                    result = execute_service(
                        env.layout, env.plan, service_ref, session.services
                    )
                    write_document(
                        env.layout,
                        reply,
                        ServiceReply(
                            request_digest=service.sha256,
                            status="succeeded",
                            result=result,
                            reason=None,
                        ),
                    )
            assert time.monotonic() < deadline, state
            time.sleep(0.02)
        assert state["state"] == ("complete" if show_image else "validation_failed"), (
            state
        )
        if show_image:
            final = read_document(
                env.layout,
                ArtifactRef.model_validate_json(state["result"]),
                AttemptResult,
            )
            assert final.telemetry is not None and final.outcome.adoption is not None
            assert pool.reconcile(session.request)["state"] == "complete"
        else:
            assert "actually shown" in state["reason"]
            assert env.store.current(env.context_ref, env.adapter)[1] is None
    finally:
        if pool.ledger.task(session.request.work_id)["state"] in {"running", "exited"}:
            pool.stop(session.request, reason="synthetic test teardown")
