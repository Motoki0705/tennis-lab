from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.tasks.ball_detection.annotation.video.agent import store as store_module
from src.tasks.ball_detection.annotation.video.agent.contracts import (
    AnnotationPolicy,
    EditQuery,
    VideoCompletion,
)
from src.tasks.ball_detection.annotation.video.agent.store import VideoRevisionStore
from src.tasks.ball_detection.annotation.video.agent.worker import BallVideoWorker
from src.tasks.ball_detection.annotation.video.contracts import (
    PixelPosition,
    PointProvenance,
    VideoBallFrame,
)
from src.tasks.ball_detection.annotation.video.usage import summarize_annotation
from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.editing.store import RevisionConflict
from tests.support.tasks.blank_agent import BlankSession


@pytest.fixture
def session(tmp_path: Path) -> BlankSession:
    return BlankSession(tmp_path / "annotation")


def final_rows(session: BlankSession) -> tuple[VideoBallFrame, ...]:
    return tuple(
        VideoBallFrame(
            **f.model_dump(), state="unknown", occlusion="unknown", position=None
        )
        for f in session.store.expected.frames
    )


def test_saved_progress_keeps_provenance_and_final_unknown_out_of_remaining(
    session: BlankSession,
) -> None:
    first, middle, last = session.store.expected.frames
    observed = VideoBallFrame(
        **first.model_dump(),
        state="in_frame",
        occlusion="not_occluded",
        position=PixelPosition(x=7, y=19, provenance=PointProvenance(kind="observed")),
    )
    estimated = VideoBallFrame(
        **middle.model_dump(),
        state="in_frame",
        occlusion="occluded",
        position=PixelPosition(
            x=11,
            y=23,
            provenance=PointProvenance(
                kind="occlusion_estimated", support_frames=(first.frame_index,)
            ),
        ),
    )
    session.edit((observed,))
    session.edit(
        (
            VideoBallFrame(
                **last.model_dump(), state="unknown", occlusion="unknown", position=None
            ),
        )
    )
    view = session.call("video_handoff", {"camera_id": session.camera})
    assert view.remaining == (middle.frame_index,)
    assert (
        view.summary["processed"],
        view.summary["final_unknown"],
        view.summary["unprocessed"],
    ) == (2, 1, 1)
    assert not view.summary["complete"]
    reopened = VideoRevisionStore(session.layout, session.store.run_ref)
    assert reopened.current() == session.store.current()
    session.prepare(2)
    assert session.call("video_handoff", {"camera_id": session.camera}).remaining == (
        middle.frame_index,
    )
    session.edit((estimated,))
    summary = summarize_annotation(session.store.annotation(session.store.current()[0]))
    assert (
        summary.processed,
        summary.unprocessed,
        summary.final_unknown,
        summary.position_usable,
        summary.direct_observation,
        summary.estimated,
    ) == (3, 0, 1, 2, 1, 1)
    assert summary.complete
    outcome = session.worker.validate(
        session.layout,
        session.plan,
        session.request,
        session.directory,
        session.decide(),
        session.execution(),
        None,
    )
    receipt = read_document(session.layout, outcome.adoption, VideoCompletion)
    assert receipt.revision == session.store.current()[0]
    assert session.store.annotation(receipt.revision).frames[1] == estimated


def test_batch_save_is_atomic_idempotent_and_uses_cas(session: BlankSession) -> None:
    head = session.store.current()[0]
    rows = final_rows(session)
    bad = rows[-1].model_copy(update={"pts": rows[-1].pts + 1})
    query = EditQuery(
        camera_id=session.camera,
        expected_revision=head,
        rows=(rows[0], bad),
        reason="bad source PTS",
    )
    with pytest.raises(ValueError, match="PTS"):
        session.store.apply(query, session.actor)
    assert session.store.current()[0] == head
    query = query.model_copy(
        update={"rows": (rows[0],), "reason": "first final judgment"}
    )
    saved = session.store.apply(query, session.actor)
    assert session.store.apply(query, session.actor) == saved
    with pytest.raises(RevisionConflict):
        session.store.apply(
            query.model_copy(update={"rows": (rows[1],), "reason": "stale"}),
            session.actor,
        )
    assert session.store.current()[0] == saved


@pytest.mark.parametrize("boundary", ["completion_file", "seal_commit"])
def test_recovery_after_completion_publication_or_seal_reuses_original_adoption(
    session: BlankSession, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    session.edit(final_rows(session))
    decision, execution = session.decide(), session.execution()
    with monkeypatch.context() as patch:
        if boundary == "completion_file":
            write = store_module.write_document

            def interrupted(layout, path, value):
                result = write(layout, path, value)
                if path.name == "completion.json":
                    raise RuntimeError("receipt interrupted")
                return result

            patch.setattr(store_module, "write_document", interrupted)
        else:
            seal = VideoRevisionStore.complete

            def interrupted_seal(self, *args):
                seal(self, *args)
                raise RuntimeError("receipt interrupted")

            patch.setattr(VideoRevisionStore, "complete", interrupted_seal)
        with pytest.raises(RuntimeError, match="receipt interrupted"):
            session.worker.validate(
                session.layout,
                session.plan,
                session.request,
                session.directory,
                decision,
                execution,
                None,
            )
    path = session.store.directory / "completion.json"
    original = path.stat().st_ino, path.read_bytes()
    one = session.worker.validate(
        session.layout,
        session.plan,
        session.request,
        session.directory,
        decision,
        execution,
        None,
    )
    two = session.worker.validate(
        session.layout,
        session.plan,
        session.request,
        session.directory,
        decision,
        execution,
        None,
    )
    assert one.adoption == two.adoption == session.store.current()[1]
    assert (path.stat().st_ino, path.read_bytes()) == original


def test_final_result_cannot_hide_unprocessed_or_change_pinned_policy(
    session: BlankSession,
) -> None:
    with pytest.raises(ValueError, match="unprocessed"):
        session.worker.validate(
            session.layout,
            session.plan,
            session.request,
            session.directory,
            session.decide(),
            session.execution(),
            None,
        )
    other = BallVideoWorker(
        AnnotationPolicy(
            policy_id="sparse", version="2", instructions="choose sparse anchors"
        )
    )
    with pytest.raises((ValueError, FileExistsError)):
        other.prepare(
            session.layout,
            session.plan,
            session.request,
            session.directory.parent / "attempt-0002",
            timeout_seconds=10,
        )
    with pytest.raises(ValueError, match="timeout"):
        session.worker.prepare(
            session.layout,
            session.plan,
            session.request,
            session.directory,
            timeout_seconds=1,
        )


@pytest.mark.parametrize(
    "change",
    [
        "camera",
        "frame",
        "crop",
        "path",
        "overlay",
        "prediction",
        "reference",
        "image_id",
        "oversized",
        "handoff",
    ],
)
def test_parameters_cannot_inject_other_camera_paths_overlays_or_unbounded_inputs(
    session: BlankSession, change: str
) -> None:
    parameters: dict[str, Any] = dict(
        camera_id=session.camera,
        tiles=[dict(frame_index=1, crop=dict(left=0, top=0, right=32, bottom=32))],
        columns=1,
        tile_width=64,
        tile_height=48,
        reason="native crop",
    )
    if change == "camera":
        parameters["camera_id"] = "cam2"
    elif change == "frame":
        parameters["tiles"][0]["frame_index"] = 4
    elif change == "crop":
        parameters["tiles"][0]["crop"]["right"] = 65
    elif change == "oversized":
        parameters["tiles"] *= 17
    elif change == "handoff":
        with pytest.raises(ValueError, match="camera"):
            session.call("video_handoff", {"camera_id": "cam2"})
        return
    else:
        parameters[{"overlay": "drawing"}.get(change, change)] = (
            "../cam2/reference.json"
        )
    with pytest.raises(ValueError):
        session.call("video_image", parameters)


def test_parent_stage_and_broker_repair_delivery_without_repeating_image_work(
    session: BlankSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.annotation.contracts import (
        OperationInput,
        OperationReceipt,
        ReviewOperation,
    )
    from src.tasks.annotation.operations import execute
    from src.tasks.annotation.services import ServiceBroker
    from src.tasks.base.annotation.artifacts import annotation_reference, write_document
    from src.tasks.base.annotation.workers import services
    from src.tasks.base.annotation.workers.ledger import WorkerLedger
    from src.tasks.base.annotation.workers.services import ServiceRequest

    service_ref = session.service_ref(
        "video_image",
        dict(
            camera_id=session.camera,
            tiles=[dict(frame_index=3, crop=dict(left=0, top=0, right=32, bottom=32))],
            columns=1,
            tile_width=64,
            tile_height=48,
            reason="parent stage",
        ),
    )
    service = read_document(session.layout, service_ref, ServiceRequest)
    operation = OperationInput(
        plan=session.receipt.plan,
        operation=ReviewOperation(service_request=service_ref),
    )
    op_ref = write_document(
        session.layout,
        session.layout.path(
            "campaigns",
            session.plan.campaign_id,
            "jobs",
            "test-service",
            "operation.json",
        ),
        operation,
    )
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", op_ref.digest)
    with monkeypatch.context() as patch:
        write = services.write_document

        def interrupt(layout, path, document):
            if "exchange" in path.parts and path.name == "reply.json":
                raise OSError("delivery interrupted")
            return write(layout, path, document)

        patch.setattr(services, "write_document", interrupt)
        with pytest.raises(OSError, match="delivery interrupted"):
            execute(session.layout, op_ref)
    reply_path = session.directory / "services" / service.sha256 / "reply.json"
    mirror = session.directory / "exchange" / service.sha256 / "reply.json"
    assert reply_path.exists() and not mirror.exists()
    image = session.directory / "exchange" / service.sha256 / "page.jpg"
    identity = image.stat().st_ino, image.read_bytes()

    def forbidden(*args):
        raise AssertionError("repair must reuse the completed operation")

    broker = ServiceBroker(
        session.layout, session.receipt.plan, 10, forbidden, forbidden
    )
    ledger = WorkerLedger(session.layout)
    broker.collect(session.request, ledger.task(session.request.work_id), ledger)
    assert mirror.read_bytes() == reply_path.read_bytes()
    assert (image.stat().st_ino, image.read_bytes()) == identity
    receipt = read_document(
        session.layout, execute(session.layout, op_ref), OperationReceipt
    )
    assert len(receipt.results) == 1
    assert annotation_reference(session.layout, mirror).size_bytes > 0


def test_services_reject_foreign_task_reference_and_inactive_attempt(
    session: BlankSession,
) -> None:
    import copy

    from src.tasks.ball_detection.annotation.video.agent.services import execute_service
    from src.tasks.base.annotation.artifacts import write_document
    from src.tasks.base.annotation.workers.contracts import WorkRequest
    from src.tasks.base.annotation.workers.services import ServiceRequest

    other = copy.copy(session)
    other.request = read_document(
        session.layout, session.receipt.registrations[1].request, WorkRequest
    )
    other.prepare(1)
    forged = ServiceRequest(
        task_request=other.task_ref,
        operation="video_handoff",
        parameters={"camera_id": other.camera},
    )
    reference = write_document(
        session.layout,
        session.directory / "work/service-requests" / (forged.sha256 + ".json"),
        forged,
    )
    with pytest.raises(ValueError, match="inbox"):
        execute_service(session.layout, session.plan, reference)
    old_ref = session.service_ref("video_handoff", {"camera_id": session.camera})
    session.prepare(2)
    with pytest.raises(ValueError, match="running worker attempt"):
        execute_service(session.layout, session.plan, old_ref)
