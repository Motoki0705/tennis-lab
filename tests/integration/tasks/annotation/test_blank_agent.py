from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import av
import numpy as np
from PIL import Image

from src.tasks.annotation.configuration import configure_local_videos
from src.tasks.annotation.local_videos import LocalVideoConfigurationReceipt
from src.tasks.ball_detection.annotation.video.agent.contracts import (
    AgentTask,
    VideoCompletion,
)
from src.tasks.ball_detection.annotation.video.agent.store import VideoRevisionStore
from src.tasks.ball_detection.annotation.video.agent.worker import BallVideoWorker
from src.tasks.ball_detection.annotation.video.io import read_annotation
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_asset,
)
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.images import read_rendered
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import LocalAnnotationPlan
from src.tasks.base.annotation.telemetry.contracts import RenderedImage
from src.tasks.base.annotation.workers.contracts import AttemptResult, WorkRequest
from src.tasks.base.annotation.workers.pool import WorkerPool
from tests.support.tasks.blank_agent import BlankSession, drive, fake_worker
from tests.support.tasks.local_annotation import local_configuration


def test_image_service_preserves_native_source_frames_crop_scale_and_pixels(
    tmp_path: Path,
) -> None:
    session = BlankSession(tmp_path / "annotation")
    view = session.call(
        "video_image",
        dict(
            camera_id=session.camera,
            tiles=[
                dict(frame_index=i, crop=dict(left=5, top=7, right=45, bottom=37))
                for i in (3, 1)
            ],
            columns=2,
            tile_width=80,
            tile_height=60,
            reason="reverse temporal anchors",
        ),
    )
    render, metadata = read_rendered(
        session.layout, RenderedImage(image=view.image, metadata=view.metadata)
    )
    assert render.drawing == "plain" and all(not t.drawings for t in render.tiles)
    assert [t.source_frame_index for t in metadata.tiles] == [3, 1]
    assert [(t.scale_x, t.scale_y) for t in metadata.tiles] == [("2", "2"), ("2", "2")]
    with av.open(session.configuration.videos[0].path) as video:
        source = [frame.to_image() for frame in video.decode(video=0)]
    assert view.image_path is not None
    with Image.open(BytesIO(Path(view.image_path).read_bytes())) as page:
        for tile in metadata.tiles:
            rect = tile.image_rect
            actual = np.asarray(
                page.crop(
                    (
                        rect.left,
                        rect.top,
                        rect.left + rect.width,
                        rect.top + rect.height,
                    )
                )
            ).astype(float)
            expected = np.asarray(
                source[tile.source_frame_index]
                .crop((5, 7, 45, 37))
                .resize((80, 60), Image.Resampling.BILINEAR)
            ).astype(float)
            wrong = np.asarray(
                source[2]
                .crop((5, 7, 45, 37))
                .resize((80, 60), Image.Resampling.BILINEAR)
            ).astype(float)
            assert np.abs(actual - expected).mean() < 10
            assert np.abs(actual - wrong).mean() > 25
    assert view.camera_id == "cam0"


def test_supervised_worker_blocks_direct_and_descendant_host_reads_and_completes(
    tmp_path: Path,
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    configuration = local_configuration(layout)
    receipt_ref = configure_local_videos(layout, configuration)
    receipt = read_document(layout, receipt_ref, LocalVideoConfigurationReceipt)
    plan = read_document(layout, receipt.plan, LocalAnnotationPlan)
    request = read_document(layout, receipt.registrations[0].request, WorkRequest)
    foreign = read_document(layout, receipt.registrations[1].request, WorkRequest)
    from src.tasks.base.annotation.contracts import ClipManifest
    from src.tasks.base.annotation.editing.contracts import BlankAnnotationContext

    foreign_context = read_document(layout, foreign.payload, BlankAnnotationContext)
    foreign_manifest = read_document(layout, foreign_context.manifest, ClipManifest)
    prediction = write_asset(
        layout, layout.path("predictions", "forbidden.json"), b'{"x": 99}'
    )
    reference = write_asset(
        layout, layout.path("annotations", "reference.json"), b'{"ground_truth": true}'
    )
    source_home = layout.path("cache", "codex-home")
    other_session = write_asset(
        layout,
        source_home / "sessions/another-camera.json",
        b"private other-camera history",
    )
    probes = tuple(
        str(layout.artifact_path(r))
        for r in (
            foreign_context.source_media,
            foreign_manifest.media,
            foreign_context.task_input,
            prediction,
            reference,
            other_session,
            receipt.plan,
        )
    ) + (configuration.videos[1].path,)
    pool = WorkerPool(
        layout,
        plan,
        (BallVideoWorker(),),
        test_command=fake_worker(probes=probes),
        codex_home=source_home,
        timeout_seconds=10,
    )
    assert pool.start(request)
    directory = layout.path(
        "campaigns", plan.campaign_id, "workers", request.work_id, "attempt-0001"
    )
    try:
        state = drive(layout, plan, pool, request, directory)
        assert state["state"] == "complete", state
        assert json.loads(
            (directory / "work/boundary-checked.json").read_text()
        ) == list(probes)
        final = read_document(
            layout, ArtifactRef.model_validate_json(state["result"]), AttemptResult
        )
        completion = read_document(layout, final.outcome.adoption, VideoCompletion)
        annotation = read_annotation(
            layout.artifact_path(completion.annotation), require_complete=True
        )
        assert [f.state for f in annotation.frames] == [
            "unknown",
            "in_frame",
            "in_frame",
        ]
        assert [
            f.position.provenance.kind for f in annotation.frames if f.position
        ] == ["estimated", "estimated"]
        assert pool.start(request) is None
        assert pool.reconcile(request)["result"] == state["result"]
    finally:
        if pool.ledger.task(request.work_id)["state"] in {"running", "exited"}:
            pool.stop(request, reason="test cleanup")


def test_real_interruption_before_final_response_keeps_saved_unknown_and_resumes_only_remaining(
    tmp_path: Path,
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    receipt = read_document(
        layout,
        configure_local_videos(layout, local_configuration(layout)),
        LocalVideoConfigurationReceipt,
    )
    plan = read_document(layout, receipt.plan, LocalAnnotationPlan)
    request = read_document(layout, receipt.registrations[0].request, WorkRequest)
    worker = BallVideoWorker()
    pool = WorkerPool(
        layout,
        plan,
        (worker,),
        test_command=fake_worker(pause_after_save=True),
        timeout_seconds=10,
    )
    pool.start(request)
    base = layout.path("campaigns", plan.campaign_id, "workers", request.work_id)
    first = base / "attempt-0001"
    try:
        drive(layout, plan, pool, request, first, saved_only=True)
        task = read_document(
            layout, annotation_reference(layout, first / "request.json"), AgentTask
        )
        store = VideoRevisionStore(layout, task.run)
        saved = store.current()[0]
        before = store.annotation(saved)
        assert [f.state for f in before.frames] == [
            "unknown",
            "unprocessed",
            "unprocessed",
        ]
        assert not (first / "work/result.json").exists()
        assert (
            pool.stop(request, reason="interrupt after durable edit")["state"]
            == "cancelled"
        )
        pool.test_command = fake_worker()
        pool.start(request, retry=True)
        second = base / "attempt-0002"
        assert (
            read_document(
                layout, annotation_reference(layout, second / "request.json"), AgentTask
            ).initial_revision
            == saved
        )
        state = drive(layout, plan, pool, request, second)
        assert state["state"] == "complete", state
        handoff_event = next(
            json.loads(line)["envelope"]
            for line in (second / "events.jsonl").read_text().splitlines()
            if json.loads(line).get("type") == "tool.synthetic"
        )
        assert handoff_event["view"]["remaining"] == [2, 3]
        assert handoff_event["view"]["summary"]["final_unknown"] == 1
        assert store.annotation(saved) == before
    finally:
        if pool.ledger.task(request.work_id)["state"] in {"running", "exited"}:
            pool.stop(request, reason="test cleanup")
