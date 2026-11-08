from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest

from src.tasks.annotation.benchmark.blank import (
    BlankComparison,
    BlankCondition,
    compare_blank_conditions,
    register_blank_condition,
)
from src.tasks.annotation.configuration import configure_local_videos
from src.tasks.annotation.local_videos import LocalVideoConfigurationReceipt
from src.tasks.ball_detection.annotation.video.agent.contracts import (
    AgentTask,
    AnnotationPolicy,
)
from src.tasks.ball_detection.annotation.video.agent.store import VideoRevisionStore
from src.tasks.ball_detection.annotation.video.agent.worker import BallVideoWorker
from src.tasks.ball_detection.annotation.video.contracts import (
    PixelPosition,
    PointProvenance,
    VideoAnnotationInput,
    VideoBallAnnotation,
    VideoBallFrame,
)
from src.tasks.ball_detection.annotation.video.io import write_annotation
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.publisher import (
    BallPublisher,
)
from src.tasks.base.annotation.artifacts import annotation_reference, read_document
from src.tasks.base.annotation.editing.contracts import BlankAnnotationContext
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import LocalAnnotationPlan
from src.tasks.base.annotation.telemetry.capture import verify_report
from src.tasks.base.annotation.workers.contracts import WorkRequest
from src.tasks.base.annotation.workers.ledger import WorkerLedger
from src.tasks.base.annotation.workers.pool import WorkerPool
from tests.support.tasks.blank_agent import drive, fake_worker, synthetic_telemetry
from tests.support.tasks.local_annotation import local_configuration


def test_two_strategies_share_exact_inputs_but_not_progress_and_compare_usage_and_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    receipt = read_document(
        layout,
        configure_local_videos(layout, local_configuration(layout)),
        LocalVideoConfigurationReceipt,
    )
    policies = (
        AnnotationPolicy(
            policy_id="chronological",
            version="1",
            instructions="Inspect chronological overview anchors; choose crops as needed.",
        ),
        AnnotationPolicy(
            policy_id="sparse",
            version="2",
            instructions="Choose sparse temporal anchors, infer and adaptively revisit.",
        ),
    )
    conditions = tuple(
        register_blank_condition(layout, receipt.plan, policy.policy_id, policy)
        for policy in policies
    )
    pending = read_document(
        layout, compare_blank_conditions(layout, conditions), BlankComparison
    )
    assert pending.acceptance == "unverified"
    assert all(
        camera.worker_kind == "not_started"
        and camera.usage.totals["input_tokens"] is None
        for group in pending.results
        for camera in group
    )
    references = []
    for entry in receipt.registrations:
        context = read_document(layout, entry.context, BlankAnnotationContext)
        expected = read_document(layout, context.task_input, VideoAnnotationInput)
        annotation = VideoBallAnnotation(
            schema_version="video_ball_annotation.v3",
            coordinate_system="native_pixels_top_left_endpoint",
            source=expected.source,
            target=expected.target,
            frame_range=expected.frame_range,
            frames=tuple(
                VideoBallFrame(
                    **f.model_dump(),
                    state="unknown" if index == 0 else "in_frame",
                    occlusion="unknown" if index == 0 else "not_occluded",
                    position=None
                    if index == 0
                    else PixelPosition(
                        x=15, y=13, provenance=PointProvenance(kind="observed")
                    ),
                )
                for index, f in enumerate(expected.frames)
            ),
        )
        references.append(
            write_annotation(
                layout,
                layout.path(
                    "annotations", "reference-" + expected.source.camera_id + ".json"
                ),
                annotation,
                expected_input=expected,
            )
        )
    for index, condition_ref in enumerate(conditions):
        condition = read_document(layout, condition_ref, BlankCondition)
        plan = read_document(layout, condition.plan, LocalAnnotationPlan)
        local = layout.for_benchmark(plan.campaign_id)
        worker = BallVideoWorker(condition.policy)
        pool = WorkerPool(
            local, plan, (worker,), test_command=fake_worker(), timeout_seconds=10
        )
        monkeypatch.setattr(
            pool,
            "_telemetry",
            partial(synthetic_telemetry, local, worker, missing=index == 1),
        )
        for request_ref in condition.requests:
            request = read_document(local, request_ref, WorkRequest)
            source_request = read_document(
                layout,
                next(
                    e.request
                    for e in receipt.registrations
                    if read_document(layout, e.request, WorkRequest).subject_id
                    == request.subject_id
                ),
                WorkRequest,
            )
            assert request.inputs == source_request.inputs
            assert WorkerLedger(layout).task(request.work_id)["state"] == "ready"
            assert pool.start(request)
            directory = local.path(
                "campaigns",
                plan.campaign_id,
                "workers",
                request.work_id,
                "attempt-0001",
            )
            try:
                state = drive(local, plan, pool, request, directory)
                assert state["state"] == "complete", state
                task = read_document(
                    local,
                    annotation_reference(local, directory / "request.json"),
                    AgentTask,
                )
                with pytest.raises(ValueError, match="benchmark"):
                    VideoRevisionStore(layout, task.run).initialize()
                history = pool.ledger.history(plan, request)
                report = verify_report(local, history[0].telemetry)
                assert report.responses == 4 and report.duplicate_usage_records == 1
                assert report.image_inputs == 2 and report.unbound_image_inputs == 0
            finally:
                if pool.ledger.task(request.work_id)["state"] in {"running", "exited"}:
                    pool.stop(request, reason="synthetic comparison cleanup")
        with pytest.raises(ValueError, match="benchmark"):
            BallPublisher(local)
        assert (
            register_blank_condition(
                layout, receipt.plan, plan.campaign_id, condition.policy
            )
            == condition_ref
        )
    result = compare_blank_conditions(
        layout, conditions, reference_annotations=tuple(references)
    )
    assert (
        compare_blank_conditions(
            layout, conditions, reference_annotations=tuple(references)
        )
        == result
    )
    report = read_document(layout, result, BlankComparison)
    for index, cameras in enumerate(report.results):
        assert {camera.camera_id for camera in cameras} == {"cam0", "cam2"}
        for camera in cameras:
            assert camera.worker_kind == "synthetic"
            assert (
                camera.image_requests == 2
                and camera.frames_requested == 2
                and camera.revisited_frames == 1
            )
            assert camera.usage.known_responses == 4
            assert (
                camera.usage.totals["input_tokens"],
                camera.usage.totals["output_tokens"],
                camera.usage.totals["cached_input_tokens"],
            ) == ((400, 40, 160) if index == 0 else (None, None, None))
            if index == 1:
                assert camera.usage.known_subtotals["input_tokens"] == 300
                assert camera.usage.known_subtotals["cached_input_tokens"] == 80
                assert camera.responses[1].usage is None
            assert camera.responses[-1].summary["unprocessed"] == 0
            assert camera.responses[-1].summary["final_unknown"] == 1
            assert camera.responses[-1].mean_reference_error_px == 5
            assert camera.responses[2].revisited_frames == 1
            assert [f.frame_index for f in camera.frames] == (
                [1, 2, 3] if camera.camera_id == "cam0" else [2, 3]
            )
            assert [f.error_px for f in camera.frames] == (
                [None, 5, 5] if camera.camera_id == "cam0" else [None, 5]
            )
            assert {f.camera_id for f in camera.frames} == {camera.camera_id}
    assert report.results[0][0].run != report.results[1][0].run
    for entry in receipt.registrations:
        request = read_document(layout, entry.request, WorkRequest)
        assert WorkerLedger(layout).task(request.work_id)["state"] == "ready"
        assert (
            not layout.artifact_path(entry.context).with_name("video-run.json").exists()
        )
    with pytest.raises((ValueError, FileExistsError)):
        register_blank_condition(layout, receipt.plan, "chronological", policies[1])
