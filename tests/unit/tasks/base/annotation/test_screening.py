from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from src.tasks.annotation.contracts import (
    AcquireOperation,
    OperationInput,
    OperationReceipt,
    PredictOperation,
    PrepareOperation,
    Runtime,
    ScreenOperation,
    StageCost,
)
from src.tasks.annotation.controller import Controller
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.collection.contracts import (
    PreparationConfig,
    PreparationReceipt,
    SelectionReceipt,
    SourceCandidate,
)
from src.tasks.base.annotation.collection.service import Collection, SourceExcluded
from src.tasks.base.annotation.contracts import ArtifactRef, ClipManifest
from src.tasks.base.annotation.execution.contracts import (
    QueueConfiguration,
    StageExit,
    StageLaunch,
)
from src.tasks.base.annotation.media import encode_video
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, file_digest, save_plan
from src.tasks.base.annotation.predictions import PredictionContext, PredictionReceipt
from src.tasks.base.annotation.resources import (
    BudgetLimits,
    ResourceManager,
    ResourceRequest,
)
from src.tasks.base.annotation.screening.contracts import (
    ClipDecision,
    Interval,
    PresentedImages,
    PreviewReceipt,
    ScreeningPolicy,
    ScreeningRequest,
    SourceDecision,
)
from src.tasks.base.annotation.screening.images import make_sheets
from src.tasks.base.annotation.screening.service import (
    Screening,
    attempt_directory,
    request_preview,
)
from src.tasks.base.annotation.workers.contracts import WorkOutcome, WorkRequest
from src.tasks.base.annotation.workers.services import (
    ServiceReply,
    ServiceRequest,
    request_service,
)
from src.utils.paths import PROJECT_ROOT


@dataclass
class Example:
    layout: AnnotationLayout
    screening: Screening
    source: SourceCandidate
    request_ref: ArtifactRef
    request: ScreeningRequest
    detail_ref: ArtifactRef
    detail: PreviewReceipt

    def decision(self, *, empty: bool = False) -> SourceDecision:
        return SourceDecision(
            source_sha256=self.request.source.media.digest,
            requested_maximum=2,
            status="insufficient",
            match_type="singles",
            surface="fixture",
            setting="fixture",
            camera="fixture",
            clips=()
            if empty
            else (
                ClipDecision(
                    start=0.0,
                    stop=15.0,
                    normal_speed=True,
                    in_play=(Interval(start=2.0, stop=12.0),),
                    reason="test fixture, not a real judgment",
                    evidence_images=tuple(
                        page.image.path for page in self.detail.pages
                    ),
                ),
            ),
            rejected=(),
            notes="only one fixture window qualifies" if not empty else "no play",
        )

    def finish(
        self,
        *,
        decision: SourceDecision | None = None,
        shown: tuple[ArtifactRef, ...] | None = None,
        name: str = "normal",
    ) -> ArtifactRef:
        decision = self.decision() if decision is None else decision
        directory = attempt_directory(self.layout, self.request)
        result = write_document(
            self.layout, directory / "work" / f"{name}.json", decision
        )
        overview = read_document(self.layout, self.request.overview, PreviewReceipt)
        shown = (
            tuple(
                page.image
                for preview in (overview, self.detail)
                for page in preview.pages
            )
            if shown is None
            else shown
        )
        events = write_asset(
            self.layout,
            self.layout.path("campaigns", "fixture", f"{name}.events.txt"),
            b"SYNTHETIC TEST PRESENTATION EVENTS: no real model was used\n",
        )
        presentation = write_document(
            self.layout,
            self.layout.path("campaigns", "fixture", f"{name}.presentation.json"),
            PresentedImages(
                request_digest=self.request_ref.digest, events=events, images=shown
            ),
        )
        return self.screening.finalize(
            self.request_ref, result, presentation, (self.detail_ref,)
        )


def controller_for(example: Example) -> Controller:
    layout, plan = example.layout, example.screening.plan
    resources = ResourceManager.create(
        layout,
        scope_id="preview",
        limits=BudgetLimits.from_budget(plan.resources),
        roots=(layout.repository,),
        previously_allocated_bytes=0,
        previously_used_gpu_seconds=0,
        max_pending_per_stage=10,
    )
    script = layout.repository / "queue.sh"
    script.write_text("# synthetic queue --idempotency-key\n")
    return Controller(
        layout,
        Runtime(
            plan=annotation_reference(layout, save_plan(layout, plan)),
            scope=resources.reference,
            queue=QueueConfiguration(
                script=str(script),
                script_digest=file_digest(script),
                directory=str(layout.repository / ".training_queue"),
                session="fixture",
                issue=1088,
            ),
            codex_binary="codex",
            codex_home=str(layout.path("cache", "codex")),
            costs=tuple(
                StageCost(
                    stage=stage.stage,
                    pss_bytes=256 * 1024**2,
                    storage_bytes=10 * 1024**2,
                    timeout_seconds=20.0,
                    gpu_seconds=30.0 if stage.kind == "gpu" else 0.0,
                )
                for stage in plan.resources.stages
            ),
        ),
    )


def setup(
    tmp_path: Path,
    seconds: int = 32,
    *,
    fps: int = 10,
    timestamps_ms: tuple[int, ...] | None = None,
) -> tuple[AnnotationLayout, Screening, SourceCandidate]:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    inputs = layout.path("campaigns", "inputs")
    policy = write_document(layout, inputs / "policy.json", ScreeningPolicy())
    prep = write_document(
        layout,
        inputs / "preparation.json",
        PreparationConfig(
            format_selector="bestvideo",
            crf=25,
            preset="ultrafast",
            minimum_clip_frames=32,
            maximum_unprepared_sources=1,
        ),
    )
    reference = policy.model_dump(mode="json")
    plan = CampaignPlan.model_validate_json(
        json.dumps(
            {
                "campaign_id": "fixture",
                "phase": "preview",
                "annotation_root": str(layout.root),
                "counts": {
                    "channels": 1,
                    "videos_per_channel": 1,
                    "max_clips_per_video": 2,
                },
                "selection": reference,
                "preparation": prep.model_dump(),
                "code_commit": "a" * 40,
                "code_diff_digest": "b" * 64,
                "agent_model": "gpt-6.1-sol",
                "agent_effort": "high",
                "resources": {
                    "codex_workers": 1,
                    "additional_storage_bytes": 30 * 1024**3,
                    "minimum_free_bytes": 1,
                    "total_pss_bytes": 22 * 1024**3,
                    "gpu_seconds": 43200,
                    "stages": [
                        {
                            "stage": "screen",
                            "kind": "codex",
                            "parallel": 1,
                            "threads_per_process": 1,
                        },
                        {
                            "stage": "render",
                            "kind": "cpu",
                            "parallel": 1,
                            "threads_per_process": 1,
                        },
                        {
                            "stage": "history",
                            "kind": "io",
                            "parallel": 1,
                            "threads_per_process": 1,
                        },
                        {
                            "stage": "prepare",
                            "kind": "cpu",
                            "parallel": 1,
                            "threads_per_process": 1,
                        },
                        {
                            "stage": "acquire",
                            "kind": "io",
                            "parallel": 1,
                            "threads_per_process": 1,
                        },
                        {
                            "stage": "infer_ball",
                            "kind": "gpu",
                            "parallel": 1,
                            "threads_per_process": 1,
                        },
                    ],
                },
                "tasks": [
                    {
                        "task": "ball",
                        "model": {
                            "task": "ball",
                            "version": "fixture",
                            "training_weights": reference,
                            "inference_weights": reference,
                            "configuration": reference,
                        },
                        "inference": reference,
                        "review": None,
                        "learning": None,
                        "products": [
                            {
                                "product": "ball",
                                "dataset_task": "ball_detection",
                                "dataset_version": "fixture",
                                "component": "frames",
                                "base_snapshot": None,
                                "generation_policy": reference,
                            }
                        ],
                    }
                ],
            }
        )
    )
    media_path = inputs / "video.mp4"
    if timestamps_ms is None:
        timestamps_ms = tuple(range(0, seconds * 1000, 1000 // fps))
    encode_video(
        media_path,
        width=64,
        height=32,
        time_base=Fraction(1, 1000),
        rate=Fraction(fps),
        frames=(
            (
                av.VideoFrame.from_ndarray(
                    np.full((32, 64, 3), i % 255, np.uint8), format="rgb24"
                ),
                timestamp,
                following - timestamp,
            )
            for i, (timestamp, following) in enumerate(
                zip(
                    timestamps_ms,
                    (*timestamps_ms[1:], seconds * 1000),
                    strict=True,
                )
            )
        ),
        crf=25,
        preset="ultrafast",
    )
    source = SourceCandidate(
        provider="local",
        channel_id="fixture",
        video_id="fixture",
        title="fixture",
        origin_url="local:fixture",
        selection_reason="synthetic fixture",
        evidence=policy,
        local_media=annotation_reference(layout, media_path),
    )
    collection = Collection(layout, plan)
    collection.register([source])
    collection.acquire(source.source_id)
    return layout, Screening(collection), source


@pytest.fixture
def example(tmp_path: Path) -> Example:
    layout, screening, source = setup(tmp_path)
    ref = screening.request(
        screening.prepare(source.source_id),
        layout.path(
            "campaigns",
            screening.plan.campaign_id,
            "workers",
            source.source_id,
            "attempt-1",
        ),
    )
    request = read_document(layout, ref, ScreeningRequest)
    detail = request_preview(layout, request, start=0.0, stop=15.0, step=0.5)
    return Example(
        layout,
        screening,
        source,
        ref,
        request,
        detail,
        read_document(layout, detail, PreviewReceipt),
    )


def test_real_timeline_selection_and_resume(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert len(example.detail.pages) == 2
    samples = [sample for page in example.detail.pages for sample in page.samples]
    assert [sample.seconds for sample in samples] == [index / 2 for index in range(30)]
    assert (
        example.screening.request(
            example.request.prepared_input,
            attempt_directory(example.layout, example.request),
        )
        == example.request_ref
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            example.screening.collection,
            "select",
            lambda _: (_ for _ in ()).throw(RuntimeError("ledger gap")),
        )
        with pytest.raises(RuntimeError, match="ledger gap"):
            example.finish()
    receipt = example.finish()
    assert example.finish() == receipt
    selection = read_document(example.layout, receipt, SelectionReceipt)
    assert len(selection.windows) == 1
    assert (selection.windows[0].start, selection.windows[0].stop) == (0.0, 15.0)
    assert (selection.windows[0].target_start, selection.windows[0].target_stop) == (
        1.0,
        14.0,
    )
    assert selection.shortage_reason == "only one fixture window qualifies"
    assert (
        example.screening.collection.ledger.source(
            example.screening.plan, example.source.source_id
        )["state"]
        == "screened"
    )
    prepared = example.screening.collection.prepare(example.source.source_id)
    assert prepared.size_bytes > 0


@pytest.mark.parametrize(
    ("fps", "timestamps_ms", "start"),
    [
        pytest.param(1, None, 0.0, id="low-fps"),
        pytest.param(
            10,
            tuple(t for t in range(0, 32000, 100) if not 11600 <= t < 12600),
            0.0,
            id="vfr-gap-between-pages",
        ),
        pytest.param(
            10,
            (0, *range(1000, 32000, 100)),
            0.1,
            id="vfr-window-start-gap",
        ),
        pytest.param(
            10,
            tuple(t for t in range(0, 32000, 100) if not 14100 <= t < 15000),
            0.0,
            id="vfr-window-end-gap",
        ),
    ],
)
def test_detail_requires_actual_pts_density_at_boundaries_and_between_samples(
    tmp_path: Path, fps: int, timestamps_ms: tuple[int, ...] | None, start: float
) -> None:
    layout, screening, source = setup(tmp_path, fps=fps, timestamps_ms=timestamps_ms)
    ref = screening.request(
        screening.prepare(source.source_id),
        layout.path(
            "campaigns",
            screening.plan.campaign_id,
            "workers",
            source.source_id,
            "attempt-1",
        ),
    )
    request = read_document(layout, ref, ScreeningRequest)
    detail_ref = request_preview(
        layout, request, start=start, stop=start + 15.0, step=0.5
    )
    detail = read_document(layout, detail_ref, PreviewReceipt)
    example = Example(layout, screening, source, ref, request, detail_ref, detail)
    decision = example.decision()
    clip = decision.clips[0].model_copy(update={"start": start, "stop": start + 15.0})
    with pytest.raises(ValueError, match="actual PTS"):
        example.finish(decision=decision.model_copy(update={"clips": (clip,)}))
    assert (
        screening.collection.ledger.source(screening.plan, source.source_id)["state"]
        == "acquired"
    )
    assert not (attempt_directory(layout, request) / "selection.json").exists()


@pytest.mark.parametrize("outside_gap", [False, True])
def test_vfr_detail_with_sufficient_actual_pts_density_can_be_selected(
    tmp_path: Path, outside_gap: bool
) -> None:
    # Variable frame durations, with an optional sparse interval outside the clip.
    timestamps = tuple(
        base + offset
        for base in range(0, 32000, 500)
        for offset in (0, 100, 300)
        if not (outside_gap and 25000 < base + offset < 26000)
    )
    layout, screening, source = setup(tmp_path, timestamps_ms=timestamps)
    ref = screening.request(
        screening.prepare(source.source_id),
        layout.path(
            "campaigns",
            screening.plan.campaign_id,
            "workers",
            source.source_id,
            "attempt-1",
        ),
    )
    request = read_document(layout, ref, ScreeningRequest)
    detail_ref = request_preview(layout, request, start=0.0, stop=32.0, step=0.5)
    detail = read_document(layout, detail_ref, PreviewReceipt)
    example = Example(layout, screening, source, ref, request, detail_ref, detail)
    selection = read_document(layout, example.finish(), SelectionReceipt)
    assert len(selection.windows) == 1
    assert (selection.windows[0].start, selection.windows[0].stop) == (0.0, 15.0)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"in_play": (Interval(start=0.0, stop=7.5),)}, "strictly more"),
        ({"normal_speed": False}, "slow-motion"),
        ({"stop": 16.0}, "15 seconds"),
        (
            {
                "start": 20.0,
                "stop": 35.0,
                "in_play": (Interval(start=22.0, stop=32.0),),
            },
            "beyond",
        ),
    ],
)
def test_invalid_window_is_not_selected(
    example: Example, change: dict[str, object], message: str
) -> None:
    decision = example.decision()
    clip = ClipDecision.model_validate_json(
        json.dumps(decision.clips[0].model_copy(update=change).model_dump(mode="json"))
    )
    with pytest.raises(ValueError, match=message):
        example.finish(decision=decision.model_copy(update={"clips": (clip,)}))
    assert (
        example.screening.collection.ledger.source(
            example.screening.plan, example.source.source_id
        )["state"]
        == "acquired"
    )


def test_generated_or_partly_shown_images_cannot_be_adopted(example: Example) -> None:
    overview = read_document(example.layout, example.request.overview, PreviewReceipt)
    with pytest.raises(ValueError, match="overview pages"):
        example.finish(shown=(), name="none")
    with pytest.raises(ValueError, match="unpresented"):
        example.finish(
            shown=tuple(page.image for page in overview.pages), name="overview-only"
        )
    decision = example.decision()
    clip = decision.clips[0].model_copy(
        update={"evidence_images": (example.detail.pages[0].image.path,)}
    )
    with pytest.raises(ValueError, match="whole window"):
        example.finish(
            decision=decision.model_copy(update={"clips": (clip,)}),
            shown=(
                *tuple(page.image for page in overview.pages),
                example.detail.pages[0].image,
            ),
            name="half-detail",
        )


def test_empty_selection_preserves_reason_and_releases_lead(example: Example) -> None:
    receipt = example.finish(decision=example.decision(empty=True))
    selection = read_document(example.layout, receipt, SelectionReceipt)
    assert selection.windows == () and selection.shortage_reason == "no play"
    assert (
        example.screening.collection.ledger.source(
            example.screening.plan, example.source.source_id
        )["state"]
        == "excluded"
    )


def test_source_hash_and_image_tampering_are_detected(example: Example) -> None:
    with pytest.raises(ValueError, match="different source"):
        example.finish(
            decision=example.decision().model_copy(update={"source_sha256": "f" * 64}),
            name="bad-source",
        )
    image = example.layout.artifact_path(example.detail.pages[0].image)
    image.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="size changed"):
        example.finish(name="bad-image")


def test_short_source_is_reasoned_exclusion(tmp_path: Path) -> None:
    layout, screening, source = setup(tmp_path, 3)
    with pytest.raises(SourceExcluded, match="duration"):
        screening.prepare(source.source_id)
    row = screening.collection.ledger.source(screening.plan, source.source_id)
    assert row["state"] == "excluded"
    selection = read_document(
        layout, ArtifactRef.model_validate_json(row["selection"]), SelectionReceipt
    )
    assert selection.windows == () and "preflight" in (selection.shortage_reason or "")


def test_preview_atomic_publish_cache_and_invalid_interval(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(
            "src.tasks.base.annotation.screening.images.os.rename",
            lambda *_: (_ for _ in ()).throw(OSError("interrupted")),
        )
        with pytest.raises(OSError, match="interrupted"):
            request_preview(
                example.layout, example.request, start=16.0, stop=31.0, step=0.5
            )
    directory = attempt_directory(example.layout, example.request) / "previews"
    # The source overview is shared outside the worker; only the first detail is here.
    assert len(list(directory.iterdir())) == 1
    ref = request_preview(
        example.layout, example.request, start=16.0, stop=31.0, step=0.5
    )
    monkeypatch.setattr(
        "src.tasks.base.annotation.screening.images.av.open",
        lambda *_: pytest.fail("cache must not decode"),
    )
    assert (
        request_preview(
            example.layout, example.request, start=16.0, stop=31.0, step=0.5
        )
        == ref
    )
    with pytest.raises(ValueError, match="beyond"):
        request_preview(
            example.layout, example.request, start=30.0, stop=40.0, step=0.5
        )
    with pytest.raises(ValueError, match="invalid"):
        make_sheets(
            example.layout,
            example.request.source.media,
            directory,
            origin_pts=0,
            time_base="1/10",
            start=0.0,
            stop=15.0,
            step=0.0,
        )
    example.layout.artifact_path(example.request.source.media).write_bytes(b"changed")
    with pytest.raises(ValueError, match="size changed"):
        request_preview(
            example.layout, example.request, start=16.0, stop=31.0, step=0.5
        )


def test_decision_overlap_and_preview_clock_are_rejected(example: Example) -> None:
    decision = example.decision()
    overlapping = decision.model_copy(
        update={"status": "accepted", "clips": decision.clips * 2}
    )
    with pytest.raises(ValueError, match="nonoverlapping"):
        SourceDecision.model_validate_json(overlapping.canonical_bytes())
    page = example.detail.pages[0]
    sample = page.samples[0].model_copy(update={"seconds": 0.2})
    corrupted_page = page.model_copy(update={"samples": (sample, *page.samples[1:])})
    preview = example.detail.model_copy(
        update={"pages": (corrupted_page, *example.detail.pages[1:])}
    )
    with pytest.raises(ValueError, match="source clock"):
        PreviewReceipt.model_validate_json(preview.canonical_bytes())


def test_wrong_presentation_request_and_coarse_evidence_fail(example: Example) -> None:
    directory = attempt_directory(example.layout, example.request)
    result = write_document(
        example.layout, directory / "work" / "result.json", example.decision()
    )
    events = write_asset(
        example.layout, directory / "test.events", b"synthetic fixture"
    )
    presentation = write_document(
        example.layout,
        directory / "wrong-presentation.json",
        PresentedImages(
            request_digest="d" * 64,
            events=events,
            images=(),
        ),
    )
    with pytest.raises(ValueError, match="another request"):
        example.screening.finalize(
            example.request_ref, result, presentation, (example.detail_ref,)
        )
    overview = read_document(example.layout, example.request.overview, PreviewReceipt)
    decision = example.decision()
    clip = decision.clips[0].model_copy(
        update={"evidence_images": (overview.pages[0].image.path,)}
    )
    with pytest.raises(ValueError, match="whole window"):
        example.finish(
            decision=decision.model_copy(update={"clips": (clip,)}), name="coarse-only"
        )


def test_cli_boundary_rejects_relative_and_external_request_paths(
    tmp_path: Path,
) -> None:
    from src.tasks.base.annotation.screening.__main__ import PATH_BOUNDARY
    from src.utils.configuration import PathContractError

    layout = AnnotationLayout.testing(tmp_path / "annotation")
    request = layout.path("campaigns", "fixture", "request.json")
    request.parent.mkdir(parents=True)
    request.write_text("{}")
    resolved = PATH_BOUNDARY.validate({"request": request}, resolver=layout.resolver)
    assert resolved.declared("request").path == request
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    for candidate in (Path("relative.json"), outside):
        with pytest.raises(PathContractError):
            PATH_BOUNDARY.validate({"request": candidate}, resolver=layout.resolver)


def test_parent_stage_renders_worker_images_without_treating_them_as_review(
    example: Example,
) -> None:
    controller = controller_for(example)
    layout = example.layout
    operation = OperationInput(
        plan=controller.runtime.plan,
        operation=ScreenOperation(source_id=example.source.source_id),
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        state = controller.operation(operation)
        if state["state"] == "succeeded":
            break
        assert state["state"] in {"running", "draining"}, state
        time.sleep(0.02)
    assert state["state"] == "succeeded", state
    prepared = controller.receipt(state).results[0]
    request = WorkRequest(
        adapter="source_screening",
        adapter_version="v1",
        stage="screen",
        subject_id=example.source.source_id,
        payload=prepared,
        inputs=(),
    )
    # This is a synthetic worker protocol. It does not call Codex or claim image inspection.
    script = r"""
import json, pathlib, sys
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.artifacts import annotation_reference
from src.tasks.base.annotation.workers.services import ServiceRequest, request_service
layout = AnnotationLayout.testing(pathlib.Path(sys.argv[1]))
directory = pathlib.Path.cwd().parent
ref = annotation_reference(layout, directory / "request.json")
request_service(layout, directory, ServiceRequest(task_request=ref, operation="screen_preview", parameters={"start":5.0,"stop":20.0,"step":0.5}), timeout_seconds=15)
pathlib.Path("result.json").write_text('{"synthetic":true}')
print(json.dumps({"type":"thread.started","thread_id":"synthetic-only"}))
print(json.dumps({"type":"turn.completed"}))
"""
    controller.pool.test_command = (sys.executable, "-c", script, str(layout.root))
    controller.pool.start(request)
    deadline = time.monotonic() + 20
    row = controller.pool.reconcile(request)
    try:
        while row["state"] in {"running", "exited"} and time.monotonic() < deadline:
            controller._services(request, row)
            time.sleep(0.02)
            row = controller.pool.reconcile(request)
        assert row["state"] == "validation_failed", row
        assert "actual identified rollout" in row["reason"]
        directory = layout.path(
            "campaigns",
            controller.plan.campaign_id,
            "workers",
            request.work_id,
            "attempt-0001",
        )
        (reply_path,) = (directory / "services").glob("*/reply.json")
        reply = read_document(
            layout, annotation_reference(layout, reply_path), ServiceReply
        )
        assert reply.status == "succeeded" and reply.result is not None
        preview = read_document(layout, reply.result, PreviewReceipt)
        assert (preview.start, preview.stop, preview.step) == (5.0, 20.0, 0.5)
        assert all(layout.artifact_path(page.image).is_file() for page in preview.pages)
        (service_path,) = (directory / "work/service-requests").glob("*.json")
        service = read_document(
            layout, annotation_reference(layout, service_path), ServiceRequest
        )
        assert (
            request_service(layout, directory, service, timeout_seconds=1)
            == reply.result
        )
        assert (
            controller.collection.ledger.source(
                controller.plan, example.source.source_id
            )["state"]
            == "acquired"
        )
    finally:
        if row["state"] in {"running", "exited"}:
            controller.pool.stop(request, reason="fixture cleanup")


def test_worker_image_service_rejects_other_attempt_and_excessive_sampling(
    example: Example,
) -> None:
    from src.tasks.annotation.operations import preview_input

    layout = example.layout
    directory = attempt_directory(layout, example.request)
    request = ServiceRequest(
        task_request=example.request_ref,
        operation="screen_preview",
        parameters={"start": 0.0, "stop": 15.0, "step": 0.001},
    )
    path = directory / "work/service-requests" / (request.sha256 + ".json")
    ref = write_document(layout, path, request)
    with pytest.raises(ValueError, match="sampling bounds"):
        preview_input(layout, ref)
    moved = write_document(
        layout, directory / "work" / (request.sha256 + ".json"), request
    )
    with pytest.raises(ValueError, match="binding"):
        preview_input(layout, moved)


def test_operation_cli_cannot_bypass_accounted_stage(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.annotation.operations import execute

    monkeypatch.delenv("TENNIS_ANNOTATION_STAGE_OPERATION", raising=False)
    with pytest.raises(ValueError, match="resource-accounted stage"):
        execute(example.layout, example.request_ref)


@pytest.mark.parametrize("remove_old", [False, True])
def test_service_limit_survives_inbox_reordering_deletion_and_restart(
    example: Example, monkeypatch: pytest.MonkeyPatch, remove_old: bool
) -> None:
    controller = controller_for(example)
    runtime = controller.runtime.model_copy(update={"maximum_services_per_attempt": 1})
    controller = Controller(example.layout, runtime)
    request = WorkRequest(
        adapter="source_screening",
        adapter_version="v1",
        stage="screen",
        subject_id=example.source.source_id,
        payload=example.request.prepared_input,
        inputs=(),
    )
    controller.pool.test_command = (sys.executable, "-c", "import time; time.sleep(60)")
    controller.pool.start(request)
    owning_pool = controller.pool
    row = controller.pool.ledger.task(request.work_id)
    directory = example.layout.path(
        "campaigns",
        controller.plan.campaign_id,
        "workers",
        request.work_id,
        "attempt-0001",
    )
    official = annotation_reference(example.layout, directory / "request.json")
    services = sorted(
        (
            ServiceRequest(
                task_request=official,
                operation="screen_preview",
                parameters={
                    "start": float(start),
                    "stop": float(start + 15),
                    "step": 0.5,
                },
            )
            for start in range(3)
        ),
        key=lambda service: service.sha256,
        reverse=True,
    )
    submitted: list[OperationInput] = []

    def submit(data: OperationInput) -> dict[str, Any]:
        submitted.append(data)
        return {"state": "succeeded"}

    monkeypatch.setattr(Controller, "operation", lambda self, data: submit(data))
    monkeypatch.setattr(
        Controller,
        "receipt",
        lambda self, state: OperationReceipt(
            operation=example.request_ref,
            status="complete",
            results=(example.detail_ref,),
            reason=None,
        ),
    )
    try:
        for index, service in enumerate(services):
            path = directory / "work/service-requests" / (service.sha256 + ".json")
            write_document(example.layout, path, service)
            controller._services(request, row)
            reply = read_document(
                example.layout,
                annotation_reference(
                    example.layout,
                    directory / "services" / service.sha256 / "reply.json",
                ),
                ServiceReply,
            )
            assert reply.status == ("succeeded" if index == 0 else "failed")
            if index:
                assert "request count" in (reply.reason or "")
            # Reprocessing the accepted request must not consume another slot.
            controller._services(request, row)
            if remove_old:
                path.unlink()
            controller = Controller(example.layout, runtime)
        assert len(submitted) == 1
    finally:
        deadline = time.monotonic() + 10
        while not (directory / "owner.json").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        owning_pool.stop(request, reason="fixture cleanup")


@pytest.mark.parametrize("empty", [False, True])
def test_controller_recovers_worker_after_source_finalized_before_worker_receipt(
    example: Example, monkeypatch: pytest.MonkeyPatch, empty: bool
) -> None:
    controller = controller_for(example)
    request = WorkRequest(
        adapter="source_screening",
        adapter_version="v1",
        stage="screen",
        subject_id=example.source.source_id,
        payload=example.request.prepared_input,
        inputs=(),
    )
    controller.pool.test_command = (
        sys.executable,
        "-c",
        "import pathlib; pathlib.Path('result.json').write_text('{}'); "
        'print(\'{"type":"thread.started","thread_id":"synthetic"}\'); '
        'print(\'{"type":"turn.completed"}\')',
    )

    def finalize(*args: Any) -> WorkOutcome:
        return WorkOutcome(
            status="not_applicable" if empty else "complete",
            receipt=example.finish(decision=example.decision(empty=empty)),
            reason="no play" if empty else None,
        )

    monkeypatch.setattr(
        "src.tasks.annotation.screening.ScreeningAdapter.validate", finalize
    )
    controller.pool.start(request)
    original_write = write_document

    def interrupt(layout: AnnotationLayout, path: Path, value: Any) -> ArtifactRef:
        if path.name == "result.receipt.json":
            raise RuntimeError("controller interrupted after source finalization")
        return original_write(layout, path, value)

    with monkeypatch.context() as patch:
        patch.setattr(
            "src.tasks.base.annotation.workers.pool.write_document", interrupt
        )
        with pytest.raises(RuntimeError, match="after source finalization"):
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                controller.pool.reconcile(request)
                time.sleep(0.02)
    assert controller.collection.ledger.source(
        controller.plan, example.source.source_id
    )["state"] == ("excluded" if empty else "screened")
    assert controller.pool.ledger.task(request.work_id)["state"] in {
        "running",
        "exited",
    }
    restarted = Controller(example.layout, controller.runtime)
    # Downstream processing is outside this recovery test.
    monkeypatch.setattr(restarted, "operation", lambda data: {"state": "waiting"})
    restarted.cycle()
    assert restarted.pool.ledger.task(request.work_id)["state"] == (
        "not_applicable" if empty else "complete"
    )
    assert restarted.status()["state"] != "running"


def record_stage(
    controller: Controller,
    data: OperationInput,
    *,
    attempt: int = 1,
    results: tuple[ArtifactRef, ...] | None = None,
    exited: bool = False,
    failed: bool = False,
) -> ArtifactRef:
    """Publish synthetic supervisor evidence without starting a queue or model."""
    layout = controller.layout
    request_id = f"{data.sha256}-{attempt}"
    directory = layout.path(
        "campaigns", controller.plan.campaign_id, "jobs", request_id
    )
    operation = write_document(layout, directory / "operation.json", data)
    cost = controller.runtime.cost(data.stage)
    launch = write_document(
        layout,
        directory / "launch.json",
        StageLaunch(
            scope=controller.runtime.scope,
            resource=ResourceRequest(
                request_id=request_id,
                plan=controller.runtime.plan,
                stage=data.stage,
                pss_bytes=cost.pss_bytes,
                storage_bytes=cost.storage_bytes,
                gpu_seconds=cost.gpu_seconds,
            ),
            command=("synthetic-stage",),
            input=operation,
            code_root=str(PROJECT_ROOT),
            timeout_seconds=cost.timeout_seconds,
            queue=controller.runtime.queue if cost.gpu_seconds else None,
            test_root=str(layout.root),
        ),
    )
    if results is not None:
        write_document(
            layout,
            directory / "result.json",
            OperationReceipt(
                operation=operation,
                status="complete",
                results=results,
                reason=None,
            ),
        )
    if exited:
        log = write_asset(layout, directory / "log.txt", b"synthetic supervisor\n")
        write_document(
            layout,
            directory / "exit.json",
            StageExit(
                launch=launch,
                status="failed" if failed else "succeeded",
                exit_code=1 if failed else 0,
                started_at="2026-10-16T00:00:00Z",
                ended_at="2026-10-16T00:00:01Z",
                elapsed_seconds=1.0,
                stdout=log,
                stderr=log,
                reason="synthetic failure" if failed else None,
            ),
        )
    return operation


@pytest.mark.parametrize(
    "cut", ["operation_receipt", "stage_exit", "failed_exit", "missing_result"]
)
def test_preview_requires_latest_receipts_and_explicit_prediction_recovery(
    example: Example, monkeypatch: pytest.MonkeyPatch, cut: str
) -> None:
    from src.tasks.annotation import operations

    controller = controller_for(example)
    example.finish()
    prepared = controller.collection.prepare(example.source.source_id)
    source = controller.collection.ledger.source(
        controller.plan, example.source.source_id
    )
    for operation, result in (
        (
            AcquireOperation(source_id=example.source.source_id),
            ArtifactRef.model_validate_json(source["acquisition"]),
        ),
        (
            ScreenOperation(source_id=example.source.source_id),
            example.request.prepared_input,
        ),
        (PrepareOperation(source_id=example.source.source_id), prepared),
    ):
        record_stage(
            controller,
            OperationInput(plan=controller.runtime.plan, operation=operation),
            results=(result,),
            exited=True,
        )
    clips = read_document(example.layout, prepared, PreparationReceipt).clips

    class SyntheticPredictionAdapter:
        task = "ball"
        produced = 0

        def produce(
            self,
            layout: AnnotationLayout,
            context: PredictionContext,
            manifest: ClipManifest,
            directory: Path,
        ) -> ArtifactRef:
            self.produced += 1
            return write_asset(
                layout, directory / "prediction.json", b"synthetic prediction"
            )

        def validate(
            self,
            layout: AnnotationLayout,
            context: PredictionContext,
            manifest: ClipManifest,
            result: ArtifactRef,
        ) -> None:
            assert layout.artifact_path(result).read_bytes() == b"synthetic prediction"

    adapter = SyntheticPredictionAdapter()
    monkeypatch.setattr(
        "src.tasks.ball_detection.annotation.inference.BallPredictionAdapter",
        lambda: adapter,
    )
    data = OperationInput(
        plan=controller.runtime.plan,
        operation=PredictOperation(task="ball", clips=clips),
    )
    reference = record_stage(controller, data)
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", reference.digest)

    def interrupt(layout: AnnotationLayout, path: Path, value: Any) -> ArtifactRef:
        if path.name == "result.json":
            raise RuntimeError("interrupted after prediction commit")
        return write_document(layout, path, value)

    if cut in {"operation_receipt", "missing_result"}:
        with monkeypatch.context() as patch:
            patch.setattr(operations, "write_document", interrupt)
            with pytest.raises(RuntimeError, match="after prediction commit"):
                operations.execute(example.layout, reference)
    else:
        operations.execute(example.layout, reference)
    if cut == "failed_exit":
        record_stage(controller, data, exited=True, failed=True)
    elif cut == "missing_result":
        record_stage(controller, data, exited=True)
    # A terminal queue entry without StageExit represents an unrecovered supervisor.
    monkeypatch.setattr(
        "src.tasks.base.annotation.execution.queue.queue_state",
        lambda config, ref: {"state": "interrupted", "reason": "no completion receipt"},
    )
    restarted = Controller(example.layout, controller.runtime)
    status = restarted.status()
    assert status["task_states"] == {"predicted": 1}
    assert status["state"] == "blocked"
    assert status["failures"]
    # Explicit new attempt reads and validates the immutable prediction without inference.
    record_stage(restarted, data, attempt=9, exited=True, failed=True)
    recovery = record_stage(restarted, data, attempt=10)
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", recovery.digest)
    operations.execute(example.layout, recovery)
    assert adapter.produced == 1
    record_stage(restarted, data, attempt=10, exited=True)
    assert restarted.status()["state"] == "preview_ready"
    assert not restarted.status()["failures"]
    receipt = read_document(
        example.layout,
        annotation_reference(
            example.layout,
            example.layout.artifact_path(recovery).parent / "result.json",
        ),
        OperationReceipt,
    )
    prediction = read_document(example.layout, receipt.results[0], PredictionReceipt)
    example.layout.artifact_path(prediction.prediction.artifact).write_bytes(
        b"tampered"
    )
    assert restarted.status()["state"] == "blocked"
