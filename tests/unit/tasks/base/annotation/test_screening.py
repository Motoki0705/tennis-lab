from __future__ import annotations

import json
import math
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
    StageLaunch,
)
from src.tasks.base.annotation.media import encode_video
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, file_digest, save_plan
from src.tasks.base.annotation.predictions import (
    PredictionContext,
    PredictionReceipt,
    Predictions,
)
from src.tasks.base.annotation.resources import (
    BudgetLimits,
    ResourceManager,
)
from src.tasks.base.annotation.screening.contracts import (
    ClipDecision,
    Interval,
    PresentedImages,
    PreviewReceipt,
    ScreeningInput,
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
from src.tasks.base.annotation.storage import StorageCatalog
from src.tasks.base.annotation.workers.contracts import WorkOutcome, WorkRequest
from src.tasks.base.annotation.workers.services import (
    ServiceReply,
    ServiceRequest,
    request_service,
)
from src.utils.paths import PROJECT_ROOT
from tests.support.tasks.annotation import record_stage, record_stage_exit


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
    return create_controller(example.layout, example.screening.plan)


def create_controller(layout: AnnotationLayout, plan: CampaignPlan) -> Controller:
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
    fps: int | Fraction = 10,
    timestamps_ms: tuple[int, ...] | None = None,
    jpeg: bool = False,
    acquire: bool = True,
    videos_per_channel: int = 1,
) -> tuple[AnnotationLayout, Screening, SourceCandidate]:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    inputs = layout.path("campaigns", "inputs")
    policy = write_document(layout, inputs / "policy.json", ScreeningPolicy())
    prep = write_document(
        layout,
        inputs / "preparation.json",
        PreparationConfig(
            format_selector="bestvideo",
            crf=None if jpeg else 25,
            preset=None if jpeg else "ultrafast",
            retained_media="jpeg_q90" if jpeg else "video",
            source_retention="delete_after_verified_frames" if jpeg else "keep",
            excluded_source_retention="delete_after_verified_exclusion" if jpeg else "keep",
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
                    "videos_per_channel": videos_per_channel,
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
        time_base = 1 / Fraction(fps)
        timestamps = tuple(range(math.ceil(seconds * fps)))
        end_pts = len(timestamps)
    else:
        time_base = Fraction(1, 1000)
        timestamps = timestamps_ms
        end_pts = seconds * 1000
    encode_video(
        media_path,
        width=64,
        height=32,
        time_base=time_base,
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
                    timestamps,
                    (*timestamps[1:], end_pts),
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
    if acquire:
        collection.acquire(source.source_id)
    return layout, Screening(collection), source


@pytest.fixture
def example(tmp_path: Path) -> Example:
    layout, screening, source = setup(tmp_path)
    return make_example(layout, screening, source)


def make_example(
    layout: AnnotationLayout, screening: Screening, source: SourceCandidate
) -> Example:
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


@pytest.mark.parametrize("fps", [25, Fraction(30000, 1001)], ids=["25fps", "29.97fps"])
@pytest.mark.parametrize("start", [0.0, 0.01])
def test_standard_detail_sampling_finalizes_at_fractional_frame_steps(
    tmp_path: Path,
    fps: int | Fraction,
    start: float,
) -> None:
    layout, screening, source = setup(tmp_path, fps=fps)
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
    selection = read_document(
        layout,
        example.finish(decision=decision.model_copy(update={"clips": (clip,)})),
        SelectionReceipt,
    )
    assert len(selection.windows) == 1
    assert (selection.windows[0].start, selection.windows[0].stop) == (
        start,
        start + 15.0,
    )
    times = [Fraction(str(start))]
    times.extend(
        (sample.source_pts - request.source_origin_pts) * Fraction(request.time_base)
        for page in detail.pages
        for sample in page.samples
    )
    times.append(Fraction(str(start + 15.0)))
    assert all(
        0 <= b - a <= Fraction(1, 2) for a, b in zip(times, times[1:], strict=False)
    )
    assert len(times[1:-1]) == len(set(times[1:-1]))
    assert all(len(page.samples) <= 24 for page in detail.pages)


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
        pytest.param(
            {"in_play": (Interval(start=0.8, stop=8.3),)},
            "strictly more",
            id="nonzero-start-half-play",
        ),
        pytest.param(
            {"in_play": (Interval(start=0.8, stop=4.3), Interval(start=4.8, stop=8.8))},
            "strictly more",
            id="multiple-intervals-half-play",
        ),
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


@pytest.mark.parametrize(
    "in_play",
    [
        (Interval(start=0.8, stop=8.300000001),),
        (Interval(start=0.8, stop=4.3), Interval(start=4.8, stop=8.800000001)),
    ],
)
def test_play_just_over_half_is_accepted_without_rounding(
    in_play: tuple[Interval, ...],
) -> None:
    clip = ClipDecision(
        start=0.0,
        stop=15.0,
        normal_speed=True,
        in_play=in_play,
        reason="synthetic strict boundary test",
        evidence_images=("fixture.jpg",),
    )
    clip.validate_policy(ScreeningPolicy())


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


def excluded_worker(example: Example, selection: ArtifactRef, state: str) -> None:
    from src.tasks.base.annotation.storage.contracts import Protection
    from src.tasks.base.annotation.workers.contracts import (
        AttemptResult,
        ExitReceipt,
        Launch,
        PreparedWorker,
        WorkOutcome,
    )
    from src.tasks.base.annotation.workers.ledger import WorkerLedger

    layout, plan = example.layout, example.screening.plan
    request = WorkRequest(adapter="source_screening", adapter_version="v1", stage="screen",
                          subject_id=example.source.source_id, payload=example.request.prepared_input, inputs=())
    directory = layout.path("campaigns", "fixture", "fixture-attempt")
    launch = write_document(layout, directory / "launch.json", Launch(
        work_id=request.work_id,
        attempt="fixture-attempt",
        plan=annotation_reference(layout, save_plan(layout, plan)),
        request=request,
        prepared=PreparedWorker(prompt=write_asset(layout, directory / "prompt.txt", b"fixture")),
        directory=str(directory),
        code_root=str(PROJECT_ROOT),
        codex_binary="codex",
        codex_home=str(layout.repository / ".codex-fixture"),
        timeout_seconds=10,
        test_command=(sys.executable, "-c", "pass"),
        test_root=str(layout.repository),
    ))
    execution = write_document(layout, directory / "exit.json", ExitReceipt(
        launch_digest=launch.digest,
        exit_code=0,
        status="succeeded",
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:00:00Z",
        elapsed_seconds=0,
        events=write_asset(layout, directory / "events.jsonl", b""),
        errors=write_asset(layout, directory / "stderr.log", b""),
        thread_id=None,
        reason=None,
    ))
    result = write_document(layout, layout.path("campaigns", "fixture", "worker-result.json"), AttemptResult(
        launch=launch, exit=execution, result=selection, telemetry=None,
        outcome=WorkOutcome(status="not_applicable", receipt=selection, reason="no play"),
    ))
    ledger = WorkerLedger(layout)
    with ledger.registry.transaction() as db:
        db.execute("INSERT INTO worker_tasks VALUES (?,?,?,?,?,?,?,NULL)", (
            request.work_id, plan.campaign_id, plan.sha256, request.canonical_bytes().decode(),
            state, "fixture-attempt", result.canonical_bytes().decode(),
        ))
    StorageCatalog(layout).protect(Protection(
        protection_id="worker-inputs-v1-" + request.sha256, kind="active", reason="fixture screening inputs",
        references=(example.request.source.media, example.request.prepared_input, example.request.overview),
    ))


@pytest.mark.parametrize("state", ["running", "not_applicable"])
def test_excluded_source_waits_for_worker_and_retains_review_evidence(tmp_path: Path, state: str) -> None:
    sample = make_example(*setup(tmp_path, jpeg=True))
    selection = sample.finish(decision=sample.decision(empty=True))
    collection = sample.screening.collection
    video = sample.layout.artifact_path(sample.request.source.media)
    with pytest.raises(ValueError, match="completed screening worker"):
        collection.retire_excluded(sample.source.source_id)
    excluded_worker(sample, selection, state)
    if state != "not_applicable":
        with pytest.raises(ValueError, match="completed screening worker"):
            collection.retire_excluded(sample.source.source_id)
        assert video.is_file()
        return
    receipt = collection.retire_excluded(sample.source.source_id)
    assert not video.exists()
    assert sample.source.local_media is not None
    assert sample.layout.artifact_path(sample.source.local_media).is_file()
    assert collection.retire_excluded(sample.source.source_id) == receipt
    kept = {ref for p in StorageCatalog(sample.layout).snapshot().protections for ref in p.references}
    assert selection in kept and sample.request.source.media not in kept
    assert all(page.image in kept and sample.layout.artifact_path(page.image).is_file() for page in sample.detail.pages)
    assert collection.ledger.summary(collection.plan)["prepared_clips"] == 0
    from src.tasks.base.annotation.collection.restoration import RestorationReceipt

    restored = collection.restore_source(sample.source.source_id, attempt="restore-empty")
    assert read_document(sample.layout, restored, RestorationReceipt).status == "restored"
    assert video.is_file()
    assert collection.ledger.summary(collection.plan)["prepared_clips"] == 0


@pytest.mark.parametrize("excluded", [False, True])
@pytest.mark.parametrize("invalid", [None, "missing_intent", "retained_evidence"])
def test_acquisition_receipt_validates_retired_source_evidence(
    tmp_path: Path, excluded: bool, invalid: str | None
) -> None:
    sample = make_example(*setup(tmp_path, jpeg=True))
    controller = controller_for(sample)
    layout, collection = sample.layout, sample.screening.collection
    acquisition = ArtifactRef.model_validate_json(
        collection.ledger.source(collection.plan, sample.source.source_id)["acquisition"]
    )
    operation = record_stage(
        controller,
        OperationInput(
            plan=controller.runtime.plan,
            operation=AcquireOperation(source_id=sample.source.source_id),
        ),
        results=(acquisition,), exited=True,
    )
    state = controller.stages.status(annotation_reference(
        layout, layout.artifact_path(operation).parent / "launch.json"
    ))
    assert controller.receipt(state).results == (acquisition,)
    selection = sample.finish(decision=sample.decision(empty=excluded))
    if excluded:
        excluded_worker(sample, selection, "not_applicable")
        collection.retire_excluded(sample.source.source_id)
        retained = selection
    else:
        prepared = collection.prepare(sample.source.source_id)
        clip = read_document(layout, prepared, PreparationReceipt).clips[0]
        retained = read_document(layout, clip, ClipManifest).media
    assert not layout.artifact_path(sample.request.source.media).exists()
    if invalid == "missing_intent":
        layout.path(
            "sources", sample.source.source_id,
            collection.plan.preparation.digest, "retirement-intent.json",
        ).unlink()
    elif invalid == "retained_evidence":
        layout.artifact_path(retained).write_bytes(b"tampered retained evidence")
    if invalid is None:
        assert Controller(layout, controller.runtime).receipt(state).results == (acquisition,)
    else:
        with pytest.raises((ValueError, OSError)):
            Controller(layout, controller.runtime).receipt(state)


def test_excluded_source_tampering_blocks_deletion(tmp_path: Path) -> None:
    sample = make_example(*setup(tmp_path, jpeg=True))
    selection = sample.finish(decision=sample.decision(empty=True))
    excluded_worker(sample, selection, "not_applicable")
    sample.layout.artifact_path(sample.detail.pages[0].image).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="size changed"):
        sample.screening.collection.retire_excluded(sample.source.source_id)
    assert sample.layout.artifact_path(sample.request.source.media).is_file()


def test_excluded_source_recovers_after_unlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.tasks.base.annotation.collection import retirement

    sample = make_example(*setup(tmp_path, jpeg=True))
    selected = sample.finish(decision=sample.decision(empty=True))
    excluded_worker(sample, selected, "not_applicable")
    original = retirement.write_document

    def crash(layout: AnnotationLayout, path: Path, value: object) -> ArtifactRef:
        if path.name == "retired.json":
            raise OSError("fixture crash after unlink")
        return original(layout, path, value)

    with monkeypatch.context() as patch:
        patch.setattr(retirement, "write_document", crash)
        with pytest.raises(OSError, match="fixture crash"):
            sample.screening.collection.retire_excluded(sample.source.source_id)
    assert not sample.layout.artifact_path(sample.request.source.media).exists()
    result = sample.screening.collection.retire_excluded(sample.source.source_id)
    assert sample.layout.artifact_path(result).is_file()
    controller = controller_for(sample)

    def no_screening(*args: object) -> None:
        pytest.fail("retired source must not be screened or protected as live video again")

    monkeypatch.setattr(controller, "_screen", no_screening)
    monkeypatch.setattr(controller, "operation", lambda *a, **k: {"state": "succeeded"})
    controller.cycle()


def test_preflight_excluded_source_keeps_explicit_failure(tmp_path: Path) -> None:
    layout, screening, source = setup(tmp_path, 3, jpeg=True)
    with pytest.raises(SourceExcluded, match="duration"):
        screening.prepare(source.source_id)
    row = screening.collection.ledger.source(screening.plan, source.source_id)
    selection = read_document(layout, ArtifactRef.model_validate_json(row["selection"]), SelectionReceipt)
    result = screening.collection.retire_excluded(source.source_id)
    assert layout.artifact_path(result).is_file()
    assert layout.artifact_path(selection.evidence).is_file()


def test_excluded_deletion_requires_explicit_policy(example: Example) -> None:
    example.finish(decision=example.decision(empty=True))
    with pytest.raises(ValueError, match="not enabled"):
        example.screening.collection.retire_excluded(example.source.source_id)


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
    protected = {
        ref
        for pin in StorageCatalog(layout).snapshot().protections
        for ref in pin.references
    }
    source_input = read_document(layout, prepared, ScreeningInput)
    overview = read_document(layout, source_input.overview, PreviewReceipt)
    assert source_input.source.media in protected
    assert source_input.overview in protected
    assert all(page.image in protected for page in overview.pages)
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
    # Services now dispatch through the shared broker's bound callbacks.
    monkeypatch.setattr(controller.broker, "operation", controller.operation)
    monkeypatch.setattr(controller.broker, "receipt", controller.receipt)
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


class SyntheticPredictionAdapter:
    task = "ball"

    def produce(
        self,
        layout: AnnotationLayout,
        context: PredictionContext,
        manifest: ClipManifest,
        directory: Path,
    ) -> ArtifactRef:
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


def finish_preview(controller: Controller, example: Example) -> None:
    """Finish synthetic selection and prediction, retaining real receipt validation."""
    example.finish()
    prepared = controller.collection.prepare(example.source.source_id)
    for operation, result in (
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
    predictions = Predictions(example.layout, controller.plan)
    results = tuple(
        predictions.run(clip, SyntheticPredictionAdapter()) for clip in clips
    )
    record_stage(
        controller,
        OperationInput(
            plan=controller.runtime.plan,
            operation=PredictOperation(task="ball", clips=clips),
        ),
        results=results,
        exited=True,
    )


@pytest.mark.parametrize(
    "invalid", [None, "missing_acquisition", "wrong_source", "reason"]
)
@pytest.mark.parametrize("exclusion_stage", ["acquire", "prepare"])
def test_preview_accepts_duplicate_exclusion_with_verified_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str | None,
    exclusion_stage: str,
) -> None:
    from src.tasks.annotation.operations import execute

    layout, screening, source = setup(tmp_path, videos_per_channel=2)
    controller = create_controller(layout, screening.plan)
    acquisition = ArtifactRef.model_validate_json(
        controller.collection.ledger.source(controller.plan, source.source_id)[
            "acquisition"
        ]
    )
    record_stage(
        controller,
        OperationInput(
            plan=controller.runtime.plan,
            operation=AcquireOperation(source_id=source.source_id),
        ),
        results=(acquisition,),
        exited=True,
    )
    finish_preview(controller, make_example(layout, screening, source))
    duplicate = source.model_copy(update={"video_id": "duplicate"})
    if exclusion_stage == "prepare":
        # Different source bytes, but identical decoded frames in the selected clip.
        media_path = layout.path("campaigns", "inputs", "overlapping-source.mp4")
        encode_video(
            media_path, width=64, height=32, time_base=Fraction(1, 10),
            rate=Fraction(10),
            frames=(
                (
                    av.VideoFrame.from_ndarray(
                        np.full((32, 64, 3), (i if i < 160 else i + 100) % 255, np.uint8),
                        format="rgb24",
                    ),
                    i, 1,
                )
                for i in range(320)
            ),
            crf=25, preset="ultrafast",
        )
        duplicate = duplicate.model_copy(
            update={"local_media": annotation_reference(layout, media_path)}
        )
    controller.collection.register([duplicate])
    if exclusion_stage == "prepare":
        duplicate_acquisition = controller.collection.acquire(duplicate.source_id)
        record_stage(
            controller,
            OperationInput(
                plan=controller.runtime.plan,
                operation=AcquireOperation(source_id=duplicate.source_id),
            ),
            results=(duplicate_acquisition,), exited=True,
        )
        make_example(layout, screening, duplicate).finish(name="duplicate")
    data = OperationInput(
        plan=controller.runtime.plan,
        operation=AcquireOperation(source_id=duplicate.source_id)
        if exclusion_stage == "acquire"
        else PrepareOperation(source_id=duplicate.source_id),
    )
    reference = record_stage(controller, data)
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", reference.digest)
    receipt_ref = execute(layout, reference)
    record_stage(controller, data, exited=True)
    restarted = Controller(layout, controller.runtime)
    status = restarted.status()
    assert status["collection"]["sources_by_state"] == {"excluded": 1, "prepared": 1}
    assert status["state"] == "preview_ready"
    receipt = read_document(layout, receipt_ref, OperationReceipt)
    row = controller.collection.ledger.source(controller.plan, duplicate.source_id)
    assert receipt.status == "not_applicable"
    assert receipt.reason == row["reason"]
    if exclusion_stage == "acquire":
        assert receipt.reason == f"duplicate content of {source.source_id}"
    else:
        assert receipt.reason is not None and receipt.reason.startswith("duplicate clip content")
        assert row["preparation"] is not None
        # The excluded clip's preparation remains evidence, not completed work.
        read_document(
            layout, ArtifactRef.model_validate_json(row["preparation"]), PreparationReceipt
        )
    assert receipt.results == (ArtifactRef.model_validate_json(row["acquisition"]),)
    # Recover an already-excluded source with the same evidence in a new stage attempt.
    recovery = record_stage(controller, data, attempt=2)
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", recovery.digest)
    recovered = read_document(layout, execute(layout, recovery, attempt=2), OperationReceipt)
    record_stage(controller, data, attempt=2, exited=True)
    assert (recovered.results, recovered.reason) == (receipt.results, receipt.reason)
    assert controller.collection.ledger.source(controller.plan, duplicate.source_id)["preparation"] == row["preparation"]
    assert restarted.status()["state"] == "preview_ready"
    if invalid is not None:
        # A successful supervisor cannot make mismatched exclusion evidence valid.
        forged = record_stage(controller, data, attempt=3, exited=True)
        bad_receipt = recovered.model_copy(
            update={
                "operation": forged,
                "results": ()
                if invalid == "missing_acquisition"
                else (acquisition,)
                if invalid == "wrong_source"
                else recovered.results,
                "reason": "unregistered exclusion"
                if invalid == "reason"
                else recovered.reason,
            }
        )
        write_document(
            layout, layout.artifact_path(forged).parent / "result.json", bad_receipt
        )
        blocked = restarted.status()
        assert blocked["state"] == "blocked"
        assert any(
            failure["state"] == "invalid_receipt" for failure in blocked["failures"]
        )


@pytest.mark.parametrize("failed_retry", [False, True])
def test_explicit_acquisition_retry_recovers_failed_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_retry: bool
) -> None:
    from src.tasks.annotation.operations import execute

    layout, screening, source = setup(tmp_path, acquire=False)
    controller = create_controller(layout, screening.plan)
    original = OperationInput(
        plan=controller.runtime.plan,
        operation=AcquireOperation(source_id=source.source_id),
    )
    submitted: list[StageLaunch] = []

    def submit(launch: StageLaunch) -> dict[str, Any]:
        # Execute only the local acquisition; synthesize the supervisor's exit evidence.
        assert launch.input is not None and launch.queue is None
        directory = layout.artifact_path(launch.input).parent
        launch_ref = write_document(layout, directory / "launch.json", launch)
        monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", launch.input.digest)
        failed = False
        try:
            execute(
                layout, launch.input,
                attempt=int(launch.command[launch.command.index("--attempt") + 1]),
            )
        except RuntimeError as error:
            assert str(error) == "synthetic acquisition failure"
            failed = True
        record_stage_exit(layout, launch_ref, failed=failed)
        submitted.append(launch)
        state: dict[str, Any] = controller.stages.status(launch_ref)
        return state

    monkeypatch.setattr(controller.stages, "submit", submit)
    retry = OperationInput(
        plan=controller.runtime.plan,
        operation=AcquireOperation(source_id=source.source_id, retry=True),
    )

    def fail_acquisition(*args: Any) -> ArtifactRef:
        raise RuntimeError("synthetic acquisition failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            "src.tasks.base.annotation.collection.service.acquire_source",
            fail_acquisition,
        )
        assert controller.operation(original)["state"] == "failed"
        if failed_retry:
            assert controller.operation(retry)["state"] == "failed"
    # Neither ordinary resume nor repeating the same retry directive creates an attempt.
    assert controller.operation(original)["state"] == "failed"
    if failed_retry:
        assert controller.operation(retry)["state"] == "failed"
    assert len(submitted) == (2 if failed_retry else 1)
    assert controller.status()["state"] == "blocked"
    assert (
        controller.collection.ledger.source(controller.plan, source.source_id)["state"]
        == "failed"
    )
    recovered = controller.operation(retry, attempt=3 if failed_retry else None)
    assert recovered["state"] == "succeeded"
    number = 3 if failed_retry else 2
    assert submitted[-1].resource.request_id == f"{original.logical_id}-{number}"
    assert submitted[-1].input is not None
    assert read_document(layout, submitted[-1].input, OperationInput) == retry
    restarted = Controller(layout, controller.runtime)
    assert restarted.operation(original) == recovered
    assert restarted.operation(retry) == recovered
    assert len(submitted) == number
    with pytest.raises(ValueError, match="immutable document"):
        restarted.operation(original, attempt=number)
    finish_preview(controller, make_example(layout, screening, source))
    status = restarted.status()
    assert status["state"] == "preview_ready"
    assert not status["failures"]
    assert any(stage["state"] == "failed" for stage in status["stages"])


@pytest.mark.parametrize(
    "failure", ["timed_out", "failed", "validation_failed", "quota"]
)
def test_cli_retries_screening_with_new_worker_attempt(
    example: Example,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    from src.tasks.annotation import __main__ as cli
    from src.tasks.annotation import controller as controller_module

    controller = controller_for(example)
    layout = example.layout
    data = OperationInput(
        plan=controller.runtime.plan,
        operation=ScreenOperation(source_id=example.source.source_id),
    )
    stage = record_stage(
        controller, data, results=(example.request.prepared_input,), exited=True
    )
    launch = annotation_reference(
        layout, layout.artifact_path(stage).parent / "launch.json"
    )

    def screen_stage(self: Controller, operation: OperationInput) -> dict[str, Any]:
        assert operation == data
        state: dict[str, Any] = self.stages.status(launch)
        return state

    monkeypatch.setattr(Controller, "operation", screen_stage)
    request = WorkRequest(
        adapter="source_screening",
        adapter_version="v1",
        stage="screen",
        subject_id=example.source.source_id,
        payload=example.request.prepared_input,
        inputs=(),
    )
    success = (
        "import pathlib; pathlib.Path('result.json').write_text('{}'); "
        'print(\'{"type":"thread.started","thread_id":"synthetic"}\'); '
        'print(\'{"type":"turn.completed"}\')'
    )
    controller.pool.test_command = (
        sys.executable,
        "-c",
        "import time; time.sleep(30)"
        if failure == "timed_out"
        else "raise SystemExit(1)"
        if failure == "failed"
        else 'print(\'{"type":"error","message":"usage limit exceeded"}\')'
        if failure == "quota"
        else success,
    )
    if failure == "timed_out":
        controller.pool.timeout = 0.1

    def reject(*args: Any) -> WorkOutcome:
        raise ValueError("synthetic validation failure")

    monkeypatch.setattr(
        "src.tasks.annotation.screening.ScreeningAdapter.validate", reject
    )
    controller.pool.start(request)
    deadline = time.monotonic() + 15
    row = controller.pool.reconcile(request)
    while row["state"] in {"running", "exited"} and time.monotonic() < deadline:
        time.sleep(0.02)
        row = controller.pool.reconcile(request)
    assert row["state"] == failure
    first_attempt = row["current_attempt"]
    restarted = Controller(layout, controller.runtime)
    restarted.pool.test_command = (sys.executable, "-c", success)
    restarted.cycle()
    assert (
        restarted.pool.ledger.task(request.work_id)["current_attempt"] == first_attempt
    )
    assert restarted.status()["state"] == "blocked"

    def finalize(
        self: Any,
        layout: AnnotationLayout,
        plan: CampaignPlan,
        request: WorkRequest,
        directory: Path,
        *args: Any,
    ) -> WorkOutcome:
        request_ref = annotation_reference(layout, directory / "request.json")
        body = read_document(layout, request_ref, ScreeningRequest)
        detail = request_preview(layout, body, start=0.0, stop=15.0, step=0.5)
        retry_example = Example(
            layout,
            Screening(Collection(layout, plan)),
            example.source,
            request_ref,
            body,
            detail,
            read_document(layout, detail, PreviewReceipt),
        )
        return WorkOutcome(
            status="complete", receipt=retry_example.finish(name="worker-retry")
        )

    monkeypatch.setattr(
        "src.tasks.annotation.screening.ScreeningAdapter.validate", finalize
    )
    monkeypatch.setattr(
        controller_module, "Controller", lambda layout, runtime: restarted
    )
    runtime = write_document(
        layout, layout.path("campaigns", "fixture", "runtime.json"), controller.runtime
    )
    argv = [
        "annotation",
        "retry-screen",
        "--runtime",
        str(layout.artifact_path(runtime)),
        "--sha256",
        runtime.digest,
        "--test-root",
        str(layout.root),
        "--source-id",
        example.source.source_id,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    if failure == "quota":
        cli.main()
        assert json.loads(capsys.readouterr().out)["state"] == "waiting"
        assert (
            restarted.pool.ledger.task(request.work_id)["current_attempt"]
            == first_attempt
        )
        with restarted.pool.ledger.registry.transaction() as db:
            assert (
                db.execute("SELECT count(*) FROM worker_campaign_stops").fetchone()[0]
                == 1
            )
        monkeypatch.setattr(sys, "argv", [*argv, "--resume-quota"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["state"] in {"running", "complete"}
    deadline = time.monotonic() + 15
    row = restarted.pool.reconcile(request)
    while row["state"] in {"running", "exited"} and time.monotonic() < deadline:
        time.sleep(0.02)
        row = restarted.pool.reconcile(request)
    assert row["state"] == "complete"
    assert row["current_attempt"] == f"{request.work_id}-0002"
    with restarted.pool.ledger.registry.transaction() as db:
        attempts = db.execute(
            "SELECT number,state FROM worker_attempts WHERE work_id=? ORDER BY number",
            (request.work_id,),
        ).fetchall()
        assert [tuple(attempt) for attempt in attempts] == [
            (1, failure),
            (2, "complete"),
        ]
        assert (
            db.execute("SELECT count(*) FROM worker_campaign_stops").fetchone()[0] == 0
        )
