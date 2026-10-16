from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from src.tasks.annotation.contracts import (
    OperationInput,
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
    SelectionReceipt,
    SourceCandidate,
)
from src.tasks.base.annotation.collection.service import Collection, SourceExcluded
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.execution.contracts import QueueConfiguration
from src.tasks.base.annotation.media import encode_video
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, file_digest, save_plan
from src.tasks.base.annotation.resources import BudgetLimits, ResourceManager
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
from src.tasks.base.annotation.workers.contracts import WorkRequest
from src.tasks.base.annotation.workers.services import (
    ServiceReply,
    ServiceRequest,
    request_service,
)


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
                    gpu_seconds=0.0,
                )
                for stage in plan.resources.stages
            ),
        ),
    )


def setup(
    tmp_path: Path, seconds: int = 32, *, jpeg: bool = False
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
            excluded_source_retention="delete_after_verified_exclusion"
            if jpeg
            else "keep",
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
    encode_video(
        media_path,
        width=64,
        height=32,
        time_base=Fraction(1, 1000),
        rate=Fraction(10),
        frames=(
            (
                av.VideoFrame.from_ndarray(
                    np.full((32, 64, 3), i % 255, np.uint8), format="rgb24"
                ),
                i * 100,
                100,
            )
            for i in range(seconds * 10)
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
    return make_example(tmp_path)


def make_example(tmp_path: Path, *, jpeg: bool = False) -> Example:
    layout, screening, source = setup(tmp_path, jpeg=jpeg)
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


def excluded_worker(example: Example, selection: ArtifactRef, state: str) -> None:
    from src.tasks.base.annotation.storage.contracts import Protection
    from src.tasks.base.annotation.workers.contracts import AttemptResult, WorkOutcome
    from src.tasks.base.annotation.workers.ledger import WorkerLedger

    layout, plan = example.layout, example.screening.plan
    request = WorkRequest(
        adapter="source_screening",
        adapter_version="v1",
        stage="screen",
        subject_id=example.source.source_id,
        payload=example.request.prepared_input,
        inputs=(),
    )
    result = write_document(
        layout,
        layout.path("campaigns", "fixture", "worker-result.json"),
        AttemptResult(
            launch=selection,
            exit=selection,
            result=selection,
            telemetry=None,
            outcome=WorkOutcome(
                status="not_applicable", receipt=selection, reason="no play"
            ),
        ),
    )
    ledger = WorkerLedger(layout)
    with ledger.registry.transaction() as db:
        db.execute(
            "INSERT INTO worker_tasks VALUES (?,?,?,?,?,?,?,NULL)",
            (
                request.work_id,
                plan.campaign_id,
                plan.sha256,
                request.canonical_bytes().decode(),
                state,
                "fixture-attempt",
                result.canonical_bytes().decode(),
            ),
        )
    StorageCatalog(layout).protect(
        Protection(
            protection_id="worker-inputs-v1-" + request.sha256,
            kind="active",
            reason="fixture screening inputs",
            references=(
                example.request.source.media,
                example.request.prepared_input,
                example.request.overview,
            ),
        )
    )


@pytest.mark.parametrize("state", ["running", "not_applicable"])
def test_excluded_source_waits_for_worker_and_retains_review_evidence(
    tmp_path: Path, state: str
) -> None:
    sample = make_example(tmp_path, jpeg=True)
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
    kept = {
        ref
        for p in StorageCatalog(sample.layout).snapshot().protections
        for ref in p.references
    }
    assert selection in kept and sample.request.source.media not in kept
    assert all(
        page.image in kept and sample.layout.artifact_path(page.image).is_file()
        for page in sample.detail.pages
    )
    assert collection.ledger.summary(collection.plan)["prepared_clips"] == 0
    from src.tasks.base.annotation.collection.restoration import RestorationReceipt

    restored = collection.restore_source(
        sample.source.source_id, attempt="restore-empty"
    )
    assert (
        read_document(sample.layout, restored, RestorationReceipt).status == "restored"
    )
    assert video.is_file()
    assert collection.ledger.summary(collection.plan)["prepared_clips"] == 0


def test_excluded_source_tampering_blocks_deletion(tmp_path: Path) -> None:
    sample = make_example(tmp_path, jpeg=True)
    selection = sample.finish(decision=sample.decision(empty=True))
    excluded_worker(sample, selection, "not_applicable")
    sample.layout.artifact_path(sample.detail.pages[0].image).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="size changed"):
        sample.screening.collection.retire_excluded(sample.source.source_id)
    assert sample.layout.artifact_path(sample.request.source.media).is_file()


def test_excluded_source_recovers_after_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.base.annotation.collection import retirement

    sample = make_example(tmp_path, jpeg=True)
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
        pytest.fail(
            "retired source must not be screened or protected as live video again"
        )

    monkeypatch.setattr(controller, "_screen", no_screening)
    monkeypatch.setattr(controller, "operation", lambda *a, **k: {"state": "succeeded"})
    controller.cycle()


def test_preflight_excluded_source_keeps_explicit_failure(tmp_path: Path) -> None:
    layout, screening, source = setup(tmp_path, 3, jpeg=True)
    with pytest.raises(SourceExcluded, match="duration"):
        screening.prepare(source.source_id)
    row = screening.collection.ledger.source(screening.plan, source.source_id)
    selection = read_document(
        layout, ArtifactRef.model_validate_json(row["selection"]), SelectionReceipt
    )
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


@pytest.mark.parametrize(
    "reply_status,decision_status,empty,expected",
    [
        (None, "insufficient", False, "deferred"),
        ("failed", "insufficient", False, "deferred"),
        ("succeeded", "deferred", False, "deferred"),
        (None, "insufficient", True, "deferred"),
        ("succeeded", "insufficient", False, "complete"),
    ],
)
def test_technical_screening_incompletion_does_not_seal_or_exclude_source(
    example: Example,
    monkeypatch: pytest.MonkeyPatch,
    reply_status: str | None,
    decision_status: str,
    empty: bool,
    expected: str,
) -> None:
    from src.tasks.annotation.screening import ScreeningAdapter
    from src.tasks.base.annotation.workers.contracts import ExitReceipt

    layout = example.layout
    directory = attempt_directory(layout, example.request)
    service = ServiceRequest(
        task_request=example.request_ref,
        operation="screen_preview",
        parameters={"start": 0.0, "stop": 15.0, "step": 0.5},
    )
    write_document(
        layout, directory / "work/service-requests" / f"{service.sha256}.json", service
    )
    if reply_status is not None:
        write_document(
            layout,
            directory / "services" / service.sha256 / "reply.json",
            ServiceReply(
                request_digest=service.sha256,
                status="succeeded" if reply_status == "succeeded" else "failed",
                result=example.detail_ref if reply_status == "succeeded" else None,
                reason=None
                if reply_status == "succeeded"
                else "fixture render failure",
            ),
        )
    result = write_document(
        layout,
        directory / "work/result.json",
        example.decision(empty=empty).model_copy(update={"status": decision_status}),
    )
    events = write_asset(
        layout, directory / "fixture-events.txt", b"SYNTHETIC TEST ONLY"
    )
    overview = read_document(layout, example.request.overview, PreviewReceipt)
    shown = tuple(
        page.image for preview in (overview, example.detail) for page in preview.pages
    )
    monkeypatch.setattr(
        "src.tasks.annotation.screening.presented_images", lambda *_: shown
    )
    outcome = ScreeningAdapter().validate(
        layout,
        example.screening.plan,
        WorkRequest(
            adapter="source_screening",
            adapter_version="v1",
            stage="screen",
            subject_id=example.source.source_id,
            payload=example.request.prepared_input,
            inputs=(),
        ),
        directory,
        result,
        ExitReceipt(
            launch_digest="a" * 64,
            exit_code=0,
            status="succeeded",
            started_at="2026-01-01T00:00:00Z",
            ended_at="2026-01-01T00:00:01Z",
            elapsed_seconds=1.0,
            events=events,
            errors=events,
            thread_id="synthetic",
            reason=None,
        ),
        events,
    )
    assert outcome.status == expected
    row = example.screening.collection.ledger.source(
        example.screening.plan, example.source.source_id
    )
    if expected == "deferred":
        assert row["state"] == "acquired" and row["selection"] is None
        assert not (directory / "selection.json").exists()
        assert outcome.receipt == result
        assert "technically incomplete" in (outcome.reason or "")
    else:
        assert row["state"] == "screened"
        assert (
            len(read_document(layout, outcome.receipt, SelectionReceipt).windows) == 1
        )
    assert layout.artifact_path(example.request.source.media).is_file()


def test_deferred_decision_cannot_bypass_adapter_and_finalize(example: Example) -> None:
    decision = example.decision().model_copy(update={"status": "deferred"})
    with pytest.raises(ValueError, match="technically incomplete"):
        example.finish(decision=decision)
    row = example.screening.collection.ledger.source(
        example.screening.plan, example.source.source_id
    )
    assert row["state"] == "acquired" and row["selection"] is None


def sealed_screening_attempt(
    example: Example,
) -> tuple[Controller, WorkRequest, ArtifactRef, Path]:
    """Synthetic old-code completion with real selection/launch/exit artifacts."""
    from src.tasks.annotation.screening import ScreeningAdapter
    from src.tasks.base.annotation.workers.contracts import (
        AttemptResult,
        ExitReceipt,
        Launch,
        WorkOutcome,
    )
    from src.utils.paths import PROJECT_ROOT

    controller = controller_for(example)
    layout, plan = example.layout, example.screening.plan
    request = WorkRequest(
        adapter="source_screening",
        adapter_version="v1",
        stage="screen",
        subject_id=example.source.source_id,
        payload=example.request.prepared_input,
        inputs=(),
    )
    controller.pool.register(request)
    directory = layout.path(
        "campaigns", plan.campaign_id, "workers", request.work_id, "attempt-0001"
    )
    prepared = ScreeningAdapter().prepare(layout, plan, request, directory)
    task_ref = annotation_reference(layout, directory / "request.json")
    body = read_document(layout, task_ref, ScreeningRequest)
    detail = request_preview(layout, body, start=0.0, stop=15.0, step=0.5)
    sample = Example(
        layout,
        example.screening,
        example.source,
        task_ref,
        body,
        detail,
        read_document(layout, detail, PreviewReceipt),
    )
    selection = sample.finish()
    launch = Launch(
        work_id=request.work_id,
        attempt=request.work_id + "-0001",
        plan=controller.runtime.plan,
        request=request,
        prepared=prepared,
        directory=str(directory),
        code_root=str(PROJECT_ROOT),
        codex_binary=sys.executable,
        codex_home=str(layout.path("cache", "codex")),
        timeout_seconds=20.0,
        test_command=("synthetic-no-process",),
        test_root=str(layout.root),
    )
    launch_ref = write_document(layout, directory / "launch.json", launch)
    event = write_asset(
        layout, directory / "synthetic-events.txt", b"SYNTHETIC FIXTURE ONLY"
    )
    execution = write_document(
        layout,
        directory / "exit.json",
        ExitReceipt(
            launch_digest=launch_ref.digest,
            exit_code=0,
            status="succeeded",
            started_at="2026-01-01T00:00:00Z",
            ended_at="2026-01-01T00:00:01Z",
            elapsed_seconds=1.0,
            events=event,
            errors=event,
            thread_id="synthetic",
            reason=None,
        ),
    )
    result = write_document(
        layout,
        directory / "result.receipt.json",
        AttemptResult(
            launch=launch_ref,
            exit=execution,
            result=event,
            telemetry=None,
            outcome=WorkOutcome(status="complete", receipt=selection),
        ),
    )
    with controller.pool.ledger.registry.transaction() as db:
        db.execute(
            "INSERT INTO worker_attempts VALUES (?,?,1,'complete',?,NULL)",
            (
                launch.attempt,
                request.work_id,
                launch_ref.canonical_bytes().decode(),
            ),
        )
        db.execute(
            "UPDATE worker_tasks SET state='complete',current_attempt=?,result=? WHERE id=?",
            (
                launch.attempt,
                result.canonical_bytes().decode(),
                request.work_id,
            ),
        )
    return controller, request, selection, directory


@pytest.mark.parametrize(
    "mode", ["recover_after_crash", "no_unresolved_service", "prepared_source"]
)
def test_explicit_reopen_preserves_old_evidence_and_refuses_prepared_sources(
    example: Example, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from src.tasks.annotation import recovery as recovery_module
    from src.tasks.annotation.contracts import RecoveryInput, ReopenScreening

    controller, request, selection, directory = sealed_screening_attempt(example)
    layout = example.layout
    request_ref = write_document(layout, directory / "work-request.json", request)
    before = {
        path: path.read_bytes()
        for path in (
            layout.artifact_path(selection),
            directory / "result.receipt.json",
            directory / "launch.json",
        )
    }
    operation = RecoveryInput(
        plan=controller.runtime.plan,
        reason="synthetic old code sealed an unfinished image request",
        operation=ReopenScreening(
            request=request_ref,
            after_attempt=request.work_id + "-0001",
            selection=selection,
        ),
    )
    if mode != "no_unresolved_service":
        service = ServiceRequest(
            task_request=annotation_reference(layout, directory / "request.json"),
            operation="screen_preview",
            parameters={"start": 16.0, "stop": 31.0, "step": 0.5},
        )
        write_document(
            layout,
            directory / "work/service-requests" / f"{service.sha256}.json",
            service,
        )
    if mode == "prepared_source":
        controller.collection.prepare(example.source.source_id)
    if mode != "recover_after_crash":
        with pytest.raises(ValueError, match="unresolved image|prepared or excluded"):
            recovery_module.recover(controller, operation)
        assert controller.pool.reconcile(request)["state"] == "complete"
        return
    original = recovery_module.write_document

    def fail_after_transaction(
        layout: AnnotationLayout, path: Path, value: object
    ) -> ArtifactRef:
        if path.name == "result.json":
            raise OSError("synthetic crash after atomic invalidation")
        return original(layout, path, value)

    with monkeypatch.context() as patch:
        patch.setattr(recovery_module, "write_document", fail_after_transaction)
        with pytest.raises(OSError, match="synthetic crash"):
            recovery_module.recover(controller, operation)
    result = recovery_module.recover(controller, operation)
    assert recovery_module.recover(controller, operation) == result
    row = controller.collection.ledger.source(controller.plan, example.source.source_id)
    assert row["state"] == "acquired" and row["selection"] is None
    assert controller.pool.reconcile(request)["state"] == "validation_failed"
    assert all(path.read_bytes() == content for path, content in before.items())
    with controller.resources.registry.transaction() as db:
        assert db.execute("SELECT count(*) FROM screening_reopens").fetchone()[0] == 1
