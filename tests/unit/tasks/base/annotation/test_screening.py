from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

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
from src.tasks.base.annotation.media import encode_video, probe_video
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan
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


def setup(
    tmp_path: Path,
    seconds: int = 32,
    *,
    fps: int | Fraction = 10,
    frame_count: int | None = None,
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
                    "minimum_free_bytes": 30 * 1024**3,
                    "total_pss_bytes": 22 * 1024**3,
                    "gpu_seconds": 43200,
                    "stages": [
                        {
                            "stage": "screening",
                            "kind": "codex",
                            "parallel": 1,
                            "threads_per_process": 1,
                        }
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
        timestamps = tuple(
            range(math.ceil(seconds * fps) if frame_count is None else frame_count)
        )
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
    collection.acquire(source.source_id)
    return layout, Screening(collection), source


@pytest.fixture
def example(tmp_path: Path) -> Example:
    layout, screening, source = setup(tmp_path)
    ref = screening.request(source.source_id, "attempt-1")
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
        example.screening.request(example.source.source_id, "attempt-1")
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


def test_rounded_source_end_finalizes_and_prepares_last_frame(tmp_path: Path) -> None:
    layout, screening, source = setup(tmp_path, fps=30, frame_count=500)
    ref = screening.request(source.source_id, "attempt-1")
    request = read_document(layout, ref, ScreeningRequest)
    timeline = probe_video(layout.artifact_path(request.source.media))
    assert len(timeline.pts) == 500
    assert timeline.boundary(500) == Fraction(50, 3)
    assert Fraction(str(request.source_duration)) > timeline.boundary(500)
    stop = request.source_duration
    start = stop - 15.0
    detail_ref = request_preview(layout, request, start=start, stop=stop, step=0.5)
    detail = read_document(layout, detail_ref, PreviewReceipt)
    example = Example(layout, screening, source, ref, request, detail_ref, detail)
    decision = example.decision()
    clip = decision.clips[0].model_copy(update={"start": start, "stop": stop})
    selection_ref = example.finish(
        decision=decision.model_copy(update={"clips": (clip,)})
    )
    selection = read_document(layout, selection_ref, SelectionReceipt)
    assert selection.windows[0].stop == stop
    prepared_ref = screening.collection.prepare(source.source_id)
    prepared = read_document(layout, prepared_ref, PreparationReceipt)
    manifest = read_document(layout, prepared.clips[0], ClipManifest)
    assert manifest.source_range.stop == 500
    assert manifest.frames[-1].source_frame_index == 499
    assert manifest.frames[-1].source_pts == timeline.pts[-1]
    assert manifest.frames[-1].duration_pts == timeline.durations[-1]
    assert screening.collection.prepare(source.source_id) == prepared_ref


@pytest.mark.parametrize("fps", [25, Fraction(30000, 1001)], ids=["25fps", "29.97fps"])
@pytest.mark.parametrize("start", [0.0, 0.01])
def test_standard_detail_sampling_finalizes_at_fractional_frame_steps(
    tmp_path: Path,
    fps: int | Fraction,
    start: float,
) -> None:
    layout, screening, source = setup(tmp_path, fps=fps)
    ref = screening.request(source.source_id, "attempt-1")
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
    ref = screening.request(source.source_id, "attempt-1")
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
    ref = screening.request(source.source_id, "attempt-1")
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
        screening.request(source.source_id, "attempt-1")
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
    assert (
        len(list(directory.iterdir())) == 2
    )  # overview and first detail, no half receipt
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
