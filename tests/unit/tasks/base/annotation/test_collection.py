from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    write_document,
)
from src.tasks.base.annotation.collection.contracts import (
    AcquisitionReceipt,
    PreparationConfig,
    PreparationReceipt,
    SelectionReceipt,
    SourceCandidate,
    TimeWindow,
)
from src.tasks.base.annotation.collection.ledger import Backpressure, CollectionLedger
from src.tasks.base.annotation.collection.preparation import window_ranges
from src.tasks.base.annotation.collection.service import (
    Collection,
    SourceExcluded,
    round_robin,
)
from src.tasks.base.annotation.contracts import ArtifactRef, ClipManifest
from src.tasks.base.annotation.media import Timeline, encode_video, probe_video
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan


@pytest.fixture
def layout(tmp_path: Path) -> AnnotationLayout:
    result = AnnotationLayout.testing(tmp_path / "annotation")
    result.path("campaigns", "inputs").mkdir(parents=True)
    return result


def plan_for(
    layout: AnnotationLayout, name: str = "campaign-a", *, lead: int = 2
) -> CampaignPlan:
    policy = layout.path("campaigns", "inputs", "policy.txt")
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text("fixture selection evidence")
    reference = annotation_reference(layout, policy)
    preparation = write_document(
        layout,
        layout.path("campaigns", "inputs", f"prepare-{lead}.json"),
        PreparationConfig(
            format_selector="bestvideo",
            crf=25,
            preset="ultrafast",
            minimum_clip_frames=32,
            maximum_unprepared_sources=lead,
        ),
    )
    return CampaignPlan.model_validate_json(
        json.dumps(
            {
                "campaign_id": name,
                "phase": "preview",
                "annotation_root": str(layout.root),
                "counts": {
                    "channels": 2,
                    "videos_per_channel": 2,
                    "max_clips_per_video": 2,
                },
                "selection": reference.model_dump(),
                "preparation": preparation.model_dump(),
                "code_commit": "a" * 40,
                "code_diff_digest": "b" * 64,
                "agent_model": "gpt-6.1-sol",
                "agent_effort": "high",
                "resources": {
                    "codex_workers": 2,
                    "additional_storage_bytes": 30 * 1024**3,
                    "minimum_free_bytes": 30 * 1024**3,
                    "total_pss_bytes": 22 * 1024**3,
                    "gpu_seconds": 43200,
                    "stages": [
                        {
                            "stage": "review",
                            "kind": "codex",
                            "parallel": 2,
                            "threads_per_process": 1,
                        }
                    ],
                },
                "tasks": [
                    {
                        "task": task,
                        "model": {
                            "task": task,
                            "version": "v1",
                            "training_weights": reference.model_dump(),
                            "inference_weights": reference.model_dump(),
                            "configuration": reference.model_dump(),
                        },
                        "inference": reference.model_dump(),
                        "review": None,
                        "learning": None,
                        "products": [
                            {
                                "product": task,
                                "dataset_task": task,
                                "dataset_version": "v2",
                                "component": "frames",
                                "base_snapshot": None,
                                "generation_policy": reference.model_dump(),
                            }
                        ],
                    }
                    for task in ("ball", "player")
                ],
            }
        )
    )


def candidate(
    layout: AnnotationLayout,
    plan: CampaignPlan,
    video: str,
    channel: str = "channel-a",
    *,
    media: ArtifactRef | None = None,
) -> SourceCandidate:
    if media is None:
        path = layout.path("campaigns", "inputs", video + ".mp4")
        seed = sum(map(ord, video))
        frames = (
            av.VideoFrame.from_ndarray(
                np.full((32, 64, 3), (seed + index) % 256, np.uint8),
                format="rgb24",
            )
            for index in range(320)
        )
        encode_video(
            path,
            width=64,
            height=32,
            time_base=Fraction(1, 1000),
            rate=Fraction(10),
            frames=((frame, index * 100, 100) for index, frame in enumerate(frames)),
            crf=25,
            preset="ultrafast",
        )
        media = annotation_reference(layout, path)
    return SourceCandidate(
        provider="local",
        channel_id=channel,
        video_id=video,
        title=video,
        origin_url=f"local:{video}",
        selection_reason="test selection",
        evidence=plan.selection,
        local_media=media,
    )


def select(
    collection: Collection,
    source: SourceCandidate,
    *,
    windows: tuple[TimeWindow, ...] | None = None,
) -> ArtifactRef:
    acquired = read_document(
        collection.layout, collection.acquire(source.source_id), AcquisitionReceipt
    )
    if windows is None:
        windows = (
            TimeWindow(start=0.0, stop=15.0, target_start=1.0, target_stop=14.0),
            TimeWindow(start=16.0, stop=31.0, target_start=17.0, target_stop=30.0),
        )
    decision = SelectionReceipt(
        source_id=source.source_id,
        source_digest=acquired.source.media.digest,
        policy=collection.plan.selection,
        evidence=collection.plan.selection,
        requested_maximum=2,
        windows=windows,
        shortage_reason="not enough play" if len(windows) < 2 else None,
    )
    ref = write_document(
        collection.layout,
        collection.layout.path(
            "campaigns",
            collection.plan.campaign_id,
            source.source_id + ".selection.json",
        ),
        decision,
    )
    collection.select(ref)
    return ref


def test_cross_campaign_registration_uses_canonical_identity(
    layout: AnnotationLayout,
) -> None:
    plans = [plan_for(layout, name) for name in ("one", "two")]
    ledger = CollectionLedger(layout)
    for plan in plans:
        ledger.register_campaign(plan)
    source = candidate(layout, plans[0], "video-a")
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(
            pool.map(lambda n: ledger.register_source(plans[n % 2], source), range(16))
        )
    assert sum(item.disposition == "registered" for item in outcomes) == 1
    assert {item.disposition for item in outcomes} == {
        "registered",
        "existing",
        "duplicate_source",
    }
    owner = next(
        item.owner_campaign for item in outcomes if item.disposition == "registered"
    )
    alias = source.model_copy(
        update={"channel_id": "renamed", "title": "different title"}
    )
    plan = next(item for item in plans if item.campaign_id != owner)
    assert ledger.register_source(plan, alias).disposition == "duplicate_source"
    another = source.model_copy(update={"video_id": "different-video"})
    assert ledger.register_source(plan, another).disposition == "duplicate_channel"


def test_collection_limits_and_staged_registration(layout: AnnotationLayout) -> None:
    plan = plan_for(layout)
    collection = Collection(layout, plan)
    source = candidate(layout, plan, "first")
    assert collection.register([source])[0].disposition == "registered"
    select(collection, source)
    prepared = collection.prepare(source.source_id)
    assert collection.prepare(source.source_id) == prepared
    second = candidate(layout, plan, "second")
    assert collection.register([second])[0].disposition == "registered"
    third = second.model_copy(update={"video_id": "third"})
    assert collection.register([third])[0].disposition == "limit"
    status = collection.ledger.summary(plan)
    assert status["prepared_clips"] == 2
    assert status["annotation_tasks"] == 4
    assert status["missing_prepared_videos"] == 3
    assert status["sources_by_state"] == {"prepared": 1, "selected": 1}


def test_content_duplicate_is_excluded_without_second_blob(
    layout: AnnotationLayout,
) -> None:
    collection = Collection(layout, plan_for(layout))
    first = candidate(layout, collection.plan, "first")
    duplicate = candidate(
        layout, collection.plan, "reupload", "channel-b", media=first.local_media
    )
    collection.register([first, duplicate])
    collection.acquire(first.source_id)
    with pytest.raises(SourceExcluded, match="duplicate content"):
        collection.acquire(duplicate.source_id)
    assert len(list(layout.path("sources", "blobs").iterdir())) == 1
    assert collection.ledger.summary(collection.plan)["sources_by_state"] == {
        "acquired": 1,
        "excluded": 1,
    }


def test_historical_source_and_channel_cannot_be_collected_as_new(
    layout: AnnotationLayout,
) -> None:
    collection = Collection(layout, plan_for(layout, "old"))
    source = candidate(layout, collection.plan, "old-video")
    collection.register([source])
    receipt = read_document(
        layout, collection.acquire(source.source_id), AcquisitionReceipt
    )
    # Use a second isolated inventory while retaining the same pinned source files.
    history_layout = AnnotationLayout.testing(layout.repository / "history")
    old_media = receipt.source.media.model_copy(
        update={"role": "artifact", "path": "placeholder"}
    )
    # History may refer to existing data/checkpoint/artifact roots, not only new media.
    path = layout.repository / "data" / "historical.mp4"
    path.parent.mkdir()
    path.write_bytes(layout.artifact_path(receipt.source.media).read_bytes())
    old_media = old_media.model_copy(update={"role": "data", "path": "historical.mp4"})
    evidence = old_media
    ledger = CollectionLedger(history_layout)
    ledger.import_history(
        receipt.source.model_copy(update={"media": old_media}), evidence
    )
    new_plan = plan_for(history_layout, "new")
    ledger.register_campaign(new_plan)
    old_candidate = source.model_copy(
        update={"evidence": new_plan.selection, "local_media": old_media}
    )
    assert (
        ledger.register_source(new_plan, old_candidate).disposition
        == "duplicate_source"
    )
    assert (
        ledger.register_source(
            new_plan, old_candidate.model_copy(update={"video_id": "another"})
        ).disposition
        == "duplicate_channel"
    )


def test_unprepared_lead_is_global_and_exclusion_releases_it(
    layout: AnnotationLayout,
) -> None:
    collection = Collection(layout, plan_for(layout, lead=1))
    sources = [candidate(layout, collection.plan, name) for name in ("first", "second")]
    collection.register(sources)
    collection.acquire(sources[0].source_id)
    with pytest.raises(Backpressure):
        collection.acquire(sources[1].source_id)
    select(collection, sources[0], windows=())
    collection.acquire(sources[1].source_id)
    assert collection.ledger.summary(collection.plan)["prepared_clips"] == 0


def test_acquisition_publish_before_ledger_failure_recovers(
    layout: AnnotationLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = Collection(layout, plan_for(layout))
    source = candidate(layout, collection.plan, "first")
    collection.register([source])
    with monkeypatch.context() as patch:
        patch.setattr(
            collection.ledger,
            "record_acquisition",
            lambda *_: (_ for _ in ()).throw(RuntimeError("receipt gap")),
        )
        with pytest.raises(RuntimeError, match="receipt gap"):
            collection.acquire(source.source_id)
    monkeypatch.setattr(
        "src.tasks.base.annotation.collection.service.acquire_source",
        lambda *_: pytest.fail("must not reacquire"),
    )
    collection.acquire(source.source_id)
    assert collection.ledger.summary(collection.plan)["sources_by_state"] == {
        "acquired": 1
    }


def test_clip_publication_is_atomic_and_recovers_ledger_gap(
    layout: AnnotationLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = Collection(layout, plan_for(layout))
    source = candidate(layout, collection.plan, "first")
    collection.register([source])
    select(collection, source)
    with monkeypatch.context() as patch:
        patch.setattr(
            "src.tasks.base.annotation.collection.preparation.os.rename",
            lambda *_: (_ for _ in ()).throw(OSError("publish failed")),
        )
        with pytest.raises(OSError, match="publish failed"):
            collection.prepare(source.source_id)
    assert not list(layout.path("clips").iterdir())
    assert collection.ledger.summary(collection.plan)["prepared_clips"] == 0
    with monkeypatch.context() as patch:
        patch.setattr(
            collection.ledger,
            "record_preparation",
            lambda *_: (_ for _ in ()).throw(RuntimeError("receipt gap")),
        )
        with pytest.raises(RuntimeError, match="receipt gap"):
            collection.prepare(source.source_id)
    monkeypatch.setattr(
        "src.tasks.base.annotation.collection.preparation.encode_video",
        lambda *_args, **_kwargs: pytest.fail("must not reencode"),
    )
    receipt = collection.prepare(source.source_id)
    assert collection.prepare(source.source_id) == receipt
    assert collection.ledger.summary(collection.plan)["annotation_tasks"] == 4


def test_manifest_preserves_real_frames_pts_and_context(
    layout: AnnotationLayout,
) -> None:
    collection = Collection(layout, plan_for(layout))
    source = candidate(layout, collection.plan, "first")
    collection.register([source])
    select(collection, source)
    prepared = read_document(
        layout, collection.prepare(source.source_id), PreparationReceipt
    )
    manifests = [read_document(layout, ref, ClipManifest) for ref in prepared.clips]
    for manifest, start in zip(manifests, (0, 160), strict=True):
        assert len(manifest.frames) == 150
        assert (manifest.source_range.start, manifest.source_range.stop) == (
            start,
            start + 150,
        )
        assert (manifest.target_range.start, manifest.target_range.stop) == (10, 140)
        timeline = probe_video(layout.artifact_path(manifest.media))
        assert timeline.boundary(150) == 15
        assert [f.source_frame_index for f in manifest.frames] == list(
            range(start, start + 150)
        )


def test_short_or_outside_windows_do_not_pad_or_publish(
    layout: AnnotationLayout,
) -> None:
    collection = Collection(layout, plan_for(layout))
    source = candidate(layout, collection.plan, "first")
    collection.register([source])
    select(
        collection,
        source,
        windows=(TimeWindow(start=0.0, stop=1.0, target_start=0.0, target_stop=1.0),),
    )
    with pytest.raises(ValueError, match="too few real frames"):
        collection.prepare(source.source_id)
    assert collection.ledger.summary(collection.plan)["prepared_clips"] == 0
    timeline = Timeline(64, 32, Fraction(1, 10), Fraction(10), (0, 1, 3), (1, 2, 2))
    with pytest.raises(ValueError, match="beyond"):
        window_ranges(
            timeline, TimeWindow(start=0.0, stop=1.0, target_start=0.0, target_stop=1.0)
        )
    with pytest.raises(ValueError, match="no complete"):
        window_ranges(
            timeline,
            TimeWindow(start=0.15, stop=0.25, target_start=0.15, target_stop=0.25),
        )


def test_acquisition_failure_requires_explicit_retry(layout: AnnotationLayout) -> None:
    collection = Collection(layout, plan_for(layout))
    source = candidate(layout, collection.plan, "first")
    assert source.local_media is not None
    path = layout.artifact_path(source.local_media)
    original = path.read_bytes()
    collection.register([source])
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="size changed"):
        collection.acquire(source.source_id)
    path.write_bytes(original)
    with pytest.raises(ValueError, match="explicit"):
        collection.acquire(source.source_id)
    collection.acquire(source.source_id, retry=True)
    assert collection.ledger.summary(collection.plan)["sources_by_state"] == {
        "acquired": 1
    }


def test_local_source_changed_during_copy_is_not_published(
    layout: AnnotationLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = Collection(layout, plan_for(layout))
    source = candidate(layout, collection.plan, "first")
    collection.register([source])
    monkeypatch.setattr(
        "src.tasks.base.annotation.collection.acquisition.shutil.copyfile",
        lambda _source, destination: destination.write_bytes(b"changed in flight"),
    )
    with pytest.raises(ValueError, match="changed while copying"):
        collection.acquire(source.source_id)
    assert not layout.path("sources", "blobs").exists()
    assert collection.ledger.summary(collection.plan)["sources_by_state"] == {
        "failed": 1
    }


def test_source_claim_and_round_robin_are_explicit(layout: AnnotationLayout) -> None:
    collection = Collection(layout, plan_for(layout))
    source = candidate(layout, collection.plan, "first")
    collection.register([source])
    with (
        artifact_lock(layout, "source-" + source.source_id),
        pytest.raises(BlockingIOError),
    ):
        collection.acquire(source.source_id)
    with pytest.raises(ValueError, match="repeats"):
        round_robin([source, source.model_copy(update={"title": "alias"})])
    a2 = source.model_copy(update={"video_id": "second"})
    b1 = source.model_copy(update={"channel_id": "b", "video_id": "third"})
    assert [item.video_id for item in round_robin([source, a2, b1])] == [
        "first",
        "third",
        "second",
    ]
