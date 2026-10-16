from __future__ import annotations

import json
import shutil
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
    HistoricalSource,
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
    layout: AnnotationLayout,
    name: str = "campaign-a",
    *,
    lead: int = 2,
    jpeg: bool = False,
    videos_per_channel: int = 2,
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
            crf=None if jpeg else 25,
            preset=None if jpeg else "ultrafast",
            retained_media="jpeg_q90" if jpeg else "video",
            source_retention="delete_after_verified_frames" if jpeg else "keep",
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
                    "videos_per_channel": videos_per_channel,
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
                    for task in (("ball",) if jpeg else ("ball", "player"))
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


def test_incomplete_historical_download_still_reserves_source_and_channel(
    layout: AnnotationLayout,
) -> None:
    plan = plan_for(layout)
    ledger = CollectionLedger(layout)
    historical = HistoricalSource(
        provider="youtube",
        channel_id="old-channel",
        video_id="old-video",
        origin_url="https://www.youtube.com/watch?v=old-video",
    )
    ledger.import_history(historical, plan.selection)
    ledger.import_history(historical, plan.selection)
    ledger.register_campaign(plan)
    candidate = SourceCandidate(
        provider="youtube",
        channel_id="old-channel",
        video_id="old-video",
        title="fixture",
        origin_url=historical.origin_url,
        selection_reason="fixture",
        evidence=plan.selection,
    )
    assert ledger.register_source(plan, candidate).disposition == "duplicate_source"
    assert (
        ledger.register_source(
            plan, candidate.model_copy(update={"video_id": "different"})
        ).disposition
        == "duplicate_channel"
    )
    with ledger.registry.transaction() as db:
        assert db.execute("SELECT count(*) FROM source_content").fetchone()[0] == 0
    with pytest.raises(ValueError, match="historical source"):
        ledger.import_history(
            historical.model_copy(update={"channel_id": "changed"}), plan.selection
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


def test_failed_retry_cannot_exceed_channel_limit_after_replacement(
    layout: AnnotationLayout,
) -> None:
    collection = Collection(layout, plan_for(layout, videos_per_channel=1))
    first = candidate(layout, collection.plan, "first")
    assert first.local_media is not None
    path = layout.artifact_path(first.local_media)
    original = path.read_bytes()
    collection.register([first])
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="size changed"):
        collection.acquire(first.source_id)
    path.write_bytes(original)

    replacement = candidate(layout, collection.plan, "replacement")
    assert collection.register([replacement])[0].disposition == "registered"
    select(collection, replacement)
    collection.prepare(replacement.source_id)
    with pytest.raises(ValueError, match="videos-per-channel limit"):
        collection.acquire(first.source_id, retry=True)
    summary = collection.ledger.summary(collection.plan)
    assert summary["sources_by_state"] == {"failed": 1, "prepared": 1}
    assert summary["prepared_clips"] == 2
    assert summary["annotation_tasks"] == 4


def test_permanent_preparation_exclusion_preserves_selection_and_releases_lead(
    layout: AnnotationLayout,
) -> None:
    collection = Collection(layout, plan_for(layout, lead=1, videos_per_channel=1))
    first = candidate(layout, collection.plan, "first")
    collection.register([first])
    selection = select(
        collection,
        first,
        windows=(TimeWindow(start=0.0, stop=1.0, target_start=0.0, target_stop=1.0),),
    )
    with pytest.raises(ValueError, match="too few real frames") as failure:
        collection.prepare(first.source_id)
    second = candidate(layout, collection.plan, "second", "channel-b")
    collection.register([second])
    with pytest.raises(Backpressure):
        collection.acquire(second.source_id)
    with pytest.raises(ValueError, match="reason"):
        collection.exclude_preparation(first.source_id, reason=" ")

    collection.exclude_preparation(first.source_id, reason=str(failure.value))
    row = collection.ledger.source(collection.plan, first.source_id)
    assert row["state"] == "excluded"
    assert row["reason"] == str(failure.value)
    assert ArtifactRef.model_validate_json(row["selection"]) == selection
    collection.select(selection)
    with pytest.raises(SourceExcluded, match="too few real frames"):
        collection.acquire(first.source_id, retry=True)
    with pytest.raises(ValueError, match="selection"):
        collection.prepare(first.source_id)
    replacement = candidate(layout, collection.plan, "replacement")
    assert collection.register([replacement])[0].disposition == "registered"
    collection.acquire(second.source_id)
    summary = collection.ledger.summary(collection.plan)
    assert summary["sources_by_state"] == {
        "excluded": 1,
        "acquired": 1,
        "selected": 1,
    }
    assert summary["annotation_tasks"] == 0


@pytest.mark.parametrize("registered", [False, True])
def test_published_preparation_cannot_be_excluded(
    layout: AnnotationLayout, monkeypatch: pytest.MonkeyPatch, registered: bool
) -> None:
    collection = Collection(layout, plan_for(layout))
    source = candidate(layout, collection.plan, "first")
    collection.register([source])
    select(collection, source)
    if registered:
        collection.prepare(source.source_id)
    else:
        with monkeypatch.context() as patch:
            patch.setattr(
                collection.ledger,
                "record_preparation",
                lambda *_: (_ for _ in ()).throw(RuntimeError("receipt gap")),
            )
            with pytest.raises(RuntimeError, match="receipt gap"):
                collection.prepare(source.source_id)
    with pytest.raises(ValueError, match="published preparation"):
        collection.exclude_preparation(source.source_id, reason="cannot prepare")
    collection.prepare(source.source_id)
    assert collection.ledger.summary(collection.plan)["annotation_tasks"] == 4


@pytest.mark.parametrize(
    ("registered", "missing"),
    [(True, "receipt")]
    + [
        (registered, missing)
        for registered in (False, True)
        for missing in ("clip_directory", "clip_manifest", "clip_media")
    ],
)
def test_preparation_resume_rejects_missing_artifacts_before_generation(
    layout: AnnotationLayout,
    monkeypatch: pytest.MonkeyPatch,
    registered: bool,
    missing: str,
) -> None:
    collection = Collection(layout, plan_for(layout))
    source = candidate(layout, collection.plan, "first")
    collection.register([source])
    select(collection, source)
    if registered:
        reference = collection.prepare(source.source_id)
    else:
        with monkeypatch.context() as patch:
            patch.setattr(
                collection.ledger,
                "record_preparation",
                lambda *_: (_ for _ in ()).throw(RuntimeError("receipt gap")),
            )
            with pytest.raises(RuntimeError, match="receipt gap"):
                collection.prepare(source.source_id)
        reference = annotation_reference(
            layout,
            layout.path(
                "sources",
                source.source_id,
                collection.plan.preparation.digest,
                "prepared.json",
            ),
        )
    receipt = read_document(layout, reference, PreparationReceipt)
    manifest = read_document(layout, receipt.clips[0], ClipManifest)
    if missing == "receipt":
        removed = layout.artifact_path(reference)
    elif missing == "clip_manifest":
        removed = layout.artifact_path(receipt.clips[0])
    else:
        removed = layout.artifact_path(manifest.media)
    if missing == "clip_directory":
        removed = removed.parent
        shutil.rmtree(removed)
    else:
        removed.unlink()
    monkeypatch.setattr(
        "src.tasks.base.annotation.collection.service.prepare_source",
        lambda *_: pytest.fail("must verify published artifacts before generation"),
    )
    with pytest.raises(ValueError, match="artifact missing"):
        collection.prepare(source.source_id)
    assert not removed.exists()
    assert collection.ledger.summary(collection.plan)["annotation_tasks"] == (
        4 if registered else 0
    )


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


def jpeg_collection(layout: AnnotationLayout):
    collection = Collection(layout, plan_for(layout, jpeg=True))
    source = candidate(layout, collection.plan, "jpeg-source")
    collection.register([source])
    acquired_ref = collection.acquire(source.source_id)
    acquired = read_document(layout, acquired_ref, AcquisitionReceipt)
    select(collection, source)
    return collection, source, acquired_ref, acquired


def test_jpeg_preparation_retires_only_owned_video_and_resumes(
    layout: AnnotationLayout,
) -> None:
    from src.tasks.base.annotation.contracts import SourceProvenance
    from src.tasks.base.annotation.media import check_clip, decode_clip
    from src.tasks.base.annotation.plan import references

    collection, source, acquired_ref, acquired = jpeg_collection(layout)
    prepared_ref = collection.prepare(source.source_id)
    prepared = read_document(layout, prepared_ref, PreparationReceipt)
    assert not layout.artifact_path(acquired.source.media).exists()
    assert (
        source.local_media is not None
        and layout.artifact_path(source.local_media).is_file()
    )
    assert not list(layout.path("clips").rglob("*.mp4"))
    for ref in prepared.clips:
        manifest = read_document(layout, ref, ClipManifest)
        assert isinstance(manifest.source, SourceProvenance)
        assert acquired.source.media not in set(references(manifest))
        path = layout.artifact_path(manifest.media)
        check_clip(path, manifest)
        decoded = list(decode_clip(path, manifest, 0, len(manifest.frames)))
        assert [frame.pts for frame in decoded] == [
            frame.clip_pts for frame in manifest.frames
        ]
    assert collection.acquire(source.source_id) == acquired_ref
    assert collection.prepare(source.source_id) == prepared_ref


def test_source_survives_uncommitted_jpeg_preparation(
    layout: AnnotationLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, source, _, acquired = jpeg_collection(layout)

    def fail(*args):
        raise RuntimeError("before source timing validation commit")

    monkeypatch.setattr(collection.ledger, "record_preparation", fail)
    with pytest.raises(RuntimeError, match="before source"):
        collection.prepare(source.source_id)
    assert layout.artifact_path(acquired.source.media).is_file()
    assert not list(layout.path("sources").rglob("retirement-intent.json"))


def test_source_retirement_recovers_after_unlink(
    layout: AnnotationLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.base.annotation.collection import retirement

    collection, source, _, acquired = jpeg_collection(layout)
    original = retirement.write_document

    def crash(layout, path, document):
        if path.name == "retired.json":
            raise RuntimeError("after unlink")
        return original(layout, path, document)

    with monkeypatch.context() as patch:
        patch.setattr(retirement, "write_document", crash)
        with pytest.raises(RuntimeError, match="after unlink"):
            collection.prepare(source.source_id)
    assert not layout.artifact_path(acquired.source.media).exists()
    assert (
        collection.ledger.source(collection.plan, source.source_id)["state"]
        == "prepared"
    )
    assert not list(layout.path("sources").rglob("retired.json"))
    collection.prepare(source.source_id)
    assert len(list(layout.path("sources").rglob("retired.json"))) == 1


def test_missing_source_without_retirement_proof_fails(
    layout: AnnotationLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, source, _, acquired = jpeg_collection(layout)
    with monkeypatch.context() as patch:
        patch.setattr(
            "src.tasks.base.annotation.collection.retirement.retire_source",
            lambda *args: None,
        )
        collection.prepare(source.source_id)
    layout.artifact_path(acquired.source.media).unlink()
    with pytest.raises(ValueError, match="without a verified retirement"):
        collection.prepare(source.source_id)


def test_corrupt_retained_frame_blocks_retirement(
    layout: AnnotationLayout, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection, source, _, acquired = jpeg_collection(layout)
    with monkeypatch.context() as patch:
        patch.setattr(
            "src.tasks.base.annotation.collection.retirement.retire_source",
            lambda *args: None,
        )
        ref = collection.prepare(source.source_id)
    prepared = read_document(layout, ref, PreparationReceipt)
    manifest = read_document(layout, prepared.clips[0], ClipManifest)
    path = layout.artifact_path(manifest.media)
    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match="hash changed"):
        collection.prepare(source.source_id)
    assert layout.artifact_path(acquired.source.media).is_file()


def test_live_reader_prevents_source_retirement(layout: AnnotationLayout) -> None:
    from src.tasks.base.annotation.storage import StorageCatalog
    from src.tasks.base.annotation.storage.contracts import Protection

    collection, source, _, acquired = jpeg_collection(layout)
    StorageCatalog(layout).protect(
        Protection(
            protection_id="external-reader",
            kind="reader",
            reason="fixture live consumer",
            references=(acquired.source.media,),
        )
    )
    with pytest.raises(ValueError, match="live storage protection"):
        collection.prepare(source.source_id)
    assert layout.artifact_path(acquired.source.media).is_file()


@pytest.mark.parametrize("state", ["complete", "validation_failed"])
def test_only_completed_screening_can_transfer_its_source_pin(
    layout: AnnotationLayout, state: str
) -> None:
    from src.tasks.base.annotation.storage import StorageCatalog
    from src.tasks.base.annotation.storage.contracts import Protection
    from src.tasks.base.annotation.workers.contracts import (
        AttemptResult,
        WorkOutcome,
        WorkRequest,
    )
    from src.tasks.base.annotation.workers.ledger import WorkerLedger

    collection, source, acquired_ref, acquired = jpeg_collection(layout)
    selected = ArtifactRef.model_validate_json(
        collection.ledger.source(collection.plan, source.source_id)["selection"]
    )
    request = WorkRequest(
        adapter="source_screening",
        adapter_version="v1",
        stage="screen",
        subject_id=source.source_id,
        payload=selected,
        inputs=(),
    )
    result = write_document(
        layout,
        layout.path("campaigns", "inputs", "screen-result.json"),
        AttemptResult(
            launch=selected,
            exit=selected,
            result=selected,
            outcome=WorkOutcome(status="complete", receipt=selected),
            telemetry=None,
        ),
    )
    ledger = WorkerLedger(layout)
    with ledger.registry.transaction() as db:
        db.execute(
            "INSERT INTO worker_tasks VALUES (?,?,?,?,?,?,?,NULL)",
            (
                request.work_id,
                collection.plan.campaign_id,
                collection.plan.sha256,
                request.canonical_bytes().decode(),
                state,
                "fixture-attempt",
                result.canonical_bytes().decode(),
            ),
        )
    original = Protection(
        protection_id="worker-inputs-v1-" + request.sha256,
        kind="active",
        reason="original screening inputs",
        references=(acquired.source.media, acquired_ref, selected),
    )
    catalog = StorageCatalog(layout)
    catalog.protect(original)
    if state != "complete":
        with pytest.raises(ValueError, match="completed screening worker"):
            collection.prepare(source.source_id)
        assert layout.artifact_path(acquired.source.media).is_file()
        assert original in catalog.snapshot().protections
        return
    prepared = collection.prepare(source.source_id)
    assert not layout.artifact_path(acquired.source.media).exists()
    retained = catalog.snapshot().protections
    assert original not in retained
    assert len(retained) == 1
    assert acquired.source.media not in retained[0].references
    assert set((prepared, selected, acquired_ref)).issubset(retained[0].references)
    assert collection.prepare(source.source_id) == prepared


@pytest.mark.parametrize("outcome", ["unavailable", "different_content", "restored"])
def test_explicit_source_restore_is_terminal_and_does_not_replace_jpegs(
    layout: AnnotationLayout, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    import subprocess

    from src.tasks.base.annotation.collection import restoration

    collection, source, _, acquired = jpeg_collection(layout)
    prepared = collection.prepare(source.source_id)
    calls = 0

    def stage(candidate, config, staging, layout):
        nonlocal calls
        calls += 1
        if outcome == "unavailable":
            raise subprocess.CalledProcessError(1, ["fixture-youtube-download"])
        target = staging / "media"
        assert candidate.local_media is not None
        target.write_bytes(
            b"changed bytes"
            if outcome == "different_content"
            else layout.artifact_path(candidate.local_media).read_bytes()
        )
        return target

    monkeypatch.setattr(restoration, "stage_source", stage)
    ref = collection.restore_source(source.source_id, attempt="first")
    receipt = read_document(layout, ref, restoration.RestorationReceipt)
    assert receipt.status == outcome
    assert collection.restore_source(source.source_id, attempt="first") == ref
    assert calls == 1
    assert layout.artifact_path(acquired.source.media).exists() == (
        outcome == "restored"
    )
    assert collection.prepare(source.source_id) == prepared
