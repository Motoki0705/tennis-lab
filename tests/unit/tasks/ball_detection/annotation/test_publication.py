"""Real synthetic review -> sealed adoption -> immutable store, on CPU only."""

from __future__ import annotations

import errno
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tasks.ball_detection.annotation.review.contracts import (
    BallEdits,
    BallObservation,
    Note,
)
from src.tasks.ball_detection.data.store import BallFrameStore, shard_name
from src.tasks.ball_detection.data.teacher_masks import LEGACY
from src.tasks.ball_detection.generate_dataset.frame_store.publishing import (
    publisher as publishing,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
    BallPublicationPolicy,
    BallSnapshot,
    LegacySourceGroup,
    PublishRequest,
    SourceSplit,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.publisher import (
    BallPublisher,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.snapshot import (
    freeze_legacy_store,
    load_snapshot,
)
from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    ProductOutcome,
    PublicationReceipt,
    content_digest,
)
from src.tasks.base.annotation.learning.contracts import (
    CampaignCompletion,
    TaskProducts,
)
from src.tasks.base.annotation.learning.validation import check_completion
from src.tasks.base.annotation.paths import AnnotationLayout
from tests.support.tasks.ball_detection.annotation import Environment
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


def seal(env: Environment, *, sealed: bool = True) -> ArtifactRef:
    image = env.page(tuple(range(8)))
    visible = BallObservation(
        track_id="ball", state="visible", center=(16.0, 12.0), origin="manual"
    )
    env.apply(
        BallEdits(
            operations=(
                env.replace(1, image, observations=(visible,), boundaries=("hit",)),
                env.replace(2, image, presence="absent", boundaries=("hit", "bounce")),
                env.replace(3, image, presence="unknown"),
                env.replace(
                    4,
                    image,
                    observations=(visible,),
                    notes=(
                        Note(
                            note_id="blur",
                            text="uncertain exact location",
                            blocking=True,
                        ),
                    ),
                ),
                env.replace(
                    5,
                    image,
                    observations=(
                        BallObservation(
                            track_id="ball",
                            state="occluded",
                            center=None,
                            origin="manual",
                        ),
                    ),
                ),
                env.replace(
                    6,
                    image,
                    observations=(
                        BallObservation(
                            track_id="ball",
                            state="occluded",
                            center=(16.0, 12.0),
                            origin="manual",
                        ),
                    ),
                ),
                env.replace(
                    7,
                    image,
                    observations=(
                        visible,
                        BallObservation(
                            track_id="other",
                            state="visible",
                            center=(32.0, 12.0),
                            origin="manual",
                        ),
                    ),
                ),
            ),
            images=(image,),
        )
    )
    adopted: ArtifactRef = env.adapter.prepare_adoption(
        env.layout,
        env.context,
        env.manifest,
        env.head(),
        (env.telemetry((image,)),),
        reason="synthetic publication test",
    )
    if sealed:
        env.store.seal(env.head(), adopted, env.adapter)
    return adopted


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Environment, PublishRequest, Path]:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    base = write_store_clip(
        tmp_path / "data/ball_detection/baseline",
        "old/clip",
        [frame(0, ball()), frame(1), frame(2, ball("interpolated"))],
        split="val",
    )
    base_ref = freeze_legacy_store(
        layout,
        base,
        (LegacySourceGroup(source="tracknet", group_id="old", source_id="c" * 64),),
    )
    policy = BallPublicationPolicy(
        jpeg_quality=95,
        source_splits=(
            SourceSplit(source_id=content_digest(["local", "fixture"]), split="train"),
        ),
    )
    env = Environment(
        layout.root,
        publication_policy=policy,
        base_snapshot=base_ref,
        dataset_version="incremental",
    )
    request = PublishRequest(
        plan=env.plan_ref, product="frames", adoptions=(seal(env),)
    )
    return env, request, base


def destination(env: Environment) -> Path:
    path: Path = env.layout.repository / "data/ball_detection/incremental/frames"
    return path


def test_jpeg_only_review_and_publication_share_exact_image_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import os

    policy = BallPublicationPolicy(
        jpeg_quality=90,
        source_splits=(
            SourceSplit(source_id=content_digest(["local", "fixture"]), split="train"),
        ),
    )
    env = Environment(tmp_path / "annotation", jpeg=True, publication_policy=policy)
    assert not list(env.layout.root.rglob("*.mp4"))
    # Both fresh renderer pages and cached images work after video deletion.
    first = env.page((0, 1, 2))
    assert env.page((0, 1, 2)) == first
    request = PublishRequest(
        plan=env.plan_ref, product="frames", adoptions=(seal(env),)
    )

    def no_encode(*args, **kwargs):
        raise AssertionError("publication must not JPEG-encode saved frames again")

    monkeypatch.setattr("cv2.imencode", no_encode)
    receipt_ref = BallPublisher(env.layout).publish(request)
    receipt, snapshot, store = result(env, receipt_ref)
    assert receipt.snapshot is not None and len(snapshot.clips) == 1
    assert env.manifest.jpeg is not None
    source_path = env.layout.artifact_path(env.manifest.media)
    published_path = store.image_path(store.clips[0])
    assert os.path.samefile(source_path, published_path)
    assert published_path.stat().st_nlink == 2
    for row, jpeg in zip(
        store.clip_rows(store.clips[0]), env.manifest.jpeg.frames, strict=True
    ):
        assert hashlib.sha256(store.read_jpeg(int(row))).hexdigest() == jpeg.digest
    assert BallPublisher(env.layout).publish(request) == receipt_ref


def test_publication_pointer_and_protection_are_atomic_after_files_appear(
    setup: tuple[Environment, PublishRequest, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.base.annotation.storage.catalog import StorageCatalog

    env, request, _ = setup
    publisher = BallPublisher(env.layout)
    original = publisher.catalog.protect
    before = StorageCatalog(env.layout).snapshot().digest

    def crash(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        raise RuntimeError("after publication protection")

    with monkeypatch.context() as patch:
        patch.setattr(publisher.catalog, "protect", crash)
        with pytest.raises(RuntimeError, match="after publication protection"):
            publisher.publish(request)
    assert (destination(env) / "snapshot.json").is_file()
    assert StorageCatalog(env.layout).snapshot().digest == before
    with publisher.registry.transaction() as db:
        assert db.execute("SELECT receipt FROM ball_publications").fetchone()[0] is None

    def forbid(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("recoverable publication must not re-encode")

    monkeypatch.setattr(publisher, "_build", forbid)
    result = publisher.publish(request)
    first = StorageCatalog(env.layout).snapshot().digest
    assert publisher.publish(request) == result
    assert StorageCatalog(env.layout).snapshot().digest == first


def result(
    env: Environment, reference: ArtifactRef
) -> tuple[PublicationReceipt, BallSnapshot, BallFrameStore]:
    receipt = read_document(env.layout, reference, PublicationReceipt)
    snapshot, store = load_snapshot(env.layout, receipt.snapshot)
    return receipt, snapshot, store


def test_publication_preserves_base_bytes_rows_and_sealed_review_masks(
    setup: tuple[Environment, PublishRequest, Path],
) -> None:
    env, request, base = setup
    old = BallFrameStore(base)
    before = {
        p: p.read_bytes()
        for p in (
            base / "metadata.json",
            base / "index.npz",
            base / "shards" / shard_name(0),
        )
    }
    reference = BallPublisher(env.layout).publish(request)
    completion = CampaignCompletion(
        plan=env.plan_ref,
        tasks=(
            TaskProducts(
                task="ball",
                products=(
                    ProductOutcome(
                        product="frames",
                        status="published",
                        publication=reference,
                        reason=None,
                    ),
                ),
            ),
        ),
    )
    assert completion.complete and check_completion(env.layout, completion) == env.plan
    receipt, snapshot, store = result(env, reference)
    assert receipt.added_clips == (env.manifest.clip_id,)
    from tests.support.annotation_storage import assert_expired_but_protected

    assert_expired_but_protected(
        env.layout,
        (
            reference,
            receipt.snapshot,
            snapshot.index,
            snapshot.metadata,
            *(clip.shard for clip in snapshot.clips),
        ),
        "publication",
    )
    assert len(receipt.reused_clips) == 1 and len(snapshot.clips) == 2
    assert store.clips[0] == old.clips[0]
    assert store.metadata["clips"][0] == old.metadata["clips"][0]
    for path, payload in before.items():
        assert path.read_bytes() == payload
    for row in range(len(old)):
        np.testing.assert_array_equal(store.read_jpeg(row), old.read_jpeg(row))
    assert (store.directory / "shards" / shard_name(0)).stat().st_ino == (
        base / "shards" / shard_name(0)
    ).stat().st_ino
    assert store.teacher_masks is not None
    assert store.teacher_masks.heatmap[:3].tolist() == [LEGACY] * 3
    added = store.clips[1]
    assert (added.width, added.height, added.source_width, added.source_height) == (
        64,
        32,
        64,
        32,
    )
    rows = store.clip_rows(added)
    assert store.frames["pts"][rows].tolist() == [
        f.clip_pts for f in env.manifest.frames
    ]
    assert store.frames["annotated"][rows[:8]].tolist() == [False] + [True] * 7
    assert store.teacher_masks.heatmap[rows[:8]].tolist() == [0, 2, 1, 0, 0, 0, 0, 2]
    assert store.teacher_masks.coordinate[rows[:8]].tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    assert store.instances_of(int(rows[4])).xy.tolist() == [[16.0, 12.0]]
    assert store.instances_of(int(rows[5])).occluded.tolist() == [True]
    assert store.frames["segment_break"][rows[2]]
    assert len(store.metadata["append_history"]) == 1
    assert snapshot.clips[0].split == "val" and snapshot.clips[1].split == "train"
    with BallPublisher(env.layout).registry.transaction() as db:
        registered = db.execute("SELECT receipt FROM ball_publications").fetchone()
    assert ArtifactRef.model_validate_json(registered[0]) == reference


def test_replay_and_concurrent_publications_encode_new_clip_only_once(
    setup: tuple[Environment, PublishRequest, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    env, request, _ = setup
    calls = []
    original = publishing.encode_clip

    def encode(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[0].clip_id)
        return original(*args, **kwargs)

    monkeypatch.setattr(publishing, "encode_clip", encode)
    service = BallPublisher(env.layout)
    with ThreadPoolExecutor(max_workers=3) as pool:
        refs = list(pool.map(lambda _: service.publish(request), range(3)))
    assert refs == [refs[0]] * 3
    assert service.publish(request) == refs[0]
    assert calls == [env.manifest.clip_id]
    assert len(result(env, refs[0])[2].clips) == 2


@pytest.mark.parametrize(
    "phase",
    [
        "shard",
        "metadata",
        "before_rename",
        "after_rename",
        "before_receipt",
        "after_receipt",
    ],
)
def test_interrupted_publication_recovers_without_duplicate_additions(
    setup: tuple[Environment, PublishRequest, Path],
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    env, request, base = setup
    original_encode, original_json, original_rename, original_write = (
        publishing.encode_clip,
        publishing._durable_document,
        publishing.os.rename,
        publishing.write_document,
    )
    calls = []

    def encode(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[0].clip_id)
        table = original_encode(*args, **kwargs)
        if phase == "shard":
            raise RuntimeError("injected interruption")
        return table

    def durable(path: Path, document: Any) -> None:
        original_json(path, document)
        if phase == "metadata" and path.name == "metadata.json":
            raise RuntimeError("injected interruption")

    def rename(source: Path, target: Path) -> None:
        if phase == "before_rename":
            raise RuntimeError("injected interruption")
        original_rename(source, target)
        if phase == "after_rename":
            raise RuntimeError("injected interruption")

    def write(layout: AnnotationLayout, path: Path, document: Any) -> ArtifactRef:
        if phase == "before_receipt" and path.name == "receipt.json":
            raise RuntimeError("injected interruption")
        reference: ArtifactRef = original_write(layout, path, document)
        if phase == "after_receipt" and path.name == "receipt.json":
            raise RuntimeError("injected interruption")
        return reference

    service = BallPublisher(env.layout)
    with monkeypatch.context() as faults:
        faults.setattr(publishing, "encode_clip", encode)
        faults.setattr(publishing, "_durable_document", durable)
        faults.setattr(publishing.os, "rename", rename)
        faults.setattr(publishing, "write_document", write)
        with pytest.raises(RuntimeError, match="injected"):
            service.publish(request)
    assert destination(env).exists() == (
        phase in {"after_rename", "before_receipt", "after_receipt"}
    )
    assert len(BallFrameStore(base).clips) == 1
    if phase not in {"shard", "metadata"}:
        monkeypatch.setattr(
            publishing,
            "encode_clip",
            lambda *a, **kw: pytest.fail("completed staging must not be encoded again"),
        )
    reference = service.publish(request)
    receipt, _, store = result(env, reference)
    assert receipt.added_clips == (env.manifest.clip_id,)
    assert len(store.clips) == 2
    assert len(store.metadata["append_history"]) == 1
    assert calls == [env.manifest.clip_id]
    assert service.publish(request) == reference


def test_foreign_destination_reservation_is_not_overwritten(
    setup: tuple[Environment, PublishRequest, Path],
) -> None:
    env, request, _ = setup
    service = BallPublisher(env.layout)
    with service.registry.transaction() as db:
        db.execute(
            "INSERT INTO ball_publications VALUES (?,?,?,NULL)",
            ("f" * 64, "ball_detection/incremental/frames", env.blob.model_dump_json()),
        )
    with pytest.raises(ValueError, match="reserved by another"):
        service.publish(request)
    assert not destination(env).exists()


def test_changed_base_shard_is_rejected_before_staging(
    setup: tuple[Environment, PublishRequest, Path],
) -> None:
    env, request, base = setup
    (base / "shards" / shard_name(0)).write_bytes(b"changed")
    with pytest.raises(ValueError, match="size changed|hash changed"):
        BallPublisher(env.layout).publish(request)
    assert not destination(env).exists()


@pytest.mark.parametrize("mode", ["unsealed", "missing_split", "changed_split"])
def test_publication_requires_sealed_adoption_and_fixed_source_splits(
    tmp_path: Path, mode: str
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    base = write_store_clip(
        tmp_path / "data/ball_detection/baseline",
        "old/clip",
        [frame(0, ball())],
        split="val",
    )
    imported = freeze_legacy_store(
        layout,
        base,
        (LegacySourceGroup(source="tracknet", group_id="old", source_id="c" * 64),),
    )
    splits: tuple[SourceSplit, ...] = (
        ()
        if mode == "missing_split"
        else (
            SourceSplit(source_id=content_digest(["local", "fixture"]), split="train"),
        )
    )
    if mode == "changed_split":
        splits += (SourceSplit(source_id="c" * 64, split="train"),)
    env = Environment(
        layout.root,
        publication_policy=BallPublicationPolicy(jpeg_quality=95, source_splits=splits),
        base_snapshot=imported,
        dataset_version="incremental",
    )
    request = PublishRequest(
        plan=env.plan_ref,
        product="frames",
        adoptions=(seal(env, sealed=mode != "unsealed"),),
    )
    with pytest.raises(ValueError, match="sealed|fixed split|existing source split"):
        BallPublisher(layout).publish(request)
    assert not destination(env).exists()


def test_legacy_import_requires_complete_explicit_source_mapping(
    tmp_path: Path,
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    base = write_store_clip(tmp_path / "data/baseline", "old/clip", [frame(0, ball())])
    with pytest.raises(ValueError, match="each stored group"):
        freeze_legacy_store(layout, base, ())
    group = LegacySourceGroup(source="tracknet", group_id="old", source_id="c" * 64)
    with pytest.raises(ValueError, match="each stored group"):
        freeze_legacy_store(layout, base, (group, group))


def test_second_append_preserves_explicit_masks_and_existing_snapshot(
    setup: tuple[Environment, PublishRequest, Path],
) -> None:
    env, request, _ = setup
    first_receipt, first_manifest, first_store = result(
        env, BallPublisher(env.layout).publish(request)
    )
    assert first_store.teacher_masks is not None
    policy = BallPublicationPolicy(
        jpeg_quality=90,
        source_splits=(
            SourceSplit(source_id=content_digest(["local", "second"]), split="train"),
        ),
    )
    second = Environment(
        env.layout.root,
        publication_policy=policy,
        base_snapshot=first_receipt.snapshot,
        dataset_version="second",
        video_id="second",
        campaign_id="second",
    )
    request2 = PublishRequest(
        plan=second.plan_ref, product="frames", adoptions=(seal(second),)
    )
    receipt, manifest, store = result(
        second, BallPublisher(second.layout).publish(request2)
    )
    assert len(store.clips) == 3 and len(receipt.reused_clips) == 2
    assert store.teacher_masks is not None
    np.testing.assert_array_equal(
        store.teacher_masks.heatmap[: len(first_store)],
        first_store.teacher_masks.heatmap,
    )
    np.testing.assert_array_equal(
        store.teacher_masks.coordinate[: len(first_store)],
        first_store.teacher_masks.coordinate,
    )
    for old, copied in zip(first_manifest.clips, manifest.clips[:2], strict=True):
        assert old.semantic_digest == copied.semantic_digest
        assert old.clip_id == copied.clip_id and old.source_id == copied.source_id
        assert old.shard.digest == copied.shard.digest
    assert len(store.metadata["append_history"]) == 2
    load_snapshot(env.layout, first_receipt.snapshot)


def test_first_publication_needs_no_legacy_store(tmp_path: Path) -> None:
    policy = BallPublicationPolicy(
        jpeg_quality=90,
        source_splits=(
            SourceSplit(source_id=content_digest(["local", "fixture"]), split="train"),
        ),
    )
    env = Environment(
        tmp_path / "annotation",
        publication_policy=policy,
        dataset_version="incremental",
    )
    request = PublishRequest(
        plan=env.plan_ref, product="frames", adoptions=(seal(env),)
    )
    receipt, _, store = result(env, BallPublisher(env.layout).publish(request))
    assert receipt.base_snapshot is None and not receipt.reused_clips
    assert len(store.clips) == 1


def test_missing_owner_creation_gap_recovers_only_empty_staging(
    setup: tuple[Environment, PublishRequest, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    env, request, _ = setup
    original = publishing._durable_document

    def interrupted(path: Path, document: Any) -> None:
        if path.name == "publication-intent.json":
            raise RuntimeError("owner interruption")
        original(path, document)

    with monkeypatch.context() as faults:
        faults.setattr(publishing, "_durable_document", interrupted)
        with pytest.raises(RuntimeError, match="owner interruption"):
            BallPublisher(env.layout).publish(request)
    assert len(result(env, BallPublisher(env.layout).publish(request))[2].clips) == 2


def test_cross_filesystem_sharing_fails_without_copy_fallback(
    setup: tuple[Environment, PublishRequest, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    env, request, base = setup
    original = publishing.os.link

    def cross_device(source: Path, target: Path) -> None:
        if source.parent == base / "shards":
            raise OSError(errno.EXDEV, "fixture different filesystem")
        original(source, target)

    monkeypatch.setattr(publishing.os, "link", cross_device)
    with pytest.raises(ValueError, match="same filesystem"):
        BallPublisher(env.layout).publish(request)
    assert not destination(env).exists()


@pytest.mark.parametrize("kind", ["index", "metadata", "shard", "membership"])
def test_published_snapshot_rejects_changed_files_and_membership(
    setup: tuple[Environment, PublishRequest, Path], kind: str
) -> None:
    env, request, _ = setup
    service = BallPublisher(env.layout)
    receipt, snapshot, store = result(env, service.publish(request))
    if kind == "membership":
        path = env.layout.artifact_path(receipt.snapshot)
        document = json.loads(path.read_text())
        document["clips"] = document["clips"][:1]
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="size changed|hash changed"):
            load_snapshot(env.layout, receipt.snapshot)
        with pytest.raises(ValueError, match="membership"):
            service.publish(request)
    else:
        target = env.layout.artifact_path(
            snapshot.index
            if kind == "index"
            else snapshot.metadata
            if kind == "metadata"
            else snapshot.clips[1].shard
        )
        target.write_bytes(b"corrupted")
        with pytest.raises(ValueError, match="size changed|hash changed"):
            load_snapshot(env.layout, receipt.snapshot)
        with pytest.raises(ValueError, match="size changed|hash changed"):
            service.publish(request)
    assert store.directory.exists()
