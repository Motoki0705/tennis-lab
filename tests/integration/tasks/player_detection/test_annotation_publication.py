from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from src.submodules.models import Pose2DResult
from src.tasks.base.annotation.artifacts import read_document
from src.tasks.base.annotation.contracts import PublicationReceipt
from src.tasks.person_tracking.contracts import ScoreOrigin
from src.tasks.player_detection.annotation.pose import approved
from src.tasks.player_detection.annotation.pose.approved import (
    generate_pose,
    validate_result,
)
from src.tasks.player_detection.annotation.pose.corrected import (
    BboxInput,
    TrackingReceipt,
    finalize_review,
    load_final,
    load_tracking,
    real_observations,
    track_corrected,
)
from src.tasks.player_detection.annotation.pose.publisher import (
    PosePublicationRequest,
    PosePublisher,
    PublishedPoseStore,
)
from src.tasks.player_detection.annotation.review.id_review import IdReviewBinding
from src.tasks.player_detection.configuration import (
    FrameSelectionConfig,
    InputSizeConfig,
)
from src.tasks.player_detection.data.detection_dataset import (
    PlayerDetectionDataset,
    collate_detection,
    select_detection_frames,
)
from src.tasks.player_detection.data.snapshots.store import load_snapshot
from src.tasks.player_detection.generate_dataset.publishing.publisher import (
    PlayerPublisher,
    PublishRequest,
)
from tests.support.tasks.player_detection.pose_publication import (
    Linker,
    PublicationFixture,
)


@pytest.fixture
def fixture(tmp_path):
    return PublicationFixture(tmp_path / "annotation")


class Pose:
    def __init__(self, calls, *, fail_at=None):
        self.calls = calls
        self.fail_at = fail_at

    def predict(self, request):
        self.calls.extend(request.frame_indices.tolist())
        if self.fail_at is not None and len(self.calls) >= self.fail_at:
            raise RuntimeError("injected pose failure")
        return Pose2DResult(torch.ones(len(request.frame_indices), 17, 3))


def test_corrected_tracking_preserves_all_rows_boxes_scores_and_never_detects(
    tmp_path, monkeypatch
):
    from src.submodules.models.dino import person_detector

    monkeypatch.setattr(
        person_detector,
        "DinoPersonDetector",
        lambda *a, **k: pytest.fail("must not detect"),
    )
    fixture = PublicationFixture(tmp_path / "annotation")
    inputs, receipt, raw = load_tracking(fixture.env.layout, fixture.tracking)
    items = real_observations(inputs.labels)
    assert receipt.observations == 6
    assert receipt.unassigned_rows == tuple(range(6))  # tracker n_init is unchanged
    assert sorted(raw["detection_rows"][raw["detection_rows"] >= 0].tolist()) == list(
        range(6)
    )
    for item in (fixture.corrected, fixture.added):
        at, frame = np.argwhere(raw["observation_ids"] == item.observation_id)[0]
        assert (
            raw["score_origin"][at, frame] == ScoreOrigin.MISSING
            and raw["scores"][at, frame] == 0
        )
        assert raw["bbox_hashes"][at, frame] == item.box.sha256
    assert len({o.observation_id for _, o in items}) == 6
    monkeypatch.setattr(
        fixture.extractor,
        "extract",
        lambda *a, **k: pytest.fail("resume must use saved tracking"),
    )
    assert (
        track_corrected(
            fixture.env.layout,
            fixture.input,
            models=fixture.models,
            extractor=fixture.extractor,
            aflink=Linker(),
        )
        == fixture.tracking
    )


def test_final_id_decisions_publish_negative_positive_and_ignored_reader_frames(
    fixture,
):
    env = fixture.env
    _, final, labels = load_final(env.layout, fixture.final)
    assert [r.supervision for r in final.teachers] == [
        "ignore",
        "positive",
        "negative",
        "ignore",
        "ignore",
    ]
    assert {
        o.observation_id: o.classification
        for o in labels.frames[1].decision.observations
    } == {
        fixture.corrected.observation_id: "player",
        fixture.duplicate.observation_id: "duplicate",
        fixture.added.observation_id: "player",
        fixture.audience.observation_id: "non_player",
    }
    receipt_ref = PlayerPublisher(env.layout).publish(
        PublishRequest(plan=env.plan_ref, reviews=(fixture.final,))
    )
    receipt = read_document(env.layout, receipt_ref, PublicationReceipt)
    snapshot, store = load_snapshot(env.layout, receipt.snapshot)
    assert receipt.added_clips == (env.manifest.clip_id,)
    assert snapshot.clips[0].clip_id == env.manifest.clip_id
    selected = select_detection_frames(
        store,
        "train",
        FrameSelectionConfig(
            require_reviewed=False,
            require_located_players=False,
            min_visible_box_px=0.0,
        ),
        frame_stride=1,
    )
    assert selected.frames.tolist() == [1, 2]
    assert (
        selected.stats["negative_frames"] == 1
        and selected.stats["dropped_annotation_ignore"] == 1
    )
    assert store.frames["reviewed"].tolist() == [False, True, True, True, False]
    data = PlayerDetectionDataset(
        store,
        selected,
        input_size=InputSizeConfig(short_side=32, max_long_side=64),
        augmentation=None,
    )
    batch = collate_detection([data[0], data[1]])
    assert [tuple(b.shape) for b in batch.boxes_cxcywh] == [(2, 4), (0, 4)]
    assert batch.labels[1].shape == (0,)
    assert store.clips[0].track_ids == ("player_1", "player_2")
    inputs, _, _ = load_final(env.layout, fixture.final)
    for row in range(5):
        np.testing.assert_array_equal(store.read_jpeg(row), inputs.store.read_jpeg(row))
    assert (store.directory / "shards/clip-00000.bin").samefile(
        env.layout.artifact_path(inputs.shard)
    )


def test_id_unknown_and_ungenerated_pose_are_not_publishable(fixture):
    env = fixture.env
    with pytest.raises(ValueError, match="completed identity"):
        finalize_review(
            env.layout, fixture.identity_review(status="needs_review", unknown=True)
        )
    with pytest.raises(ValueError, match="unresolved identities"):
        finalize_review(env.layout, fixture.identity_review(unknown=True))
    with pytest.raises(ValueError, match="ungenerated"):
        PosePublisher(env.layout).publish(
            PosePublicationRequest(
                plan=env.plan_ref, ball_snapshot=fixture.ball_snapshot, results=()
            )
        )


def test_pose_only_calls_approved_real_boxes_reuses_and_reads_receipt(fixture):
    env = fixture.env
    calls: list[int] = []
    policy = fixture.pose_policy
    result = generate_pose(
        env.layout,
        fixture.final,
        policy_ref=policy,
        weights=env.blob,
        model_factory=lambda: Pose(calls),
    )
    assert calls == [1, 1]
    _, arrays = validate_result(env.layout, result)
    assert arrays is not None
    assert arrays["observed"].sum() == 2
    assert {str(x) for x in arrays["observation_ids"][arrays["observed"]]} == {
        fixture.corrected.observation_id,
        fixture.added.observation_id,
    }
    reused = generate_pose(
        env.layout,
        fixture.final,
        policy_ref=policy,
        weights=env.blob,
        model_factory=lambda: pytest.fail("reused pose must not load a model"),
        reuse=result,
    )
    assert validate_result(env.layout, reused)[0].status == "reused"
    receipt = PosePublisher(env.layout).publish(
        PosePublicationRequest(
            plan=env.plan_ref, ball_snapshot=fixture.ball_snapshot, results=(result,)
        )
    )
    reader = PublishedPoseStore(env.layout, receipt)
    loaded = reader.read_clip(env.manifest.clip_id)
    assert loaded is not None
    np.testing.assert_array_equal(loaded["keypoints"], arrays["keypoints"])
    assert not (
        env.layout.repository / "data/ball_detection/one/player_pose/shards"
    ).exists()


def test_pose_not_applicable_does_not_block_bbox_teachers(tmp_path):
    fixture = PublicationFixture(tmp_path / "annotation", presence_threshold=1.0)
    env = fixture.env
    policy = fixture.pose_policy
    result = generate_pose(
        env.layout,
        fixture.final,
        policy_ref=policy,
        weights=env.blob,
        model_factory=lambda: pytest.fail("excluded"),
    )
    outcome, arrays = validate_result(env.layout, result)
    assert (
        outcome.status == "not_applicable"
        and outcome.reason == "presence_at_or_below_threshold"
        and arrays is None
    )
    bbox = PlayerPublisher(env.layout).publish(
        PublishRequest(plan=env.plan_ref, reviews=(fixture.final,))
    )
    assert read_document(env.layout, bbox, PublicationReceipt).added_clips
    pose = PosePublisher(env.layout).publish(
        PosePublicationRequest(
            plan=env.plan_ref, ball_snapshot=fixture.ball_snapshot, results=(result,)
        )
    )
    assert PublishedPoseStore(env.layout, pose).read_clip(env.manifest.clip_id) is None


def test_pose_chunk_resume_after_receipt_failure_does_not_repeat_inference(
    fixture, monkeypatch
):
    env = fixture.env
    calls: list[int] = []
    policy = fixture.pose_policy
    original = approved.write_document

    def fail_receipt(layout, path, value):
        if path.name == "receipt.json":
            raise RuntimeError("injected after pose chunks")
        return original(layout, path, value)

    with monkeypatch.context() as m:
        m.setattr(approved, "write_document", fail_receipt)
        with pytest.raises(RuntimeError, match="after pose chunks"):
            generate_pose(
                env.layout,
                fixture.final,
                policy_ref=policy,
                weights=env.blob,
                model_factory=lambda: Pose(calls),
            )
    result = generate_pose(
        env.layout,
        fixture.final,
        policy_ref=policy,
        weights=env.blob,
        model_factory=lambda: pytest.fail("completed chunks"),
    )
    assert calls == [1, 1] and validate_result(env.layout, result)[1] is not None


def test_incremental_bbox_publication_preserves_existing_snapshot_and_concurrent_receipt(
    fixture,
):
    env = fixture.env
    first_ref = PlayerPublisher(env.layout).publish(
        PublishRequest(plan=env.plan_ref, reviews=(fixture.final,))
    )
    first = read_document(env.layout, first_ref, PublicationReceipt)
    before, old_store = load_snapshot(env.layout, first.snapshot)
    next_fixture = PublicationFixture(
        env.layout.root, version="two", video="new", base=first.snapshot
    )
    request = PublishRequest(
        plan=next_fixture.env.plan_ref, reviews=(next_fixture.final,)
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(
            pool.map(lambda _: PlayerPublisher(env.layout).publish(request), range(2))
        )
    assert receipts[0] == receipts[1]
    next_receipt = read_document(env.layout, receipts[0], PublicationReceipt)
    after, store = load_snapshot(env.layout, next_receipt.snapshot)
    assert next_receipt.reused_clips == first.added_clips
    assert before == load_snapshot(env.layout, first.snapshot)[0]
    assert before.clips[0].semantic_digest == after.clips[0].semantic_digest
    for key, values in old_store.frames.items():
        np.testing.assert_array_equal(store.frames[key][: len(values)], values)
    assert env.layout.artifact_path(before.clips[0].shard).samefile(
        env.layout.artifact_path(after.clips[0].shard)
    )


@pytest.mark.parametrize("stage", ["build", "rename", "receipt"])
def test_bbox_publication_recovers_owned_staging_and_receipt_gaps(
    fixture, monkeypatch, stage
):
    from src.tasks.base.annotation import publication
    from src.tasks.player_detection.generate_dataset.publishing import publisher

    env = fixture.env
    request = PublishRequest(plan=env.plan_ref, reviews=(fixture.final,))
    with monkeypatch.context() as m:
        if stage == "build":
            m.setattr(
                publisher,
                "merge_tables",
                lambda *a: (_ for _ in ()).throw(RuntimeError("injected build")),
            )
        elif stage == "rename":
            original = publication.os.rename

            def rename(a, b):
                original(a, b)
                raise RuntimeError("injected rename")

            m.setattr(publication.os, "rename", rename)
        else:
            original_write = publisher.write_document

            def write(layout, path, value):
                if path.name == "receipt.json":
                    raise RuntimeError("injected receipt")
                return original_write(layout, path, value)

            m.setattr(publisher, "write_document", write)
        with pytest.raises(RuntimeError, match="injected"):
            PlayerPublisher(env.layout).publish(request)
    ref = PlayerPublisher(env.layout).publish(request)
    assert ref == PlayerPublisher(env.layout).publish(request)
    load_snapshot(
        env.layout, read_document(env.layout, ref, PublicationReceipt).snapshot
    )


def test_all_empty_clip_survives_publication_and_pose_has_explicit_reason(tmp_path):
    fixture = PublicationFixture(tmp_path / "annotation", all_empty=True)
    env = fixture.env
    ref = PlayerPublisher(env.layout).publish(
        PublishRequest(plan=env.plan_ref, reviews=(fixture.final,))
    )
    _, store = load_snapshot(
        env.layout, read_document(env.layout, ref, PublicationReceipt).snapshot
    )
    assert len(store.instances["bbox_xyxy"]) == 0
    selected = select_detection_frames(
        store, "train", FrameSelectionConfig(True, True, 0.0), frame_stride=1
    )
    assert (
        selected.frames.tolist() == [1, 2, 3] and selected.stats["negative_frames"] == 3
    )
    result = generate_pose(
        env.layout,
        fixture.final,
        policy_ref=fixture.pose_policy,
        weights=env.blob,
        model_factory=lambda: pytest.fail("all empty clip cannot call pose"),
    )
    assert validate_result(env.layout, result)[0].reason == "no_approved_real_players"


def test_estimated_observation_is_not_a_pose_crop_or_positive_teacher(tmp_path):
    fixture = PublicationFixture(tmp_path / "annotation", estimated=True)
    env = fixture.env
    calls: list[int] = []
    _, final, labels = load_final(env.layout, fixture.final)
    assert final.teachers[1].supervision == "ignore"
    assert any(
        o.box_state == "estimated" for o in labels.frames[1].decision.observations
    )
    generate_pose(
        env.layout,
        fixture.final,
        policy_ref=fixture.pose_policy,
        weights=env.blob,
        model_factory=lambda: Pose(calls),
    )
    assert calls == [1, 1]


def test_final_id_commit_resumes_after_seal_before_receipt(tmp_path, monkeypatch):
    from src.tasks.player_detection.annotation.pose import corrected

    fixture = PublicationFixture(tmp_path / "annotation", finalize=False)
    original = corrected.write_document

    def fail_receipt(layout, path, value):
        if path.name == "receipt.json":
            raise RuntimeError("after ID seal")
        return original(layout, path, value)

    with monkeypatch.context() as m:
        m.setattr(corrected, "write_document", fail_receipt)
        with pytest.raises(RuntimeError, match="after ID seal"):
            finalize_review(fixture.env.layout, fixture.review)
    reference = finalize_review(fixture.env.layout, fixture.review)
    assert reference == finalize_review(fixture.env.layout, fixture.review)
    assert load_final(fixture.env.layout, reference)[1].player_ids


@pytest.mark.parametrize(
    "field", ["media", "frames", "image", "bbox_revision", "id_review", "selection"]
)
def test_pose_rejects_retargeted_input_identity(fixture, field):
    env = fixture.env
    original = generate_pose(
        env.layout,
        fixture.final,
        policy_ref=fixture.pose_policy,
        weights=env.blob,
        model_factory=lambda: Pose([]),
    )
    result, _ = validate_result(env.layout, original)
    forged = env.save(
        result.model_copy(
            update={"identity": result.identity.model_copy(update={field: "f" * 64})}
        )
    )
    with pytest.raises(ValueError, match="Pose identity changed"):
        validate_result(env.layout, forged)


def test_pose_reuse_checks_id_review_evidence_and_exact_pose_reference(fixture):
    from src.tasks.base.annotation.artifacts import annotation_reference, write_asset
    from src.tasks.player_detection.annotation.pose.corrected import (
        IdentityReview,
    )
    from src.tasks.player_detection.annotation.pose.storage import write_npz

    env = fixture.env
    original = generate_pose(
        env.layout,
        fixture.final,
        policy_ref=fixture.pose_policy,
        weights=env.blob,
        model_factory=lambda: Pose([]),
    )
    result, arrays = validate_result(env.layout, original)
    assert arrays is not None
    # The same arrays at a different path are not the published reused artifact.
    copy = env.layout.path("annotations", "pose-copy.npz")
    write_npz(copy, **arrays)
    forged = env.save(
        result.model_copy(
            update={
                "status": "reused",
                "reused_from": original,
                "poses": annotation_reference(env.layout, copy),
            }
        )
    )
    with pytest.raises(ValueError, match="Reused pose input identity"):
        validate_result(env.layout, forged)
    _, final, _ = load_final(env.layout, fixture.final)
    review = read_document(env.layout, final.identity_review, IdentityReview)
    changed_image = write_asset(
        env.layout,
        env.layout.path("annotations", "other", review.required_sheets[0]),
        b"different review image",
    )
    changed_review = env.save(review.model_copy(update={"images": (changed_image,)}))
    changed_final = env.save(
        final.model_copy(update={"identity_review": changed_review})
    )
    with pytest.raises(ValueError, match="reuse requires identical"):
        generate_pose(
            env.layout,
            changed_final,
            policy_ref=fixture.pose_policy,
            weights=env.blob,
            reuse=original,
            model_factory=lambda: pytest.fail("reuse cannot infer"),
        )


@pytest.mark.parametrize("stage", ["rename", "receipt"])
def test_pose_publication_resumes_without_inference_and_rechecks_receipt(
    fixture, monkeypatch, stage
):
    from src.tasks.base.annotation import publication
    from src.tasks.player_detection.annotation.pose import publisher

    env = fixture.env
    result = generate_pose(
        env.layout,
        fixture.final,
        policy_ref=fixture.pose_policy,
        weights=env.blob,
        model_factory=lambda: Pose([]),
    )
    request = PosePublicationRequest(
        plan=env.plan_ref, ball_snapshot=fixture.ball_snapshot, results=(result,)
    )
    with monkeypatch.context() as m:
        if stage == "rename":
            rename = publication.os.rename

            def fail_rename(a, b):
                rename(a, b)
                raise RuntimeError("publication interrupted")

            m.setattr(publication.os, "rename", fail_rename)
        else:
            original = publisher.write_document

            def fail_receipt(layout, path, value):
                if path.name == "receipt.json":
                    raise RuntimeError("publication interrupted")
                return original(layout, path, value)

            m.setattr(publisher, "write_document", fail_receipt)
        with pytest.raises(RuntimeError, match="publication interrupted"):
            PosePublisher(env.layout).publish(request)
    ref = PosePublisher(env.layout).publish(request)
    assert (
        PublishedPoseStore(env.layout, ref).read_clip(env.manifest.clip_id) is not None
    )
    receipt = read_document(env.layout, ref, PublicationReceipt)
    with pytest.raises(ValueError, match="receipt"):
        PublishedPoseStore(
            env.layout, env.save(receipt.model_copy(update={"added_clips": ()}))
        )


def test_bbox_receipt_rejects_retargeting_and_preserves_legacy_baseline(fixture):
    from src.tasks.player_detection.data.snapshots.contracts import SourceBinding
    from src.tasks.player_detection.data.snapshots.store import freeze_existing_store
    from src.tasks.player_detection.generate_dataset.publishing.publisher import (
        load_publication,
    )

    env = fixture.env
    receipt = PlayerPublisher(env.layout).publish(
        PublishRequest(plan=env.plan_ref, reviews=(fixture.final,))
    )
    publication, snapshot, store = load_publication(env.layout, receipt)
    with pytest.raises(ValueError, match="receipt differs"):
        load_publication(
            env.layout,
            env.save(publication.model_copy(update={"adoptions": (env.blob,)})),
        )
    # Explicitly frozen v2 input is also a valid cumulative base, with its legacy identity unchanged.
    baseline = freeze_existing_store(
        env.layout,
        store.directory,
        (
            SourceBinding(
                store_source_id=store.clips[0].source_id,
                source_id=snapshot.clips[0].source_id,
            ),
        ),
    )
    second = PublicationFixture(
        env.layout.root, version="legacy-base", video="different", base=baseline
    )
    after = PlayerPublisher(env.layout).publish(
        PublishRequest(plan=second.env.plan_ref, reviews=(second.final,))
    )
    _, next_snapshot, next_store = load_publication(env.layout, after)
    original, _ = load_snapshot(env.layout, baseline)
    assert next_snapshot.clips[0].clip_id == original.clips[0].clip_id
    assert next_store.clips[0].split == store.clips[0].split


@pytest.mark.parametrize(
    "mutation", ["missing_column", "unknown_schema", "positive_empty", "mixed_legacy"]
)
def test_reader_rejects_corrupt_explicit_supervision(fixture, mutation, tmp_path):
    import json
    import shutil

    from src.tasks.player_detection.data.store import PlayerFrameStore

    env = fixture.env
    ref = PlayerPublisher(env.layout).publish(
        PublishRequest(plan=env.plan_ref, reviews=(fixture.final,))
    )
    _, original = load_snapshot(
        env.layout, read_document(env.layout, ref, PublicationReceipt).snapshot
    )
    destination = tmp_path / "corrupt"
    shutil.copytree(original.directory, destination)
    metadata = json.loads((destination / "metadata.json").read_text())
    with np.load(destination / "index.npz", allow_pickle=False) as archive:
        columns = dict(archive)
    if mutation == "missing_column":
        del columns["teacher_supervision"]
    elif mutation == "unknown_schema":
        metadata["teacher_schema"] = "unrecognized"
    elif mutation == "positive_empty":
        columns["teacher_supervision"][2] = 1
    else:
        columns["teacher_supervision"][1] = 0
    np.savez(destination / "index.npz", **columns)
    (destination / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="teacher|supervision"):
        PlayerFrameStore(destination)


def test_bbox_edit_after_tracking_invalidates_id_review(tmp_path):
    from src.tasks.player_detection.annotation.review.contracts import PlayerEdits

    fixture = PublicationFixture(tmp_path / "annotation", finalize=False)
    env = fixture.env
    image = env.page((1,))
    items = env.labels().frames[1].decision.observations
    changed = tuple(
        item.model_copy(update={"box": item.box.model_copy(update={"x1": 3.0})})
        if item.observation_id == fixture.corrected.observation_id
        else item
        for item in items
    )
    env.apply(
        PlayerEdits(
            operations=(env.replace(1, image, observations=changed),), images=(image,)
        )
    )
    with pytest.raises(ValueError, match="Bbox revision changed"):
        finalize_review(env.layout, fixture.review)


@pytest.mark.parametrize(
    "field", ["context", "revision", "frame", "frame_digest", "observation_ids"]
)
def test_tracking_rejects_retargeted_bbox_worker_handoff(tmp_path, field):
    fixture = PublicationFixture(tmp_path / "annotation", finalize=False)
    env = fixture.env
    binding = read_document(env.layout, fixture.binding, IdReviewBinding)
    if field in {"context", "revision"}:
        binding = binding.model_copy(update={field: env.blob})
    else:
        frame = binding.frames[1]
        changed = {
            "frame": frame.frame.model_copy(
                update={"source_pts": frame.frame.source_pts + 1}
            ),
            "frame_digest": "0" * 64,
            "observation_ids": tuple(reversed(frame.observation_ids)),
        }[field]
        binding = binding.model_copy(
            update={
                "frames": (
                    binding.frames[0],
                    frame.model_copy(update={field: changed}),
                    *binding.frames[2:],
                )
            }
        )
    value = read_document(env.layout, fixture.input, BboxInput)
    forged_input = env.save(value.model_copy(update={"id_review": env.save(binding)}))
    receipt = read_document(env.layout, fixture.tracking, TrackingReceipt)
    with pytest.raises(ValueError, match="ID review"):
        load_tracking(
            env.layout, env.save(receipt.model_copy(update={"input": forged_input}))
        )
