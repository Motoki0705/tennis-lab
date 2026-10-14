from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from src.tasks.ball_detection.annotation.review.contracts import (
    BallEdits,
    BallObservation,
)
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.windows.dataset import FrozenBallWindowDataset
from src.tasks.ball_detection.data.windows.intervals import PlayIntervalConfig
from src.tasks.ball_detection.data.windows.learning.contracts import (
    HoldoutWindows,
    TrainingWindows,
    WindowPolicy,
)
from src.tasks.ball_detection.data.windows.learning.holdout import (
    freeze_holdout,
    validate_holdout,
)
from src.tasks.ball_detection.data.windows.learning.preparation import (
    prepare_training_windows,
    require_additional_teachers,
    training_partition,
    validate_training_windows,
)
from src.tasks.ball_detection.data.windows.manifest import (
    build_coordinate_manifest,
    clip_semantic_digest,
)
from src.tasks.ball_detection.data.windows.temporal import TemporalSamplingConfig
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
    BallPublicationPolicy,
    LegacySourceGroup,
    PublishRequest,
    SourceSplit,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.publisher import (
    BallPublisher,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.snapshot import (
    freeze_legacy_store,
)
from src.tasks.base.annotation.artifacts import (
    read_document,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    PublicationReceipt,
    canonical_bytes,
    content_digest,
)
from src.tasks.base.annotation.learning.validation import check_partitions
from src.tasks.base.annotation.paths import AnnotationLayout
from tests.support.tasks.ball_detection.annotation import Environment
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


def publish(tmp_path: Path, *, absent: bool = False, frame_count: int = 34) -> tuple[Environment, ArtifactRef, ArtifactRef]:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    directory = tmp_path / "data/ball_detection/base"
    for split in ("train", "val", "test"):
        write_store_clip(directory, f"{split}/clip", [frame(i, ball()) for i in range(32)], split=split)
    base = freeze_legacy_store(layout, directory, tuple(
        LegacySourceGroup(source="tracknet", group_id=split, source_id=content_digest(split))
        for split in ("train", "val", "test")))
    policy = BallPublicationPolicy(jpeg_quality=95,
        source_splits=(SourceSplit(source_id=content_digest(["local", "fixture"]), split="train"),))
    env = Environment(layout.root, publication_policy=policy, base_snapshot=base, frame_count=frame_count)
    visible = BallObservation(track_id="ball", state="visible", center=(16., 12.), origin="manual")
    images = []
    indices = list(range(1, frame_count - 1))
    for start in range(0, len(indices), 16):
        page_indices = tuple(indices[start:start + 16])
        image = env.page(page_indices)
        images.append(image)
        env.apply(BallEdits(images=(image,), operations=tuple(
            env.replace(i, image, presence="absent" if absent else "present", observations=() if absent else (visible,))
            for i in page_indices)))
    adopted = env.adapter.prepare_adoption(env.layout, env.context, env.manifest, env.head(),
        (env.telemetry(tuple(images)),), reason="synthetic learning-input test")
    env.store.seal(env.head(), adopted, env.adapter)
    publication = BallPublisher(layout).publish(PublishRequest(plan=env.plan_ref, product="frames", adoptions=(adopted,)))
    return env, publication, base


def policy() -> WindowPolicy:
    return WindowPolicy(**asdict(PlayIntervalConfig()), frame_steps=(1,))


def holdout(layout: AnnotationLayout, snapshot: ArtifactRef, *, split: str = "validation") -> HoldoutWindows:
    store = BallFrameStore(layout.repository / "data/ball_detection/base")
    val = store.clip_by_id("val/clip")
    raw = build_coordinate_manifest(store, PlayIntervalConfig(), TemporalSamplingConfig((1,)), input_kind="mdd_only",
        common_reference=dict(clip_ids=[val.clip_id], clip_semantic_sha256={val.clip_id: clip_semantic_digest(store, val)}))
    manifest = write_asset(layout, layout.path("models", "fixed-windows.json"), canonical_bytes(raw) + b"\n")
    return HoldoutWindows(snapshot=snapshot, manifest=manifest, split=split)


def test_new_and_old_train_windows_are_cumulative_and_holdouts_stay_fixed(tmp_path: Path) -> None:
    env, publication, base = publish(tmp_path)
    fixed = freeze_holdout(env.layout, holdout(env.layout, base))
    fixed_test = freeze_holdout(env.layout, holdout(env.layout, base, split="test"))
    original = env.layout.artifact_path(fixed.manifest).read_bytes()
    receipt = read_document(env.layout, publication, PublicationReceipt)
    partition = training_partition(env.layout, receipt.snapshot)
    check_partitions(partition, fixed, fixed_test)
    reference = prepare_training_windows(env.layout, publication, policy(), env.layout.path("campaigns", "prepared"))
    prepared = validate_training_windows(env.layout, reference)
    assert prepared.old_train_clips == prepared.added_train_clips == 1
    assert prepared.added_eligible_windows == 1
    manifest = require_additional_teachers(prepared)
    data = FrozenBallWindowDataset(env.layout.artifact_path(manifest), split="train", requires_pose=False)
    assert len(data.records) == 2
    assert all(record["clip"]["split"] == "train" for record in data.records)
    assert data.pose_directory is None
    assert sum(c.added for c in prepared.coverage) == 1
    new = next(c for c in prepared.coverage if c.added)
    assert new.native_frames == 34 and new.coordinate_teacher_frames == 32
    assert prepare_training_windows(env.layout, publication, policy(), env.layout.path("campaigns", "prepared")) == reference
    assert validate_holdout(env.layout, fixed).snapshot == base
    assert env.layout.artifact_path(fixed.manifest).read_bytes() == original


@pytest.mark.parametrize("absent,frames", [(True, 34), (False, 32)])
def test_unusable_new_clips_do_not_start_old_only_training(tmp_path: Path, absent: bool, frames: int) -> None:
    env, publication, _ = publish(tmp_path, absent=absent, frame_count=frames)
    reference = prepare_training_windows(env.layout, publication, policy(), env.layout.path("campaigns", "prepared"))
    prepared = read_document(env.layout, reference, TrainingWindows)
    assert prepared.status == "no_additional_coordinate_teachers"
    assert prepared.old_train_clips == 1 and prepared.added_train_clips == 1
    assert prepared.manifest is None and prepared.added_eligible_windows == 0
    assert next(c for c in prepared.coverage if c.added).status == "no_eligible_window"
    assert next(c for c in prepared.coverage if not c.added).status == "eligible"
    with pytest.raises(ValueError, match="old-only training is forbidden"):
        require_additional_teachers(prepared)


def test_validation_rejects_missing_prepared_inputs_without_repairing_them(tmp_path: Path) -> None:
    env, publication, _ = publish(tmp_path)
    reference = prepare_training_windows(env.layout, publication, policy(), env.layout.path("campaigns", "prepared"))
    prepared = read_document(env.layout, reference, TrainingWindows)
    path = env.layout.artifact_path(require_additional_teachers(prepared))
    path.unlink()
    with pytest.raises(ValueError, match="missing or size changed"):
        validate_training_windows(env.layout, reference)
    assert not path.exists()


def test_fixed_window_subset_and_partition_identity_cannot_be_changed(tmp_path: Path) -> None:
    env, _, base = publish(tmp_path)
    binding = holdout(env.layout, base)
    fixed = freeze_holdout(env.layout, binding)
    forged = fixed.model_copy(update={"sources": ("f" * 64,)})
    with pytest.raises(ValueError, match="partition membership changed"):
        validate_holdout(env.layout, forged)
    raw = json.loads(env.layout.artifact_path(binding.manifest).read_text())
    raw["clips"] = [r for r in raw["clips"] if r["clip"]["split"] != "val"]
    replaced = write_asset(env.layout, env.layout.path("models", "different-windows.json"), canonical_bytes(raw))
    with pytest.raises(ValueError, match="windows differ"):
        freeze_holdout(env.layout, binding.model_copy(update={"manifest": replaced}))


@pytest.mark.parametrize("fault,match", [("task", "newly published"), ("request", "pinned request"), ("base", "reused clips require")])
def test_publication_identity_is_checked_before_deriving_training_inputs(tmp_path: Path, fault: str, match: str) -> None:
    env, publication, _ = publish(tmp_path)
    original = read_document(env.layout, publication, PublicationReceipt)
    updates = {"task": "player"} if fault == "task" else {"request_id": "f" * 64} if fault == "request" else {"base_snapshot": None}
    changed = write_document(env.layout, env.layout.path("campaigns", "changed-publication.json"), original.model_copy(update=updates))
    with pytest.raises(ValueError, match=match):
        prepare_training_windows(env.layout, changed, policy(), env.layout.path("campaigns", "prepared"))
    assert not env.layout.path("campaigns", "prepared").exists()


@pytest.mark.parametrize("frame_steps", [(2,), (1, 1), (1, 0), (2, 1)])
def test_sampling_policy_is_explicit_and_validated(frame_steps: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        WindowPolicy(**asdict(PlayIntervalConfig()), frame_steps=frame_steps)
