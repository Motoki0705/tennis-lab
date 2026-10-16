"""Synthetic published teachers and fixed holdouts for learning tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from src.tasks.ball_detection.annotation.review.contracts import (
    BallEdits,
    BallObservation,
)
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.windows.intervals import PlayIntervalConfig
from src.tasks.ball_detection.data.windows.learning.contracts import (
    HoldoutWindows,
    WindowPolicy,
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
    write_asset,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    canonical_bytes,
    content_digest,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from tests.support.tasks.ball_detection.annotation import Environment
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


def publish(tmp_path: Path, *, absent: bool = False, frame_count: int = 34, setup: Callable[[AnnotationLayout, ArtifactRef], dict[str, Any]] | None = None, new_splits: tuple[str, ...] = ("train",), image_storage: Literal["jpeg", "video_q90"] = "jpeg", base_splits: tuple[str, ...] = ("train", "val", "test")) -> tuple[Environment, ArtifactRef, ArtifactRef]:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    directory = tmp_path / "data/ball_detection/base"
    for split in base_splits:
        write_store_clip(directory, f"{split}/clip", [frame(i, ball()) for i in range(32)], split=split)
    base = freeze_legacy_store(layout, directory, tuple(
        LegacySourceGroup(source="tracknet", group_id=split, source_id=content_digest(split))
        for split in base_splits))
    videos = tuple("fixture" if index == 0 else f"fixture-{index}" for index in range(len(new_splits)))
    publication_policy = BallPublicationPolicy(jpeg_quality=90 if image_storage == "video_q90" else 95, image_storage=image_storage,
        source_splits=tuple(SourceSplit(source_id=content_digest(["local", video]), split=split)
                            for video, split in zip(videos, new_splits, strict=True)))
    settings = setup(layout, base) if setup else {}
    adoptions = []
    environments = []
    for video in videos:
        env = Environment(layout.root, publication_policy=publication_policy, base_snapshot=base,
                          frame_count=frame_count, video_id=video, actor_attempt=f"first-{video}", **settings)
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
        adoptions.append(adopted)
        environments.append(env)
    env = environments[0]
    publication = BallPublisher(layout).publish(PublishRequest(plan=env.plan_ref, product="frames", adoptions=tuple(sorted(adoptions, key=lambda ref: ref.digest))))
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


