"""Build MDD-only inputs without making unavailable new teachers look usable."""

from __future__ import annotations

from pathlib import Path

from src.tasks.ball_detection.data.annotation_states import annotation_states
from src.tasks.ball_detection.data.windows.manifest import build_coordinate_manifest
from src.tasks.ball_detection.data.windows.temporal import select_sampled_windows
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.contracts import (
    PublishRequest,
)
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.snapshot import (
    load_snapshot,
)
from src.tasks.base.annotation.artifacts import (
    read_document,
    verify_reference,
    write_asset,
    write_document,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    PublicationReceipt,
    canonical_bytes,
)
from src.tasks.base.annotation.learning.contracts import DatasetPartition
from src.tasks.base.annotation.paths import AnnotationLayout

from .contracts import ClipCoverage, TrainingWindows, WindowPolicy


def training_partition(layout: AnnotationLayout, reference: ArtifactRef) -> DatasetPartition:
    snapshot, _ = load_snapshot(layout, reference)
    clips = tuple(c for c in snapshot.clips if c.split == "train")
    result = DatasetPartition(manifest=reference, split="train",
        sources=tuple(sorted({c.source_id for c in clips})), clips=tuple(sorted(c.clip_id for c in clips)))
    return result


def prepare_training_windows(
    layout: AnnotationLayout, publication_ref: ArtifactRef, policy: WindowPolicy, directory: Path,
) -> ArtifactRef:
    publication = read_document(layout, publication_ref, PublicationReceipt)
    if (publication.task, publication.product) != ("ball", "frames") or not publication.added_clips:
        raise ValueError("ball coordinate learning requires newly published frame teachers")
    snapshot, store = load_snapshot(layout, publication.snapshot)
    if snapshot.request is None:
        raise ValueError("training publication must refer to an adopted snapshot")
    request = read_document(layout, snapshot.request, PublishRequest)
    if (request.sha256, request.product, request.adoptions) != (
        publication.request_id, publication.product, publication.adoptions
    ):
        raise ValueError("publication receipt differs from its pinned request")
    if set(publication.added_clips + publication.reused_clips) != {c.clip_id for c in snapshot.clips}:
        raise ValueError("publication membership differs from cumulative snapshot")
    if not set(publication.added_clips).intersection(c.clip_id for c in snapshot.clips if c.split == "train"):
        raise ValueError("publication must include new training teachers")
    if publication.base_snapshot is not None:
        base, _ = load_snapshot(layout, publication.base_snapshot)
        if publication.reused_clips != tuple(c.clip_id for c in base.clips):
            raise ValueError("reused publication membership differs from its base")
        for old, current in zip(base.clips, snapshot.clips, strict=False):
            if (old.model_dump(exclude={"shard"}) != current.model_dump(exclude={"shard"})
                    or (old.shard.digest, old.shard.size_bytes) != (current.shard.digest, current.shard.size_bytes)):
                raise ValueError("cumulative snapshot changed an existing clip")
        if len(snapshot.clips) < len(base.clips):
            raise ValueError("cumulative snapshot removed existing clips")
    elif publication.reused_clips:
        raise ValueError("reused clips require an explicit base snapshot")
    coverage = []
    for identity in snapshot.clips:
        if identity.split != "train":
            continue
        clip = store.clip_by_id(identity.store_clip_id)
        states = annotation_states(store, clip)
        _, windows = select_sampled_windows(states, policy.play(), policy.sampling())
        coverage.append(ClipCoverage(clip_id=identity.clip_id, source_id=identity.source_id,
            store_clip_id=clip.clip_id, added=identity.clip_id in publication.added_clips,
            native_frames=clip.frame_count, coordinate_teacher_frames=int(states.supervision.sum()),
            windows_by_frame_step={str(step): sum(w.frame_step == step for w in windows) for step in policy.frame_steps},
            status="eligible" if windows else "no_eligible_window"))
    added_windows = sum(sum(c.windows_by_frame_step.values()) for c in coverage if c.added)
    manifest_ref: ArtifactRef | None = None
    if added_windows:
        raw = build_coordinate_manifest(store, policy.play(), policy.sampling(), input_kind="mdd_only",
            shard_hashes={c.store_clip_id: c.shard.digest for c in snapshot.clips})
        train_ids = {c.store_clip_id for c in snapshot.clips if c.split == "train"}
        # The immutable store contains all splits; this derived manifest exposes
        # only cumulative train windows. Fixed holdouts are separate inputs.
        raw["clips"] = [r for r in raw["clips"] if r["clip"]["split"] == "train"]
        raw["skipped"] = [r for r in raw["skipped"] if r["clip_id"] in train_ids]
        raw["counts"] = {k: v for k, v in raw["counts"].items() if k.endswith("/train")}
        manifest_ref = write_asset(layout, directory / "windows.json", canonical_bytes(raw) + b"\n")
    result = TrainingWindows(publication=publication_ref, snapshot=publication.snapshot, policy=policy,
        manifest=manifest_ref, coverage=tuple(coverage),
        old_train_clips=sum(not c.added for c in coverage), added_train_clips=sum(c.added for c in coverage),
        added_eligible_windows=added_windows, status="ready" if added_windows else "no_additional_coordinate_teachers")
    reference: ArtifactRef = write_document(layout, directory / "coverage.json", result)
    return reference


def require_additional_teachers(prepared: TrainingWindows) -> ArtifactRef:
    if prepared.status != "ready" or prepared.manifest is None:
        raise ValueError("new clips contain no eligible MDD coordinate window; old-only training is forbidden")
    return prepared.manifest


def validate_training_windows(layout: AnnotationLayout, reference: ArtifactRef) -> TrainingWindows:
    prepared: TrainingWindows = read_document(layout, reference, TrainingWindows)
    if prepared.manifest is not None:
        verify_reference(layout, prepared.manifest)
    # Deterministic regeneration writes identical metadata only. RGB is never
    # decoded/copied here. Different persisted content is an error, not a repair.
    rebuilt = prepare_training_windows(layout, prepared.publication, prepared.policy, layout.artifact_path(reference).parent)
    if rebuilt != reference:
        raise ValueError("prepared training inputs changed")
    return prepared
