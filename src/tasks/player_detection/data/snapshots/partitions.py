"""Fixed source/clip membership and ordered frame/target support for shared learning."""

from __future__ import annotations

import hashlib

from src.tasks.base.annotation.artifacts import read_document, write_document
from src.tasks.base.annotation.contracts import ArtifactRef, canonical_bytes
from src.tasks.base.annotation.learning.contracts import DatasetPartition
from src.tasks.base.annotation.paths import AnnotationLayout

from ..detection_dataset import select_detection_frames
from .contracts import ExistingPlayerSnapshot, PlayerPartition, SelectionPolicy, Split
from .store import load_snapshot


def _support(
    layout: AnnotationLayout,
    snapshot_ref: ArtifactRef,
    split: Split,
    policy: SelectionPolicy,
) -> tuple[ExistingPlayerSnapshot, PlayerPartition]:
    snapshot, store = load_snapshot(layout, snapshot_ref)
    selected = select_detection_frames(
        store, split, policy.selection(), frame_stride=policy.frame_stride
    )
    digest = hashlib.sha256()
    for row, boxes in zip(selected.frames, selected.boxes_xyxy, strict=True):
        identity = snapshot.clips[int(store.frames["clip"][row])]
        digest.update(
            canonical_bytes(
                [
                    identity.clip_id,
                    identity.source_id,
                    int(store.frames["frame_index"][row]),
                    int(store.frames["source_frame_index"][row]),
                    int(store.frames["clip_pts"][row]),
                    bool(store.frames["confirmed_empty"][row]),
                    boxes.tolist(),
                ]
            )
            + b"\n"
        )
    return snapshot, PlayerPartition(
        snapshot=snapshot_ref,
        split=split,
        policy=policy,
        selected_frames=selected.stats["frames"],
        selected_instances=selected.stats["instances"],
        negative_frames=selected.stats["negative_frames"],
        support_digest=digest.hexdigest(),
    )


def _membership(
    snapshot: ExistingPlayerSnapshot, partition: PlayerPartition, reference: ArtifactRef
) -> DatasetPartition:
    clips = [clip for clip in snapshot.clips if clip.split == partition.split]
    return DatasetPartition(
        manifest=reference,
        split="validation" if partition.split == "val" else partition.split,
        sources=tuple(sorted({clip.source_id for clip in clips})),
        clips=tuple(sorted(clip.clip_id for clip in clips)),
    )


def freeze_partition(
    layout: AnnotationLayout,
    snapshot: ArtifactRef,
    split: Split,
    policy: SelectionPolicy,
) -> DatasetPartition:
    pinned, partition = _support(layout, snapshot, split, policy)
    reference = write_document(
        layout,
        layout.path("campaigns", "player-partitions", partition.sha256 + ".json"),
        partition,
    )
    return _membership(pinned, partition, reference)


def validate_partition(
    layout: AnnotationLayout, partition: DatasetPartition
) -> PlayerPartition:
    value = read_document(layout, partition.manifest, PlayerPartition)
    snapshot, expected = _support(layout, value.snapshot, value.split, value.policy)
    if (
        value != expected
        or _membership(snapshot, value, partition.manifest) != partition
    ):
        raise ValueError(
            "player partition membership, frame policy or target support changed"
        )
    return value
