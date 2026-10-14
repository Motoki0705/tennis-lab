"""Pin existing MDD evaluation windows and canonical source/clip memberships."""

from __future__ import annotations

import json
from typing import Any

from src.tasks.ball_detection.data.windows.dataset import FrozenBallWindowDataset
from src.tasks.ball_detection.data.windows.manifest import build_coordinate_manifest
from src.tasks.ball_detection.generate_dataset.frame_store.publishing.snapshot import (
    load_snapshot,
)
from src.tasks.base.annotation.artifacts import (
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.contracts import canonical_bytes
from src.tasks.base.annotation.learning.contracts import DatasetPartition
from src.tasks.base.annotation.paths import AnnotationLayout

from .contracts import HoldoutWindows, WindowPolicy


def _check(layout: AnnotationLayout, holdout: HoldoutWindows) -> tuple[tuple[str, ...], tuple[str, ...]]:
    snapshot, store = load_snapshot(layout, holdout.snapshot)
    path = verify_reference(layout, holdout.manifest)
    raw: dict[str, Any] = json.loads(path.read_bytes())
    policy = WindowPolicy.model_validate_json(json.dumps({**raw["config"], **raw["sampling"]}))
    expected = build_coordinate_manifest(store, policy.play(), policy.sampling(), input_kind="mdd_only",
        common_reference=raw["common_evaluation"], shard_hashes={c.store_clip_id: c.shard.digest for c in snapshot.clips})
    if canonical_bytes(raw) != canonical_bytes(expected):
        raise ValueError("fixed holdout windows differ from their snapshot or declared sampling policy")
    split = "val" if holdout.split == "validation" else "test"
    data = FrozenBallWindowDataset(path, split=split, requires_pose=False)
    if holdout.split == "validation" and not any(r["common_evaluation"] for r in data.records):
        raise ValueError("fixed validation has no declared Common selection subset")
    selected = tuple(c for c in snapshot.clips if c.split == split)
    return tuple(sorted({c.source_id for c in selected})), tuple(sorted(c.clip_id for c in selected))


def freeze_holdout(layout: AnnotationLayout, holdout: HoldoutWindows) -> DatasetPartition:
    sources, clips = _check(layout, holdout)
    reference = write_document(layout, layout.path("models", "ball", "evaluation", holdout.sha256 + ".json"), holdout)
    result = DatasetPartition(manifest=reference, split=holdout.split, sources=sources, clips=clips)
    return result


def validate_holdout(layout: AnnotationLayout, partition: DatasetPartition) -> HoldoutWindows:
    holdout: HoldoutWindows = read_document(layout, partition.manifest, HoldoutWindows)
    sources, clips = _check(layout, holdout)
    if (partition.split, partition.sources, partition.clips) != (holdout.split, sources, clips):
        raise ValueError("fixed holdout partition membership changed")
    return holdout
