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
    if holdout.existing is not None:
        bind_cohorts(layout, holdout, expected)
    if canonical_bytes(raw) != canonical_bytes(expected):
        raise ValueError("fixed holdout windows differ from their snapshot or declared sampling policy")
    split = "val" if holdout.split == "validation" else "test"
    data = FrozenBallWindowDataset(path, split=split, requires_pose=False)
    if holdout.existing is None and holdout.split == "validation" and not any(r["common_evaluation"] for r in data.records):
        raise ValueError("fixed validation has no declared Common selection subset")
    selected = tuple(c for c in snapshot.clips if c.split == split)
    return tuple(sorted({c.source_id for c in selected})), tuple(sorted(c.clip_id for c in selected))


def bind_cohorts(layout: AnnotationLayout, holdout: HoldoutWindows, raw: dict[str, Any]) -> None:
    """Rebase unchanged old holdouts onto a cumulative store; name both cohorts."""
    if holdout.existing is None:
        raise ValueError("Cohort evaluation requires the original holdout binding")
    old = read_document(layout, holdout.existing, HoldoutWindows)
    if old.existing is not None or old.split != holdout.split:
        raise ValueError("Original holdout must have the same split and no redefined cohorts")
    _check(layout, old)
    original = json.loads(verify_reference(layout, old.manifest).read_bytes())
    if any(canonical_bytes(raw[key]) != canonical_bytes(original[key]) for key in ("config", "sampling", "common_evaluation")):
        raise ValueError("Old holdout sampling or Common membership changed")
    before, _ = load_snapshot(layout, old.snapshot)
    after, _ = load_snapshot(layout, holdout.snapshot)
    current = {c.clip_id: c for c in after.clips}
    for previous in before.clips:
        if previous.split not in {"val", "test"}:
            continue
        now = current.get(previous.clip_id)
        if now is None or (previous.model_dump(exclude={"shard"}) != now.model_dump(exclude={"shard"})
                           or previous.shard.digest != now.shard.digest):
            raise ValueError("Existing holdout clip/teacher identity changed")
    original_records = {r["clip"]["clip_id"]: r for r in original["clips"]
                        if r["clip"]["split"] in {"val", "test"}}
    current_records = {r["clip"]["clip_id"]: r for r in raw["clips"]}
    for name, record in original_records.items():
        if name not in current_records or any(record[key] != current_records[name][key]
                for key in ("semantic_sha256", "rgb_shard_sha256", "windows", "common_evaluation")):
            raise ValueError("Existing holdout window membership changed")
    old_ids = {c.store_clip_id for c in before.clips if c.split in {"val", "test"}}
    eligible_splits = {c.store_clip_id for c in after.clips if c.split in {"val", "test"}}
    raw["evaluation_cohorts"] = {"existing": sorted(old_ids), "new": sorted(eligible_splits - old_ids)}
    split = "val" if holdout.split == "validation" else "test"
    for group in raw["evaluation_cohorts"].values():
        records = [r for r in raw["clips"] if r["clip"]["clip_id"] in group and r["clip"]["split"] == split]
        if any(not any(w["frame_step"] == step for r in records for w in r["windows"])
               for step in raw["sampling"]["frame_steps"]):
            raise ValueError("Every existing/new holdout cohort must have windows at each FPS")


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
