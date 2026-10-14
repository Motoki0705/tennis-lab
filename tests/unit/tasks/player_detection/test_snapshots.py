from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.tasks.base.annotation.artifacts import (
    data_reference,
    read_document,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef, content_digest
from src.tasks.base.annotation.learning.contracts import (
    FixedEvaluation,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.player_detection.data.snapshots.contracts import (
    PlayerPartition,
    SelectionPolicy,
    SourceBinding,
)
from src.tasks.player_detection.data.snapshots.partitions import (
    freeze_partition,
    validate_partition,
)
from src.tasks.player_detection.data.snapshots.store import (
    freeze_existing_store,
    load_snapshot,
    validate_snapshot,
)
from src.tasks.player_detection.data.store import (
    INDEX_FILE,
    METADATA_FILE,
)
from src.tasks.player_detection.generate_dataset.builder import build_dataset
from tests.support.tasks.player_detection.chat_root import (
    SyntheticRoot,
    player,
    publish_player_annotation,
    write_clip,
)
from tests.support.tasks.player_detection.store import legacy_store

POLICY = SelectionPolicy(
    require_reviewed=True,
    require_located_players=True,
    min_visible_box_px=4.0,
    frame_stride=1,
)


@dataclass
class Example:
    layout: AnnotationLayout
    directory: Path
    bindings: tuple[SourceBinding, ...]

    def freeze(self) -> ArtifactRef:
        result: ArtifactRef = freeze_existing_store(
            self.layout, self.directory, self.bindings
        )
        return result


@pytest.fixture
def example(synthetic_root: SyntheticRoot) -> Example:
    for source, clip in (("src_c", "f020-026"), ("src_d", "f030-036")):
        manifest = write_clip(synthetic_root.root, source, clip)
        publish_player_annotation(
            synthetic_root.root, manifest, {1: [player("p1", [10.0, 10.0, 30.0, 40.0])]}
        )
    directory = build_dataset(synthetic_root.config)
    metadata = json.loads((directory / METADATA_FILE).read_bytes())
    splits = {"src_a": "train", "src_b": "train", "src_c": "val", "src_d": "test"}
    for clip in metadata["clips"]:
        clip["split"] = splits[clip["source_id"]]
    for key, field in (("clips_by_split", None), ("frames_by_split", "stored_frames")):
        metadata["counts"][key] = {
            split: sum(
                c[field] if field else 1
                for c in metadata["clips"]
                if c["split"] == split
            )
            for split in ("train", "val", "test")
        }
    (directory / METADATA_FILE).write_text(json.dumps(metadata))
    layout = AnnotationLayout.testing(directory.parents[2] / "annotation")
    return Example(
        layout,
        directory,
        tuple(
            SourceBinding(
                store_source_id=key, source_id=content_digest(["source", key])
            )
            for key in splits
        ),
    )


def test_freeze_preserves_files_native_rows_splits_and_negative_teachers(
    example: Example,
) -> None:
    before = {
        path: path.read_bytes()
        for path in example.directory.rglob("*")
        if path.is_file()
    }
    reference = example.freeze()
    snapshot, store = load_snapshot(example.layout, reference)
    assert example.freeze() == reference
    assert all(path.read_bytes() == value for path, value in before.items())
    assert [c.split for c in snapshot.clips] == [c.split for c in store.clips]
    assert len(store) == 24 and store.frames["confirmed_empty"].sum() == 19
    assert {c.source_id for c in snapshot.clips} == {
        b.source_id for b in example.bindings
    }
    assert not list(example.directory.glob("**/*.mp4"))


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate"])
def test_source_bindings_must_be_exact_without_inferred_identity(
    example: Example, kind: str
) -> None:
    bindings = (
        example.bindings[:-1]
        if kind == "missing"
        else example.bindings
        + (
            (SourceBinding(store_source_id="extra", source_id="a" * 64),)
            if kind == "extra"
            else example.bindings[:1]
        )
    )
    with pytest.raises(ValueError, match="exactly once"):
        freeze_existing_store(example.layout, example.directory, bindings)


def test_aliases_cannot_leak_one_canonical_source_across_fixed_splits(
    example: Example,
) -> None:
    shared = "c" * 64
    bindings = tuple(
        binding.model_copy(update={"source_id": shared})
        if binding.store_source_id in {"src_a", "src_c"}
        else binding
        for binding in example.bindings
    )
    with pytest.raises(ValueError, match="crosses dataset partitions"):
        freeze_existing_store(example.layout, example.directory, bindings)


@pytest.mark.parametrize("file", [INDEX_FILE, METADATA_FILE, "shards/clip-00000.bin"])
def test_any_pinned_store_file_change_is_rejected(example: Example, file: str) -> None:
    reference = example.freeze()
    path = example.directory / file
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="hash changed"):
        load_snapshot(example.layout, reference)


@pytest.mark.parametrize(
    "change", ["semantic_digest", "source_id", "split", "clip_id", "shard"]
)
def test_snapshot_cannot_reassign_clip_or_teacher_identity(
    example: Example, change: str
) -> None:
    snapshot, _ = load_snapshot(example.layout, example.freeze())
    value = (
        "test"
        if change == "split"
        else snapshot.clips[1].shard
        if change == "shard"
        else "d" * 64
    )
    clips = (snapshot.clips[0].model_copy(update={change: value}), *snapshot.clips[1:])
    with pytest.raises(ValueError, match="changed|another RGB"):
        validate_snapshot(example.layout, snapshot.model_copy(update={"clips": clips}))


def test_sparse_v1_import_never_fills_missing_frames_or_negative_flags(
    example: Example,
) -> None:
    sparse = legacy_store(
        example.directory, example.directory.with_name("legacy-fixture")
    )
    bindings = tuple(
        binding for binding in example.bindings if binding.store_source_id != "src_b"
    )
    reference = freeze_existing_store(example.layout, sparse, bindings)
    _, store = load_snapshot(example.layout, reference)
    assert len(store) == 5 and not store.frames["confirmed_empty"].any()
    partition = freeze_partition(example.layout, reference, "train", POLICY)
    verified = validate_partition(example.layout, partition)
    assert verified.selected_frames == 2 and verified.negative_frames == 0


def test_fixed_partitions_pin_policy_frame_order_gt_and_source_membership(
    example: Example,
) -> None:
    reference = example.freeze()
    train = freeze_partition(example.layout, reference, "train", POLICY)
    validation = freeze_partition(example.layout, reference, "val", POLICY)
    test = freeze_partition(example.layout, reference, "test", POLICY)
    fixed = FixedEvaluation(task="player", validation=validation, test=test)
    assert not set(train.sources) & (set(fixed.validation.sources) | set(test.sources))
    value = validate_partition(example.layout, train)
    assert (value.selected_frames, value.selected_instances, value.negative_frames) == (
        11,
        3,
        9,
    )
    assert validate_partition(example.layout, validation).selected_instances == 1
    stride = freeze_partition(
        example.layout,
        reference,
        "train",
        POLICY.model_copy(update={"frame_stride": 2}),
    )
    assert (
        validate_partition(example.layout, stride).support_digest
        != value.support_digest
    )
    assert stride.clips == train.clips
    with pytest.raises(ValueError, match="membership"):
        validate_partition(
            example.layout, train.model_copy(update={"sources": ("e" * 64,)})
        )


def test_forged_partition_support_or_policy_requires_recomputed_proof(
    example: Example,
) -> None:
    partition = freeze_partition(example.layout, example.freeze(), "train", POLICY)
    value = read_document(example.layout, partition.manifest, PlayerPartition)
    for change in (
        {"support_digest": "f" * 64},
        {"policy": POLICY.model_copy(update={"frame_stride": 2})},
    ):
        changed = value.model_copy(update=change)
        ref = write_document(
            example.layout,
            example.layout.path("campaigns", "forged", changed.sha256 + ".json"),
            changed,
        )
        with pytest.raises(ValueError, match="support changed"):
            validate_partition(
                example.layout, partition.model_copy(update={"manifest": ref})
            )


def test_all_negative_validation_is_not_a_defined_ap_holdout(example: Example) -> None:
    metadata = json.loads((example.directory / METADATA_FILE).read_bytes())
    for clip in metadata["clips"]:
        if clip["source_id"] == "src_b":
            clip["split"] = "val"
        elif clip["source_id"] == "src_c":
            clip["split"] = "train"
    (example.directory / METADATA_FILE).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="positive ground-truth"):
        freeze_partition(example.layout, example.freeze(), "val", POLICY)


@pytest.mark.parametrize("field", ["clip_pts", "source_frame_index", "offset"])
def test_freezing_rejects_invalid_native_timeline_or_rgb_ranges(
    example: Example, field: str
) -> None:
    path = example.directory / INDEX_FILE
    with np.load(path) as source:
        data = {key: source[key].copy() for key in source.files}
    data[field][1] = data[field][0]
    np.savez(path, **data)
    with pytest.raises(ValueError, match="sequence must increase|byte ranges"):
        example.freeze()


def test_snapshot_rejects_data_symlink_and_outside_root(example: Example) -> None:
    alias = example.directory.parent / "alias"
    alias.symlink_to(example.directory, target_is_directory=True)
    with pytest.raises(ValueError):
        freeze_existing_store(example.layout, alias, example.bindings)
    with pytest.raises(ValueError):
        data_reference(example.layout, example.layout.repository / "outside.npz")
