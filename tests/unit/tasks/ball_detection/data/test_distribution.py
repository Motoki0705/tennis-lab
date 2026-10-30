"""Portable subsets preserve inputs and fail closed on incomplete transfers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.ball_detection.data.distribution import (
    load_distribution,
    prepare_distribution,
    sample_windows,
    source_payloads,
    verify_distribution,
)
from src.tasks.ball_detection.data.distribution.manifest import distribution_files
from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.data.teacher_masks import EXPLICIT_MASK_SCHEMA
from src.utils.checksum import dual_sha256
from src.utils.data.verification import file_version
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


@pytest.fixture
def source(tmp_path: Path) -> Path:
    directory = tmp_path / "source"
    write_store_clip(directory, "tracknet/test/clip", [frame(0, ball())], split="test")
    write_store_clip(directory, "chat_annotation/train/clip", [
        frame(0, ball()), frame(1), frame(2, ball("unresolved", None)),
        frame(3, ball(), ball(track="second")),
    ], source="chat_annotation")
    write_store_clip(directory, "tracknet/validation/clip", [frame(0, ball()), frame(1, ball())], split="val")
    original = tmp_path / "original.json"
    original.write_text(json.dumps(dict(notes="uncertain empty frame", frames=[dict(notes=""), dict(notes="uncertain")])))
    metadata = json.loads((directory / "metadata.json").read_text())
    metadata["version"] = "source-v1"
    metadata["clips"][1].update(annotation_path=str(original), annotation_sha256=dual_sha256(original))
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return directory


def prepare(source: Path, destination: Path, *, clips: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = prepare_distribution(
        source, destination, version="train-val-v1",
        clip_ids=clips if clips is not None else ["tracknet/validation/clip", "chat_annotation/train/clip"],
        inputs={},
    )
    return result


def materialize(source: Path, staging: Path, destination: Path) -> None:
    for name, path in distribution_files(staging, source_directory=source).items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(staging / "distribution.json", destination / "distribution.json")


def test_subset_is_readable_without_excluded_shards_after_relocation(source: Path, tmp_path: Path) -> None:
    paths = [p for p in source.rglob("*") if p.is_file()] + [tmp_path / "original.json"]
    before = {p: file_version(p) for p in paths}
    stage = tmp_path / "stage"
    manifest = prepare(source, stage)
    assert [c["source_index"] for c in manifest["clips"]] == [1, 2]
    assert [c["index"] for c in manifest["clips"]] == [0, 1]
    assert not (stage / "shards").exists()
    with pytest.raises(ValueError, match="Shard size"):
        BallFrameStore(stage)
    preview = BallFrameStore(stage, external_payloads=source_payloads(stage, source))
    windows = [(c.clip_id, 0, c.frame_count) for c in preview.clips]
    expected = sample_windows(BallFrameStore(source), windows)
    assert sample_windows(preview, windows) == expected
    assert verify_distribution(stage, source_directory=source)["mode"] == "source_payloads"
    assert {p: file_version(p) for p in paths} == before

    destination = tmp_path / "moved" / "distribution"
    materialize(source, stage, destination)
    shutil.rmtree(stage)
    shutil.rmtree(source)
    (tmp_path / "original.json").unlink()
    store = BallFrameStore(destination)
    assert sample_windows(store, windows) == expected
    receipt = verify_distribution(destination)
    assert receipt["mode"] == "self_contained" and receipt["clips"] == 2
    assert sorted(c.split for c in store.clips) == ["train", "val"]
    assert len(list((destination / "shards").iterdir())) == 2
    assert all(not Path(c.get("annotation_path", "")).is_absolute()
               for c in json.loads((destination / "metadata.json").read_text())["clips"])


def test_explicit_teacher_masks_are_reindexed_and_preserved(source: Path, tmp_path: Path) -> None:
    store = BallFrameStore(source)
    with np.load(source / "index.npz") as archive:
        columns = {k: archive[k] for k in archive.files}
    heatmap: NDArray[np.uint8] = np.full(len(store), 255, dtype=np.uint8)
    coordinate = heatmap.copy()
    rows = store.clip_rows(store.clips[2])
    heatmap[rows], coordinate[rows] = 2, 1
    savez: Any = np.savez
    savez(source / "index.npz", **columns, teacher_heatmap=heatmap, teacher_coordinate=coordinate)
    metadata = json.loads((source / "metadata.json").read_text())
    metadata["schema_version"] = EXPLICIT_MASK_SCHEMA
    (source / "metadata.json").write_text(json.dumps(metadata))
    prepare(source, tmp_path / "stage", clips=[store.clips[2].clip_id])
    result = BallFrameStore(tmp_path / "stage", external_payloads=source_payloads(tmp_path / "stage", source))
    assert result.teacher_masks is not None
    np.testing.assert_array_equal(result.teacher_masks.heatmap, [2, 2])
    np.testing.assert_array_equal(result.teacher_masks.coordinate, [1, 1])
    assert sample_windows(result, [(store.clips[2].clip_id, 0, 2)]) == sample_windows(
        BallFrameStore(source), [(store.clips[2].clip_id, 0, 2)])


def test_omitting_a_middle_clip_remaps_instance_offsets(source: Path, tmp_path: Path) -> None:
    original = BallFrameStore(source)
    selected = [original.clips[i].clip_id for i in (0, 2)]
    stage = tmp_path / "stage"
    manifest = prepare(source, stage, clips=selected)
    assert [c["source_index"] for c in manifest["clips"]] == [0, 2]
    result = BallFrameStore(stage, external_payloads=source_payloads(stage, source))
    assert result.frames["inst_start"].tolist() == [0, 1, 2]
    windows = [(c.clip_id, 0, c.frame_count) for c in result.clips]
    assert sample_windows(result, windows) == sample_windows(original, windows)


def test_sample_receipts_survive_json_round_trip(source: Path) -> None:
    store = BallFrameStore(source)
    receipt = sample_windows(store, [(clip.clip_id, 0, clip.frame_count) for clip in store.clips])
    assert json.loads(json.dumps(receipt)) == receipt


def test_parallel_verification_matches_the_serial_receipt(source: Path, tmp_path: Path) -> None:
    stage, destination = tmp_path / "stage", tmp_path / "distribution"
    prepare(source, stage)
    materialize(source, stage, destination)
    assert verify_distribution(destination, workers=2) == verify_distribution(destination)
    with pytest.raises(ValueError, match="positive"):
        verify_distribution(destination, workers=0)


def test_relative_roots_verify_source_preview_and_relocated_store(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, destination = tmp_path / "stage", tmp_path / "distribution"
    prepare(source, stage)
    materialize(source, stage, destination)
    expected_preview = verify_distribution(stage, source_directory=source)
    expected_store = verify_distribution(destination)
    monkeypatch.chdir(tmp_path)
    assert verify_distribution(Path("stage"), source_directory=Path("source")) == expected_preview
    assert verify_distribution(Path("distribution")) == expected_store
    assert all(p.is_absolute() for p in distribution_files(Path("distribution")).values())


def test_root_symlink_is_rejected_before_path_normalization(source: Path, tmp_path: Path) -> None:
    stage, destination = tmp_path / "stage", tmp_path / "distribution"
    prepare(source, stage)
    materialize(source, stage, destination)
    alias = tmp_path / "alias"
    alias.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        verify_distribution(alias)


@pytest.mark.parametrize("damage", ["missing", "same_size_corruption", "extra", "symlink"])
def test_incomplete_or_changed_distribution_is_rejected(source: Path, tmp_path: Path, damage: str) -> None:
    stage, destination = tmp_path / "stage", tmp_path / "distribution"
    prepare(source, stage)
    materialize(source, stage, destination)
    shard = destination / "shards/clip-00001.bin"
    if damage == "missing":
        shard.unlink()
    elif damage == "same_size_corruption":
        data = shard.read_bytes()
        shard.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    elif damage == "extra":
        (destination / "unexpected.txt").write_text("not manifested")
    else:
        shard.unlink()
        shard.symlink_to(source / "shards/clip-00002.bin")
    with pytest.raises(ValueError):
        verify_distribution(destination)


def test_source_identity_and_exact_external_payload_set_are_required(source: Path, tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    prepare(source, stage)
    payloads = source_payloads(stage, source)
    with pytest.raises(ValueError, match="every clip exactly"):
        BallFrameStore(stage, external_payloads={})
    with pytest.raises(ValueError, match="every clip exactly"):
        BallFrameStore(stage, external_payloads={**payloads, "unknown": tmp_path / "missing"})
    metadata = source / "metadata.json"
    metadata.write_text(metadata.read_text() + " ")
    with pytest.raises(ValueError, match="pinned identity"):
        source_payloads(stage, source)


def test_manifest_inputs_are_copied_and_checked(source: Path, tmp_path: Path) -> None:
    inputs = tmp_path / "common.jsonl"
    inputs.write_text('["tracknet/validation/clip",[[0,2]]]\n')
    stage = tmp_path / "stage"
    manifest = prepare_distribution(source, stage, version="validation-v1",
                                    clip_ids=["tracknet/validation/clip"], inputs={"inputs/common.jsonl": inputs})
    assert (stage / "inputs/common.jsonl").read_bytes() == inputs.read_bytes()
    entry = next(e for e in manifest["files"] if e["path"] == "inputs/common.jsonl")
    assert entry["sha256"] == dual_sha256(inputs)
    (stage / "inputs/common.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="mismatch"):
        verify_distribution(stage, source_directory=source)


@pytest.mark.parametrize("path", ["/absolute.json", "inputs/../escape", "inputs//duplicate", "metadata.json"])
def test_input_paths_cannot_escape_or_override_the_store(source: Path, tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError):
        prepare_distribution(source, tmp_path / "distribution", version="subset-v1",
                             clip_ids=["tracknet/validation/clip"], inputs={path: tmp_path / "original.json"})
    assert not (tmp_path / "distribution").exists()


def test_selection_and_version_are_explicit(source: Path, tmp_path: Path) -> None:
    for clips in ([], ["tracknet/validation/clip"] * 2):
        with pytest.raises(ValueError, match="unique clip selection"):
            prepare(source, tmp_path / "stage", clips=clips)
    with pytest.raises(ValueError, match="differ"):
        prepare_distribution(source, tmp_path / "stage", version="source-v1",
                             clip_ids=["tracknet/validation/clip"], inputs={})
    with pytest.raises(KeyError):
        prepare(source, tmp_path / "stage", clips=["unknown"])
    prepare(source, tmp_path / "stage")
    with pytest.raises(FileExistsError):
        prepare(source, tmp_path / "stage")


def test_legacy_annotations_require_an_explicit_absolute_path(source: Path, tmp_path: Path) -> None:
    store = BallFrameStore(source)
    clip = store.clip_by_id("chat_annotation/train/clip")
    assert store.annotation_path(clip) == tmp_path / "original.json"
    (tmp_path / "original.json").unlink()
    assert store.annotation_path(clip, must_exist=False) == tmp_path / "original.json"
    with pytest.raises(ValueError, match="Original annotation is missing"):
        store.annotation_path(clip)
    metadata = json.loads((source / "metadata.json").read_text())
    metadata["clips"][clip.index]["annotation_path"] = "../original.json"
    (source / "metadata.json").write_text(json.dumps(metadata))
    store = BallFrameStore(source)
    with pytest.raises(ValueError, match="Legacy annotation paths must be absolute"):
        store.annotation_path(clip)


@pytest.mark.parametrize("value", [
    "../original.json", "/absolute.json", "annotations/../original.json",
    "annotations//original.json", "./annotations/original.json",
])
def test_relative_annotation_paths_require_a_declared_base_and_stay_inside(
    source: Path, tmp_path: Path, value: str,
) -> None:
    stage, destination = tmp_path / "stage", tmp_path / "distribution"
    prepare(source, stage)
    materialize(source, stage, destination)
    metadata = json.loads((destination / "metadata.json").read_text())
    metadata["clips"][0]["annotation_path"] = value
    (destination / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="relative to the store"):
        BallFrameStore(destination)


@pytest.mark.parametrize("symlink", ["annotation", "parent"])
def test_distribution_annotations_reject_symlinks_and_parent_escapes(
    source: Path, tmp_path: Path, symlink: str,
) -> None:
    stage, destination = tmp_path / "stage", tmp_path / "distribution"
    prepare(source, stage)
    materialize(source, stage, destination)
    store = BallFrameStore(destination)
    annotation = store.annotation_path(store.clip_by_id("chat_annotation/train/clip"))
    if symlink == "annotation":
        annotation.unlink()
        annotation.symlink_to(tmp_path / "original.json")
    else:
        moved = tmp_path / "moved-annotations"
        annotation.parent.rename(moved)
        annotation.parent.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the store or is a symlink"):
        BallFrameStore(destination)


def test_unknown_manifest_version_and_duplicate_paths_are_rejected(source: Path, tmp_path: Path) -> None:
    directory = tmp_path / "stage"
    manifest = prepare(source, directory)
    manifest["schema_version"] = "ball_frame_store_distribution.future"
    (directory / "distribution.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema"):
        load_distribution(directory)
    manifest["schema_version"] = "ball_frame_store_distribution.v1"
    manifest["files"].append(manifest["files"][0])
    (directory / "distribution.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unique"):
        load_distribution(directory)
