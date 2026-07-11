"""Contract reader/writer tests for the issue #634 dataset layout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.tasks.slcs.data.contract import (
    ClipManifest,
    DatasetContractError,
    DatasetIndex,
    IncompleteAnnotationError,
    UnsupportedFormatVersionError,
    has_tennis_scene_annotation,
    load_tennis_scene_annotation,
    split_clip_id,
    tennis_scene_dir,
    validate_id_component,
)
from src.tasks.slcs.data.contract_writer import append_dataset_index
from src.tasks.slcs.data.synthetic import (
    SyntheticDatasetConfig,
    build_synthetic_dataset,
)


def test_split_clip_id_valid() -> None:
    assert split_clip_id("rec-a/clip_000") == ("rec-a", "clip_000")


@pytest.mark.parametrize(
    "clip_id",
    ["noslash", "a/b/c", "../x/clip", "rec/..", "rec/cli p", "rec/", "/clip"],
)
def test_split_clip_id_rejects_malformed(clip_id: str) -> None:
    with pytest.raises(DatasetContractError):
        split_clip_id(clip_id)


def test_validate_id_component_rejects_traversal() -> None:
    with pytest.raises(DatasetContractError):
        validate_id_component("..", field_name="recording_id")


def test_index_roundtrip(synthetic_dataset: DatasetIndex) -> None:
    index = DatasetIndex.load(synthetic_dataset.root)
    assert [ref.clip_id for ref in index.clips] == [
        "rec-a/clip_000",
        "rec-b/clip_000",
        "rec-c/clip_000",
    ]
    assert index.recording_ids() == ("rec-a", "rec-b", "rec-c")


def test_manifest_roundtrip(synthetic_dataset: DatasetIndex) -> None:
    manifest = ClipManifest.load(
        synthetic_dataset.clip_dir(synthetic_dataset.clips[0])
    )
    assert manifest.clip_id == "rec-a/clip_000"
    assert manifest.camera_ids == ("cam0",)
    assert manifest.media_path("cam0").is_file()
    with pytest.raises(DatasetContractError):
        manifest.media_path("cam9")


def test_scene_annotation_roundtrip(synthetic_dataset: DatasetIndex) -> None:
    clip_dir = synthetic_dataset.clip_dir(synthetic_dataset.clips[0])
    manifest = ClipManifest.load(clip_dir)
    assert has_tennis_scene_annotation(clip_dir)
    scene = load_tennis_scene_annotation(manifest)
    assert scene.num_frames == manifest.num_frames
    assert scene.court_kp.shape[0] == len(manifest.camera_ids)


def test_duplicate_index_registration_fails(synthetic_dataset: DatasetIndex) -> None:
    manifest = ClipManifest.load(
        synthetic_dataset.clip_dir(synthetic_dataset.clips[0])
    )
    with pytest.raises(DatasetContractError, match="already registered"):
        append_dataset_index(synthetic_dataset.root, manifest)


def test_appending_new_recording_keeps_existing(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    build_synthetic_dataset(root, SyntheticDatasetConfig(recordings=("first",)))
    build_synthetic_dataset(root, SyntheticDatasetConfig(recordings=("second",)))
    index = DatasetIndex.load(root)
    assert [ref.recording_id for ref in index.clips] == ["first", "second"]


def test_unsupported_dataset_version(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    build_synthetic_dataset(root, SyntheticDatasetConfig(recordings=("only",)))
    index_path = root / "dataset.json"
    payload = json.loads(index_path.read_text())
    payload["format_version"] = 99
    index_path.write_text(json.dumps(payload))
    with pytest.raises(UnsupportedFormatVersionError):
        DatasetIndex.load(root)


def test_missing_marker_is_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    index = build_synthetic_dataset(root, SyntheticDatasetConfig(recordings=("only",)))
    clip_dir = index.clip_dir(index.clips[0])
    (tennis_scene_dir(clip_dir) / "annotation.json").unlink()
    assert not has_tennis_scene_annotation(clip_dir)
    with pytest.raises(IncompleteAnnotationError):
        load_tennis_scene_annotation(ClipManifest.load(clip_dir))


def test_manifest_digest_mismatch_detected(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    index = build_synthetic_dataset(root, SyntheticDatasetConfig(recordings=("only",)))
    clip_dir = index.clip_dir(index.clips[0])
    manifest_path = clip_dir / "clip.json"
    payload = json.loads(manifest_path.read_text())
    payload["source"] = {"origin": "edited-after-annotation"}
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(DatasetContractError, match="different clip.json"):
        load_tennis_scene_annotation(ClipManifest.load(clip_dir))


def test_marker_shape_mismatch_detected(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    index = build_synthetic_dataset(root, SyntheticDatasetConfig(recordings=("only",)))
    clip_dir = index.clip_dir(index.clips[0])
    marker_path = tennis_scene_dir(clip_dir) / "annotation.json"
    marker = json.loads(marker_path.read_text())
    marker["arrays"]["ball_uv"]["shape"] = [9, 9, 9]
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(DatasetContractError, match="ball_uv"):
        load_tennis_scene_annotation(
            ClipManifest.load(clip_dir), verify_manifest_digest=False
        )


def test_calibrated_camera_block_rejected(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    index = build_synthetic_dataset(root, SyntheticDatasetConfig(recordings=("only",)))
    clip_dir = index.clip_dir(index.clips[0])
    manifest_path = clip_dir / "clip.json"
    payload = json.loads(manifest_path.read_text())
    payload["cameras"] = {"cam0": {"calibrated": True, "focal": 1000.0}}
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(DatasetContractError, match="calibrated"):
        ClipManifest.load(clip_dir)


def test_media_path_traversal_rejected(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    index = build_synthetic_dataset(root, SyntheticDatasetConfig(recordings=("only",)))
    clip_dir = index.clip_dir(index.clips[0])
    manifest_path = clip_dir / "clip.json"
    payload = json.loads(manifest_path.read_text())
    payload["media"]["cam0"] = "../../../etc/passwd"
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(DatasetContractError, match="escapes"):
        ClipManifest.load(clip_dir)


def test_scene_frame_count_mismatch_detected(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    index = build_synthetic_dataset(root, SyntheticDatasetConfig(recordings=("only",)))
    clip_dir = index.clip_dir(index.clips[0])
    manifest_path = clip_dir / "clip.json"
    payload = json.loads(manifest_path.read_text())
    payload["num_frames"] = payload["num_frames"] + 5
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(DatasetContractError, match="num_frames"):
        load_tennis_scene_annotation(
            ClipManifest.load(clip_dir), verify_manifest_digest=False
        )


def test_required_scene_arrays_enforced(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    index = build_synthetic_dataset(root, SyntheticDatasetConfig(recordings=("only",)))
    clip_dir = index.clip_dir(index.clips[0])
    scene_path = tennis_scene_dir(clip_dir) / "scene.npz"
    data = dict(np.load(scene_path, allow_pickle=False))
    del data["ball_uv"]
    np.savez_compressed(scene_path, **data)
    with pytest.raises(DatasetContractError, match="ball_uv"):
        load_tennis_scene_annotation(
            ClipManifest.load(clip_dir), verify_manifest_digest=False
        )
