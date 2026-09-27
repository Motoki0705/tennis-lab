"""Tests for the canonical SLCS annotation reader boundary."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tasks.slcs.data.annotation import (
    SLCS_ANNOTATION_FILENAME,
    IncompleteAnnotationError,
    SLCSDataIndex,
    has_slcs_annotation,
    load_slcs_annotation,
    slcs_annotation_dir,
)
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    DatasetManifestError,
    UnsupportedDatasetVersionError,
    register_exported_clip,
    split_clip_id,
    validate_id_component,
)
from src.tennis_scene.schema import SceneResult
from tests.support.tasks.slcs.dataset import (
    SLCSFixtureDatasetConfig,
    build_slcs_dataset_fixture,
    make_fixture_scene,
)
from tests.support.tennis_scene.annotations import (
    publish_dataset_annotations,
    publish_scene_to_clip_store,
)


def test_manifest_symbols_are_consumed_from_canonical_module() -> None:
    assert split_clip_id("video_000/clip_000") == ("video_000", "clip_000")
    with pytest.raises(DatasetManifestError):
        split_clip_id("../x/clip")
    with pytest.raises(DatasetManifestError):
        validate_id_component("..", field_name="video_id")


def test_removed_contract_module_has_no_forwarding_path() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.tasks.slcs.data.contract")


def test_test_only_dataset_builder_has_no_production_module() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.tasks.slcs.data.synthetic")


def test_index_and_manifest_roundtrip(synthetic_dataset: SLCSDataIndex) -> None:
    index = SLCSDataIndex.load(synthetic_dataset.root)
    assert [record.clip_id for record in index.clips] == [
        "video_000/clip_000",
        "video_001/clip_000",
        "video_002/clip_000",
    ]
    assert index.video_ids() == ("video_000", "video_001", "video_002")
    manifest = ClipManifest.load(index.clip_dir(index.clips[0]))
    assert manifest.camera_ids == ("cam0", "cam1")
    assert manifest.media_path("cam0").is_file()
    with pytest.raises(DatasetManifestError):
        manifest.media_path("cam9")


def test_annotation_roundtrip(synthetic_dataset: SLCSDataIndex) -> None:
    clip_dir = synthetic_dataset.clip_dir(synthetic_dataset.clips[0])
    manifest = ClipManifest.load(clip_dir)
    assert has_slcs_annotation(clip_dir)
    scene = load_slcs_annotation(manifest)
    assert scene.num_frames == manifest.num_frames
    assert scene.court_kp.shape[0] == len(manifest.camera_ids)


def test_fixture_publishes_through_the_component_store(
    synthetic_dataset: SLCSDataIndex,
) -> None:
    clip_dir = synthetic_dataset.clip_dir(synthetic_dataset.clips[0])
    clip_payload = json.loads((clip_dir / "clip.json").read_text())
    marker_path = slcs_annotation_dir(clip_dir) / SLCS_ANNOTATION_FILENAME
    marker = json.loads(marker_path.read_text())

    assert clip_payload["sync_source"] == "clip_studio"
    assert marker["generator"] == "src.tennis_scene"
    assert marker["scene_index"] == "scene.json" and marker["scene_result"].startswith("exports/")
    scene = load_slcs_annotation(ClipManifest.load(clip_dir))
    assert scene.schema_version == 2 and scene.ball_uv is not None
    assert marker["arrays"]["ball_uv"] == {
        "shape": list(scene.ball_uv.shape),
        "dtype": str(scene.ball_uv.dtype),
    }


def test_v1_layout_marker_is_rejected(tmp_path: Path) -> None:
    """A marker naming scene.npz directly (the retired v1 layout) is never read."""
    index = build_slcs_dataset_fixture(
        tmp_path / "dataset", SLCSFixtureDatasetConfig(videos=("video_000",))
    )
    clip_dir = index.clip_dir(index.clips[0])
    marker_path = slcs_annotation_dir(clip_dir) / SLCS_ANNOTATION_FILENAME
    marker = json.loads(marker_path.read_text())
    del marker["scene_index"]
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="v1 layout is not readable"):
        load_slcs_annotation(ClipManifest.load(clip_dir))


def test_duplicate_registration_is_idempotent(
    synthetic_dataset: SLCSDataIndex,
) -> None:
    manifest = ClipManifest.load(synthetic_dataset.clip_dir(synthetic_dataset.clips[0]))
    register_exported_clip(synthetic_dataset.root, manifest.manifest_path)
    index = SLCSDataIndex.load(synthetic_dataset.root)
    assert [record.clip_id for record in index.clips].count(manifest.clip_id) == 1


def test_appending_new_video_keeps_existing(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    build_slcs_dataset_fixture(root, SLCSFixtureDatasetConfig(videos=("video_000",)))
    build_slcs_dataset_fixture(root, SLCSFixtureDatasetConfig(videos=("video_001",)))
    assert [record.video_id for record in SLCSDataIndex.load(root).clips] == [
        "video_000",
        "video_001",
    ]


def test_unsupported_dataset_version_uses_canonical_error(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    build_slcs_dataset_fixture(root, SLCSFixtureDatasetConfig(videos=("video_000",)))
    index_path = root / "dataset.json"
    payload = json.loads(index_path.read_text())
    payload["version"] = 99
    index_path.write_text(json.dumps(payload))
    with pytest.raises(UnsupportedDatasetVersionError):
        SLCSDataIndex.load(root)


def test_missing_marker_is_explicitly_incomplete(tmp_path: Path) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "dataset", SLCSFixtureDatasetConfig(videos=("video_000",))
    )
    clip_dir = index.clip_dir(index.clips[0])
    (slcs_annotation_dir(clip_dir) / SLCS_ANNOTATION_FILENAME).unlink()
    assert not has_slcs_annotation(clip_dir)
    with pytest.raises(IncompleteAnnotationError):
        load_slcs_annotation(ClipManifest.load(clip_dir))


def test_manifest_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "dataset", SLCSFixtureDatasetConfig(videos=("video_000",))
    )
    clip_dir = index.clip_dir(index.clips[0])
    manifest_path = clip_dir / "clip.json"
    payload = json.loads(manifest_path.read_text())
    payload["exported_at"] = "edited-after-annotation"
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(DatasetManifestError, match="different clip.json"):
        load_slcs_annotation(ClipManifest.load(clip_dir))


def test_marker_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "dataset", SLCSFixtureDatasetConfig(videos=("video_000",))
    )
    clip_dir = index.clip_dir(index.clips[0])
    marker_path = slcs_annotation_dir(clip_dir) / SLCS_ANNOTATION_FILENAME
    marker = json.loads(marker_path.read_text())
    marker["arrays"]["ball_uv"]["shape"] = [9, 9, 9]
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(DatasetManifestError, match="ball_uv"):
        load_slcs_annotation(ClipManifest.load(clip_dir), verify_manifest_digest=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["cameras"][0].update(calibrated=True), "calibrated"),
        (
            lambda payload: payload["video_paths"].__setitem__(
                0, "../../../etc/passwd"
            ),
            "escapes",
        ),
        (
            lambda payload: payload.__setitem__(
                "num_frames", payload["num_frames"] + 5
            ),
            "num_frames",
        ),
    ],
)
def test_clip_manifest_or_scene_mismatch_is_rejected(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "dataset", SLCSFixtureDatasetConfig(videos=("video_000",))
    )
    clip_dir = index.clip_dir(index.clips[0])
    manifest_path = clip_dir / "clip.json"
    payload = json.loads(manifest_path.read_text())
    mutation(payload)
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(DatasetManifestError, match=message):
        manifest = ClipManifest.load(clip_dir)
        load_slcs_annotation(manifest, verify_manifest_digest=False)


def test_marker_must_record_every_required_array(tmp_path: Path) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "dataset", SLCSFixtureDatasetConfig(videos=("video_000",))
    )
    clip_dir = index.clip_dir(index.clips[0])
    marker_path = slcs_annotation_dir(clip_dir) / SLCS_ANNOTATION_FILENAME
    marker = json.loads(marker_path.read_text())
    del marker["arrays"]["ball_uv"]
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(DatasetManifestError, match="does not record array 'ball_uv'"):
        load_slcs_annotation(ClipManifest.load(clip_dir), verify_manifest_digest=False)


def test_scene_requires_calibration_for_every_manifest_camera(tmp_path: Path) -> None:
    config = SLCSFixtureDatasetConfig(videos=("video_000",))
    index = build_slcs_dataset_fixture(tmp_path / "dataset", config)
    manifest = ClipManifest.load(index.clip_dir(index.clips[0]))
    partial = make_fixture_scene(config, np.random.default_rng(0), manifest.camera_ids[:-1])
    publish_dataset_annotations(index.root, {manifest.clip_id: partial}, overwrite=True)
    with pytest.raises(DatasetManifestError, match="calibration for every manifest camera"):
        load_slcs_annotation(manifest)


def test_v1_scene_is_rejected_even_in_the_component_store(tmp_path: Path) -> None:
    config = SLCSFixtureDatasetConfig(videos=("video_000",))
    index = build_slcs_dataset_fixture(tmp_path / "dataset", config)
    manifest = ClipManifest.load(index.clip_dir(index.clips[0]))
    v2 = make_fixture_scene(config, np.random.default_rng(0), manifest.camera_ids)
    fields = {name: getattr(v2, name) for name in SceneResult.__dataclass_fields__}
    for name in ("player_observed", "player_valid", "player_heading_valid", "player_kp_3d_vis",
                 "player_smpl_valid", "ball_3d_valid", "player_rejection_code",
                 "player_kp_3d_rejection_code", "ball_rejection_code"):
        fields[name] = None
    fields["metadata"] = {"court_reference": {"camera_ids": list(manifest.camera_ids)}}
    export = publish_scene_to_clip_store(manifest.clip_dir, manifest.clip_id, SceneResult(**fields))
    annotation = slcs_annotation_dir(manifest.clip_dir)
    marker_path = annotation / SLCS_ANNOTATION_FILENAME
    marker = json.loads(marker_path.read_text())
    marker["scene_result"] = str(export.relative_to(annotation))
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(DatasetManifestError, match="only SceneResult v2"):
        load_slcs_annotation(manifest)
