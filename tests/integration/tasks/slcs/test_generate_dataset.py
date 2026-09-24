"""Scene-to-SLCS preparation with real tiny videos and a CPU fake encoder."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from numpy.typing import NDArray
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationToMissingValueError

from src.tasks.slcs.configuration import SLCSGenerationConfig
from src.tasks.slcs.data.dataset import SLCSWindowDataset
from src.tasks.slcs.data.dino_precompute import FrameEncoder
from src.tasks.slcs.data.dino_tokens import dino_dir, load_dino_tokens
from src.tasks.slcs.data.splits import load_split_assignments
from src.tasks.slcs.generate_dataset.preparation import prepare_dataset
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    DatasetManifestError,
)
from src.utils.paths import PROJECT_ROOT
from tests.support.tasks.slcs.dataset import (
    DEFAULT_FIXTURE_DINO_SPEC,
    FIXTURE_DINO_CHECKPOINT_BYTES,
    SLCSFixtureDatasetConfig,
    build_slcs_dataset_fixture,
)


def _runtime(root: Path) -> SLCSGenerationConfig:
    spec = DEFAULT_FIXTURE_DINO_SPEC
    checkpoint = root / "weights/fake-dino.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        checkpoint.write_bytes(FIXTURE_DINO_CHECKPOINT_BYTES)
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tasks/slcs/configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="generate_dataset",
            overrides=[
                f"paths.data_root={root}",
                f"paths.checkpoint_root={checkpoint.parent}",
                f"precompute.checkpoint_path={checkpoint.name}",
                "data.dataset_root=clips",
                "data.window_size=6",
                "data.train_stride=6",
                "data.eval_stride=6",
                f"data.dino.image_height={spec.image_height}",
                f"data.dino.image_width={spec.image_width}",
                f"data.dino.embed_dim={spec.embed_dim}",
                "precompute.device=cpu",
                "splits.val_ratio=0.34",
                "splits.test_ratio=0.33",
            ],
        )
    return SLCSGenerationConfig.from_config(cfg)


def test_dataset_root_requires_explicit_configuration() -> None:
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tasks/slcs/configs"), version_base="1.3"
    ):
        cfg = compose(config_name="generate_dataset")
    assert OmegaConf.is_missing(cfg.data, "dataset_root")
    with pytest.raises(InterpolationToMissingValueError, match="data.dataset_root"):
        SLCSGenerationConfig.from_config(cfg)


def _no_encoder() -> FrameEncoder:
    raise AssertionError("valid cached features must not load a model")


def _scene_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and path.parent.name == "tennis_scene"
    }


def test_generation_reads_published_scenes_and_reuses_complete_results(
    tmp_path: Path,
) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "clips", SLCSFixtureDatasetConfig(num_frames=6, clips_per_video=2)
    )
    before = _scene_hashes(index.root)
    for ref in index.clips:
        shutil.rmtree(dino_dir(index.clip_dir(ref)))
    runtime = _runtime(tmp_path)
    spec = runtime.precompute.data.pipeline.dino_spec
    encoded = 0

    def encode(frames: NDArray[np.uint8]) -> NDArray[np.float16]:
        nonlocal encoded
        encoded += len(frames)
        assert frames.shape[1:] == (spec.image_height, spec.image_width, 3)
        return np.ones((len(frames), spec.num_tokens, spec.embed_dim), dtype=np.float16)

    first = prepare_dataset(runtime, encoder_factory=lambda: encode)
    assert first.ok and encoded > 0
    assert len(first.features.processed) == len(index.clips)
    assert _scene_hashes(index.root) == before
    split_file = runtime.precompute.data.split_file
    split_bytes = split_file.read_bytes()
    assignments = load_split_assignments(split_file, index)
    assert set(assignments) == set(index.video_ids())
    for split in ("train", "val", "test"):
        dataset = SLCSWindowDataset(
            dataset_root=index.root,
            split_file=split_file,
            split=split,
            config=runtime.precompute.data.pipeline,
            stride=6,
            augment=False,
        )
        assert len(dataset) > 0
        assert dataset[0]["target_ball_position"].shape == (6, 3)
        assert {assignments[meta.video_id] for meta in dataset.metas} == {split}
    second = prepare_dataset(runtime, encoder_factory=_no_encoder)
    assert second.ok and second.split_reused
    assert len(second.features.skipped_existing) == len(index.clips)
    assert _scene_hashes(index.root) == before
    assert split_file.read_bytes() == split_bytes


@pytest.mark.parametrize("fault", ["digest", "camera", "file", "samples", "spec"])
def test_cached_feature_mismatch_is_reported_before_split_publication(
    tmp_path: Path, fault: str
) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "clips", SLCSFixtureDatasetConfig(num_frames=6)
    )
    marker = dino_dir(index.clip_dir(index.clips[0])) / "annotation.json"
    data = json.loads(marker.read_text())
    if fault == "digest":
        data["input_manifest_digest"] = "outdated"
    elif fault == "camera":
        data["cameras"]["cam9"] = data["cameras"]["cam0"]
    elif fault == "file":
        data["cameras"]["cam0"]["file"] = "wrong.npz"
    elif fault == "samples":
        data["cameras"]["cam0"]["num_samples"] += 1
    else:
        data["spec"]["frame_stride"] += 1
    marker.write_text(json.dumps(data))
    report = prepare_dataset(_runtime(tmp_path), encoder_factory=_no_encoder)
    assert not report.ok
    assert index.clips[0].clip_id in report.features.failed
    assert not (index.root / "splits.json").exists()


@pytest.mark.parametrize("fault", ["missing_marker", "missing_ball", "shape"])
def test_incomplete_scene_fails_before_feature_work(tmp_path: Path, fault: str) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "clips", SLCSFixtureDatasetConfig(num_frames=6)
    )
    annotation = index.clip_dir(index.clips[0]) / "annotations/tennis_scene"
    marker = annotation / "annotation.json"
    if fault == "missing_marker":
        marker.unlink()
    elif fault == "missing_ball":
        scene_path = annotation / "scene.npz"
        with np.load(scene_path) as scene:
            arrays = {key: scene[key] for key in scene.files if key != "ball_3d"}
        np.savez_compressed(scene_path, **arrays)
    else:
        data = json.loads(marker.read_text())
        data["arrays"]["ball_uv"]["shape"] = [99, 99, 2]
        marker.write_text(json.dumps(data))
    with pytest.raises(DatasetManifestError):
        prepare_dataset(_runtime(tmp_path), encoder_factory=_no_encoder)
    assert not (index.root / "splits.json").exists()


def test_split_changes_require_explicit_overwrite(tmp_path: Path) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "clips", SLCSFixtureDatasetConfig(num_frames=6)
    )
    runtime = _runtime(tmp_path)
    assert prepare_dataset(runtime, encoder_factory=_no_encoder).ok
    old_bytes = (index.root / "splits.json").read_bytes()
    changed = replace(runtime, splits=replace(runtime.splits, seed=42))
    with pytest.raises(DatasetManifestError, match="splits.overwrite"):
        prepare_dataset(changed, encoder_factory=_no_encoder)
    assert (index.root / "splits.json").read_bytes() == old_bytes
    changed = replace(changed, splits=replace(changed.splits, overwrite=True))
    assert prepare_dataset(changed, encoder_factory=_no_encoder).ok
    assert json.loads((index.root / "splits.json").read_text())["seed"] == 42


def test_feature_overwrite_regenerates_stale_cache(tmp_path: Path) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "clips", SLCSFixtureDatasetConfig(num_frames=6)
    )
    marker = dino_dir(index.clip_dir(index.clips[0])) / "annotation.json"
    document = json.loads(marker.read_text())
    document["input_manifest_digest"] = "stale"
    marker.write_text(json.dumps(document))
    runtime = _runtime(tmp_path)
    assert not prepare_dataset(runtime, encoder_factory=_no_encoder).ok
    before = _scene_hashes(index.root)
    runtime = replace(runtime, precompute=replace(runtime.precompute, overwrite=True))
    spec = runtime.precompute.data.pipeline.dino_spec

    def encode(frames: NDArray[np.uint8]) -> NDArray[np.float16]:
        return np.ones((len(frames), spec.num_tokens, spec.embed_dim), dtype=np.float16)

    report = prepare_dataset(runtime, encoder_factory=lambda: encode)
    assert report.ok
    assert len(report.features.processed) == len(index.clips)
    assert _scene_hashes(index.root) == before
    assert json.loads(marker.read_text())["input_manifest_digest"] != "stale"


def test_real_hydra_cli_returns_nonzero_for_invalid_cached_features(
    tmp_path: Path,
) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "clips", SLCSFixtureDatasetConfig(num_frames=6)
    )
    marker = dino_dir(index.clip_dir(index.clips[0])) / "annotation.json"
    document = json.loads(marker.read_text())
    document["input_manifest_digest"] = "stale"
    marker.write_text(json.dumps(document))
    checkpoint = _runtime(tmp_path).precompute.checkpoint_path
    completed = subprocess.run(
        [
            sys.executable, "-m", "src.tasks.slcs.scripts.generate_dataset",
            f"paths.data_root={tmp_path}",
            f"paths.output_root={tmp_path / 'logs'}",
            f"paths.checkpoint_root={checkpoint.parent}",
            f"precompute.checkpoint_path={checkpoint.name}",
            "data.dataset_root=clips",
            "data.dino.image_height=48",
            "data.dino.image_width=64",
            "data.dino.embed_dim=8",
            "precompute.device=cpu",
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "manifest digest" in completed.stdout + completed.stderr


@pytest.mark.parametrize("change", ["different_path", "same_path", "missing_identity"])
def test_checkpoint_cache_identity_requires_explicit_overwrite(
    tmp_path: Path, change: str
) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "clips", SLCSFixtureDatasetConfig(num_frames=6)
    )
    runtime = _runtime(tmp_path)
    assert prepare_dataset(runtime, encoder_factory=_no_encoder).ok
    before = _scene_hashes(index.root)
    checkpoint = runtime.precompute.checkpoint_path
    markers = [dino_dir(index.clip_dir(ref)) / "annotation.json" for ref in index.clips]
    if change == "missing_identity":
        marker = json.loads(markers[0].read_text())
        del marker["generator"]["checkpoint_sha256"]
        markers[0].write_text(json.dumps(marker))
    else:
        if change == "different_path":
            checkpoint = checkpoint.with_name("different-dino.pth")
        checkpoint.write_bytes(b"a different checkpoint with the same backbone")
        runtime = replace(
            runtime, precompute=replace(runtime.precompute, checkpoint_path=checkpoint)
        )
    marker_bytes = [path.read_bytes() for path in markers]
    split_bytes = runtime.precompute.data.split_file.read_bytes()
    rejected = prepare_dataset(runtime, encoder_factory=_no_encoder)
    assert not rejected.ok
    assert len(rejected.features.failed) == (1 if change == "missing_identity" else 3)
    assert all(
        "checkpoint SHA-256" in error and "precompute.overwrite=true" in error
        for error in rejected.features.failed.values()
    )
    assert [path.read_bytes() for path in markers] == marker_bytes
    assert runtime.precompute.data.split_file.read_bytes() == split_bytes
    assert _scene_hashes(index.root) == before

    spec = runtime.precompute.data.pipeline.dino_spec
    encoded = 0

    def encode(frames: NDArray[np.uint8]) -> NDArray[np.float16]:
        nonlocal encoded
        encoded += len(frames)
        return np.full((len(frames), spec.num_tokens, spec.embed_dim), 2, dtype=np.float16)

    runtime = replace(runtime, precompute=replace(runtime.precompute, overwrite=True))
    regenerated = prepare_dataset(runtime, encoder_factory=lambda: encode)
    assert regenerated.ok and len(regenerated.features.processed) == len(index.clips)
    assert encoded > 0
    expected_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for ref, path in zip(index.clips, markers, strict=True):
        assert json.loads(path.read_text())["generator"]["checkpoint_sha256"] == expected_digest
        manifest = ClipManifest.load(index.clip_dir(ref))
        tokens, _, _ = load_dino_tokens(
            manifest, "cam0", expected_checkpoint_sha256=expected_digest
        )
        assert np.all(tokens == 2)
    assert _scene_hashes(index.root) == before
    runtime = replace(runtime, precompute=replace(runtime.precompute, overwrite=False))
    assert prepare_dataset(runtime, encoder_factory=_no_encoder).ok


def test_identical_checkpoint_contents_can_move_without_reencoding(tmp_path: Path) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "clips", SLCSFixtureDatasetConfig(num_frames=6)
    )
    runtime = _runtime(tmp_path)
    assert prepare_dataset(runtime, encoder_factory=_no_encoder).ok
    moved = tmp_path / "weights/moved-dino.pth"
    runtime.precompute.checkpoint_path.rename(moved)
    runtime = replace(runtime, precompute=replace(runtime.precompute, checkpoint_path=moved))
    reused = prepare_dataset(runtime, encoder_factory=_no_encoder)
    assert reused.ok and len(reused.features.skipped_existing) == len(index.clips)


@pytest.mark.parametrize("indices", [[0, 9, 20], [0, 20]])
def test_sample_grid_mismatch_is_reported_despite_matching_marker_count(
    tmp_path: Path, indices: list[int]
) -> None:
    index = build_slcs_dataset_fixture(
        tmp_path / "clips", SLCSFixtureDatasetConfig(num_frames=21)
    )
    folder = dino_dir(index.clip_dir(index.clips[0]))
    archive = folder / "cam0.npz"
    with np.load(archive) as data:
        tokens = data["tokens"][:len(indices)]
    np.savez_compressed(archive, tokens=tokens, frame_idx=np.asarray(indices, dtype=np.int64))
    marker = json.loads((folder / "annotation.json").read_text())
    marker["cameras"]["cam0"]["num_samples"] = len(indices)
    (folder / "annotation.json").write_text(json.dumps(marker))
    report = prepare_dataset(_runtime(tmp_path), encoder_factory=_no_encoder)
    assert not report.ok
    assert list(report.features.failed) == [index.clips[0].clip_id]
    assert "configured sampling sequence" in report.features.failed[index.clips[0].clip_id]
    assert len(report.features.skipped_existing) == len(index.clips) - 1
    assert not (index.root / "splits.json").exists()


def test_individual_precompute_cli_records_and_checks_checkpoint_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tasks.slcs.scripts import precompute_dino_tokens

    index = build_slcs_dataset_fixture(
        tmp_path / "clips", SLCSFixtureDatasetConfig(num_frames=6)
    )
    for ref in index.clips:
        shutil.rmtree(dino_dir(index.clip_dir(ref)))
    runtime = _runtime(tmp_path)
    spec = runtime.precompute.data.pipeline.dino_spec
    encoded = 0

    def encode(frames: NDArray[np.uint8]) -> NDArray[np.float16]:
        nonlocal encoded
        encoded += len(frames)
        return np.ones((len(frames), spec.num_tokens, spec.embed_dim), dtype=np.float16)

    monkeypatch.setattr(
        precompute_dino_tokens.SLCSPrecomputeConfig, "from_config", lambda _: runtime.precompute
    )
    monkeypatch.setattr(
        precompute_dino_tokens, "create_slcs_frame_token_encoder", lambda _: encode
    )
    assert precompute_dino_tokens.run(OmegaConf.create({})) == 0
    assert encoded > 0
    assert prepare_dataset(runtime, encoder_factory=_no_encoder).ok
    encoded = 0
    runtime.precompute.checkpoint_path.write_bytes(b"changed checkpoint")
    assert precompute_dino_tokens.run(OmegaConf.create({})) == 1
    assert encoded == 0
