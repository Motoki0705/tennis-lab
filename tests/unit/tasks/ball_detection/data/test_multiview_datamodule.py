"""Video/annotation alignment, missing-label semantics and clip-grouped splits."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from src.tasks.ball_detection.configuration import validate_training
from src.tasks.ball_detection.data import build_ball_detection_datamodule
from src.tasks.ball_detection.data.components.multiview import (
    frame_cache_path,
    prepare_frame_cache,
    read_object,
)
from src.tasks.ball_detection.data.dataset import BallDetectionDataset
from src.tasks.ball_detection.data.multiview_datamodule import MultiviewBallDataModule
from tests.support.tasks.ball_detection.multiview import make_multiview_config


@pytest.fixture
def config(tmp_path: Path) -> DictConfig:
    return make_multiview_config(tmp_path)


def test_shared_dataset_preserves_coordinates_and_never_bridges_unknowns(
    config: DictConfig,
) -> None:
    dm = build_ball_detection_datamodule(config)
    assert isinstance(dm, MultiviewBallDataModule)
    dm.prepare_data()
    dm.setup()
    assert type(dm.train_dataset) is BallDetectionDataset
    assert dm.train_dataset is not None and dm.val_dataset is not None
    windows = dm.train_dataset.windows
    assert [w.start_index for w in windows[:6]] == [0, 1, 2, 5, 7, 8]
    assert len(windows) == 12
    sample = dm.train_dataset[1]  # interpolated + occlusion_estimated
    assert sample["images"].shape == (2, 3, 32, 64)
    assert torch.all(sample["visibility"] == 1)
    assert torch.allclose(
        sample["coords"], torch.tensor([[[40.0, 24.0]], [[40.0, 24.0]]])
    )
    assert sample["original_size"].tolist() == [80, 48]
    assert sample["heatmaps"].amax().item() > 0.8
    assert 0 <= sample["images"].min() <= sample["images"].max() <= 1
    assert dm.train_dataset[0]["images"].mean() < dm.train_dataset[2]["images"].mean()
    # Same clip number and camera in different recordings still have unique IDs.
    assert dm.train_dataset[0]["window_id"] != dm.val_dataset[0]["window_id"]
    assert len(next(iter(dm.train_dataloader()))["window_id"]) == 1
    dm.prepare_data()  # exact cache reuse


def test_observed_only_excludes_estimates_without_compressing_time(
    config: DictConfig,
) -> None:
    config.data.accepted_statuses = ["observed"]
    dm = MultiviewBallDataModule(config)
    stream = dm.streams["r0/clip_000"][0]
    assert stream.valid_starts(2, 1) == [5, 7, 8]


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate_split", "Duplicate clip"),
        ("missing_split", "partition"),
        ("bad_frame_index", "frame_index"),
        ("bad_coordinates", "center_px"),
        ("unknown_status", "Unknown annotation status"),
        ("missing_coordinates", "center_px"),
        ("missing_annotation", ""),
        ("bad_count", "frame count"),
    ],
)
def test_invalid_supervision_fails_loudly(
    config: DictConfig, mutation: str, match: str
) -> None:
    root = Path(config.paths.data_root) / "fixture"
    path = root / "clips/r0/clip_000/outsource/cam0_annotations.json"
    document = read_object(path)
    if mutation == "duplicate_split":
        (root / "val.txt").write_text("r0/clip_000\n")
    elif mutation == "missing_split":
        (root / "val.txt").write_text("r9/clip_000\n")
    elif mutation == "missing_annotation":
        path.unlink()
    else:
        if mutation == "bad_frame_index":
            document["frames"][1]["frame_index"] = 0
        elif mutation == "bad_coordinates":
            document["frames"][0]["center_px"]["x"] = float("nan")
        elif mutation == "unknown_status":
            document["frames"][0]["status"] = "not_checked"
        elif mutation == "missing_coordinates":
            document["frames"][0]["center_px"] = None
        elif mutation == "bad_count":
            document["source"]["frame_count"] = 11
        path.write_text(json.dumps(document))
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        _ = MultiviewBallDataModule(config).streams


def test_video_digest_and_cache_completeness(config: DictConfig) -> None:
    dm = MultiviewBallDataModule(config)
    stream = dm.streams["r0/clip_000"][0]
    cache = frame_cache_path(dm.frame_cache_dir, stream, dm.image_size)
    prepare_frame_cache(cache, stream, dm.image_size)
    (cache / "000003.bmp").unlink()
    with pytest.raises(ValueError, match="Incomplete frame cache"):
        prepare_frame_cache(cache, stream, dm.image_size)
    with stream.video_path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_frame_cache(cache.with_name("new"), stream, dm.image_size)


def test_preprocess_materializes_all_frames_losslessly_and_reuses_them(
    config: DictConfig,
) -> None:
    dm = MultiviewBallDataModule(config)
    dm.prepare_data()
    assert dm.preparation_report["new_frames"] == 60
    assert len(list(dm.frame_cache_dir.rglob("*.bmp"))) == 60
    stream = dm.streams["r0/clip_000"][0]
    cache = frame_cache_path(dm.frame_cache_dir, stream, dm.image_size)
    capture = cv2.VideoCapture(str(stream.video_path))
    try:
        for index in range(10):
            ok, frame = capture.read()
            assert ok
            expected = cv2.resize(frame, (64, 32))
            # Includes unresolved frames; these are cached but never supervised.
            assert np.array_equal(cv2.imread(str(cache / f"{index:06d}.bmp")), expected)
    finally:
        capture.release()
    before = (cache / "000000.bmp").stat().st_mtime_ns
    dm.prepare_data()
    assert dm.preparation_report["new_frames"] == 0
    assert (cache / "000000.bmp").stat().st_mtime_ns == before


def test_eight_persistent_workers_can_read_two_complete_epochs(config: DictConfig) -> None:
    config.data.num_workers = 8
    dm = MultiviewBallDataModule(config)
    dm.prepare_data()
    dm.setup("fit")
    loader = dm.val_dataloader()
    first = [batch["window_id"] for batch in loader]
    second = [batch["window_id"] for batch in loader]
    assert first == second
    assert len(first) == 12
    assert loader.persistent_workers
    assert loader.prefetch_factor == 1


@pytest.mark.parametrize("key", ["prepare_workers", "prefetch_factor"])
def test_preprocessing_settings_must_be_positive(config: DictConfig, key: str) -> None:
    config.data[key] = 0
    with pytest.raises(ValueError, match=key):
        validate_training(config)


def test_empty_supervised_split_fails(config: DictConfig) -> None:
    config.model.num_frames = 10
    dm = MultiviewBallDataModule(config)
    dm.prepare_data()
    with pytest.raises(RuntimeError, match="No supervised"):
        dm.setup("fit")


@pytest.mark.parametrize("statuses", [[], ["unresolved"], ["observed", "observed"]])
def test_config_rejects_invalid_status_policy(
    config: DictConfig, statuses: list[str]
) -> None:
    config.data.accepted_statuses = statuses
    with pytest.raises(ValueError, match="accepted_statuses"):
        validate_training(config)
