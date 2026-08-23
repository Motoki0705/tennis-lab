"""Tests for the sole composable Court DataModule."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data import datamodule as datamodule_module
from src.tasks.court_detection.data.contracts import (
    CourtSampleRecord,
    CourtTargetBundleSpec,
    CourtTargetSpec,
)
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule

pytestmark = pytest.mark.unit

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/court_detection/configs"


def _compose(tmp_path: Path, *, processing: str = "kp"):
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=[f"data/processing={processing}"],
        )
    config.paths.project_root = str(tmp_path)
    config.paths.data_root = "data"
    config.paths.output_root = "outputs"
    config.paths.checkpoint_root = "checkpoints"
    config.paths.artifact_root = "artifacts"
    config.data.batch_size = 2
    config.data.num_workers = 0
    config.data.pin_memory = False
    return config


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("kp", ("kp",)),
        ("seg", ("seg",)),
        ("line", ("line",)),
        ("kp_seg", ("kp", "seg")),
        ("kp_line", ("kp", "line")),
        ("seg_line", ("seg", "line")),
        ("all", ("kp", "seg", "line")),
    ],
)
def test_all_non_empty_target_subsets_are_strictly_configurable(
    tmp_path: Path,
    preset: str,
    expected: tuple[str, ...],
) -> None:
    runtime = CourtTrainingConfig.from_config(_compose(tmp_path, processing=preset))

    assert tuple(target.kind for target in runtime.data.processing.targets) == expected


def test_setup_test_requests_explicit_test_split_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = CourtTargetBundleSpec(
        {
            "kp": CourtTargetSpec(
                kind="kp",
                schema="test_kp",
                output_channels=2,
                channel_names=("left", "right"),
                target_dtype=torch.float32,
                precomputed=False,
            )
        }
    )
    record_calls: list[str] = []
    record = CourtSampleRecord(
        sample_id="sample",
        split="test",
        image_path=tmp_path / "unused.png",
        annotation_path=tmp_path / "unused.json",
        derived_key="test/sample",
        dense_target_refs={},
        payload={},
    )

    class _Input:
        def records(self, split: str):
            record_calls.append(split)
            return (record,)

    class _Pipeline:
        target_bundle_spec = bundle
        input_layer = _Input()

        def preflight(self, records):
            assert records == (record,)

        def process(self, selected):
            raise AssertionError(f"Dataset item should not load: {selected}")

    monkeypatch.setattr(
        datamodule_module,
        "build_court_processing_pipeline",
        lambda config, *, is_train: _Pipeline(),
    )

    datamodule = CourtDetectionDataModule(_compose(tmp_path))
    datamodule.setup("test")

    assert record_calls == ["test"]
    assert datamodule.test_dataset is not None
    assert datamodule.target_bundle_spec == bundle


def test_query_datamodule_scans_all_authority_before_model_or_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = CourtTargetBundleSpec(
        {
            "kp": CourtTargetSpec(
                kind="kp",
                schema=(
                    "synthetic_camera_view_kp14_v3_target_court:gaussian_max_v1"
                ),
                output_channels=14,
                channel_names=tuple(f"kp_{index}" for index in range(14)),
                target_dtype=torch.float32,
                precomputed=False,
            )
        }
    )
    records = {
        split: (
            CourtSampleRecord(
                sample_id=f"sample-{split}",
                split=split,
                image_path=tmp_path / "unused.npy",
                annotation_path=tmp_path / "unused.json",
                derived_key=f"test/{split}",
                dense_target_refs={},
                payload={},
            ),
        )
        for split in ("train", "val", "test")
    }
    preflight_calls: list[tuple[bool, str]] = []

    class _Input:
        available_splits = ("train", "val", "test")

        def records(self, split):
            return records[split]

    class _Pipeline:
        target_bundle_spec = bundle
        input_layer = _Input()

        def __init__(self, is_train: bool) -> None:
            self.is_train = is_train

        def preflight(self, selected):
            preflight_calls.append((self.is_train, selected[0].split))

    def _factory(config, *, is_train, require_pose=False):
        _ = config
        assert require_pose
        return _Pipeline(is_train)

    monkeypatch.setattr(
        datamodule_module,
        "build_court_processing_pipeline",
        _factory,
    )
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        query = compose(
            config_name="train",
            overrides=[
                "data/source=synthetic_court",
                "data.source.keypoint_court_scope=target_court",
                "data/processing=kp",
                "data/augmentation=pose_safe",
                "loss=query_pose",
                "model=query_encoder",
            ],
        )
    query.paths.project_root = str(tmp_path)
    query.paths.data_root = "data"
    query.paths.output_root = "outputs"
    query.paths.checkpoint_root = "checkpoints"
    query.paths.artifact_root = "artifacts"

    datamodule = CourtDetectionDataModule(query)

    assert datamodule.num_workers == 4
    assert preflight_calls == [
        (True, "train"),
        (False, "val"),
        (False, "test"),
    ]
