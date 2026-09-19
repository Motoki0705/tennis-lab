"""Tests for the SLCS DataModule split wiring."""

from pathlib import Path

import pytest

from src.tasks.slcs.configuration import SLCSDataRuntimeConfig
from src.tasks.slcs.data.annotation import SLCSDataIndex
from src.tasks.slcs.data.datamodule import SLCSDataModule
from src.tasks.slcs.data.dataset import SLCSDataConfig, SLCSWindowDataset
from src.tasks.slcs.data.splits import generate_overfit_splits, save_split_file
from src.utils.configuration import PathResolver, RuntimePathRoots
from tests.support.tasks.slcs.dataset import (
    DEFAULT_FIXTURE_DINO_SPEC,
    SLCSFixtureDatasetConfig,
    build_slcs_dataset_fixture,
)


def test_explicit_overfit_mode_reuses_train_windows_for_all_stages(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    index = build_slcs_dataset_fixture(
        root,
        SLCSFixtureDatasetConfig(videos=("video_000",), num_frames=16),
    )
    split_file = root / "splits.json"
    save_split_file(
        split_file,
        generate_overfit_splits(index),
        seed=0,
        val_ratio=0.0,
        test_ratio=0.0,
    )
    spec = DEFAULT_FIXTURE_DINO_SPEC
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(
            {
                "project_root": ".",
                "data_root": ".",
                "checkpoint_root": "checkpoints",
                "artifact_root": "artifacts",
                "output_root": "outputs",
                "cache_root": "cache",
                "external_asset_root": "external-assets",
            },
            repository_root=tmp_path,
        )
    )
    data: dict[str, object] = {
        "dataset_root": "dataset",
        "split_file": "dataset/splits.json",
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": False,
        "overfit": True,
        "window_size": 8,
        "train_stride": 8,
        "eval_stride": 16,
        "num_players": 2,
        "num_court_kp": 14,
        "require_dino": True,
        "cache_dino_tokens": True,
        "on_incomplete": "error",
        "dino": {
            "backbone": spec.backbone,
            "patch_size": spec.patch_size,
            "image_height": spec.image_height,
            "image_width": spec.image_width,
            "embed_dim": spec.embed_dim,
            "frame_stride": spec.frame_stride,
        },
        "quality": {
            "min_player_confidence": 0.3,
            "min_ball_cameras": 1,
            "label_weight_power": 1.0,
            "min_window_label_ratio": 0.1,
        },
        "augmentation": {
            "enabled": True,
            "joint_dropout": 0.0,
            "ball_dropout": 0.0,
            "uv_std": 0.0,
            "burst_probability": 0.0,
            "burst_max_frames": 1,
            "rgb_only_probability": 1.0,
            "rgb_dropout_probability": 0.0,
        },
    }
    config = SLCSDataRuntimeConfig.from_mapping(data, resolver)

    datamodule = SLCSDataModule(config)
    datamodule.setup()

    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None
    assert datamodule.train_dataset.scenes == datamodule.val_dataset.scenes
    assert datamodule.train_dataset.scenes == datamodule.test_dataset.scenes
    datasets = (
        datamodule.train_dataset,
        datamodule.val_dataset,
        datamodule.test_dataset,
    )
    assert len({id(dataset) for dataset in datasets}) == 3
    assert [dataset.apply_augmentation for dataset in datasets] == [True, False, False]
    assert all(dataset.split == "train" and dataset.stride == 8 for dataset in datasets)
    for stage in ("validate", "fit", "test", None):
        datamodule.setup(stage)
        assert datamodule.train_dataset is datasets[0]
        assert datamodule.val_dataset is datasets[1]
        assert datamodule.test_dataset is datasets[2]
        assert [dataset.apply_augmentation for dataset in datasets] == [
            True,
            False,
            False,
        ]
    train, val, test = (dataset[0] for dataset in datasets)
    assert not train["player_kp_vis"].any()
    assert val["player_kp_vis"].any()
    assert test["player_kp_vis"].equal(val["player_kp_vis"])
    assert train["target_ball_position"].equal(val["target_ball_position"])


@pytest.mark.parametrize(
    "stages",
    [
        ("fit", "fit"),
        ("validate", "fit", "fit"),
        ("fit", "test", "fit"),
        (None, None),
        ("test", "validate", None, "fit"),
    ],
)
def test_setup_reuses_datasets_and_token_caches_across_stages(
    synthetic_dataset: SLCSDataIndex,
    synthetic_split_file: Path,
    data_config: SLCSDataConfig,
    monkeypatch: pytest.MonkeyPatch,
    stages: tuple[str | None, ...],
) -> None:
    config = SLCSDataRuntimeConfig(
        dataset_root=synthetic_dataset.root,
        split_file=synthetic_split_file,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        overfit=False,
        pipeline=data_config,
    )
    datamodule = SLCSDataModule(config)
    original_build = datamodule._build_dataset
    calls: list[str] = []
    built: dict[str, SLCSWindowDataset] = {}

    def build(split: str) -> SLCSWindowDataset:
        calls.append(split)
        dataset = original_build(split)
        assert dataset._dino_cache, "Fixture must exercise real eager token caching"
        built[split] = dataset
        return dataset

    monkeypatch.setattr(datamodule, "_build_dataset", build)
    required: set[str] = set()
    for stage in stages:
        before = {
            split: (dataset, dataset._dino_cache, dict(dataset._dino_cache))
            for split, dataset in built.items()
        }
        datamodule.setup(stage)
        required.update(
            {
                None: {"train", "val", "test"},
                "fit": {"train", "val"},
                "validate": {"val"},
                "test": {"test"},
            }[stage]
        )
        assert set(calls) == required
        assert len(calls) == len(required), (
            "Each requested split must be built only once"
        )
        for split in ("train", "val", "test"):
            dataset = getattr(datamodule, f"{split}_dataset")
            if split not in required:
                assert dataset is None
            else:
                assert dataset is built[split]
                assert dataset.apply_augmentation == (split == "train")
                assert dataset.stride == (
                    data_config.train_stride
                    if split == "train"
                    else data_config.eval_stride
                )
        for split, (dataset, cache, entries) in before.items():
            assert built[split] is dataset
            assert dataset._dino_cache is cache
            assert dataset._dino_cache.keys() == entries.keys()
            for key, value in entries.items():
                assert dataset._dino_cache[key] is value
