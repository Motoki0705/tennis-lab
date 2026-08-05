"""Tests for the SLCS DataModule split wiring."""

from pathlib import Path

from src.tasks.slcs.configuration import SLCSDataRuntimeConfig
from src.tasks.slcs.data.datamodule import SLCSDataModule
from src.tasks.slcs.data.splits import generate_overfit_splits, save_split_file
from src.tasks.slcs.data.synthetic import (
    DEFAULT_TEST_DINO_SPEC,
    SyntheticDatasetConfig,
    build_synthetic_dataset,
)
from src.utils.configuration import PathResolver, RuntimePathRoots


def test_explicit_overfit_mode_reuses_train_windows_for_all_stages(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    index = build_synthetic_dataset(
        root,
        SyntheticDatasetConfig(recordings=("only",), num_frames=16),
    )
    split_file = root / "splits.json"
    save_split_file(
        split_file,
        generate_overfit_splits(index),
        seed=0,
        val_ratio=0.0,
        test_ratio=0.0,
    )
    spec = DEFAULT_TEST_DINO_SPEC
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
    }
    config = SLCSDataRuntimeConfig.from_mapping(data, resolver)

    datamodule = SLCSDataModule(config)
    datamodule.setup()

    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None
    assert datamodule.train_dataset.scenes == datamodule.val_dataset.scenes
    assert datamodule.train_dataset.scenes == datamodule.test_dataset.scenes
