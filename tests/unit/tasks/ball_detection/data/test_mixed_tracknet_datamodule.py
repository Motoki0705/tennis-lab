"""Unit tests for the controlled real-plus-synthetic TrackNet adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from torch.utils.data import Dataset

import src.tasks.ball_detection.data.mixed_tracknet_datamodule as mixed_module
from src.tasks.ball_detection.data.mixed_tracknet_datamodule import (
    MixedTrackNetDataModule,
)

_CONFIG_DIR = Path(__file__).resolve().parents[5] / "src/tasks/ball_detection/configs"


class _TaggedDataset(Dataset[dict[str, str]]):
    def __init__(self, source: str, size: int) -> None:
        self.source = source
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, str]:
        return {"source": self.source, "index": str(index)}


class _FakeTrackNetDataModule:
    instances: list[_FakeTrackNetDataModule] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        data_dir = str(config.data.data_dir)
        self.source = "synthetic" if "synthetic" in data_dir else "real"
        self.train_split_file = str(config.data.split.train_file)
        self.augmentation_config = config.data.augmentation
        self.train_dataset: _TaggedDataset | None = None
        self.val_dataset: _TaggedDataset | None = None
        self.test_dataset: _TaggedDataset | None = None
        self.instances.append(self)

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = _TaggedDataset(
                self.source, 4 if self.source == "synthetic" else 8
            )
            self.val_dataset = _TaggedDataset(self.source, 3)
        if stage in (None, "validate"):
            self.val_dataset = _TaggedDataset(self.source, 3)
        if stage in (None, "test"):
            self.test_dataset = _TaggedDataset(self.source, 2)

    def create_dataset(
        self,
        *,
        split_name: str,
        split_file: str,
        augmentation: Any,
    ) -> _TaggedDataset:
        assert split_name == "train"
        assert split_file.endswith("train.txt")
        assert augmentation is not None
        return _TaggedDataset(self.source, 4)


@pytest.fixture(autouse=True)
def fake_tracknet(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeTrackNetDataModule.instances.clear()
    monkeypatch.setattr(
        mixed_module,
        "TrackNetDataModule",
        _FakeTrackNetDataModule,
    )


def _config(
    *,
    synthetic_per_batch: int,
    synthetic_batch_period: int = 1,
) -> Any:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=["data=3dgs_blcs_mixed", "model=stunet"],
        )
    config.data.data_dir = "real"
    config.data.split.root_role = "data"
    config.data.split.train_file = "real/train.txt"
    config.data.split.val_file = "real/val.txt"
    config.data.split.test_file = "real/test.txt"
    config.data.synthetic.data_dir = "synthetic"
    config.data.synthetic.split.root_role = "data"
    config.data.synthetic.split.train_file = "synthetic/train.txt"
    config.data.batch_size = 3
    config.data.synthetic_per_batch = synthetic_per_batch
    config.data.synthetic_batch_period = synthetic_batch_period
    config.data.steps_per_epoch = 2
    config.data.sampling_seed = 17
    config.data.num_workers = 0
    config.data.pin_memory = False
    return config


def test_control_does_not_read_synthetic_source() -> None:
    datamodule = MixedTrackNetDataModule(_config(synthetic_per_batch=0))

    datamodule.setup("fit")
    batches = list(datamodule.train_dataloader())

    assert len(_FakeTrackNetDataModule.instances) == 1
    assert datamodule.synthetic_train_size == 0
    assert all(source == "real" for batch in batches for source in batch["source"])


def test_treatment_uses_synthetic_only_in_training() -> None:
    datamodule = MixedTrackNetDataModule(_config(synthetic_per_batch=1))

    datamodule.setup("fit")
    train_batches = list(datamodule.train_dataloader())

    assert len(_FakeTrackNetDataModule.instances) == 2
    assert datamodule.real_train_size == 8
    assert datamodule.synthetic_train_size == 4
    assert all(
        list(batch["source"]).count("synthetic") == 1 for batch in train_batches
    )

    val_batch = next(iter(datamodule.val_dataloader()))
    assert set(val_batch["source"]) == {"real"}

    datamodule.setup("test")
    test_batch = next(iter(datamodule.test_dataloader()))
    assert set(test_batch["source"]) == {"real"}


def test_periodic_treatment_halves_synthetic_batch_exposure() -> None:
    datamodule = MixedTrackNetDataModule(
        _config(synthetic_per_batch=1, synthetic_batch_period=2)
    )

    datamodule.setup("fit")
    synthetic_counts = [
        list(batch["source"]).count("synthetic")
        for batch in datamodule.train_dataloader()
    ]

    assert synthetic_counts == [1, 0]


@pytest.mark.parametrize("synthetic_per_batch", [-1, 3])
def test_invalid_mix_count_fails(synthetic_per_batch: int) -> None:
    with pytest.raises(ValueError, match="synthetic_per_batch"):
        MixedTrackNetDataModule(_config(synthetic_per_batch=synthetic_per_batch))


def test_invalid_mix_period_fails() -> None:
    with pytest.raises(ValueError, match="synthetic_batch_period"):
        MixedTrackNetDataModule(
            _config(synthetic_per_batch=1, synthetic_batch_period=0)
        )
