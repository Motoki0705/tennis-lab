"""Controlled real-plus-synthetic TrackNet training data adapter."""

from __future__ import annotations

from typing import Any, cast

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.tasks.ball_detection.configuration import validate_data
from src.tasks.ball_detection.data.components.augmentation import (
    BallDetectionAugmentation,
)
from src.tasks.ball_detection.data.components.source_mix_sampler import (
    ExactSourceMixBatchSampler,
)
from src.tasks.ball_detection.data.tracknet_datamodule import TrackNetDataModule
from src.tasks.ball_detection.data.types import BallDetectionSample

_MIXING_KEYS = {
    "sampling_seed",
    "steps_per_epoch",
    "synthetic",
    "synthetic_batch_period",
    "synthetic_per_batch",
}


class MixedTrackNetDataModule(pl.LightningDataModule):
    """Use real TrackNet evaluation splits and an exact mixed training stream.

    The synthetic source is never used for validation or testing. Setting
    ``data.synthetic_per_batch=0`` creates the paired real-only control and does
    not read the synthetic artifact.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        data_cfg = validate_data(config)

        self.batch_size = int(data_cfg["batch_size"])
        self.num_workers = int(data_cfg["num_workers"])
        self.pin_memory = bool(data_cfg["pin_memory"])
        self.steps_per_epoch = int(data_cfg["steps_per_epoch"])
        self.synthetic_per_batch = int(data_cfg["synthetic_per_batch"])
        self.synthetic_batch_period = int(data_cfg["synthetic_batch_period"])
        self.sampling_seed = int(data_cfg["sampling_seed"])

        if self.batch_size <= 0:
            raise ValueError("data.batch_size must be positive.")
        if self.steps_per_epoch <= 0:
            raise ValueError("data.steps_per_epoch must be positive.")
        if not 0 <= self.synthetic_per_batch < self.batch_size:
            raise ValueError(
                "data.synthetic_per_batch must be in [0, data.batch_size)."
            )
        if self.synthetic_batch_period <= 0:
            raise ValueError("data.synthetic_batch_period must be positive.")

        self.real_module = TrackNetDataModule(self._sub_config(synthetic=False))
        self.synthetic_module: TrackNetDataModule | None = None
        self.train_dataset: Dataset[BallDetectionSample] | None = None
        self.val_dataset: Dataset[BallDetectionSample] | None = None
        self.test_dataset: Dataset[BallDetectionSample] | None = None
        self.real_train_size = 0
        self.synthetic_train_size = 0

    def _resolved_config_dict(self) -> dict[str, Any]:
        resolved = OmegaConf.to_container(self.config, resolve=True)
        if not isinstance(resolved, dict):
            raise TypeError("The training config must resolve to a mapping.")
        return cast(dict[str, Any], resolved)

    def _sub_config(self, *, synthetic: bool) -> DictConfig:
        config_dict = self._resolved_config_dict()
        data_cfg = dict(config_dict["data"])
        for key in _MIXING_KEYS:
            del data_cfg[key]
        data_cfg["source"] = "tracknet"

        if synthetic:
            synthetic_cfg = self.config.data.synthetic
            data_cfg["data_dir"] = str(synthetic_cfg["data_dir"])
            synthetic_split = OmegaConf.to_container(
                synthetic_cfg["split"], resolve=True
            )
            if not isinstance(synthetic_split, dict):
                raise TypeError("data.synthetic.split must resolve to a mapping.")
            split = dict(data_cfg["split"])
            split.update(synthetic_split)
            data_cfg["split"] = split
            data_cfg["sample_stride"] = int(synthetic_cfg["sample_stride"])

        config_dict["data"] = data_cfg
        sub_config = OmegaConf.create(config_dict)
        if not isinstance(sub_config, DictConfig):
            raise TypeError("Mixed TrackNet sub-config must be a mapping.")
        return sub_config

    def _ensure_synthetic_module(self) -> TrackNetDataModule:
        if self.synthetic_module is None:
            self.synthetic_module = TrackNetDataModule(self._sub_config(synthetic=True))
        return self.synthetic_module

    def setup(self, stage: str | None = None) -> None:
        """Build real splits and the optional training-only synthetic split."""
        self.real_module.setup(stage=stage)

        if stage in (None, "fit"):
            real_dataset = self.real_module.train_dataset
            if real_dataset is None:
                raise RuntimeError("The real source did not produce a train dataset.")
            self.real_train_size = len(real_dataset)
            datasets: list[Dataset[BallDetectionSample]] = [real_dataset]

            if self.synthetic_per_batch > 0:
                synthetic_module = self._ensure_synthetic_module()
                aug_cfg = synthetic_module.augmentation_config
                synthetic_dataset = synthetic_module.create_dataset(
                    split_name="train",
                    split_file=synthetic_module.train_split_file,
                    augmentation=BallDetectionAugmentation(aug_cfg),
                )
                self.synthetic_train_size = len(synthetic_dataset)
                datasets.append(synthetic_dataset)
            else:
                self.synthetic_train_size = 0

            self.train_dataset = ConcatDataset(datasets)
            self.val_dataset = self.real_module.val_dataset

        if stage in (None, "validate"):
            self.val_dataset = self.real_module.val_dataset
        if stage in (None, "test"):
            self.test_dataset = self.real_module.test_dataset

    def train_dataloader(self) -> DataLoader:
        """Return the deterministic exact-ratio training loader."""
        if self.train_dataset is None:
            raise RuntimeError("setup('fit') must run before train_dataloader().")
        sampler = ExactSourceMixBatchSampler(
            real_size=self.real_train_size,
            synthetic_size=self.synthetic_train_size,
            batch_size=self.batch_size,
            synthetic_per_batch=self.synthetic_per_batch,
            synthetic_batch_period=self.synthetic_batch_period,
            steps_per_epoch=self.steps_per_epoch,
            seed=self.sampling_seed,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the untouched real validation loader."""
        if self.val_dataset is None:
            raise RuntimeError("setup('fit') must run before val_dataloader().")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the untouched real test loader."""
        if self.test_dataset is None:
            raise RuntimeError("setup('test') must run before test_dataloader().")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


__all__ = ["MixedTrackNetDataModule"]
