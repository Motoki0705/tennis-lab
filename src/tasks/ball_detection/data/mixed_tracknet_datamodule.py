"""Controlled real-plus-synthetic TrackNet training data adapter."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset

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

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__()
        self.config = config or OmegaConf.create({})
        data_cfg = self.config.get("data", {}) or {}

        self.batch_size = int(data_cfg.get("batch_size", 0))
        self.num_workers = int(data_cfg.get("num_workers", 0))
        self.pin_memory = bool(data_cfg.get("pin_memory", True))
        self.steps_per_epoch = int(data_cfg.get("steps_per_epoch", 0))
        self.synthetic_per_batch = int(data_cfg.get("synthetic_per_batch", 0))
        self.synthetic_batch_period = int(
            data_cfg.get("synthetic_batch_period", 1)
        )
        self.sampling_seed = int(data_cfg.get("sampling_seed", 0))

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
        return resolved

    def _sub_config(self, *, synthetic: bool) -> DictConfig:
        config_dict = self._resolved_config_dict()
        data_cfg = dict(config_dict.get("data", {}))
        for key in _MIXING_KEYS:
            data_cfg.pop(key, None)
        data_cfg["source"] = "tracknet"

        if synthetic:
            synthetic_cfg = self.config.get("data", {}).get("synthetic", {}) or {}
            required = ("data_dir", "split")
            missing = [key for key in required if key not in synthetic_cfg]
            if missing:
                raise ValueError(
                    f"data.synthetic is missing required keys: {missing}."
                )
            data_cfg["data_dir"] = str(synthetic_cfg["data_dir"])
            data_cfg["split"] = OmegaConf.to_container(
                synthetic_cfg["split"], resolve=True
            )
            if "sample_stride" in synthetic_cfg:
                data_cfg["sample_stride"] = int(synthetic_cfg["sample_stride"])

        config_dict["data"] = data_cfg
        return OmegaConf.create(config_dict)

    def _ensure_synthetic_module(self) -> TrackNetDataModule:
        if self.synthetic_module is None:
            self.synthetic_module = TrackNetDataModule(
                self._sub_config(synthetic=True)
            )
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
                aug_cfg = synthetic_module.config.get("data", {}).get(
                    "augmentation", {}
                )
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
