"""Shared Lightning DataModule base for scene-directory datasets.

Extracts the common structure of :class:`PLCSDataModule` and
:class:`BLCSDataModule`: reading loader settings from ``config["data"]``,
checking the scene directory exists, building train/val/test datasets from
split files, and constructing the DataLoaders with identical kwargs.

Subclasses provide the task-specific dataset construction
(:meth:`_build_dataset`), the collate function (:meth:`_build_collate_fn`),
and a human-readable name (:meth:`_dataset_name`).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.tasks.base.configuration import (
    BaseDataConfig,
    as_config_mapping,
    require_config_mapping,
)
from src.tasks.base.data.rng import (
    derive_seed,
    require_run_seed,
    require_worker_seeded_dataset,
    seed_scene_dataset_worker,
)
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT


class SceneDirectoryDataModule(pl.LightningDataModule):
    """Lightning DataModule backed by a scene directory with split files.

    Loader settings and the scene path must already exist in the composed
    configuration.  The scene path is resolved against ``data_root``.
    """

    def __init__(self, config: object) -> None:
        super().__init__()
        root = as_config_mapping(config, path="configuration")
        self.config = config
        resolver = PathResolver(
            RuntimePathRoots.from_mapping(
                require_config_mapping(root, "paths", path="configuration"),
                repository_root=PROJECT_ROOT,
            )
        )
        self.data_config = BaseDataConfig.from_validated_task_mapping(
            require_config_mapping(root, "data", path="configuration"),
            resolver=resolver,
        )
        self.seed = require_run_seed(config)

        self.batch_size = self.data_config.batch_size
        self.num_workers = self.data_config.num_workers
        self.pin_memory = self.data_config.pin_memory
        self.scene_dir = self.data_config.scene_dir

        self.collate_fn: Callable[..., Any] | None = self._build_collate_fn()

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self._loader_generators = {
            stage: torch.Generator().manual_seed(
                derive_seed(self.seed, "loader", stage)
            )
            for stage in ("train", "val", "test")
        }

    # -- abstract hooks --------------------------------------------------------

    @abstractmethod
    def _build_collate_fn(self) -> Callable[..., Any] | None:
        """Return the collate function (or None) for the DataLoaders."""

    @abstractmethod
    def _build_dataset(
        self,
        scene_dir: Path,
        split_file: str,
        augment: bool,
        seed: int | None = None,
    ) -> Dataset:
        """Build a task-specific dataset for the given split."""

    @abstractmethod
    def _dataset_name(self) -> str:
        """Return the task name used in error messages (e.g. ``'plcs'``)."""

    def _dataset_seed(self, scene_dir: Path, split_file: str) -> int:
        """Derive one stable dataset stream from run seed and persisted split."""
        return derive_seed(
            self.seed,
            "dataset",
            split_file,
            scene_dir.resolve().as_posix(),
        )

    # -- shared lifecycle ------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        if not self.scene_dir.exists():
            raise RuntimeError(
                f"Scene directory not found: {self.scene_dir}. "
                f"Run {self._dataset_name()}.scripts.generate_dataset "
                "to create the dataset."
            )

        if stage == "fit" or stage is None:
            train_split = self.scene_dir / "train.txt"
            if not train_split.exists():
                raise RuntimeError(f"Missing required split file: {train_split}")
            self.train_dataset = self._build_dataset(
                scene_dir=self.scene_dir,
                split_file="train.txt",
                augment=True,
                seed=self._dataset_seed(self.scene_dir, "train.txt"),
            )

            val_split = self.scene_dir / "val.txt"
            if not val_split.exists():
                raise RuntimeError(f"Missing required split file: {val_split}")
            self.val_dataset = self._build_dataset(
                scene_dir=self.scene_dir,
                split_file="val.txt",
                augment=False,
                seed=self._dataset_seed(self.scene_dir, "val.txt"),
            )

        if stage == "test" or stage is None:
            test_split = self.scene_dir / "test.txt"
            if not test_split.exists():
                raise RuntimeError(f"Missing required split file: {test_split}")
            self.test_dataset = self._build_dataset(
                scene_dir=self.scene_dir,
                split_file="test.txt",
                augment=False,
                seed=self._dataset_seed(self.scene_dir, "test.txt"),
            )

    def _build_loader(
        self,
        dataset: Dataset,
        *,
        stage: Literal["train", "val", "test"],
    ) -> DataLoader:
        require_worker_seeded_dataset(dataset)
        train = stage == "train"
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=train,
            collate_fn=self.collate_fn,
            generator=self._loader_generators[stage],
            worker_init_fn=seed_scene_dataset_worker,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before train_dataloader()")
        return self._build_loader(self.train_dataset, stage="train")

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before val_dataloader()")
        return self._build_loader(self.val_dataset, stage="val")

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before test_dataloader()")
        return self._build_loader(self.test_dataset, stage="test")
