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
from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from omegaconf import DictConfig


class SceneDirectoryDataModule(pl.LightningDataModule):
    """Lightning DataModule backed by a scene directory with split files.

    Class-level overridable defaults:
        default_scene_dir: Fallback ``data.scene_dir`` value.
        default_batch_size: Fallback ``data.batch_size`` value.
    """

    default_scene_dir: str = "data"
    default_batch_size: int = 32

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__()
        self.config = config or {}

        data_cfg = self.config.get("data", {})
        self.batch_size = int(data_cfg.get("batch_size", self.default_batch_size))
        self.num_workers = int(data_cfg.get("num_workers", 4))
        self.pin_memory = bool(data_cfg.get("pin_memory", True))
        self.scene_dir = Path(data_cfg.get("scene_dir", self.default_scene_dir))

        self.collate_fn: Callable[..., Any] | None = self._build_collate_fn()

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

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
    ) -> Dataset:
        """Build a task-specific dataset for the given split."""

    @abstractmethod
    def _dataset_name(self) -> str:
        """Return the task name used in error messages (e.g. ``'plcs'``)."""

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
            )

            val_split = self.scene_dir / "val.txt"
            if val_split.exists():
                self.val_dataset = self._build_dataset(
                    scene_dir=self.scene_dir,
                    split_file="val.txt",
                    augment=False,
                )
            else:
                self.val_dataset = self.train_dataset

        if stage == "test" or stage is None:
            test_split = self.scene_dir / "test.txt"
            if not test_split.exists():
                raise RuntimeError(f"Missing required split file: {test_split}")
            self.test_dataset = self._build_dataset(
                scene_dir=self.scene_dir,
                split_file="test.txt",
                augment=False,
            )

    def _build_loader(self, dataset: Dataset, *, train: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=train,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before train_dataloader()")
        return self._build_loader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before val_dataloader()")
        return self._build_loader(self.val_dataset, train=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before test_dataloader()")
        return self._build_loader(self.test_dataset, train=False)
