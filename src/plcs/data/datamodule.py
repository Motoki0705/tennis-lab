"""PyTorch Lightning DataModule for PLCS."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.plcs.data.dataset import SceneDataset
from src.plcs.data.scene_batch_sampler import SceneBatchSampler
from src.plcs.data.multiview_dataset import (
    MultiViewSceneDataset,
    MultiViewSequenceDataset,
    collate_multiview,
    collate_multiview_sequence,
)
from src.plcs.data.sequence_dataset import SceneSequenceDataset

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSDataModule(pl.LightningDataModule):
    """Lightning DataModule for PLCS training.

    This module handles the creation of training, validation, and test
    dataloaders from pre-generated scene files.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        """Initialize the DataModule.

        Args:
            config: Configuration dictionary with data parameters.

        """
        super().__init__()
        self.config = config or {}

        data_cfg = self.config.get("data", {})
        self.batch_size = data_cfg.get("batch_size", 64)
        self.num_workers = data_cfg.get("num_workers", 4)
        self.scene_dir = Path(data_cfg.get("scene_dir", "data/plcs_scenes"))
        self.val_split = data_cfg.get("val_split", 0.1)
        self.test_split = data_cfg.get("test_split", 0.1)
        self.camera_mode = data_cfg.get("camera_mode", "random")
        self.scene_batch_sampler = bool(data_cfg.get("scene_batch_sampler", False))

        self.train_dataset: SceneDataset | None = None
        self.val_dataset: SceneDataset | None = None
        self.test_dataset: SceneDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for the given stage.

        Args:
            stage: Either 'fit', 'validate', 'test', or None for all.

        """
        # Load full dataset
        full_dataset = SceneDataset(
            scene_dir=self.scene_dir,
            config=self.config,
            augment=True,
            camera_mode=self.camera_mode,
        )

        # Split into train/val/test
        total_len = len(full_dataset)
        val_len = int(total_len * self.val_split)
        test_len = int(total_len * self.test_split)
        train_len = total_len - val_len - test_len

        train_ds, val_ds, test_ds = random_split(
            full_dataset, [train_len, val_len, test_len]
        )

        if stage == "fit" or stage is None:
            self.train_dataset = train_ds
            self.val_dataset = val_ds

        if stage == "test" or stage is None:
            self.test_dataset = test_ds

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader.

        Returns:
            DataLoader: Training dataloader.

        """
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before train_dataloader()")

        if self.scene_batch_sampler:
            batch_sampler = SceneBatchSampler(
                self.train_dataset,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader.

        Returns:
            DataLoader: Validation dataloader.

        """
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before val_dataloader()")

        if self.scene_batch_sampler:
            batch_sampler = SceneBatchSampler(
                self.val_dataset,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=False,
            )
            return DataLoader(
                self.val_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader.

        Returns:
            DataLoader: Test dataloader.

        """
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before test_dataloader()")

        if self.scene_batch_sampler:
            batch_sampler = SceneBatchSampler(
                self.test_dataset,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=False,
            )
            return DataLoader(
                self.test_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
            )

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class PLCSSequenceDataModule(pl.LightningDataModule):
    """Lightning DataModule for sequential PLCS training.

    This module creates train/val/test dataloaders using SceneSequenceDataset
    to provide fixed-length temporal clips.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        """Initialize the sequence DataModule.

        Args:
            config: Configuration dictionary with data parameters.

        """
        super().__init__()
        self.config = config or {}

        data_cfg = self.config.get("data", {})
        self.batch_size = data_cfg.get("batch_size", 64)
        self.num_workers = data_cfg.get("num_workers", 4)
        self.scene_dir = Path(data_cfg.get("scene_dir", "data/plcs_scenes"))
        self.val_split = data_cfg.get("val_split", 0.1)
        self.test_split = data_cfg.get("test_split", 0.1)
        self.camera_mode = data_cfg.get("camera_mode", "random")

        self.train_dataset: SceneSequenceDataset | None = None
        self.val_dataset: SceneSequenceDataset | None = None
        self.test_dataset: SceneSequenceDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for the given stage.

        Args:
            stage: Either 'fit', 'validate', 'test', or None for all.

        """
        # Load full sequence dataset
        full_dataset = SceneSequenceDataset(
            scene_dir=self.scene_dir,
            config=self.config,
            augment=True,
            camera_mode=self.camera_mode,
        )

        # Split into train/val/test
        total_len = len(full_dataset)
        val_len = int(total_len * self.val_split)
        test_len = int(total_len * self.test_split)
        train_len = total_len - val_len - test_len

        train_ds, val_ds, test_ds = random_split(
            full_dataset, [train_len, val_len, test_len]
        )

        if stage == "fit" or stage is None:
            self.train_dataset = train_ds
            self.val_dataset = val_ds

        if stage == "test" or stage is None:
            self.test_dataset = test_ds

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader.

        Returns:
            DataLoader: Training dataloader.

        """
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before train_dataloader()")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader.

        Returns:
            DataLoader: Validation dataloader.

        """
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before val_dataloader()")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader.

        Returns:
            DataLoader: Test dataloader.

        """
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before test_dataloader()")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class PLCSMultiViewDataModule(pl.LightningDataModule):
    """Lightning DataModule for multi-view PLCS training.

    This module creates train/val/test dataloaders using MultiViewSceneDataset
    to provide observations from multiple cameras simultaneously.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        """Initialize the multi-view DataModule.

        Args:
            config: Configuration dictionary with data parameters.

        """
        super().__init__()
        self.config = config or {}

        data_cfg = self.config.get("data", {})
        self.batch_size = data_cfg.get("batch_size", 32)
        self.num_workers = data_cfg.get("num_workers", 4)
        self.pin_memory = True
        self.scene_dir = Path(data_cfg.get("scene_dir", "data/plcs"))
        self.val_split = data_cfg.get("val_split", 0.1)
        self.test_split = data_cfg.get("test_split", 0.1)
        self.num_views = data_cfg.get("num_views", 2)
        self.min_cameras = data_cfg.get("min_cameras", 2)

        self.train_dataset: MultiViewSceneDataset | None = None
        self.val_dataset: MultiViewSceneDataset | None = None
        self.test_dataset: MultiViewSceneDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for the given stage.

        Args:
            stage: Either 'fit', 'validate', 'test', or None for all.

        """
        # Load full multi-view dataset
        full_dataset = MultiViewSceneDataset(
            scene_dir=self.scene_dir,
            config=self.config,
            augment=True,
            num_views=self.num_views,
            min_cameras=self.min_cameras,
        )

        # Split into train/val/test
        total_len = len(full_dataset)
        val_len = int(total_len * self.val_split)
        test_len = int(total_len * self.test_split)
        train_len = total_len - val_len - test_len

        train_ds, val_ds, test_ds = random_split(
            full_dataset, [train_len, val_len, test_len]
        )

        if stage == "fit" or stage is None:
            self.train_dataset = train_ds
            self.val_dataset = val_ds

        if stage == "test" or stage is None:
            self.test_dataset = test_ds

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader.

        Returns:
            DataLoader: Training dataloader.

        """
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before train_dataloader()")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=collate_multiview,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader.

        Returns:
            DataLoader: Validation dataloader.

        """
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before val_dataloader()")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_multiview,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader.

        Returns:
            DataLoader: Test dataloader.

        """
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before test_dataloader()")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_multiview,
        )


class PLCSMultiViewSequenceDataModule(pl.LightningDataModule):
    """Lightning DataModule for multi-view sequential PLCS training.

    This module creates train/val/test dataloaders using MultiViewSequenceDataset
    to provide observations from multiple cameras over temporal sequences.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        """Initialize the multi-view sequence DataModule.

        Args:
            config: Configuration dictionary with data parameters.

        """
        super().__init__()
        self.config = config or {}

        data_cfg = self.config.get("data", {})
        self.batch_size = data_cfg.get("batch_size", 32)
        self.num_workers = data_cfg.get("num_workers", 4)
        self.pin_memory = True
        self.scene_dir = Path(data_cfg.get("scene_dir", "data/plcs"))
        self.val_split = data_cfg.get("val_split", 0.1)
        self.test_split = data_cfg.get("test_split", 0.1)
        self.num_views = data_cfg.get("num_views", 2)
        self.min_cameras = data_cfg.get("min_cameras", 2)

        self.train_dataset: MultiViewSequenceDataset | None = None
        self.val_dataset: MultiViewSequenceDataset | None = None
        self.test_dataset: MultiViewSequenceDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for the given stage.

        Args:
            stage: Either 'fit', 'validate', 'test', or None for all.

        """
        # Load full multi-view sequence dataset
        full_dataset = MultiViewSequenceDataset(
            scene_dir=self.scene_dir,
            config=self.config,
            augment=True,
            num_views=self.num_views,
            min_cameras=self.min_cameras,
        )

        # Split into train/val/test
        total_len = len(full_dataset)
        val_len = int(total_len * self.val_split)
        test_len = int(total_len * self.test_split)
        train_len = total_len - val_len - test_len

        train_ds, val_ds, test_ds = random_split(
            full_dataset, [train_len, val_len, test_len]
        )

        if stage == "fit" or stage is None:
            self.train_dataset = train_ds
            self.val_dataset = val_ds

        if stage == "test" or stage is None:
            self.test_dataset = test_ds

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader.

        Returns:
            DataLoader: Training dataloader.

        """
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before train_dataloader()")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=collate_multiview_sequence,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader.

        Returns:
            DataLoader: Validation dataloader.

        """
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before val_dataloader()")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_multiview_sequence,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader.

        Returns:
            DataLoader: Test dataloader.

        """
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before test_dataloader()")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_multiview_sequence,
        )


if __name__ == "__main__":
    # quick smoke test for SceneBatchSampler functionality
    from omegaconf import OmegaConf
    
    # test config with scene_batch_sampler enabled
    test_config = OmegaConf.create({
        "data": {
            "batch_size": 4,
            "num_workers": 0,
            "scene_dir": "data/plcs_scenes",
            "val_split": 0.2,
            "test_split": 0.2,
            "camera_mode": "random",
            "scene_batch_sampler": True
        }
    })
    
    # test PLCSDataModule initialization
    datamodule = PLCSDataModule(test_config)
    assert datamodule.scene_batch_sampler == True
    assert datamodule.batch_size == 4
    
    print("✓ PLCSDataModule with SceneBatchSampler initialized successfully")
    print("✓ SceneBatchSampler will be applied to train, val, and test dataloaders")
