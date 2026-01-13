"""PyTorch Lightning DataModule for MAE training.

Provides train/val data loaders with variable resolution support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.mae.data.dataset import TennisVideoDataset, VideoFrameDataset


class MAEDataModule(pl.LightningDataModule):
    """Lightning DataModule for MAE pre-training.

    Handles data loading with:
    - Variable resolution training
    - Train/val split
    - Efficient multi-worker loading
    """

    def __init__(
        self,
        video_dir: str | Path = "data/tennis/raw/videos",
        min_resolution: int = 160,
        max_resolution: int = 320,
        frames_per_video: int = 100,
        patch_size: int = 16,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        pin_memory: bool = True,
        use_decord: bool = True,
    ) -> None:
        """Initialize DataModule.

        Args:
            video_dir: Directory containing video files.
            min_resolution: Minimum training resolution.
            max_resolution: Maximum training resolution.
            frames_per_video: Frames to sample per video.
            patch_size: Patch size for resolution rounding.
            batch_size: Batch size for training.
            num_workers: Number of data loading workers.
            val_split: Fraction of data for validation.
            pin_memory: Pin memory for faster GPU transfer.
            use_decord: Use decord for video reading.

        """
        super().__init__()
        self.save_hyperparameters()

        self.video_dir = Path(video_dir)
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.frames_per_video = frames_per_video
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.pin_memory = pin_memory
        self.use_decord = use_decord

        self.train_dataset: Optional[VideoFrameDataset] = None
        self.val_dataset: Optional[VideoFrameDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup train and validation datasets.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'.

        """
        if stage == "fit" or stage is None:
            # Create full dataset
            full_dataset = TennisVideoDataset(
                video_dir=self.video_dir,
                min_resolution=self.min_resolution,
                max_resolution=self.max_resolution,
                frames_per_video=self.frames_per_video,
                patch_size=self.patch_size,
                use_decord=self.use_decord,
            )

            # Split into train/val
            num_samples = len(full_dataset)
            num_val = int(num_samples * self.val_split)
            num_train = num_samples - num_val

            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [num_train, num_val],
            )

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch: list[dict]) -> dict:
        """Custom collate function for variable resolution.

        Groups samples by resolution and pads to max size in batch.

        Args:
            batch: List of sample dictionaries.

        Returns:
            Batched dictionary with padded images.

        """
        import torch

        # Find max resolution in this batch
        max_res = max(sample["resolution"] for sample in batch)

        # Pad all images to max resolution
        images = []
        masks = []  # Mask for valid (non-padded) regions

        for sample in batch:
            img = sample["image"]
            C, H, W = img.shape

            if H == max_res and W == max_res:
                images.append(img)
                masks.append(torch.ones(1, max_res, max_res))
            else:
                # Pad to max_res
                padded = torch.zeros(C, max_res, max_res)
                padded[:, :H, :W] = img
                images.append(padded)

                # Create mask
                mask = torch.zeros(1, max_res, max_res)
                mask[:, :H, :W] = 1.0
                masks.append(mask)

        return {
            "image": torch.stack(images),
            "padding_mask": torch.stack(masks),
            "resolutions": torch.tensor([s["resolution"] for s in batch]),
        }

    @classmethod
    def from_config(cls, config) -> "MAEDataModule":
        """Create DataModule from Hydra config.

        Args:
            config: Hydra configuration.

        Returns:
            Initialized DataModule.

        """
        data_cfg = config.get("data", config)
        return cls(
            video_dir=data_cfg.get("video_dir", "data/tennis/raw/videos"),
            min_resolution=data_cfg.get("min_resolution", 160),
            max_resolution=data_cfg.get("max_resolution", 320),
            frames_per_video=data_cfg.get("frames_per_video", 100),
            patch_size=data_cfg.get("patch_size", 16),
            batch_size=data_cfg.get("batch_size", 32),
            num_workers=data_cfg.get("num_workers", 4),
            val_split=data_cfg.get("val_split", 0.1),
            pin_memory=data_cfg.get("pin_memory", True),
            use_decord=data_cfg.get("use_decord", True),
        )
