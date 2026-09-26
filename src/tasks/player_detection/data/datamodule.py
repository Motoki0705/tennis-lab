"""Lightning DataModule over the player frame store."""

from __future__ import annotations

import json

import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler

from src.tasks.player_detection.configuration import PlayerDataConfig
from src.tasks.player_detection.data.detection_dataset import (
    PlayerDetectionDataset,
    collate_detection,
    select_detection_frames,
)
from src.tasks.player_detection.data.store import PlayerFrameStore, Split


class PlayerDetectionDataModule(pl.LightningDataModule):
    """Train on a fresh random frame subset per epoch; evaluate on strided frames.

    Consecutive video frames are nearly identical, so an "epoch" is
    ``train_samples_per_epoch`` frames drawn without replacement from the full
    train pool rather than one pass over every frame.
    """

    def __init__(self, config: PlayerDataConfig) -> None:
        super().__init__()
        self.config = config
        self.store = PlayerFrameStore(config.dataset_dir)
        self.selection_stats: dict[str, dict[str, int]] = {}
        self._datasets: dict[Split, PlayerDetectionDataset] = {}

    def _dataset(self, split: Split) -> PlayerDetectionDataset:
        dataset = self._datasets.get(split)
        if dataset is None:
            stride = {
                "train": 1,
                "val": self.config.val_frame_stride,
                "test": self.config.test_frame_stride,
            }[split]
            selection = select_detection_frames(
                self.store, split, self.config.selection, frame_stride=stride
            )
            self.selection_stats[split] = selection.stats
            print(f"[player_detection] {split} selection: {json.dumps(selection.stats)}")
            dataset = PlayerDetectionDataset(
                self.store,
                selection,
                input_size=self.config.input_size,
                augmentation=self.config.augmentation if split == "train" else None,
            )
            self._datasets[split] = dataset
        return dataset

    @property
    def test_dataset(self) -> PlayerDetectionDataset:
        return self._dataset("test")

    def setup(self, stage: str | None = None) -> None:
        if stage in {"fit", None}:
            self._dataset("train")
            self._dataset("val")
        if stage in {"test", None}:
            self._dataset("test")

    def _eval_loader(self, split: Split) -> DataLoader[object]:
        return DataLoader(
            self._dataset(split),
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_detection,
            persistent_workers=False,
        )

    def train_dataloader(self) -> DataLoader[object]:
        dataset = self._dataset("train")
        if self.config.train_samples_per_epoch > len(dataset):
            raise ValueError(
                "data.train_samples_per_epoch exceeds the train pool "
                f"({self.config.train_samples_per_epoch} > {len(dataset)})"
            )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=RandomSampler(
                dataset, replacement=False, num_samples=self.config.train_samples_per_epoch
            ),
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_detection,
            drop_last=True,
            persistent_workers=self.config.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[object]:
        return self._eval_loader("val")

    def test_dataloader(self) -> DataLoader[object]:
        return self._eval_loader("test")
