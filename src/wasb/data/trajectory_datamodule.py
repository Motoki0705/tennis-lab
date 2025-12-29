from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.wasb.data.trajectory_dataset import TrajectoryWindowDataset


class TrajectoryDataModule(pl.LightningDataModule):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        cfg = config or {}
        data_cfg = cfg.get("data", {})

        self.root_dir = Path(data_cfg.get("root_dir", "data/tennis"))
        self.train_matches: Sequence[str] = data_cfg.get("train_matches", [])
        self.val_matches: Sequence[str] = data_cfg.get("val_matches", [])
        self.test_matches: Sequence[str] = data_cfg.get("test_matches", [])

        self.sequence_length = data_cfg.get("sequence_length", 64)
        self.step = data_cfg.get("step", 8)
        self.image_ext = data_cfg.get("image_ext", ".jpg")
        self.csv_filename = data_cfg.get("csv_filename", "Label.csv")
        self.min_visible_per_window = data_cfg.get("min_visible_per_window", 1)

        self.block_mask_min_len = data_cfg.get("block_mask_min_len", 4)
        self.block_mask_max_len = data_cfg.get("block_mask_max_len", 7)
        self.sparse_mask_prob = data_cfg.get("sparse_mask_prob", 0.05)
        self.noise_prob = data_cfg.get("noise_prob", 0.3)
        self.noise_std_px = data_cfg.get("noise_std_px", 3.0)

        self.batch_size = data_cfg.get("batch_size", 32)
        self.num_workers = data_cfg.get("num_workers", 4)
        self.pin_memory = data_cfg.get("pin_memory", True)

        self.train_dataset: TrajectoryWindowDataset | None = None
        self.val_dataset: TrajectoryWindowDataset | None = None
        self.test_dataset: TrajectoryWindowDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = TrajectoryWindowDataset(
                root_dir=self.root_dir,
                matches=self.train_matches,
                sequence_length=self.sequence_length,
                step=self.step,
                image_ext=self.image_ext,
                csv_filename=self.csv_filename,
                min_visible_per_window=self.min_visible_per_window,
                block_mask_min_len=self.block_mask_min_len,
                block_mask_max_len=self.block_mask_max_len,
                sparse_mask_prob=self.sparse_mask_prob,
                noise_prob=self.noise_prob,
                noise_std_px=self.noise_std_px,
            )
            val_matches = self.val_matches or self.train_matches
            self.val_dataset = TrajectoryWindowDataset(
                root_dir=self.root_dir,
                matches=val_matches,
                sequence_length=self.sequence_length,
                step=self.step,
                image_ext=self.image_ext,
                csv_filename=self.csv_filename,
                min_visible_per_window=self.min_visible_per_window,
                block_mask_min_len=self.block_mask_min_len,
                block_mask_max_len=self.block_mask_max_len,
                sparse_mask_prob=self.sparse_mask_prob,
                noise_prob=self.noise_prob,
                noise_std_px=self.noise_std_px,
            )

        if stage in (None, "test"):
            test_matches = self.test_matches or self.val_matches or self.train_matches
            self.test_dataset = TrajectoryWindowDataset(
                root_dir=self.root_dir,
                matches=test_matches,
                sequence_length=self.sequence_length,
                step=self.step,
                image_ext=self.image_ext,
                csv_filename=self.csv_filename,
                min_visible_per_window=self.min_visible_per_window,
                block_mask_min_len=self.block_mask_min_len,
                block_mask_max_len=self.block_mask_max_len,
                sparse_mask_prob=self.sparse_mask_prob,
                noise_prob=self.noise_prob,
                noise_std_px=self.noise_std_px,
            )

    def _loader(self, dataset: TrajectoryWindowDataset | None, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Dataset is not initialized; call setup() first.")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
