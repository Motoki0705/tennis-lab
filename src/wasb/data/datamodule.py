"""Lightning DataModule for the WASB tennis dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.wasb.data.dataset import (
    TennisSequenceDataset,
    VisibilityMode,
)


class TennisDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapper around ``TennisSequenceDataset``."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        cfg = config or {}
        data_cfg = cfg.get("data", {})

        self.root_dir = Path(data_cfg.get("root_dir", "data/tennis"))
        self.train_matches: Sequence[str] = data_cfg.get("train_matches", [])
        self.val_matches: Sequence[str] = data_cfg.get("val_matches", [])
        self.test_matches: Sequence[str] = data_cfg.get("test_matches", [])
        self.frames_in = data_cfg.get("frames_in", 5)
        self.frames_out = data_cfg.get("frames_out", 1)
        self.step = data_cfg.get("step", 1)
        self.visibility_mode: VisibilityMode = data_cfg.get("visibility_mode", "none")
        self.image_ext = data_cfg.get("image_ext", ".jpg")
        self.csv_filename = data_cfg.get("csv_filename", "Label.csv")
        self.batch_size = data_cfg.get("batch_size", 8)
        self.num_workers = data_cfg.get("num_workers", 4)
        self.resize_hw = data_cfg.get("resize_hw")
        self.heatmap_hw = data_cfg.get("heatmap_hw")
        self.heatmap_sigma = data_cfg.get("heatmap_sigma")
        self.pin_memory = data_cfg.get("pin_memory", True)

        self.train_dataset: TennisSequenceDataset | None = None
        self.val_dataset: TennisSequenceDataset | None = None
        self.test_dataset: TennisSequenceDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = TennisSequenceDataset(
                root_dir=self.root_dir,
                matches=self.train_matches,
                frames_in=self.frames_in,
                frames_out=self.frames_out,
                step=self.step,
                visibility_mode=self.visibility_mode,
                image_ext=self.image_ext,
                csv_filename=self.csv_filename,
                resize_hw=self.resize_hw,
                heatmap_hw=self.heatmap_hw,
                heatmap_sigma=self.heatmap_sigma,
            )
            self.val_dataset = TennisSequenceDataset(
                root_dir=self.root_dir,
                matches=self.val_matches or self.train_matches,
                frames_in=self.frames_in,
                frames_out=self.frames_out,
                step=self.step,
                visibility_mode=self.visibility_mode,
                image_ext=self.image_ext,
                csv_filename=self.csv_filename,
                resize_hw=self.resize_hw,
                heatmap_hw=self.heatmap_hw,
                heatmap_sigma=self.heatmap_sigma,
            )

        if stage in (None, "test"):
            self.test_dataset = TennisSequenceDataset(
                root_dir=self.root_dir,
                matches=self.test_matches or self.val_matches or self.train_matches,
                frames_in=self.frames_in,
                frames_out=self.frames_out,
                step=self.step,
                visibility_mode=self.visibility_mode,
                image_ext=self.image_ext,
                csv_filename=self.csv_filename,
                resize_hw=self.resize_hw,
                heatmap_hw=self.heatmap_hw,
                heatmap_sigma=self.heatmap_sigma,
            )

    def _loader(
        self, dataset: TennisSequenceDataset | None, shuffle: bool
    ) -> DataLoader:
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
