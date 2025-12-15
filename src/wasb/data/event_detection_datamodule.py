"""Lightning DataModule for trajectory event detection."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.wasb.data.event_detection_dataset import TrajectoryEventWindowDataset


class TrajectoryEventDataModule(pl.LightningDataModule):
    """DataModule for per-frame trajectory event detection."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        cfg = config or {}
        data_cfg = cfg.get("data", {})

        self.root_dir = Path(data_cfg.get("root_dir", "data/tennis"))
        self.train_matches: Sequence[str] = data_cfg.get("train_matches", [])
        self.val_matches: Sequence[str] = data_cfg.get("val_matches", [])
        self.test_matches: Sequence[str] = data_cfg.get("test_matches", [])

        self.sequence_length = int(data_cfg.get("sequence_length", 64))
        self.step = int(data_cfg.get("step", 8))
        self.image_ext = data_cfg.get("image_ext", ".jpg")
        self.csv_filename = data_cfg.get("csv_filename", "Label.csv")
        self.min_visible_per_window = int(data_cfg.get("min_visible_per_window", 1))
        self.xy_scale = tuple(data_cfg.get("xy_scale", (1920.0, 1080.0)))
        self.ignore_invisible_targets = bool(data_cfg.get("ignore_invisible_targets", True))
        self.ignore_index = int(data_cfg.get("ignore_index", -100))

        self.batch_size = int(data_cfg.get("batch_size", 32))
        self.num_workers = int(data_cfg.get("num_workers", 4))
        self.pin_memory = bool(data_cfg.get("pin_memory", True))

        self.train_dataset: TrajectoryEventWindowDataset | None = None
        self.val_dataset: TrajectoryEventWindowDataset | None = None
        self.test_dataset: TrajectoryEventWindowDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = TrajectoryEventWindowDataset(
                root_dir=self.root_dir,
                matches=self.train_matches,
                sequence_length=self.sequence_length,
                step=self.step,
                image_ext=self.image_ext,
                csv_filename=self.csv_filename,
                min_visible_per_window=self.min_visible_per_window,
                xy_scale=self.xy_scale,  # type: ignore[arg-type]
                ignore_invisible_targets=self.ignore_invisible_targets,
                ignore_index=self.ignore_index,
            )
            val_matches = self.val_matches or self.train_matches
            self.val_dataset = TrajectoryEventWindowDataset(
                root_dir=self.root_dir,
                matches=val_matches,
                sequence_length=self.sequence_length,
                step=self.step,
                image_ext=self.image_ext,
                csv_filename=self.csv_filename,
                min_visible_per_window=self.min_visible_per_window,
                xy_scale=self.xy_scale,  # type: ignore[arg-type]
                ignore_invisible_targets=self.ignore_invisible_targets,
                ignore_index=self.ignore_index,
            )

        if stage in (None, "test"):
            test_matches = self.test_matches or self.val_matches or self.train_matches
            self.test_dataset = TrajectoryEventWindowDataset(
                root_dir=self.root_dir,
                matches=test_matches,
                sequence_length=self.sequence_length,
                step=self.step,
                image_ext=self.image_ext,
                csv_filename=self.csv_filename,
                min_visible_per_window=self.min_visible_per_window,
                xy_scale=self.xy_scale,  # type: ignore[arg-type]
                ignore_invisible_targets=self.ignore_invisible_targets,
                ignore_index=self.ignore_index,
            )

    def _loader(
        self, dataset: TrajectoryEventWindowDataset | None, *, shuffle: bool
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

    def estimate_class_counts(
        self, *, max_windows: int | None = None
    ) -> torch.Tensor:
        """Estimate class counts over the train split (classes 0/1/2)."""
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized; call setup('fit') first.")

        counts = torch.zeros(3, dtype=torch.int64)
        limit = max_windows if max_windows is not None else len(self.train_dataset)
        limit = max(0, min(limit, len(self.train_dataset)))

        for i in range(limit):
            sample = self.train_dataset[i]
            target: torch.Tensor = sample["target_status"]  # type: ignore[assignment]
            valid = target != self.ignore_index
            if not valid.any():
                continue
            vals = target[valid].clamp(min=0, max=2)
            counts += torch.bincount(vals, minlength=3)
        return counts

    @staticmethod
    def class_weights_from_counts(
        counts: torch.Tensor, *, eps: float = 1.0
    ) -> torch.Tensor:
        """Convert class counts (0/1/2) to inverse-frequency weights."""
        counts_f = counts.to(dtype=torch.float32)
        weights = (counts_f.sum() + 3.0 * eps) / (counts_f + eps)
        # Normalize to keep weight[0] ~= 1.0 as a stable reference.
        weights = weights / (weights[0].clamp(min=1e-6))
        return weights

