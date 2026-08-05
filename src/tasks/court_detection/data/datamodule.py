"""Lightning DataModule for court detection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.court_kp_dataset import CourtKPDataset
from src.tasks.court_detection.data.court_line_dataset import CourtLineDataset
from src.tasks.court_detection.data.court_seg_dataset import CourtSegDataset

# ── Padding helpers (shared by collate functions) ─────────────────


def _align8(n: int) -> int:
    """Round ``n`` up to the nearest multiple of 8."""
    return ((n + 7) // 8) * 8


def _pad_max_hw(batch: list[dict], key: str = "image") -> tuple[int, int]:
    """Compute the 8-aligned max height/width across a batch."""
    max_h = max(b[key].shape[1] for b in batch)
    max_w = max(b[key].shape[2] for b in batch)
    return _align8(max_h), _align8(max_w)


def _pad_image(img: torch.Tensor, max_h: int, max_w: int) -> torch.Tensor:
    """Zero-pad a ``[C, H, W]`` tensor to ``[C, max_h, max_w]``."""
    c, h, w = img.shape
    padded = torch.zeros(c, max_h, max_w, dtype=img.dtype)
    padded[:, :h, :w] = img
    return padded


# ── Custom collate functions (variable-size images) ───────────────


def _pad_collate_seg(batch: list[dict]) -> dict:
    """Pad variable-size images/masks to the max size in the batch."""
    max_h, max_w = _pad_max_hw(batch)

    images, masks, sizes, ids = [], [], [], []
    for b in batch:
        images.append(_pad_image(b["image"], max_h, max_w))

        _, h, w = b["image"].shape
        padded_mask = torch.zeros(max_h, max_w, dtype=b["mask"].dtype)
        padded_mask[:h, :w] = b["mask"]
        masks.append(padded_mask)
        sizes.append(b["image_size"])
        ids.append(b["image_id"])

    return {
        "image": torch.stack(images),
        "mask": torch.stack(masks),
        "image_size": torch.stack(sizes),
        "image_id": ids,
    }


def _pad_collate_line(batch: list[dict]) -> dict:
    """Pad variable-size images/binary masks to the max size in the batch."""
    max_h, max_w = _pad_max_hw(batch)

    images, masks, sizes, ids = [], [], [], []
    for b in batch:
        images.append(_pad_image(b["image"], max_h, max_w))

        _, mh, mw = b["mask"].shape
        padded_mask = torch.zeros(1, max_h, max_w, dtype=b["mask"].dtype)
        padded_mask[:, :mh, :mw] = b["mask"]
        masks.append(padded_mask)
        _, h, w = b["image"].shape
        sizes.append(b["image_size"])
        ids.append(b["image_id"])

    return {
        "image": torch.stack(images),
        "mask": torch.stack(masks),
        "image_size": torch.stack(sizes),
        "image_id": ids,
    }


def _pad_collate_kp(batch: list[dict]) -> dict:
    """Pad variable-size images/heatmaps to the max size in the batch."""
    max_h, max_w = _pad_max_hw(batch)

    images, heatmaps, keypoints, visibles, sizes, ids = [], [], [], [], [], []
    for b in batch:
        images.append(_pad_image(b["image"], max_h, max_w))

        n, hh, hw = b["heatmap"].shape
        padded_hm = torch.zeros(n, max_h, max_w, dtype=b["heatmap"].dtype)
        padded_hm[:, :hh, :hw] = b["heatmap"]
        heatmaps.append(padded_hm)
        keypoints.append(b["keypoints"])
        visibles.append(b["kp_visible"])
        _, h, w = b["image"].shape
        sizes.append(b["image_size"])
        ids.append(b["image_id"])

    return {
        "image": torch.stack(images),
        "heatmap": torch.stack(heatmaps),
        "keypoints": torch.stack(keypoints),
        "kp_visible": torch.stack(visibles),
        "image_size": torch.stack(sizes),
        "image_id": ids,
    }


# ── DataModule ────────────────────────────────────────────────────


class CourtDetectionDataModule(pl.LightningDataModule):
    """Lightning DataModule for court detection tasks.

    Supports three tasks via ``config.data.task``:
    ``seg``, ``kp``, ``line``.
    """

    def __init__(self, config: object) -> None:
        super().__init__()
        runtime = CourtTrainingConfig.from_config(config)
        self.data_config = runtime.data
        self.task = runtime.data.task
        self.data_dir = runtime.data.data_dir
        self.batch_size = runtime.data.batch_size
        self.num_workers = runtime.data.num_workers
        self.pin_memory = runtime.data.pin_memory

        self.train_dataset: Dataset[Any] | None = None
        self.val_dataset: Dataset[Any] | None = None
        self.test_dataset: Dataset[Any] | None = None

    def create_dataset(self, *, split: str, is_train: bool) -> Dataset[Any]:
        """Build one task-specific dataset for ``split``.

        ``is_train`` selects the full training augmentation pipeline; with
        ``False`` only the deterministic validation resize is applied.
        """
        if self.task == "seg":
            return CourtSegDataset(
                self.data_dir,
                split=split,
                is_train=is_train,
                config=self.data_config,
            )
        if self.task == "kp":
            return CourtKPDataset(
                self.data_dir,
                split=split,
                is_train=is_train,
                config=self.data_config,
            )
        if self.task == "line":
            mask_dir = self.data_config.mask_dir_name
            if mask_dir is None:
                raise AssertionError("line data requires mask_dir_name")
            return CourtLineDataset(
                self.data_dir,
                split=split,
                is_train=is_train,
                config=self.data_config,
                mask_dir_name=mask_dir,
            )
        raise ValueError(f"Unknown task: {self.task!r}")

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage."""
        if stage not in ("fit", "validate", "test", None):
            return

        if stage in ("fit", None):
            self.train_dataset = self.create_dataset(split="train", is_train=True)
        if stage in ("fit", "validate", None):
            self.val_dataset = self.create_dataset(split="val", is_train=False)
        if stage in ("test", None):
            self.test_dataset = self.create_dataset(split="val", is_train=False)

    def _collate_fn(self) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        if self.task == "seg":
            return _pad_collate_seg
        if self.task == "kp":
            return _pad_collate_kp
        return _pad_collate_line

    @staticmethod
    def _require_dataset(dataset: Dataset[Any] | None, *, stage: str) -> Dataset[Any]:
        if dataset is None:
            raise RuntimeError(
                f"CourtDetectionDataModule.setup({stage!r}) was not called."
            )
        return dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._require_dataset(self.train_dataset, stage="fit"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn(),
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._require_dataset(self.val_dataset, stage="validate"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._require_dataset(self.test_dataset, stage="test"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn(),
        )
