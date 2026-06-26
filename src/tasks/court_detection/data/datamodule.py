"""Lightning DataModule for court detection."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.tasks.court_detection.data.court_kp_dataset import CourtKPDataset
from src.tasks.court_detection.data.court_line_dataset import CourtLineDataset
from src.tasks.court_detection.data.court_seg_dataset import CourtSegDataset

if TYPE_CHECKING:
    from omegaconf import DictConfig


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
        sizes.append(b.get("image_size", torch.tensor([h, w], dtype=torch.int64)))
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
        sizes.append(b.get("image_size", torch.tensor([h, w], dtype=torch.int64)))
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

    images, heatmaps, keypoints, sizes, ids = [], [], [], [], []
    for b in batch:
        images.append(_pad_image(b["image"], max_h, max_w))

        n, hh, hw = b["heatmap"].shape
        padded_hm = torch.zeros(n, max_h, max_w, dtype=b["heatmap"].dtype)
        padded_hm[:, :hh, :hw] = b["heatmap"]
        heatmaps.append(padded_hm)
        keypoints.append(b["keypoints"])
        _, h, w = b["image"].shape
        sizes.append(b.get("image_size", torch.tensor([h, w], dtype=torch.int64)))
        ids.append(b["image_id"])

    return {
        "image": torch.stack(images),
        "heatmap": torch.stack(heatmaps),
        "keypoints": torch.stack(keypoints),
        "image_size": torch.stack(sizes),
        "image_id": ids,
    }


# ── DataModule ────────────────────────────────────────────────────


class CourtDetectionDataModule(pl.LightningDataModule):
    """Lightning DataModule for court detection tasks.

    Supports three tasks via ``config.data.task``:
    ``seg``, ``kp``, ``line``.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__()
        self.config = config or {}

        data_cfg = self.config.get("data", {})
        self.task = str(data_cfg.get("task", "seg"))
        self.data_dir = Path(str(data_cfg.get("data_dir", "data/court")))
        self.batch_size = int(data_cfg.get("batch_size", 8))
        self.num_workers = int(data_cfg.get("num_workers", 4))
        self.pin_memory = bool(data_cfg.get("pin_memory", True))

        self.train_dataset: Dataset[Any] | None = None
        self.val_dataset: Dataset[Any] | None = None
        self.test_dataset: Dataset[Any] | None = None

    def _build_config_dict(self) -> dict:
        """Build a flat config dict for dataset constructors."""
        data_cfg = self.config.get("data", {})
        aug_cfg = data_cfg.get("augmentation", {})
        return {
            "train_scales": list(aug_cfg.get("train_scales", [288])),
            "val_short_side": int(aug_cfg.get("val_short_side", 288)),
            "crop_scale": tuple(aug_cfg.get("crop_scale", (0.2, 1.0))),
            "crop_ratio": tuple(aug_cfg.get("crop_ratio", (0.5, 2.0))),
            "hflip_prob": float(aug_cfg.get("hflip_prob", 0.7)),
            "hflip_swap_pairs": [tuple(pair) for pair in data_cfg.get("hflip_swap_pairs", [])],
            "affine_degrees": float(aug_cfg.get("affine_degrees", 25.0)),
            "affine_translate": tuple(aug_cfg.get("affine_translate", (0.18, 0.18))),
            "affine_scale": tuple(aug_cfg.get("affine_scale", (0.65, 1.5))),
            "affine_shear": float(aug_cfg.get("affine_shear", 18.0)),
            "perspective_distortion": float(aug_cfg.get("perspective_distortion", 0.25)),
            "perspective_prob": float(aug_cfg.get("perspective_prob", 0.6)),
            "color_jitter": tuple(aug_cfg.get("color_jitter", (0.5, 0.5, 0.5, 0.2))),
            "gaussian_blur_kernel": list(aug_cfg.get("gaussian_blur_kernel", [3, 5, 7, 9])),
            "gaussian_blur_sigma": tuple(aug_cfg.get("gaussian_blur_sigma", (0.1, 3.0))),
            "gaussian_blur_prob": float(aug_cfg.get("gaussian_blur_prob", 0.5)),
            "sigma_ratio": float(self.config.get("data", {}).get("sigma_ratio", 0.01)),
        }

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage."""
        if stage not in ("fit", "validate", "test", None):
            return

        cfg_dict = self._build_config_dict()
        data_cfg = self.config.get("data", {})
        build_train = stage in ("fit", None)
        build_val = stage in ("fit", "validate", None)
        build_test = stage in ("test", None)

        if self.task == "seg":
            if build_train:
                self.train_dataset = CourtSegDataset(
                    self.data_dir, split="train", is_train=True, config=cfg_dict,
                )
            if build_val:
                self.val_dataset = CourtSegDataset(
                    self.data_dir, split="val", is_train=False, config=cfg_dict,
                )
            if build_test:
                self.test_dataset = CourtSegDataset(
                    self.data_dir, split="val", is_train=False, config=cfg_dict,
                )
        elif self.task == "kp":
            if build_train:
                self.train_dataset = CourtKPDataset(
                    self.data_dir, split="train", is_train=True, config=cfg_dict,
                )
            if build_val:
                self.val_dataset = CourtKPDataset(
                    self.data_dir, split="val", is_train=False, config=cfg_dict,
                )
            if build_test:
                self.test_dataset = CourtKPDataset(
                    self.data_dir, split="val", is_train=False, config=cfg_dict,
                )
        elif self.task == "line":
            mask_dir = str(data_cfg.get("mask_dir_name", "line_masks"))
            if build_train:
                self.train_dataset = CourtLineDataset(
                    self.data_dir, split="train", is_train=True,
                    config=cfg_dict, mask_dir_name=mask_dir,
                )
            if build_val:
                self.val_dataset = CourtLineDataset(
                    self.data_dir, split="val", is_train=False,
                    config=cfg_dict, mask_dir_name=mask_dir,
                )
            if build_test:
                self.test_dataset = CourtLineDataset(
                    self.data_dir, split="val", is_train=False,
                    config=cfg_dict, mask_dir_name=mask_dir,
                )
        else:
            raise ValueError(f"Unknown task: {self.task!r}")

    def _collate_fn(self) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        if self.task == "seg":
            return _pad_collate_seg
        if self.task == "kp":
            return _pad_collate_kp
        return _pad_collate_line

    @staticmethod
    def _require_dataset(dataset: Dataset[Any] | None, *, stage: str) -> Dataset[Any]:
        if dataset is None:
            raise RuntimeError(f"CourtDetectionDataModule.setup({stage!r}) was not called.")
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
