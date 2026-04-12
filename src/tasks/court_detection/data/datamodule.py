"""Lightning DataModule for court detection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.tasks.court_detection.data.court_kp_dataset import CourtKPDataset
from src.tasks.court_detection.data.court_line_dataset import CourtLineDataset
from src.tasks.court_detection.data.court_seg_dataset import CourtSegDataset

if TYPE_CHECKING:
    from omegaconf import DictConfig


# ── Custom collate functions (variable-size images) ───────────────


def _pad_collate_seg(batch: list[dict]) -> dict:
    """Pad variable-size images/masks to the max size in the batch."""
    max_h = max(b["image"].shape[1] for b in batch)
    max_w = max(b["image"].shape[2] for b in batch)
    max_h = ((max_h + 7) // 8) * 8
    max_w = ((max_w + 7) // 8) * 8

    images, masks, ids = [], [], []
    for b in batch:
        c, h, w = b["image"].shape
        padded_img = torch.zeros(c, max_h, max_w, dtype=b["image"].dtype)
        padded_img[:, :h, :w] = b["image"]
        images.append(padded_img)

        padded_mask = torch.zeros(max_h, max_w, dtype=b["mask"].dtype)
        padded_mask[:h, :w] = b["mask"]
        masks.append(padded_mask)
        ids.append(b["image_id"])

    return {"image": torch.stack(images), "mask": torch.stack(masks), "image_id": ids}


def _pad_collate_line(batch: list[dict]) -> dict:
    """Pad variable-size images/binary masks to the max size in the batch."""
    max_h = max(b["image"].shape[1] for b in batch)
    max_w = max(b["image"].shape[2] for b in batch)
    max_h = ((max_h + 7) // 8) * 8
    max_w = ((max_w + 7) // 8) * 8

    images, masks, ids = [], [], []
    for b in batch:
        c, h, w = b["image"].shape
        padded_img = torch.zeros(c, max_h, max_w, dtype=b["image"].dtype)
        padded_img[:, :h, :w] = b["image"]
        images.append(padded_img)

        _, mh, mw = b["mask"].shape
        padded_mask = torch.zeros(1, max_h, max_w, dtype=b["mask"].dtype)
        padded_mask[:, :mh, :mw] = b["mask"]
        masks.append(padded_mask)
        ids.append(b["image_id"])

    return {"image": torch.stack(images), "mask": torch.stack(masks), "image_id": ids}


def _pad_collate_kp(batch: list[dict]) -> dict:
    """Pad variable-size images/heatmaps to the max size in the batch."""
    max_h = max(b["image"].shape[1] for b in batch)
    max_w = max(b["image"].shape[2] for b in batch)
    max_h = ((max_h + 7) // 8) * 8
    max_w = ((max_w + 7) // 8) * 8

    images, heatmaps, keypoints, ids = [], [], [], []
    for b in batch:
        c, h, w = b["image"].shape
        padded_img = torch.zeros(c, max_h, max_w, dtype=b["image"].dtype)
        padded_img[:, :h, :w] = b["image"]
        images.append(padded_img)

        n, hh, hw = b["heatmap"].shape
        padded_hm = torch.zeros(n, max_h, max_w, dtype=b["heatmap"].dtype)
        padded_hm[:, :hh, :hw] = b["heatmap"]
        heatmaps.append(padded_hm)
        keypoints.append(b["keypoints"])
        ids.append(b["image_id"])

    return {
        "image": torch.stack(images),
        "heatmap": torch.stack(heatmaps),
        "keypoints": torch.stack(keypoints),
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

        self.train_dataset = None
        self.val_dataset = None

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
            "gaussian_sigma": float(self.config.get("data", {}).get("gaussian_sigma", 3.0)),
        }

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage."""
        if stage not in ("fit", None):
            return

        cfg_dict = self._build_config_dict()
        data_cfg = self.config.get("data", {})

        if self.task == "seg":
            self.train_dataset = CourtSegDataset(
                self.data_dir, split="train", is_train=True, config=cfg_dict,
            )
            self.val_dataset = CourtSegDataset(
                self.data_dir, split="val", is_train=False, config=cfg_dict,
            )
        elif self.task == "kp":
            self.train_dataset = CourtKPDataset(
                self.data_dir, split="train", is_train=True, config=cfg_dict,
            )
            self.val_dataset = CourtKPDataset(
                self.data_dir, split="val", is_train=False, config=cfg_dict,
            )
        elif self.task == "line":
            mask_dir = str(data_cfg.get("mask_dir_name", "line_masks"))
            self.train_dataset = CourtLineDataset(
                self.data_dir, split="train", is_train=True,
                config=cfg_dict, mask_dir_name=mask_dir,
            )
            self.val_dataset = CourtLineDataset(
                self.data_dir, split="val", is_train=False,
                config=cfg_dict, mask_dir_name=mask_dir,
            )
        else:
            raise ValueError(f"Unknown task: {self.task!r}")

    def _collate_fn(self):
        if self.task == "seg":
            return _pad_collate_seg
        if self.task == "kp":
            return _pad_collate_kp
        return _pad_collate_line

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn(),
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn(),
        )
