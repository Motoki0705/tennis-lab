"""Lightning DataModule for WASB ball detection training."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Sequence

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from src.wasb.data.ball_detection_dataset import (
    BallDetectionSequenceDataset,
    VisibilityMode,
)


class BallDetectionDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapper around ``BallDetectionSequenceDataset``."""

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

        self.augment_cfg = data_cfg.get("augment", {})

        self.train_dataset: BallDetectionSequenceDataset | None = None
        self.val_dataset: BallDetectionSequenceDataset | None = None
        self.test_dataset: BallDetectionSequenceDataset | None = None

    def _build_transform(self, train: bool) -> Callable:
        data_aug = self.augment_cfg or {}
        enabled = bool(data_aug.get("enabled", False)) and train

        ops: list[Callable] = []
        if self.resize_hw is not None:
            ops.append(transforms.Resize(self.resize_hw))

        if enabled:
            cj = data_aug.get("color_jitter", {})
            cj_prob = float(cj.get("prob", 0.0))
            if cj_prob > 0:
                ops.append(
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=float(cj.get("brightness", 0.0)),
                                contrast=float(cj.get("contrast", 0.0)),
                                saturation=float(cj.get("saturation", 0.0)),
                                hue=float(cj.get("hue", 0.0)),
                            )
                        ],
                        p=cj_prob,
                    )
                )

            gs = data_aug.get("random_grayscale", {})
            gs_prob = float(gs.get("prob", 0.0))
            if gs_prob > 0:
                ops.append(transforms.RandomGrayscale(p=gs_prob))

            gb = data_aug.get("gaussian_blur", {})
            gb_prob = float(gb.get("prob", 0.0))
            if gb_prob > 0:
                kernel_size = int(gb.get("kernel_size", 3))
                sigma_min = float(gb.get("sigma_min", 0.1))
                sigma_max = float(gb.get("sigma_max", 2.0))
                ops.append(
                    transforms.RandomApply(
                        [
                            transforms.GaussianBlur(
                                kernel_size=kernel_size,
                                sigma=(sigma_min, sigma_max),
                            )
                        ],
                        p=gb_prob,
                    )
                )

        ops.append(transforms.ToTensor())

        if enabled:
            re_cfg = data_aug.get("random_erasing", {})
            re_prob = float(re_cfg.get("prob", 0.0))
            if re_prob > 0:
                scale = re_cfg.get("scale", [0.02, 0.2])
                ratio = re_cfg.get("ratio", [0.3, 3.3])
                value = re_cfg.get("value", 0)
                ops.append(
                    transforms.RandomErasing(
                        p=re_prob,
                        scale=(float(scale[0]), float(scale[1])),
                        ratio=(float(ratio[0]), float(ratio[1])),
                        value=value,
                    )
                )

        return transforms.Compose(ops)

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
        self.train_dataset = BallDetectionSequenceDataset(
                root_dir=self.root_dir,
                matches=self.train_matches,
                frames_in=self.frames_in,
                frames_out=self.frames_out,
                step=self.step,
                visibility_mode=self.visibility_mode,
                image_ext=self.image_ext,
                csv_filename=self.csv_filename,
                transform=self._build_transform(train=True),
                resize_hw=self.resize_hw,
                heatmap_hw=self.heatmap_hw,
                heatmap_sigma=self.heatmap_sigma,
            )
        self.val_dataset = BallDetectionSequenceDataset(
                root_dir=self.root_dir,
                matches=self.val_matches or self.train_matches,
                frames_in=self.frames_in,
                frames_out=self.frames_out,
                step=self.step,
                visibility_mode=self.visibility_mode,
                image_ext=self.image_ext,
                csv_filename=self.csv_filename,
                transform=self._build_transform(train=False),
                resize_hw=self.resize_hw,
                heatmap_hw=self.heatmap_hw,
                heatmap_sigma=self.heatmap_sigma,
            )

        if stage in (None, "test"):
        self.test_dataset = BallDetectionSequenceDataset(
                root_dir=self.root_dir,
                matches=self.test_matches or self.val_matches or self.train_matches,
                frames_in=self.frames_in,
                frames_out=self.frames_out,
                step=self.step,
                visibility_mode=self.visibility_mode,
                image_ext=self.image_ext,
                csv_filename=self.csv_filename,
                transform=self._build_transform(train=False),
                resize_hw=self.resize_hw,
                heatmap_hw=self.heatmap_hw,
                heatmap_sigma=self.heatmap_sigma,
            )

    def _loader(
        self, dataset: BallDetectionSequenceDataset | None, shuffle: bool
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
