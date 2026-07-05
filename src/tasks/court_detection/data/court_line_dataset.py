"""Court white-line segmentation dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.tasks.court_detection.data.augmentation import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    _pil_to_tensor_image,
    build_seg_transforms,
)
from src.utils.io import find_existing_file


class CourtLineDataset(Dataset):
    """PyTorch Dataset for court white-line segmentation."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        is_train: bool = True,
        config: dict[str, Any] | None = None,
        mask_dir_name: str = "line_masks",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.is_train = is_train
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / mask_dir_name

        json_path = self.root / f"data_{split}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {json_path}")
        with open(json_path) as f:
            self._entries: list[dict] = json.load(f)

        self._entries = [
            e for e in self._entries
            if (self.masks_dir / f"{e['id']}.png").exists()
        ]

        cfg = config or {}
        self.spatial_transforms, self.image_transforms = build_seg_transforms(
            is_train=is_train,
            train_scales=cfg.get("train_scales"),
            val_short_side=cfg.get("val_short_side", 640),
            crop_scale=cfg.get("crop_scale", (0.3, 1.0)),
            crop_ratio=cfg.get("crop_ratio", (0.75, 1.333)),
            hflip_prob=cfg.get("hflip_prob", 0.5),
            swap_pairs=cfg.get("hflip_swap_pairs"),
            affine_degrees=cfg.get("affine_degrees", 15.0),
            affine_translate=cfg.get("affine_translate", (0.1, 0.1)),
            affine_scale=cfg.get("affine_scale", (0.8, 1.2)),
            affine_shear=cfg.get("affine_shear", 10.0),
            perspective_distortion=cfg.get("perspective_distortion", 0.15),
            perspective_prob=cfg.get("perspective_prob", 0.3),
            color_jitter=cfg.get("color_jitter", (0.3, 0.3, 0.3, 0.1)),
            gaussian_blur_kernel=cfg.get("gaussian_blur_kernel"),
            gaussian_blur_sigma=cfg.get("gaussian_blur_sigma", (0.1, 2.0)),
            gaussian_blur_prob=cfg.get("gaussian_blur_prob", 0.3),
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        entry = self._entries[idx]
        image_id: str = entry["id"]

        img_path = find_existing_file(
            self.images_dir, image_id, (".png", ".jpg")
        ) or (self.images_dir / f"{image_id}.jpg")
        img = Image.open(img_path).convert("RGB")

        mask_path = self.masks_dir / f"{image_id}.png"
        mask = Image.open(mask_path).convert("L")

        for t in self.spatial_transforms:
            img, mask = t(img, mask)

        for t in self.image_transforms:
            img = t(img)

        img_tensor = _pil_to_tensor_image(img)
        img_tensor = TF.normalize(img_tensor, IMAGENET_MEAN, IMAGENET_STD)
        _, h, w = img_tensor.shape

        mask_np = (np.array(mask, dtype=np.uint8) > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "image_size": torch.tensor([h, w], dtype=torch.int64),
            "image_id": image_id,
        }
