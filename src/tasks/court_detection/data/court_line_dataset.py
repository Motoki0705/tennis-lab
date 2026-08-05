"""Court white-line segmentation dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.tasks.court_detection.configuration import CourtDataConfig
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
        split: str,
        is_train: bool,
        config: CourtDataConfig,
        mask_dir_name: str,
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
            e for e in self._entries if (self.masks_dir / f"{e['id']}.png").exists()
        ]

        aug = config.augmentation
        self.spatial_transforms, self.image_transforms = build_seg_transforms(
            is_train=is_train,
            train_scales=list(aug.train_scales),
            val_short_side=aug.val_short_side,
            crop_scale=aug.crop_scale,
            crop_ratio=aug.crop_ratio,
            hflip_prob=aug.hflip_prob,
            swap_pairs=list(config.hflip_swap_pairs),
            affine_degrees=aug.affine_degrees,
            affine_translate=aug.affine_translate,
            affine_scale=aug.affine_scale,
            affine_shear=aug.affine_shear,
            perspective_distortion=aug.perspective_distortion,
            perspective_prob=aug.perspective_prob,
            color_jitter=aug.color_jitter,
            gaussian_blur_kernel=list(aug.gaussian_blur_kernel),
            gaussian_blur_sigma=aug.gaussian_blur_sigma,
            gaussian_blur_prob=aug.gaussian_blur_prob,
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        entry = self._entries[idx]
        image_id: str = entry["id"]

        img_path = find_existing_file(self.images_dir, image_id, (".png", ".jpg"))
        if img_path is None:
            raise FileNotFoundError(
                f"Image not found for image_id={image_id!r} under {self.images_dir}."
            )
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
