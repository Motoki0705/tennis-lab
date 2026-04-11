"""Court keypoint heatmap dataset.

Loads court images and 14-keypoint annotations, generates per-keypoint
Gaussian heatmaps after spatial augmentation.
"""

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
    build_kp_transforms,
)


def generate_gaussian_heatmap(
    h: int,
    w: int,
    cx: float,
    cy: float,
    sigma: float = 3.0,
) -> np.ndarray:
    """Generate a Gaussian heatmap centred at ``(cx, cy)``.

    Returns
    -------
    np.ndarray
        ``(H, W)`` float32 in ``[0, 1]``.
    """
    cx_int, cy_int = int(round(cx)), int(round(cy))
    if cx_int < 0 or cy_int < 0 or cx_int >= w or cy_int >= h:
        return np.zeros((h, w), dtype=np.float32)

    size = int(3 * sigma)
    yy, xx = np.mgrid[-size : size + 1, -size : size + 1]
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.max()

    heatmap = np.zeros((h, w), dtype=np.float32)

    y_start = max(0, cy_int - size)
    y_end = min(h, cy_int + size + 1)
    x_start = max(0, cx_int - size)
    x_end = min(w, cx_int + size + 1)

    ky_start = size - (cy_int - y_start)
    ky_end = ky_start + (y_end - y_start)
    kx_start = size - (cx_int - x_start)
    kx_end = kx_start + (x_end - x_start)

    heatmap[y_start:y_end, x_start:x_end] = kernel[ky_start:ky_end, kx_start:kx_end]
    return heatmap


class CourtKPDataset(Dataset):
    """PyTorch Dataset for court keypoint heatmap regression.

    Parameters
    ----------
    root:
        Path to ``data/court/``.
    split:
        ``"train"`` or ``"val"``.
    is_train:
        Whether to apply training augmentations.
    config:
        Configuration dict (from :class:`CourtKPConfig`).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        is_train: bool = True,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.is_train = is_train
        self.images_dir = self.root / "images"

        json_path = self.root / f"data_{split}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {json_path}")
        with open(json_path) as f:
            self._entries: list[dict] = json.load(f)

        cfg = config or {}
        self.gaussian_sigma = cfg.get("gaussian_sigma", 3.0)

        self.spatial_transforms, self.image_transforms = build_kp_transforms(
            is_train=is_train,
            train_scales=cfg.get("train_scales"),
            val_short_side=cfg.get("val_short_side", 640),
            crop_scale=cfg.get("crop_scale", (0.3, 1.0)),
            crop_ratio=cfg.get("crop_ratio", (0.75, 1.333)),
            hflip_prob=cfg.get("hflip_prob", 0.5),
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

    def __getitem__(self, idx: int) -> dict[str, Tensor | str]:
        entry = self._entries[idx]
        image_id: str = entry["id"]
        kps = np.array(entry["kps"], dtype=np.float32)  # (14, 2)

        img_path = self.images_dir / f"{image_id}.png"
        if not img_path.exists():
            img_path = self.images_dir / f"{image_id}.jpg"
        img = Image.open(img_path).convert("RGB")

        for t in self.spatial_transforms:
            img, kps = t(img, kps)

        for t in self.image_transforms:
            img = t(img)

        w, h = img.size
        heatmaps = np.zeros((len(kps), h, w), dtype=np.float32)
        for k_idx, (cx, cy) in enumerate(kps):
            heatmaps[k_idx] = generate_gaussian_heatmap(
                h, w, float(cx), float(cy), sigma=self.gaussian_sigma,
            )

        img_tensor = _pil_to_tensor_image(img)
        img_tensor = TF.normalize(img_tensor, IMAGENET_MEAN, IMAGENET_STD)
        heatmap_tensor = torch.from_numpy(heatmaps)

        return {
            "image": img_tensor,
            "heatmap": heatmap_tensor,
            "keypoints": torch.from_numpy(kps),
            "image_id": image_id,
        }
