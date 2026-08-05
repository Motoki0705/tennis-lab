"""Court keypoint heatmap dataset."""

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
    build_kp_transforms,
)
from src.utils.data.heatmaps import generate_gaussian_heatmaps
from src.utils.io import find_existing_file


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
        split: str,
        is_train: bool,
        config: CourtDataConfig,
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

        if config.sigma_ratio is None:
            raise ValueError("CourtKPDataset requires keypoint data configuration.")
        self.sigma_ratio = config.sigma_ratio
        aug = config.augmentation

        self.spatial_pipeline, self.image_transforms = build_kp_transforms(
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
            min_visible_kp=aug.min_visible_kp,
            visibility_max_retries=aug.visibility_max_retries,
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Tensor | str]:
        entry = self._entries[idx]
        image_id: str = entry["id"]
        kps = np.array(entry["kps"], dtype=np.float32)  # (14, 2)

        img_path = find_existing_file(self.images_dir, image_id, (".png", ".jpg"))
        if img_path is None:
            raise FileNotFoundError(
                f"Image not found for image_id={image_id!r} under {self.images_dir}."
            )
        img = Image.open(img_path).convert("RGB")

        img, kps, kp_visible = self.spatial_pipeline.transform_with_visibility(img, kps)

        for t in self.image_transforms:
            img = t(img)

        w, h = img.size
        img_tensor = _pil_to_tensor_image(img)
        img_tensor = TF.normalize(img_tensor, IMAGENET_MEAN, IMAGENET_STD)
        kps_tensor = torch.from_numpy(kps)

        normalized_kps = kps_tensor.clone()
        if w > 1:
            normalized_kps[:, 0] = normalized_kps[:, 0] / float(w - 1)
        else:
            normalized_kps[:, 0] = 0.0
        if h > 1:
            normalized_kps[:, 1] = normalized_kps[:, 1] / float(h - 1)
        else:
            normalized_kps[:, 1] = 0.0

        kp_visible_tensor = torch.from_numpy(kp_visible)
        heatmap_tensor = generate_gaussian_heatmaps(
            size_hw=(h, w),
            centers_xy=normalized_kps,
            sigma_ratio=self.sigma_ratio,
            visibility=kp_visible_tensor,
        )

        return {
            "image": img_tensor,
            "heatmap": heatmap_tensor,
            "keypoints": kps_tensor,
            "kp_visible": kp_visible_tensor,
            "image_size": torch.tensor([h, w], dtype=torch.int64),
            "image_id": image_id,
        }
