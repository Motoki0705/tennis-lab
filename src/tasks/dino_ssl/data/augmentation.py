"""DINO multi-crop augmentation and iBOT block-masking generator.

The augmentation follows the DINO/DINOv2 multi-crop recipe: two high-resolution
*global* crops and several low-resolution *local* crops per image, with the
standard photometric and blur augmentations. The masking generator produces the
boolean patch masks consumed by the iBOT objective.
"""

from __future__ import annotations

import math
import random

import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GaussianBlur:
    """Apply Gaussian blur with a configurable probability."""

    def __init__(
        self, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0
    ):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        from PIL import ImageFilter

        radius = random.uniform(self.radius_min, self.radius_max)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _normalize() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class DataAugmentationDINO:
    """Produce two global crops and ``local_crops_number`` local crops."""

    def __init__(
        self,
        *,
        global_crops_scale: tuple[float, float] = (0.32, 1.0),
        local_crops_scale: tuple[float, float] = (0.05, 0.32),
        local_crops_number: int = 6,
        global_size: int = 224,
        local_size: int = 96,
    ) -> None:
        self.local_crops_number = int(local_crops_number)
        self.global_size = int(global_size)
        self.local_size = int(local_size)

        flip_and_color = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue=0.1,
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = _normalize()

        self.global_transform_1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.global_size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color,
                GaussianBlur(p=1.0),
                normalize,
            ]
        )
        self.global_transform_2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.global_size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color,
                GaussianBlur(p=0.1),
                normalize,
            ]
        )
        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.local_size,
                    scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color,
                GaussianBlur(p=0.5),
                normalize,
            ]
        )

    def __call__(self, image: Image.Image) -> dict[str, list[torch.Tensor]]:
        image = image.convert("RGB")
        global_crops = [
            self.global_transform_1(image),
            self.global_transform_2(image),
        ]
        local_crops = [
            self.local_transform(image) for _ in range(self.local_crops_number)
        ]
        return {"global_crops": global_crops, "local_crops": local_crops}


class MaskingGenerator:
    """Random block masking over a square patch grid (iBOT-style).

    Produces a flat boolean mask of length ``num_patches`` per call, masking a
    fraction sampled uniformly from ``[mask_ratio_min, mask_ratio_max]``.
    """

    def __init__(
        self,
        *,
        input_size: int,
        patch_size: int,
        mask_ratio_min: float = 0.1,
        mask_ratio_max: float = 0.5,
    ) -> None:
        if input_size % patch_size != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by "
                f"patch_size ({patch_size})."
            )
        self.grid = input_size // patch_size
        self.num_patches = self.grid * self.grid
        self.mask_ratio_min = float(mask_ratio_min)
        self.mask_ratio_max = float(mask_ratio_max)

    def __call__(self) -> torch.Tensor:
        ratio = random.uniform(self.mask_ratio_min, self.mask_ratio_max)
        num_mask = int(math.ceil(self.num_patches * ratio))
        # Guarantee at least one masked and one visible patch.
        num_mask = max(1, min(num_mask, self.num_patches - 1))
        mask = torch.zeros(self.num_patches, dtype=torch.bool)
        indices = torch.randperm(self.num_patches)[:num_mask]
        mask[indices] = True
        return mask


__all__ = [
    "DataAugmentationDINO",
    "MaskingGenerator",
    "GaussianBlur",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
