"""Shared image preprocessing for court-detection predictors."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor

from src.tasks.court_detection.data.augmentation import IMAGENET_MEAN, IMAGENET_STD
from src.utils.geometry.image_size import resize_short_side_aligned


def preprocess_court_image(
    image: np.ndarray | Image.Image,
    *,
    short_side: int,
    device: torch.device,
) -> tuple[Tensor, int, int]:
    """Resize (short-side, multiple of 8) and ImageNet-normalize an image.

    Args:
        image: Input image as a numpy array or PIL image.
        short_side: Target length of the shorter image side.
        device: Device the returned tensor is moved to.

    Returns:
        Tuple of (tensor ``[1, 3, H', W']``, original_height, original_width).
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    orig_w, orig_h = image.size

    new_w, new_h = resize_short_side_aligned(orig_w, orig_h, short_side)
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    img_tensor = TF.to_tensor(image)
    img_tensor = TF.normalize(img_tensor, IMAGENET_MEAN, IMAGENET_STD)
    batched = cast("Tensor", img_tensor.unsqueeze(0).to(device))
    return batched, orig_h, orig_w
