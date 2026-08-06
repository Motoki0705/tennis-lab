"""Canonical raw-image boundary for court model inference."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor

from src.tasks.court_detection.data.augmentation import IMAGENET_MEAN, IMAGENET_STD
from src.tasks.court_detection.model_io.contracts import CourtModelIOError
from src.utils.geometry.image_size import resize_short_side_aligned


def prepare_court_image(
    image: np.ndarray | Image.Image,
    *,
    short_side: int,
    device: torch.device,
) -> tuple[Tensor, int, int]:
    """Validate, resize, normalize, and batch one RGB court image."""
    if short_side <= 0:
        raise CourtModelIOError("Court preprocessing short_side must be positive.")
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise CourtModelIOError(
                "Court numpy images must have shape (H, W, 3) and dtype uint8."
            )
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise CourtModelIOError("Court preprocessing requires a numpy or PIL image.")
    if image.mode != "RGB":
        raise CourtModelIOError(
            f"Court PIL images must use RGB mode, got {image.mode!r}."
        )

    original_width, original_height = image.size
    if original_height <= 0 or original_width <= 0:
        raise CourtModelIOError("Court input image dimensions must be positive.")
    new_width, new_height = resize_short_side_aligned(
        original_width,
        original_height,
        short_side,
    )
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    tensor = TF.normalize(TF.to_tensor(resized), IMAGENET_MEAN, IMAGENET_STD)
    batched = cast(Tensor, tensor.unsqueeze(0).to(device))
    return batched, original_height, original_width


__all__ = ["prepare_court_image"]
