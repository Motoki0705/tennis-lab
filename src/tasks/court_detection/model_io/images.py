"""Canonical raw-image boundary for court model inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor

from src.tasks.court_detection.data.augmentation import IMAGENET_MEAN, IMAGENET_STD
from src.tasks.court_detection.model_io.contracts import (
    CourtModelIOError,
    CourtModelSpec,
)
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


@dataclass(frozen=True)
class PreparedCourtImage:
    images: Tensor
    original_size_hw: tuple[int, int]
    source_from_model_xy: tuple[float, float]
    content_size_hw: tuple[int, int]


def prepare_court_input(image: np.ndarray | Image.Image, *, spec: CourtModelSpec,
                        device: torch.device) -> PreparedCourtImage:
    """Match the checkpoint's validation geometry, including pose patch padding."""
    if not spec.pose_long_side:
        images, height, width = prepare_court_image(image, short_side=spec.short_side, device=device)
        model_height, model_width = images.shape[-2:]
        return PreparedCourtImage(images, (height, width),
                                  ((width - 1) / max(model_width - 1, 1),
                                   (height - 1) / max(model_height - 1, 1)),
                                  (model_height, model_width))
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            raise CourtModelIOError("Court PIL images must use RGB mode")
        image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise CourtModelIOError("Court numpy images must be uint8 RGB (H,W,3)")
    height, width = image.shape[:2]
    if min(height, width, spec.short_side, spec.patch_size) <= 0:
        raise CourtModelIOError("Court image and configured sizes must be positive")
    scale = spec.short_side / max(height, width)
    content_height = max(1, round(height * scale))
    content_width = max(1, round(width * scale))
    matrix = np.diag([scale, scale, 1.0])
    content = cv2.warpPerspective(image, matrix, (content_width, content_height), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    padded = cv2.copyMakeBorder(content, 0, (-content_height) % spec.patch_size,
                               0, (-content_width) % spec.patch_size, cv2.BORDER_REPLICATE)
    tensor = TF.normalize(TF.to_tensor(padded), IMAGENET_MEAN, IMAGENET_STD)
    return PreparedCourtImage(tensor[None].to(device), (height, width), (1 / scale, 1 / scale),
                              (content_height, content_width))
