"""Canonical raw-image boundaries for Court model inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

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


@dataclass(frozen=True, slots=True)
class PreparedCourtPoseImage:
    """One pose-safe image plus the geometry needed to decode its prediction."""

    images: Tensor
    original_size_hw: tuple[int, int]
    content_size_hw: tuple[int, int]
    model_size_hw: tuple[int, int]
    source_to_model_scale: float


def prepare_court_pose_image(
    image: np.ndarray | Image.Image,
    *,
    long_side: int,
    patch_size: int,
    device: torch.device,
) -> PreparedCourtPoseImage:
    """Apply the validation geometry used by pose-supervised Court training.

    Pose targets encode one scalar focal length, so inference must preserve an
    isotropic image transform. The content is resized by its long side and only
    minimally padded on the right and bottom to reach the model patch grid.
    """

    if long_side <= 0:
        raise CourtModelIOError("Court pose long_side must be positive.")
    if patch_size <= 0:
        raise CourtModelIOError("Court pose patch_size must be positive.")
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
    scale = long_side / float(max(original_width, original_height))
    content_width = max(1, int(round(original_width * scale)))
    content_height = max(1, int(round(original_height * scale)))
    model_width = content_width + (-content_width) % patch_size
    model_height = content_height + (-content_height) % patch_size

    resized = image.resize(
        (content_width, content_height),
        Image.Resampling.BILINEAR,
    )
    tensor = TF.to_tensor(resized)
    pad_right = model_width - content_width
    pad_bottom = model_height - content_height
    if pad_right or pad_bottom:
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode="replicate")
    normalized = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return PreparedCourtPoseImage(
        images=cast(Tensor, normalized.unsqueeze(0).to(device)),
        original_size_hw=(original_height, original_width),
        content_size_hw=(content_height, content_width),
        model_size_hw=(model_height, model_width),
        source_to_model_scale=scale,
    )


__all__ = [
    "PreparedCourtPoseImage",
    "prepare_court_image",
    "prepare_court_pose_image",
]
