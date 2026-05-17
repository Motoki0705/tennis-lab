"""Frame transforms for video streaming pipelines."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import torch
from numpy.typing import NDArray


class BgrToTensorTransform:
    """Convert OpenCV BGR frames into CHW float RGB tensors."""

    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        normalize_imagenet: bool = False,
    ) -> None:
        self.image_size = image_size
        self.normalize_imagenet = normalize_imagenet

    def __call__(self, frame_bgr: NDArray) -> torch.Tensor:
        image_h, image_w = self.image_size
        resized = cv2.resize(frame_bgr, (image_w, image_h))
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()
        tensor = tensor.to(dtype=torch.float32).div_(255.0)
        if self.normalize_imagenet:
            tensor = normalize_tensor_imagenet(tensor)
        return tensor


def normalize_tensor_imagenet(
    images: torch.Tensor,
    *,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Apply ImageNet normalization to ``(..., 3, H, W)`` tensors."""
    if images.ndim < 3 or images.shape[-3] != 3:
        raise ValueError(
            "Expected images with shape (..., 3, H, W) for ImageNet normalization, "
            f"got {tuple(images.shape)}."
        )
    view_shape = [1] * images.ndim
    view_shape[-3] = 3
    mean_tensor = images.new_tensor(mean).view(*view_shape)
    std_tensor = images.new_tensor(std).view(*view_shape)
    return (images - mean_tensor) / std_tensor
