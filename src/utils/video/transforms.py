"""Frame transforms for video streaming pipelines."""

from __future__ import annotations

import cv2
import torch
from numpy.typing import NDArray

from src.utils.data.augmentation import normalize_tensor_images_imagenet


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
            tensor = normalize_tensor_images_imagenet(tensor)
        return tensor
