"""Frame transforms for video streaming pipelines."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from src.utils.data.augmentation import normalize_tensor_images_imagenet
from src.utils.inference_recording import measured


class BgrToRgbUint8Transform:
    """Resize BGR pixels with OpenCV linear interpolation, retaining uint8 RGB."""

    def __init__(self, *, image_size: tuple[int, int]) -> None:
        self.image_size = image_size

    def __call__(self, frame_bgr: NDArray) -> torch.Tensor:
        if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Video frames must be H,W,3 BGR uint8")
        image_h, image_w = self.image_size
        resized = cv2.resize(frame_bgr, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()


class BgrToTensorTransform(BgrToRgbUint8Transform):
    """Convert OpenCV BGR frames into CHW float RGB tensors."""

    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        normalize_imagenet: bool = False,
    ) -> None:
        super().__init__(image_size=image_size)
        self.normalize_imagenet = normalize_imagenet

    @measured("preprocess")
    def __call__(self, frame_bgr: NDArray) -> torch.Tensor:
        tensor = super().__call__(frame_bgr)
        tensor = tensor.to(dtype=torch.float32).div_(255.0)
        if self.normalize_imagenet:
            tensor = normalize_tensor_images_imagenet(tensor)
        return tensor
