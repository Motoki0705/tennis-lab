"""Unit tests for :mod:`src.utils.video.transforms`."""

from __future__ import annotations

import numpy as np
import torch

from src.utils.data.augmentation import normalize_tensor_images_imagenet
from src.utils.video.transforms import (
    BgrToTensorTransform,
    normalize_tensor_imagenet,
)


class TestBgrToTensorTransform:
    def test_output_shape_dtype_range(self) -> None:
        frame_bgr = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        transform = BgrToTensorTransform(image_size=(4, 6))
        out = transform(frame_bgr)
        assert out.shape == (3, 4, 6)
        assert out.dtype == torch.float32
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_bgr_to_rgb_channel_order(self) -> None:
        # Pure-blue BGR image (B=255). After BGR->RGB it lands in channel index 2.
        frame_bgr: np.ndarray = np.zeros((4, 4, 3), dtype=np.uint8)
        frame_bgr[..., 0] = 255
        out = BgrToTensorTransform(image_size=(4, 4))(frame_bgr)
        assert torch.allclose(out[2], torch.ones(4, 4))
        assert torch.allclose(out[0], torch.zeros(4, 4))

    def test_imagenet_normalization_changes_values(self) -> None:
        frame_bgr: np.ndarray = np.full((4, 4, 3), 128, dtype=np.uint8)
        plain = BgrToTensorTransform(image_size=(4, 4))(frame_bgr)
        normed = BgrToTensorTransform(image_size=(4, 4), normalize_imagenet=True)(
            frame_bgr
        )
        assert not torch.allclose(plain, normed)


class TestNormalizeTensorImagenetAlias:
    def test_matches_canonical_implementation(self) -> None:
        images = torch.rand(3, 5, 5)
        assert torch.allclose(
            normalize_tensor_imagenet(images),
            normalize_tensor_images_imagenet(images),
        )
