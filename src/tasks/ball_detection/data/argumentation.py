"""Sequence argumentation utilities for supervised ball detection."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import get_worker_info


Frames = list[np.ndarray]
Coords = list[tuple[float, float]]
Visibility = list[float]


class BaseArgumentation:
    """Base interface for one sequence augmentation."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def forward(
        self,
        frames: Frames,
        coords: Coords,
        visibility: Visibility,
        *,
        rng: random.Random,
    ) -> tuple[Frames, Coords, Visibility]:
        """Apply the augmentation to images and targets."""
        raise NotImplementedError


class CameraRotationArgumentation(BaseArgumentation):
    """Apply sequence-consistent camera rotation."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.prob = float(self.config.get("prob", 0.0))
        self.max_center_angle_deg = float(self.config.get("max_center_angle_deg", 0.0))
        self.max_angular_velocity_deg_per_frame = float(
            self.config.get("max_angular_velocity_deg_per_frame", 0.0)
        )
        self.border_mode = str(self.config.get("border_mode", "reflect101"))

    def forward(
        self,
        frames: Frames,
        coords: Coords,
        visibility: Visibility,
        *,
        rng: random.Random,
    ) -> tuple[Frames, Coords, Visibility]:
        """Rotate the full sequence around the image center."""
        if not self.enabled or rng.random() >= self.prob:
            return frames, coords, visibility

        height, width = frames[0].shape[:2]
        theta0 = rng.uniform(-self.max_center_angle_deg, self.max_center_angle_deg)
        omega = rng.uniform(
            -self.max_angular_velocity_deg_per_frame,
            self.max_angular_velocity_deg_per_frame,
        )
        t_ref = (len(frames) - 1) / 2.0
        angles_deg = [theta0 + omega * (frame_idx - t_ref) for frame_idx in range(len(frames))]

        border_mode = {
            "constant": cv2.BORDER_CONSTANT,
            "reflect": cv2.BORDER_REFLECT,
            "reflect101": cv2.BORDER_REFLECT_101,
            "replicate": cv2.BORDER_REPLICATE,
        }.get(self.border_mode, cv2.BORDER_REFLECT_101)

        out_frames: Frames = []
        out_coords: Coords = []
        out_visibility: Visibility = []
        for frame, (x, y), vis, angle in zip(frames, coords, visibility, angles_deg):
            matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
            rotated = cv2.warpAffine(
                frame,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=border_mode,
            )
            if vis > 0:
                new_x = float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2])
                new_y = float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2])
                if new_x < 0 or new_x >= width or new_y < 0 or new_y >= height:
                    out_coords.append((0.0, 0.0))
                    out_visibility.append(0.0)
                else:
                    out_coords.append((new_x, new_y))
                    out_visibility.append(float(vis))
            else:
                out_coords.append((0.0, 0.0))
                out_visibility.append(0.0)
            out_frames.append(rotated.astype(np.float32, copy=False))
        return out_frames, out_coords, out_visibility


class HorizontalFlipArgumentation(BaseArgumentation):
    """Apply sequence-consistent horizontal flipping."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.prob = float(self.config.get("prob", 0.0))

    def forward(
        self,
        frames: Frames,
        coords: Coords,
        visibility: Visibility,
        *,
        rng: random.Random,
    ) -> tuple[Frames, Coords, Visibility]:
        """Flip all frames horizontally with the configured probability."""
        if not self.enabled or rng.random() >= self.prob:
            return frames, coords, visibility

        width = frames[0].shape[1]
        out_frames = [cv2.flip(frame, 1).astype(np.float32, copy=False) for frame in frames]
        out_coords: Coords = []
        out_visibility: Visibility = []
        for (x, y), vis in zip(coords, visibility):
            if vis > 0:
                out_coords.append(((width - 1) - x, y))
                out_visibility.append(float(vis))
            else:
                out_coords.append((0.0, 0.0))
                out_visibility.append(0.0)
        return out_frames, out_coords, out_visibility


class BrightnessGainArgumentation(BaseArgumentation):
    """Apply sequence-consistent brightness gain jitter."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.jitter = float(self.config.get("jitter", 0.0))

    def forward(
        self,
        frames: Frames,
        coords: Coords,
        visibility: Visibility,
        *,
        rng: random.Random,
    ) -> tuple[Frames, Coords, Visibility]:
        """Adjust brightness gain for the entire sequence."""
        if not self.enabled or self.jitter <= 0:
            return frames, coords, visibility
        gain = rng.uniform(1.0 - self.jitter, 1.0 + self.jitter)
        out_frames = [np.clip(frame * gain, 0.0, 1.0).astype(np.float32) for frame in frames]
        return out_frames, coords, visibility


class ContrastArgumentation(BaseArgumentation):
    """Apply sequence-consistent contrast jitter."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.jitter = float(self.config.get("jitter", 0.0))

    def forward(
        self,
        frames: Frames,
        coords: Coords,
        visibility: Visibility,
        *,
        rng: random.Random,
    ) -> tuple[Frames, Coords, Visibility]:
        """Adjust contrast for the entire sequence."""
        if not self.enabled or self.jitter <= 0:
            return frames, coords, visibility
        factor = rng.uniform(1.0 - self.jitter, 1.0 + self.jitter)
        out_frames = [
            np.clip((frame - 0.5) * factor + 0.5, 0.0, 1.0).astype(np.float32)
            for frame in frames
        ]
        return out_frames, coords, visibility


class GammaArgumentation(BaseArgumentation):
    """Apply sequence-consistent gamma jitter."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.jitter = float(self.config.get("jitter", 0.0))

    def forward(
        self,
        frames: Frames,
        coords: Coords,
        visibility: Visibility,
        *,
        rng: random.Random,
    ) -> tuple[Frames, Coords, Visibility]:
        """Adjust gamma for the entire sequence."""
        if not self.enabled or self.jitter <= 0:
            return frames, coords, visibility
        gamma = rng.uniform(1.0 - self.jitter, 1.0 + self.jitter)
        out_frames = [np.power(np.clip(frame, 0.0, 1.0), gamma).astype(np.float32) for frame in frames]
        return out_frames, coords, visibility


class GaussianNoiseArgumentation(BaseArgumentation):
    """Apply sequence-consistent Gaussian noise scale."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.std = float(self.config.get("std", 0.0))

    def forward(
        self,
        frames: Frames,
        coords: Coords,
        visibility: Visibility,
        *,
        rng: random.Random,
    ) -> tuple[Frames, Coords, Visibility]:
        """Add Gaussian noise to each frame."""
        if not self.enabled or self.std <= 0:
            return frames, coords, visibility

        noise_scale = rng.uniform(0.0, self.std)
        np_rng = np.random.default_rng(rng.randrange(0, 2**32))
        out_frames = [
            np.clip(
                frame + np_rng.normal(0.0, noise_scale, size=frame.shape).astype(np.float32),
                0.0,
                1.0,
            ).astype(np.float32)
            for frame in frames
        ]
        return out_frames, coords, visibility


class GaussianBlurArgumentation(BaseArgumentation):
    """Apply sequence-consistent Gaussian blur."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.prob = float(self.config.get("prob", 0.0))
        self.kernel_size = int(self.config.get("kernel_size", 3))

    def forward(
        self,
        frames: Frames,
        coords: Coords,
        visibility: Visibility,
        *,
        rng: random.Random,
    ) -> tuple[Frames, Coords, Visibility]:
        """Blur the full sequence with the configured probability."""
        if not self.enabled or rng.random() >= self.prob:
            return frames, coords, visibility

        kernel_size = self.kernel_size
        if kernel_size < 3:
            return frames, coords, visibility
        if kernel_size % 2 == 0:
            kernel_size += 1
        out_frames = [
            cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0).astype(np.float32)
            for frame in frames
        ]
        return out_frames, coords, visibility


class BallDetectionArgumentation:
    """Compose and apply all configured sequence augmentations."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.transforms: list[BaseArgumentation] = [
            CameraRotationArgumentation(dict(config.get("camera_rotation", {}) or {})),
            HorizontalFlipArgumentation(dict(config.get("horizontal_flip", {}) or {})),
            BrightnessGainArgumentation(dict(config.get("brightness_gain", {}) or {})),
            ContrastArgumentation(dict(config.get("contrast", {}) or {})),
            GammaArgumentation(dict(config.get("gamma", {}) or {})),
            GaussianNoiseArgumentation(dict(config.get("gaussian_noise", {}) or {})),
            GaussianBlurArgumentation(dict(config.get("gaussian_blur", {}) or {})),
        ]

    def forward(
        self,
        frames: Frames,
        coords: Coords,
        visibility: Visibility,
        *,
        rng: random.Random,
    ) -> tuple[Frames, Coords, Visibility]:
        """Apply all configured augmentations in sequence."""
        out_frames = [frame.astype(np.float32, copy=True) for frame in frames]
        out_coords = list(coords)
        out_visibility = list(visibility)
        for transform in self.transforms:
            out_frames, out_coords, out_visibility = transform.forward(
                out_frames,
                out_coords,
                out_visibility,
                rng=rng,
            )
        return out_frames, out_coords, out_visibility


def make_sample_rng(sample_idx: int) -> random.Random:
    """Create a deterministic RNG per sample and dataloader worker."""
    worker_info = get_worker_info()
    base_seed = int(torch.initial_seed())
    if worker_info is not None:
        base_seed += int(worker_info.id) * 1_000_003
    return random.Random(base_seed + int(sample_idx))


__all__ = [
    "BallDetectionArgumentation",
    "BaseArgumentation",
    "BrightnessGainArgumentation",
    "CameraRotationArgumentation",
    "ContrastArgumentation",
    "GammaArgumentation",
    "GaussianBlurArgumentation",
    "GaussianNoiseArgumentation",
    "HorizontalFlipArgumentation",
    "make_sample_rng",
]
