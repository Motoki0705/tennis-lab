"""Sequence augmentation utilities for supervised ball detection."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import get_worker_info

Frames = list[np.ndarray]
Coords = list[list[tuple[float, float]]]
Visibility = list[list[float]]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_frames_imagenet(
    frames: Frames,
    *,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
) -> Frames:
    """Apply ImageNet normalization to HWC float frames."""
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    return [((frame - mean_arr) / std_arr).astype(np.float32) for frame in frames]


def normalize_tensor_images_imagenet(
    images: torch.Tensor,
    *,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Apply ImageNet normalization to ``(..., 3, H, W)`` image tensors."""
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


def denormalize_tensor_images_imagenet(
    images: torch.Tensor,
    *,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Undo ImageNet normalization for ``(..., 3, H, W)`` image tensors."""
    if images.ndim < 3 or images.shape[-3] != 3:
        raise ValueError(
            "Expected images with shape (..., 3, H, W) for ImageNet denormalization, "
            f"got {tuple(images.shape)}."
        )
    view_shape = [1] * images.ndim
    view_shape[-3] = 3
    mean_tensor = images.new_tensor(mean).view(*view_shape)
    std_tensor = images.new_tensor(std).view(*view_shape)
    return images * std_tensor + mean_tensor


def _resolve_border_mode(name: str) -> int:
    """Map config border mode names to OpenCV constants."""
    return {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT,
        "reflect101": cv2.BORDER_REFLECT_101,
        "replicate": cv2.BORDER_REPLICATE,
    }.get(name, cv2.BORDER_REFLECT_101)


def _parse_float_range(value: Any, name: str) -> tuple[float, float]:
    """Parse a length-2 sequence into a validated (min, max) float tuple."""
    if not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{name} must be a sequence with two elements.")
    low = float(value[0])
    high = float(value[1])
    if low > high:
        raise ValueError(f"{name} must satisfy min <= max.")
    return low, high


def _apply_affine_to_sequence(
    frames: Frames,
    coords: Coords,
    visibility: Visibility,
    *,
    matrix: np.ndarray,
    border_mode: int,
) -> tuple[Frames, Coords, Visibility]:
    """Warp frames and transform visible coordinates with one affine matrix."""
    height, width = frames[0].shape[:2]
    out_frames: Frames = []
    out_coords: Coords = []
    out_visibility: Visibility = []

    for frame, frame_coords, frame_visibility in zip(
        frames,
        coords,
        visibility,
        strict=True,
    ):
        warped = cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=border_mode,
        )
        transformed_coords: list[tuple[float, float]] = []
        transformed_visibility: list[float] = []
        for (x, y), vis in zip(frame_coords, frame_visibility, strict=True):
            if vis > 0:
                new_x = float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2])
                new_y = float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2])
                if new_x < 0 or new_x >= width or new_y < 0 or new_y >= height:
                    transformed_coords.append((0.0, 0.0))
                    transformed_visibility.append(0.0)
                else:
                    transformed_coords.append((new_x, new_y))
                    transformed_visibility.append(float(vis))
            else:
                transformed_coords.append((0.0, 0.0))
                transformed_visibility.append(0.0)
        out_coords.append(transformed_coords)
        out_visibility.append(transformed_visibility)
        out_frames.append(warped.astype(np.float32, copy=False))
    return out_frames, out_coords, out_visibility


class BaseAugmentation:
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


class CameraRotationAugmentation(BaseAugmentation):
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

        border_mode = _resolve_border_mode(self.border_mode)

        out_frames: Frames = []
        out_coords: Coords = []
        out_visibility: Visibility = []
        for frame, frame_coords, frame_visibility, angle in zip(
            frames,
            coords,
            visibility,
            angles_deg,
            strict=True,
        ):
            matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
            rotated = cv2.warpAffine(
                frame,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=border_mode,
            )
            transformed_coords: list[tuple[float, float]] = []
            transformed_visibility: list[float] = []
            for (x, y), vis in zip(frame_coords, frame_visibility, strict=True):
                if vis > 0:
                    new_x = float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2])
                    new_y = float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2])
                    if new_x < 0 or new_x >= width or new_y < 0 or new_y >= height:
                        transformed_coords.append((0.0, 0.0))
                        transformed_visibility.append(0.0)
                    else:
                        transformed_coords.append((new_x, new_y))
                        transformed_visibility.append(float(vis))
                else:
                    transformed_coords.append((0.0, 0.0))
                    transformed_visibility.append(0.0)
            out_coords.append(transformed_coords)
            out_visibility.append(transformed_visibility)
            out_frames.append(rotated.astype(np.float32, copy=False))
        return out_frames, out_coords, out_visibility


class HorizontalFlipAugmentation(BaseAugmentation):
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
        for frame_coords, frame_visibility in zip(coords, visibility, strict=True):
            flipped_coords: list[tuple[float, float]] = []
            flipped_visibility: list[float] = []
            for (x, y), vis in zip(frame_coords, frame_visibility, strict=True):
                if vis > 0:
                    flipped_coords.append(((width - 1) - x, y))
                    flipped_visibility.append(float(vis))
                else:
                    flipped_coords.append((0.0, 0.0))
                    flipped_visibility.append(0.0)
            out_coords.append(flipped_coords)
            out_visibility.append(flipped_visibility)
        return out_frames, out_coords, out_visibility


class BrightnessGainAugmentation(BaseAugmentation):
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


class ContrastAugmentation(BaseAugmentation):
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


class GammaAugmentation(BaseAugmentation):
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


class GaussianNoiseAugmentation(BaseAugmentation):
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


class AffineAugmentation(BaseAugmentation):
    """Apply one sequence-consistent affine transform."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.prob = float(self.config.get("prob", 0.0))
        self.rotation_deg_range = _parse_float_range(
            self.config.get("rotation_deg_range", (0.0, 0.0)),
            "rotation_deg_range",
        )
        self.scale_range = _parse_float_range(
            self.config.get("scale_range", (1.0, 1.0)),
            "scale_range",
        )
        self.translate_x_ratio_range = _parse_float_range(
            self.config.get("translate_x_ratio_range", (0.0, 0.0)),
            "translate_x_ratio_range",
        )
        self.translate_y_ratio_range = _parse_float_range(
            self.config.get("translate_y_ratio_range", (0.0, 0.0)),
            "translate_y_ratio_range",
        )
        self.shear_x_deg_range = _parse_float_range(
            self.config.get("shear_x_deg_range", (0.0, 0.0)),
            "shear_x_deg_range",
        )
        self.shear_y_deg_range = _parse_float_range(
            self.config.get("shear_y_deg_range", (0.0, 0.0)),
            "shear_y_deg_range",
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
        """Apply one affine transform to the whole sequence."""
        if not self.enabled or rng.random() >= self.prob:
            return frames, coords, visibility

        height, width = frames[0].shape[:2]
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0

        rotation_rad = np.deg2rad(rng.uniform(*self.rotation_deg_range))
        scale = rng.uniform(*self.scale_range)
        if scale <= 0.0:
            return frames, coords, visibility
        shear_x_rad = np.deg2rad(rng.uniform(*self.shear_x_deg_range))
        shear_y_rad = np.deg2rad(rng.uniform(*self.shear_y_deg_range))
        translate_x = width * rng.uniform(*self.translate_x_ratio_range)
        translate_y = height * rng.uniform(*self.translate_y_ratio_range)

        center_to_origin = np.array(
            [[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        scale_matrix = np.array(
            [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        shear_matrix = np.array(
            [
                [1.0, np.tan(shear_x_rad), 0.0],
                [np.tan(shear_y_rad), 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        cos_theta = float(np.cos(rotation_rad))
        sin_theta = float(np.sin(rotation_rad))
        rotation_matrix = np.array(
            [[cos_theta, -sin_theta, 0.0], [sin_theta, cos_theta, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        recenter = np.array(
            [
                [1.0, 0.0, center_x + translate_x],
                [0.0, 1.0, center_y + translate_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        full_matrix = recenter @ rotation_matrix @ shear_matrix @ scale_matrix @ center_to_origin
        affine_matrix = full_matrix[:2, :]

        return _apply_affine_to_sequence(
            frames,
            coords,
            visibility,
            matrix=affine_matrix,
            border_mode=_resolve_border_mode(self.border_mode),
        )


class GaussianBlurAugmentation(BaseAugmentation):
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


class ScaleAndCropAugmentation(BaseAugmentation):
    """Scale around the image center, then center-crop back to the original size."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.prob = float(self.config.get("prob", 0.0))
        self.scale_range = _parse_float_range(
            self.config.get("scale_range", (1.0, 1.0)),
            "scale_range",
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
        """Apply one zoom centered on the mean visible ball position."""
        if not self.enabled or rng.random() >= self.prob:
            return frames, coords, visibility

        scale = rng.uniform(*self.scale_range)
        if scale <= 0.0 or np.isclose(scale, 1.0):
            return frames, coords, visibility

        height, width = frames[0].shape[:2]
        visible_coords = [
            point
            for frame_coords, frame_visibility in zip(coords, visibility, strict=True)
            for point, vis in zip(frame_coords, frame_visibility, strict=True)
            if vis > 0
        ]
        if visible_coords:
            center_x = float(sum(x for x, _ in visible_coords) / len(visible_coords))
            center_y = float(sum(y for _, y in visible_coords) / len(visible_coords))
        else:
            center_x = (width - 1) / 2.0
            center_y = (height - 1) / 2.0
        matrix = np.array(
            [
                [scale, 0.0, center_x - scale * center_x],
                [0.0, scale, center_y - scale * center_y],
            ],
            dtype=np.float32,
        )

        return _apply_affine_to_sequence(
            frames,
            coords,
            visibility,
            matrix=matrix,
            border_mode=_resolve_border_mode(self.border_mode),
        )


class BallAreaZeroMaskAugmentation(BaseAugmentation):
    """Zero-mask a rectangle around the visible ball for sampled frames."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.prob = float(self.config.get("prob", 0.0))
        self.mask_width_ratio_range = self._parse_ratio_range(
            self.config.get("mask_width_ratio_range", (0.0, 0.0)),
            "mask_width_ratio_range",
        )
        self.mask_height_ratio_range = self._parse_ratio_range(
            self.config.get("mask_height_ratio_range", (0.0, 0.0)),
            "mask_height_ratio_range",
        )
        self.num_frames_range = self._parse_int_range(
            self.config.get("num_frames_range", (0, 0)),
            "num_frames_range",
        )

    @staticmethod
    def _parse_ratio_range(value: Any, name: str) -> tuple[float, float]:
        if not isinstance(value, Sequence) or len(value) != 2:
            raise ValueError(f"{name} must be a sequence with two elements.")
        low = float(value[0])
        high = float(value[1])
        if low < 0.0 or high < 0.0:
            raise ValueError(f"{name} must be non-negative.")
        if low > high:
            raise ValueError(f"{name} must satisfy min <= max.")
        return low, high

    @staticmethod
    def _parse_int_range(value: Any, name: str) -> tuple[int, int]:
        if not isinstance(value, Sequence) or len(value) != 2:
            raise ValueError(f"{name} must be a sequence with two elements.")
        low = int(value[0])
        high = int(value[1])
        if low < 0 or high < 0:
            raise ValueError(f"{name} must be non-negative.")
        if low > high:
            raise ValueError(f"{name} must satisfy min <= max.")
        return low, high

    def forward(
        self,
        frames: Frames,
        coords: Coords,
        visibility: Visibility,
        *,
        rng: random.Random,
    ) -> tuple[Frames, Coords, Visibility]:
        """Apply zero masks centered on sampled visible ball locations."""
        if not self.enabled or rng.random() >= self.prob:
            return frames, coords, visibility

        visible_indices = [
            idx
            for idx, frame_visibility in enumerate(visibility)
            if any(vis > 0 for vis in frame_visibility)
        ]
        if not visible_indices:
            return frames, coords, visibility

        min_frames, max_frames = self.num_frames_range
        if max_frames <= 0:
            return frames, coords, visibility

        num_frames = rng.randint(min_frames, max_frames)
        num_frames = min(num_frames, len(visible_indices))
        if num_frames <= 0:
            return frames, coords, visibility

        selected_indices = rng.sample(visible_indices, k=num_frames)
        out_frames = [frame.astype(np.float32, copy=True) for frame in frames]

        for frame_idx in selected_indices:
            frame = out_frames[frame_idx]
            height, width = frame.shape[:2]
            visible_points = [
                point
                for point, vis in zip(
                    coords[frame_idx],
                    visibility[frame_idx],
                    strict=True,
                )
                if vis > 0
            ]
            x, y = rng.choice(visible_points)
            mask_width = int(round(width * rng.uniform(*self.mask_width_ratio_range)))
            mask_height = int(round(height * rng.uniform(*self.mask_height_ratio_range)))
            mask_width = max(1, min(mask_width, width))
            mask_height = max(1, min(mask_height, height))

            center_x = int(round(x))
            center_y = int(round(y))
            x0 = max(0, center_x - mask_width // 2)
            y0 = max(0, center_y - mask_height // 2)
            x1 = min(width, x0 + mask_width)
            y1 = min(height, y0 + mask_height)
            x0 = max(0, x1 - mask_width)
            y0 = max(0, y1 - mask_height)
            frame[y0:y1, x0:x1] = 0.0

        return out_frames, coords, visibility


class ImageNetNormalizeAugmentation(BaseAugmentation):
    """Apply ImageNet channel normalization with DINO-compatible constants."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", True))
        mean = self.config.get("mean", IMAGENET_MEAN.tolist())
        std = self.config.get("std", IMAGENET_STD.tolist())
        if not isinstance(mean, Sequence) or len(mean) != 3:
            raise ValueError("normalize_imagenet.mean must contain 3 values.")
        if not isinstance(std, Sequence) or len(std) != 3:
            raise ValueError("normalize_imagenet.std must contain 3 values.")
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        if np.any(self.std <= 0.0):
            raise ValueError("normalize_imagenet.std values must be positive.")

    def forward(
        self,
        frames: Frames,
        coords: Coords,
        visibility: Visibility,
        *,
        rng: random.Random,
    ) -> tuple[Frames, Coords, Visibility]:
        """Normalize RGB frames with the same formula as TVF.normalize."""
        del rng
        if not self.enabled:
            return frames, coords, visibility
        out_frames = normalize_frames_imagenet(frames, mean=self.mean, std=self.std)
        return out_frames, coords, visibility


class BallDetectionAugmentation:
    """Compose and apply all configured sequence augmentations."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.transforms: list[BaseAugmentation] = [
            CameraRotationAugmentation(dict(config.get("camera_rotation", {}) or {})),
            HorizontalFlipAugmentation(dict(config.get("horizontal_flip", {}) or {})),
            AffineAugmentation(dict(config.get("affine", {}) or {})),
            ScaleAndCropAugmentation(dict(config.get("scale_and_crop", {}) or {})),
            BrightnessGainAugmentation(dict(config.get("brightness_gain", {}) or {})),
            ContrastAugmentation(dict(config.get("contrast", {}) or {})),
            GammaAugmentation(dict(config.get("gamma", {}) or {})),
            GaussianNoiseAugmentation(dict(config.get("gaussian_noise", {}) or {})),
            GaussianBlurAugmentation(dict(config.get("gaussian_blur", {}) or {})),
            BallAreaZeroMaskAugmentation(dict(config.get("ball_area_zero_mask", {}) or {})),
            ImageNetNormalizeAugmentation(dict(config.get("normalize_imagenet", {}) or {})),
        ]

    @classmethod
    def from_eval_config(
        cls,
        config: dict[str, Any] | None = None,
    ) -> BallDetectionAugmentation | None:
        """Build an eval-only pipeline that keeps deterministic preprocessing."""
        config = config or {}
        normalize_cfg = dict(config.get("normalize_imagenet", {}) or {})
        if not bool(normalize_cfg.get("enabled", False)):
            return None
        return cls(
            {
                "normalize_imagenet": normalize_cfg,
            }
        )

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
        out_coords = [list(frame_coords) for frame_coords in coords]
        out_visibility = [list(frame_visibility) for frame_visibility in visibility]
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
    "BallDetectionAugmentation",
    "BaseAugmentation",
    "AffineAugmentation",
    "BrightnessGainAugmentation",
    "CameraRotationAugmentation",
    "ContrastAugmentation",
    "GammaAugmentation",
    "BallAreaZeroMaskAugmentation",
    "GaussianBlurAugmentation",
    "GaussianNoiseAugmentation",
    "HorizontalFlipAugmentation",
    "ImageNetNormalizeAugmentation",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "ScaleAndCropAugmentation",
    "denormalize_tensor_images_imagenet",
    "make_sample_rng",
    "normalize_frames_imagenet",
    "normalize_tensor_images_imagenet",
]
