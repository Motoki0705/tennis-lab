"""Window-to-sample dataset for supervised ball detection."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.tasks.ball_detection.data.components.augmentation import (
    BallDetectionAugmentation,
    make_sample_rng,
)
from src.tasks.ball_detection.data.types import BallDetectionSample, ClipWindow
from src.utils.data.heatmaps import generate_gaussian_heatmaps

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BallDetectionDataset(Dataset[BallDetectionSample]):
    """Convert source-agnostic temporal windows into model-ready samples.

    Dataset-specific discovery and annotation parsing belong to DataModules.
    """

    def __init__(
        self,
        *,
        windows: Sequence[ClipWindow],
        config: DictConfig | None = None,
        augmentation: BallDetectionAugmentation | None = None,
    ) -> None:
        super().__init__()
        self.config = config or {}
        self.augmentation = augmentation

        data_cfg = self.config.get("data", {}) or {}
        model_cfg = self.config.get("model", {}) or {}

        self.num_frames = int(model_cfg.get("num_frames", 8))
        self.image_size = self._parse_size(
            data_cfg.get("image_size", [288, 512]),
            name="data.image_size",
        )
        self.heatmap_size = self._parse_size(
            data_cfg.get("heatmap_size", [144, 256]),
            name="data.heatmap_size",
        )
        self.sigma_ratio = float(data_cfg.get("sigma_ratio", 0.0066))
        self.max_instances = int(data_cfg.get("max_instances", 8))

        if self.num_frames <= 0:
            raise ValueError("model.num_frames must be positive.")
        if self.sigma_ratio <= 0:
            raise ValueError("data.sigma_ratio must be positive.")
        if self.max_instances <= 0:
            raise ValueError("data.max_instances must be positive.")

        self.windows = tuple(windows)
        if not self.windows:
            raise RuntimeError("No supervised ball detection windows were provided.")
        for window in self.windows:
            if window.start_index < 0:
                raise ValueError("ClipWindow.start_index must be non-negative.")
            if window.start_index + self.num_frames > len(window.frame_names):
                raise ValueError(
                    "ClipWindow does not contain enough frames for "
                    f"model.num_frames={self.num_frames}: {window.clip_dir}"
                )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> BallDetectionSample:
        window = self.windows[index]
        image_h, image_w = self.image_size
        heatmap_h, heatmap_w = self.heatmap_size
        original_w, original_h = window.original_size

        frames_hwc: list[np.ndarray] = []
        coords_image: list[list[tuple[float, float]]] = []
        visibility: list[list[float]] = []

        for offset in range(self.num_frames):
            frame_name = window.frame_names[window.start_index + offset]
            frame_path = window.clip_dir / frame_name
            frames_hwc.append(self._load_frame(frame_path))

            labels = [
                label
                for label in window.labels.get(frame_name, ())
                if label.role != "distractor"
            ]
            if len(labels) > self.max_instances:
                raise ValueError(
                    f"{window.clip_dir} frame={frame_name} has "
                    f"{len(labels)} trainable instances, exceeding "
                    f"data.max_instances={self.max_instances}."
                )
            frame_coords: list[tuple[float, float]] = []
            frame_visibility: list[float] = []
            for label in labels:
                if label.visibility > 0:
                    frame_coords.append(
                        (
                            label.x * image_w / max(original_w, 1),
                            label.y * image_h / max(original_h, 1),
                        )
                    )
                    frame_visibility.append(1.0)
                else:
                    frame_coords.append((0.0, 0.0))
                    frame_visibility.append(0.0)
            coords_image.append(frame_coords)
            visibility.append(frame_visibility)

        if self.augmentation is not None:
            frames_hwc, coords_image, visibility = self.augmentation.forward(
                frames_hwc,
                coords_image,
                visibility,
                rng=make_sample_rng(index),
            )

        image_tensors: list[np.ndarray] = []
        heatmaps: list[np.ndarray] = []
        coords_original: list[list[tuple[float, float]]] = []
        visibility_padded: list[list[float]] = []
        for frame, frame_coords, frame_visibility in zip(
            frames_hwc,
            coords_image,
            visibility,
            strict=True,
        ):
            image_tensors.append(np.transpose(frame, (2, 0, 1)))
            normalized_centers = [
                self._to_normalized_xy(
                    x_img=x_img,
                    y_img=y_img,
                    width=image_w,
                    height=image_h,
                )
                for x_img, y_img in frame_coords
            ]
            if normalized_centers:
                instance_heatmaps = generate_gaussian_heatmaps(
                    size_hw=self.heatmap_size,
                    centers_xy=normalized_centers,
                    sigma_ratio=self.sigma_ratio,
                    visibility=frame_visibility,
                )
                heatmaps.append(instance_heatmaps.amax(dim=0).cpu().numpy())
            else:
                heatmaps.append(np.zeros(self.heatmap_size, dtype=np.float32))

            original_points = [
                (
                    x_img * original_w / max(image_w, 1),
                    y_img * original_h / max(image_h, 1),
                )
                if vis > 0
                else (0.0, 0.0)
                for (x_img, y_img), vis in zip(
                    frame_coords,
                    frame_visibility,
                    strict=True,
                )
            ]
            padded_points = original_points + [(0.0, 0.0)] * (
                self.max_instances - len(original_points)
            )
            frame_visibility_padded = frame_visibility + [0.0] * (
                self.max_instances - len(frame_visibility)
            )
            coords_original.append(padded_points)
            visibility_padded.append(frame_visibility_padded)

        sample: BallDetectionSample = {
            "images": torch.from_numpy(np.stack(image_tensors)).to(torch.float32),
            "heatmaps": torch.from_numpy(np.stack(heatmaps)).to(torch.float32),
            "coords": torch.tensor(coords_original, dtype=torch.float32),
            "visibility": torch.tensor(visibility_padded, dtype=torch.float32),
            "original_size": torch.tensor([original_w, original_h], dtype=torch.float32),
            "heatmap_size": torch.tensor([heatmap_w, heatmap_h], dtype=torch.float32),
            "window_id": f"{window.clip_dir.parent.name}/{window.clip_dir.name}:{window.start_index}",
        }
        return sample

    def _load_frame(self, path: Path) -> np.ndarray:
        image_h, image_w = self.image_size
        image: np.ndarray | None = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Failed to read frame: {path}")
        resized: np.ndarray = cv2.resize(image, (image_w, image_h))
        rgb: np.ndarray = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized: np.ndarray = rgb.astype(np.float32) / 255.0
        return normalized

    @staticmethod
    def _to_normalized_xy(
        *,
        x_img: float,
        y_img: float,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        x_norm = 0.0 if width <= 1 else x_img / float(width - 1)
        y_norm = 0.0 if height <= 1 else y_img / float(height - 1)
        return x_norm, y_norm

    @staticmethod
    def _parse_size(value: Any, *, name: str) -> tuple[int, int]:
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
            or len(value) != 2
        ):
            raise ValueError(f"{name} must be a list or tuple with length 2.")
        return int(value[0]), int(value[1])
