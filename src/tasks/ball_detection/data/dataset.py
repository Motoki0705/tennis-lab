"""Window-to-sample conversion shared by every supervised ball dataset."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from src.tasks.ball_detection.configuration import validate_data
from src.tasks.ball_detection.data.components.augmentation import (
    BallDetectionAugmentation,
    make_sample_rng,
)
from src.tasks.ball_detection.data.types import BallDetectionSample
from src.utils.data.heatmaps import generate_gaussian_heatmaps

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass(frozen=True, slots=True)
class WindowFrame:
    """One decoded frame of a window and its training target.

    ``points`` are the positive balls in original-image pixels. When
    ``supervised`` is False the frame contributes to neither the loss nor the
    metrics, and ``points`` must be empty.
    """

    image_bgr: NDArray[np.uint8]
    points: tuple[tuple[float, float], ...]
    supervised: bool
    frame_id: int
    observed_xy: tuple[float, float] | None

    def __post_init__(self) -> None:
        if not self.supervised and self.points:
            raise ValueError("An unsupervised frame cannot carry positive balls")


@dataclass(frozen=True, slots=True)
class WindowFrames:
    """One fixed-length window, read by a source dataset."""

    frames: tuple[WindowFrame, ...]
    original_size: tuple[int, int]
    window_id: str
    source: str
    namespace: str
    camera: str
    source_scale: float
    window_start: int


class BallDetectionDataset(Dataset[BallDetectionSample], ABC):
    """Turn source windows into model-ready samples.

    Subclasses own window discovery and label semantics and implement
    :meth:`read_window`; this class owns resizing, augmentation, heatmap
    targets and the sample contract.
    """

    def __init__(
        self,
        *,
        config: DictConfig,
        augmentation: BallDetectionAugmentation | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.augmentation = augmentation

        data_cfg = validate_data(config)
        self.num_frames = int(config.model.num_frames)
        self.image_size = self._parse_size(data_cfg["image_size"], name="data.image_size")
        self.heatmap_size = self._parse_size(data_cfg["heatmap_size"], name="data.heatmap_size")
        self.sigma_ratio = float(data_cfg["sigma_ratio"])
        self.max_instances = int(data_cfg["max_instances"])

        if self.num_frames <= 0:
            raise ValueError("model.num_frames must be positive.")
        if self.sigma_ratio <= 0:
            raise ValueError("data.sigma_ratio must be positive.")
        if self.max_instances <= 0:
            raise ValueError("data.max_instances must be positive.")

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def read_window(self, index: int) -> WindowFrames:
        """Read ``model.num_frames`` frames of window ``index``."""

    def __getitem__(self, index: int) -> BallDetectionSample:
        """Build one sample of exactly ``model.num_frames`` consecutive frames."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Ball dataset index must be an integer.")
        window = self.read_window(index)
        if len(window.frames) != self.num_frames:
            raise RuntimeError(
                f"{type(self).__name__}.read_window returned {len(window.frames)} "
                f"frames for model.num_frames={self.num_frames}"
            )
        return self._make_sample(window, index)

    def _make_sample(self, window: WindowFrames, window_index: int) -> BallDetectionSample:
        image_h, image_w = self.image_size
        heatmap_h, heatmap_w = self.heatmap_size
        original_w, original_h = window.original_size

        frames_hwc: list[np.ndarray] = []
        coords_image: list[list[tuple[float, float]]] = []
        visibility: list[list[float]] = []
        for frame in window.frames:
            if frame.image_bgr.shape[:2] != (original_h, original_w):
                raise ValueError(
                    f"{window.window_id}: frame of shape {frame.image_bgr.shape} "
                    f"differs from the window size {window.original_size}"
                )
            if len(frame.points) > self.max_instances:
                raise ValueError(
                    f"{window.window_id} has {len(frame.points)} positive balls in "
                    f"one frame, exceeding data.max_instances={self.max_instances}."
                )
            frames_hwc.append(self._to_model_rgb(frame.image_bgr))
            coords_image.append(
                [(x * image_w / original_w, y * image_h / original_h) for x, y in frame.points]
            )
            visibility.append([1.0] * len(frame.points))

        if self.augmentation is not None:
            frames_hwc, coords_image, visibility = self.augmentation.forward(
                frames_hwc,
                coords_image,
                visibility,
                rng=make_sample_rng(window_index),
            )

        image_tensors: list[np.ndarray] = []
        heatmaps: list[np.ndarray] = []
        coords_original: list[list[tuple[float, float]]] = []
        visibility_padded: list[list[float]] = []
        for frame_hwc, frame_coords, frame_visibility in zip(
            frames_hwc, coords_image, visibility, strict=True
        ):
            image_tensors.append(np.transpose(frame_hwc, (2, 0, 1)))
            if frame_coords:
                normalized_centers = [
                    self._to_normalized_xy(x_img=x, y_img=y, width=image_w, height=image_h)
                    for x, y in frame_coords
                ]
                instance_heatmaps = generate_gaussian_heatmaps(
                    size_hw=self.heatmap_size,
                    centers_xy=torch.tensor(normalized_centers, dtype=torch.float32),
                    sigma_ratio=self.sigma_ratio,
                    visibility=torch.tensor(frame_visibility, dtype=torch.bool),
                )
                heatmaps.append(instance_heatmaps.amax(dim=0).cpu().numpy())
            else:
                heatmaps.append(np.zeros(self.heatmap_size, dtype=np.float32))

            original_points = [
                (x * original_w / image_w, y * original_h / image_h) if vis > 0 else (0.0, 0.0)
                for (x, y), vis in zip(frame_coords, frame_visibility, strict=True)
            ]
            padding = self.max_instances - len(original_points)
            coords_original.append(original_points + [(0.0, 0.0)] * padding)
            visibility_padded.append(list(frame_visibility) + [0.0] * padding)

        sample: BallDetectionSample = {
            "images": torch.from_numpy(np.stack(image_tensors)).to(torch.float32),
            "heatmaps": torch.from_numpy(np.stack(heatmaps)).to(torch.float32),
            "coords": torch.tensor(coords_original, dtype=torch.float32),
            "visibility": torch.tensor(visibility_padded, dtype=torch.float32),
            "supervised": torch.tensor([frame.supervised for frame in window.frames], dtype=torch.bool),
            "original_size": torch.tensor([original_w, original_h], dtype=torch.float32),
            "heatmap_size": torch.tensor([heatmap_w, heatmap_h], dtype=torch.float32),
            "window_id": window.window_id,
            "source": window.source,
            # These raw references are used only in unaugmented validation,
            # independently of the training supervision/augmentation policy.
            "candidate_reference": {
                "xy": torch.tensor([frame.observed_xy if frame.observed_xy is not None else (0.0, 0.0) for frame in window.frames], dtype=torch.float32),
                "observed": torch.tensor([frame.observed_xy is not None for frame in window.frames], dtype=torch.bool),
                "frame_id": torch.tensor([frame.frame_id for frame in window.frames], dtype=torch.int64),
                "window_start": torch.tensor(window.window_start, dtype=torch.int64),
                "source_scale": torch.tensor(window.source_scale, dtype=torch.float32),
                "namespace": window.namespace,
                "camera": window.camera,
            },
        }
        return sample

    def _to_model_rgb(self, image_bgr: NDArray[np.uint8]) -> np.ndarray:
        image_h, image_w = self.image_size
        resized: np.ndarray = cv2.resize(image_bgr, (image_w, image_h))
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
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
            raise ValueError(f"{name} must be a list or tuple with length 2.")
        return int(value[0]), int(value[1])


__all__ = ["BallDetectionDataset", "WindowFrame", "WindowFrames"]
