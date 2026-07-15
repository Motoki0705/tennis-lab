"""Shared synthetic court-line map generation and extraction for 3D tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.utils.geometry.line_segments import (
    LineExtractionResult,
    RansacLineConfig,
    extract_line_segments,
)
from src.utils.schema.court import COURT_SKELETON


@dataclass(frozen=True)
class CourtLineMapAugmentationConfig:
    """Map-space corruptions applied before the shared RANSAC extractor."""

    enabled: bool = True
    line_width_range: tuple[int, int] = (1, 3)
    partial_erasure_prob: float = 0.8
    max_partial_erasures: int = 5
    occlusion_prob: float = 0.7
    max_occlusions: int = 3
    false_positive_prob: float = 0.5
    max_false_positive_lines: int = 3
    blur_prob: float = 0.3
    morphology_prob: float = 0.4
    far_dropout_prob: float = 0.3
    near_only_prob: float = 0.15

    def __post_init__(self) -> None:
        lo, hi = self.line_width_range
        if lo <= 0 or lo > hi:
            raise ValueError("line_width_range must contain positive ordered integers.")
        if self.max_partial_erasures < 0 or self.max_occlusions < 0:
            raise ValueError("erasure and occlusion counts must be non-negative.")
        if self.max_false_positive_lines < 0:
            raise ValueError("max_false_positive_lines must be non-negative.")
        for name in (
            "partial_erasure_prob",
            "occlusion_prob",
            "false_positive_prob",
            "blur_prob",
            "morphology_prob",
            "far_dropout_prob",
            "near_only_prob",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")


@dataclass(frozen=True)
class CourtLineInputConfig:
    """Configuration for converting projected CourtKP20 into line inputs."""

    map_width: int = 160
    map_height: int = 90
    temporal_variants: int = 1
    extractor: RansacLineConfig = RansacLineConfig()
    augmentation: CourtLineMapAugmentationConfig = CourtLineMapAugmentationConfig()

    def __post_init__(self) -> None:
        if self.map_width <= 1 or self.map_height <= 1:
            raise ValueError("map_width and map_height must exceed 1.")
        if self.temporal_variants <= 0:
            raise ValueError("temporal_variants must be positive.")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> CourtLineInputConfig:
        """Build a strict typed config from a Hydra-compatible mapping."""
        map_size = config.get("map_size", [160, 90])
        if not isinstance(map_size, (list, tuple)) or len(map_size) != 2:
            raise ValueError("court_line.map_size must be [width, height].")
        extractor_cfg = _mapping(config.get("ransac", {}), "court_line.ransac")
        preprocess_cfg = _mapping(config.get("preprocess", {}), "court_line.preprocess")
        augmentation_cfg = _mapping(
            config.get("augmentation", {}), "court_line.augmentation"
        )
        line_width = augmentation_cfg.get("line_width_range", [1, 3])
        if not isinstance(line_width, (list, tuple)) or len(line_width) != 2:
            raise ValueError(
                "court_line.augmentation.line_width_range must be [min, max]."
            )
        max_lines = int(config.get("max_lines", extractor_cfg.get("max_lines", 12)))
        return cls(
            map_width=int(map_size[0]),
            map_height=int(map_size[1]),
            temporal_variants=int(config.get("temporal_variants", 1)),
            extractor=RansacLineConfig(
                probability_threshold=float(config.get("probability_threshold", 0.5)),
                max_iterations=int(extractor_cfg.get("max_iterations", 96)),
                distance_threshold_px=float(
                    extractor_cfg.get("distance_threshold_px", 1.5)
                ),
                min_inliers=int(extractor_cfg.get("min_inliers", 8)),
                min_segment_length_px=float(
                    extractor_cfg.get("min_segment_length_px", 4.0)
                ),
                max_lines=max_lines,
                skeletonize=bool(preprocess_cfg.get("skeletonize", False)),
                min_component_size=int(preprocess_cfg.get("min_component_size", 5)),
                max_points=int(preprocess_cfg.get("max_points", 2000)),
            ),
            augmentation=CourtLineMapAugmentationConfig(
                enabled=bool(augmentation_cfg.get("enabled", True)),
                line_width_range=(int(line_width[0]), int(line_width[1])),
                partial_erasure_prob=float(
                    augmentation_cfg.get("partial_erasure_prob", 0.8)
                ),
                max_partial_erasures=int(
                    augmentation_cfg.get("max_partial_erasures", 5)
                ),
                occlusion_prob=float(augmentation_cfg.get("occlusion_prob", 0.7)),
                max_occlusions=int(augmentation_cfg.get("max_occlusions", 3)),
                false_positive_prob=float(
                    augmentation_cfg.get("false_positive_prob", 0.5)
                ),
                max_false_positive_lines=int(
                    augmentation_cfg.get("max_false_positive_lines", 3)
                ),
                blur_prob=float(augmentation_cfg.get("blur_prob", 0.3)),
                morphology_prob=float(augmentation_cfg.get("morphology_prob", 0.4)),
                far_dropout_prob=float(augmentation_cfg.get("far_dropout_prob", 0.3)),
                near_only_prob=float(augmentation_cfg.get("near_only_prob", 0.15)),
            ),
        )


@dataclass(frozen=True)
class CourtLineFrameResult:
    """Rendered map and extracted segments for one camera-time observation."""

    line_map: NDArray[np.uint8]
    extraction: LineExtractionResult


class CourtLineInputBuilder:
    """Create ``(V,T,L,4)`` court-line inputs from projected CourtKP20."""

    def __init__(self, config: CourtLineInputConfig) -> None:
        self.config = config

    def build(
        self,
        court_kp: Tensor,
        *,
        augment: bool,
        rng: np.random.Generator,
    ) -> Tensor:
        """Render, corrupt, and extract line segments for each camera/time chunk."""
        if court_kp.ndim != 4 or court_kp.shape[-2:] != (20, 2):
            raise ValueError(
                f"court_kp must have shape (V, T, 20, 2), got {tuple(court_kp.shape)}."
            )
        num_views, seq_len = (int(court_kp.shape[0]), int(court_kp.shape[1]))
        output = torch.zeros(
            num_views,
            seq_len,
            self.config.extractor.max_lines,
            4,
            dtype=torch.float32,
        )
        variants = min(self.config.temporal_variants if augment else 1, seq_len)
        frame_chunks = np.array_split(np.arange(seq_len), variants)
        court_np = court_kp.detach().cpu().numpy()
        for view_index in range(num_views):
            for frame_indices in frame_chunks:
                if len(frame_indices) == 0:
                    continue
                source_frame = int(frame_indices[0])
                frame = self.build_frame(
                    court_np[view_index, source_frame],
                    augment=augment,
                    rng=rng,
                )
                segments = torch.from_numpy(frame.extraction.segments)
                output[view_index, torch.as_tensor(frame_indices)] = segments
        return output

    def build_frame(
        self,
        court_kp: Tensor | NDArray[np.floating],
        *,
        augment: bool,
        rng: np.random.Generator,
    ) -> CourtLineFrameResult:
        """Render and extract one observation using the training code path."""
        if isinstance(court_kp, Tensor):
            court_array = court_kp.detach().cpu().numpy()
        else:
            court_array = np.asarray(court_kp)
        line_map = render_court_line_map(
            court_array,
            width=self.config.map_width,
            height=self.config.map_height,
            line_width=_sample_line_width(
                self.config.augmentation,
                augment=augment,
                rng=rng,
            ),
        )
        if augment and self.config.augmentation.enabled:
            line_map = augment_court_line_map(
                line_map,
                config=self.config.augmentation,
                rng=rng,
            )
        extraction = extract_line_segments(
            line_map,
            config=self.config.extractor,
            rng=rng,
        )
        return CourtLineFrameResult(line_map=line_map, extraction=extraction)


def render_court_line_map(
    court_kp: NDArray[np.floating],
    *,
    width: int,
    height: int,
    line_width: int = 1,
) -> NDArray[np.uint8]:
    """Rasterize the ground court skeleton from normalized projected keypoints."""
    keypoints = np.asarray(court_kp, dtype=np.float32)
    if keypoints.shape != (20, 2):
        raise ValueError(f"court_kp must have shape (20, 2), got {keypoints.shape}.")
    if not np.isfinite(keypoints).all():
        raise ValueError("court_kp must contain only finite values.")
    if width <= 1 or height <= 1 or line_width <= 0:
        raise ValueError("width, height, and line_width must be positive.")

    line_map: NDArray[np.uint8] = np.zeros((height, width), dtype=np.uint8)
    for start_index, end_index in COURT_SKELETON:
        if start_index >= 14 or end_index >= 14:
            continue
        start = _pixel_point(keypoints[start_index], width=width, height=height)
        end = _pixel_point(keypoints[end_index], width=width, height=height)
        visible, clipped_start, clipped_end = cv2.clipLine(
            (0, 0, width, height), start, end
        )
        if visible:
            cv2.line(
                line_map,
                clipped_start,
                clipped_end,
                color=255,
                thickness=line_width,
                lineType=cv2.LINE_AA,
            )
    return line_map


def augment_court_line_map(
    line_map: NDArray[np.uint8],
    *,
    config: CourtLineMapAugmentationConfig,
    rng: np.random.Generator,
) -> NDArray[np.uint8]:
    """Apply line-predictor-like map corruptions without emitting extra features."""
    output = np.asarray(line_map, dtype=np.uint8).copy()
    if output.ndim != 2:
        raise ValueError(f"line_map must have shape (H, W), got {output.shape}.")
    height, width = output.shape

    if rng.random() < config.partial_erasure_prob and config.max_partial_erasures > 0:
        count = int(rng.integers(1, config.max_partial_erasures + 1))
        for _ in range(count):
            rect_w = int(rng.integers(max(2, width // 32), max(3, width // 8)))
            rect_h = int(rng.integers(max(2, height // 48), max(3, height // 12)))
            x = int(rng.integers(0, max(1, width - rect_w + 1)))
            y = int(rng.integers(0, max(1, height - rect_h + 1)))
            output[y : y + rect_h, x : x + rect_w] = 0

    if rng.random() < config.occlusion_prob and config.max_occlusions > 0:
        count = int(rng.integers(1, config.max_occlusions + 1))
        for _ in range(count):
            rect_w = int(rng.integers(max(4, width // 20), max(5, width // 4)))
            rect_h = int(rng.integers(max(6, height // 10), max(7, height // 2)))
            x = int(rng.integers(0, max(1, width - rect_w + 1)))
            y = int(rng.integers(0, max(1, height - rect_h + 1)))
            output[y : y + rect_h, x : x + rect_w] = 0

    if rng.random() < config.near_only_prob:
        cutoff = int(rng.uniform(0.45, 0.7) * height)
        output[:cutoff] = 0
    elif rng.random() < config.far_dropout_prob:
        cutoff = int(rng.uniform(0.2, 0.45) * height)
        output[:cutoff] = 0

    if (
        rng.random() < config.false_positive_prob
        and config.max_false_positive_lines > 0
    ):
        count = int(rng.integers(1, config.max_false_positive_lines + 1))
        for _ in range(count):
            start = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            end = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            cv2.line(output, start, end, color=255, thickness=1, lineType=cv2.LINE_AA)

    if rng.random() < config.blur_prob:
        output = np.asarray(
            cv2.GaussianBlur(output, (5, 5), sigmaX=float(rng.uniform(0.5, 1.5))),
            dtype=np.uint8,
        )
    if rng.random() < config.morphology_prob:
        kernel: NDArray[np.uint8] = np.ones((3, 3), dtype=np.uint8)
        operation = cv2.MORPH_ERODE if rng.random() < 0.5 else cv2.MORPH_DILATE
        output = np.asarray(
            cv2.morphologyEx(output, operation, kernel, iterations=1), dtype=np.uint8
        )
    return output


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _pixel_point(
    uv: NDArray[np.float32],
    *,
    width: int,
    height: int,
) -> tuple[int, int]:
    limit = max(width, height) * 16
    x = int(np.clip(round(float(uv[0]) * width), -limit, limit))
    y = int(np.clip(round(float(uv[1]) * height), -limit, limit))
    return x, y


def _sample_line_width(
    config: CourtLineMapAugmentationConfig,
    *,
    augment: bool,
    rng: np.random.Generator,
) -> int:
    lo, hi = config.line_width_range
    if not augment or not config.enabled:
        return lo
    return int(rng.integers(lo, hi + 1))


__all__ = [
    "CourtLineFrameResult",
    "CourtLineInputBuilder",
    "CourtLineInputConfig",
    "CourtLineMapAugmentationConfig",
    "augment_court_line_map",
    "render_court_line_map",
]
