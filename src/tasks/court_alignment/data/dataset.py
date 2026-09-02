"""Procedural binary ground-UV dataset for alignment KP14 training."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.tasks.court_alignment.data.augmentation import (
    AugmentableGroundCourtSample,
    GroundCourtAugmentation,
    GroundCourtAugmentationConfig,
    IdentityAugmentation,
    build_augmentation,
    build_augmentations,
)
from src.tasks.court_alignment.data.splits import (
    GroundCourtSplit,
    GroundCourtSplitConfig,
    stable_sample_seed,
)
from src.tasks.court_alignment.geometry.court import (
    GROUND_COURT_KP14_COUNT,
    GroundCourtInstance,
    court_doubles_footprint_for_instance,
    court_keypoints_for_instance,
    doubles_footprints_overlap,
)
from src.tasks.court_alignment.geometry.rasterization import (
    render_center_vote_targets,
    render_court_line_mask,
    render_keypoint_heatmaps,
)

GroundCourtAugmentationSpec: TypeAlias = (
    GroundCourtAugmentationConfig | Mapping[str, object] | str
)


@dataclass(frozen=True, slots=True)
class GroundCourtDatasetConfig:
    """Typed procedural data settings.

    ``scale_px_per_metre`` and all coordinates are in the full-resolution
    output pixel frame.  Keeping the renderer at full resolution is a
    deliberate contract for the small-sigma KP ablation.
    """

    image_size: tuple[int, int] | int = (256, 256)
    max_courts: int = 2
    split: GroundCourtSplitConfig = GroundCourtSplitConfig()
    sigma_px: float = 1.0
    line_width_px: float = 1.0
    vote_radius_px: float = 3.0
    min_courts: int = 1
    scale_px_per_metre_range: tuple[float, float] = (3.0, 6.0)
    rotation_seam_margin_rad: float = 0.05
    rotation_rad_range: tuple[float, float] = (0.05, math.pi - 0.05)
    min_center_distance_px: float = 32.0
    footprint_overlap_tolerance_px: float = 0.0
    max_sampling_attempts: int = 64
    court_margin_px: float = 0.0
    augmentation: GroundCourtAugmentationConfig = GroundCourtAugmentationConfig()
    augmentations: Sequence[GroundCourtAugmentationSpec] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_size", _size_tuple(self.image_size))
        if isinstance(self.max_courts, bool) or not isinstance(self.max_courts, int):
            raise TypeError("max_courts must be an integer.")
        if self.max_courts <= 0:
            raise ValueError("max_courts must be positive.")
        if isinstance(self.min_courts, bool) or not isinstance(self.min_courts, int):
            raise TypeError("min_courts must be an integer.")
        if self.min_courts <= 0 or self.min_courts > self.max_courts:
            raise ValueError("min_courts must lie in [1,max_courts].")
        _validate_positive(self.sigma_px, "sigma_px")
        _validate_positive(self.line_width_px, "line_width_px")
        _validate_positive(self.vote_radius_px, "vote_radius_px")
        _validate_positive(self.rotation_seam_margin_rad, "rotation_seam_margin_rad")
        if not math.isfinite(float(self.court_margin_px)) or self.court_margin_px < 0.0:
            raise ValueError("court_margin_px must be finite and non-negative.")
        _validate_range(
            self.scale_px_per_metre_range,
            name="scale_px_per_metre_range",
            positive=True,
        )
        _validate_rotation_range(
            self.rotation_rad_range,
            seam_margin_rad=self.rotation_seam_margin_rad,
        )
        if (
            not math.isfinite(float(self.min_center_distance_px))
            or self.min_center_distance_px < 0.0
        ):
            raise ValueError("min_center_distance_px must be finite and non-negative.")
        if (
            not math.isfinite(float(self.footprint_overlap_tolerance_px))
            or self.footprint_overlap_tolerance_px < 0.0
        ):
            raise ValueError(
                "footprint_overlap_tolerance_px must be finite and non-negative."
            )
        if (
            isinstance(self.max_sampling_attempts, bool)
            or not isinstance(self.max_sampling_attempts, int)
            or self.max_sampling_attempts <= 0
        ):
            raise ValueError("max_sampling_attempts must be a positive integer.")
        if not isinstance(self.split, GroundCourtSplitConfig):
            raise TypeError("split must be GroundCourtSplitConfig.")
        if not isinstance(self.augmentation, GroundCourtAugmentationConfig):
            raise TypeError("augmentation must be GroundCourtAugmentationConfig.")
        if isinstance(self.augmentations, (str, bytes)) or not isinstance(
            self.augmentations, Sequence
        ):
            raise TypeError("augmentations must be an ordered sequence.")
        normalized_augmentations: list[GroundCourtAugmentationConfig] = []
        for item in self.augmentations:
            if isinstance(item, GroundCourtAugmentationConfig):
                normalized_augmentations.append(item)
            elif isinstance(item, Mapping):
                normalized_augmentations.append(
                    GroundCourtAugmentationConfig.from_mapping(item)
                )
            elif isinstance(item, str):
                normalized_augmentations.append(
                    GroundCourtAugmentationConfig(name=item)
                )
            else:
                raise TypeError(
                    "augmentations must contain typed configs, mappings, or names."
                )
        object.__setattr__(self, "augmentations", tuple(normalized_augmentations))


def _size_tuple(image_size: tuple[int, int] | int) -> tuple[int, int]:
    """Normalize square integer and explicit ``(height,width)`` settings."""

    if isinstance(image_size, bool):
        raise TypeError("image_size must be a positive integer or pair.")
    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        return image_size, image_size
    if len(image_size) != 2 or any(int(value) <= 0 for value in image_size):
        raise ValueError("image_size must contain two positive dimensions.")
    return int(image_size[0]), int(image_size[1])


def _validate_positive(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")


def _validate_range(
    value: tuple[float, float], *, name: str, positive: bool = False
) -> None:
    if len(value) != 2:
        raise ValueError(f"{name} must contain two values.")
    low, high = (float(item) for item in value)
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"{name} must be finite and ordered.")
    if positive and low <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _validate_rotation_range(
    value: tuple[float, float], *, seam_margin_rad: float
) -> None:
    """Require a non-empty axial-angle interval inside configured seam bounds."""

    _validate_range(value, name="rotation_rad_range")
    margin = float(seam_margin_rad)
    if not math.isfinite(margin) or margin <= 0.0 or margin >= math.pi / 2.0:
        raise ValueError("rotation_seam_margin_rad must lie in (0, pi/2).")
    low, high = (float(item) for item in value)
    lower_bound = margin
    upper_bound = math.pi - margin
    if low < lower_bound or high > upper_bound:
        raise ValueError(
            "rotation_rad_range must span at most pi and stay inside configured seam bounds "
            f"[{lower_bound:.9g}, {upper_bound:.9g}]."
        )
    if high - low <= 1.0e-8:
        raise ValueError("rotation_rad_range span must be greater than 1e-8 rad.")


def _sample_instances(
    config: GroundCourtDatasetConfig, rng: np.random.Generator
) -> tuple[GroundCourtInstance, ...]:
    count = int(rng.integers(config.min_courts, config.max_courts + 1))
    height, width = _size_tuple(config.image_size)
    scale_low, scale_high = config.scale_px_per_metre_range
    angle_low, angle_high = config.rotation_rad_range
    instances: list[GroundCourtInstance] = []
    for instance_id in range(count):
        accepted = False
        for _attempt in range(config.max_sampling_attempts):
            scale = float(rng.uniform(scale_low, scale_high))
            angle = float(rng.uniform(angle_low, angle_high))
            # Sampling over the complete canvas intentionally includes clipped
            # courts: real ground-UV evidence routinely contains partial courts.
            margin = float(config.court_margin_px)
            x_low, x_high = margin, max(margin, width - 1.0 - margin)
            y_low, y_high = margin, max(margin, height - 1.0 - margin)
            center = (
                float(rng.uniform(x_low, x_high)),
                float(rng.uniform(y_low, y_high)),
            )
            candidate = GroundCourtInstance(
                instance_id=instance_id,
                center_xy_px=center,
                rotation_rad=angle,
                scale_px_per_metre=scale,
            )
            candidate_footprint = court_doubles_footprint_for_instance(candidate)
            if all(
                math.dist(center, existing.center_xy_px)
                >= config.min_center_distance_px
                and not doubles_footprints_overlap(
                    candidate_footprint,
                    court_doubles_footprint_for_instance(existing),
                    tolerance_px=config.footprint_overlap_tolerance_px,
                )
                for existing in instances
            ):
                instances.append(candidate)
                accepted = True
                break
        if not accepted:
            raise ValueError(
                "Unable to sample the requested number of court centers with "
                f"min_center_distance_px={config.min_center_distance_px} "
                "and non-overlapping doubles footprints "
                f"(tolerance_px={config.footprint_overlap_tolerance_px}) "
                f"within max_sampling_attempts={config.max_sampling_attempts}."
            )
    return tuple(instances)


def _pad_geometry(
    points: Tensor,
    visibility: Tensor,
    centers: Tensor,
    instance_ids: Tensor,
    *,
    max_courts: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    count = points.shape[0]
    padded_points = torch.zeros(
        (max_courts, GROUND_COURT_KP14_COUNT, 2), dtype=torch.float32
    )
    padded_visibility = torch.zeros(
        (max_courts, GROUND_COURT_KP14_COUNT), dtype=torch.bool
    )
    padded_centers = torch.zeros((max_courts, 2), dtype=torch.float32)
    padded_ids = torch.full((max_courts,), -1, dtype=torch.long)
    padded_points[:count] = points
    padded_visibility[:count] = visibility
    padded_centers[:count] = centers
    padded_ids[:count] = instance_ids
    return padded_points, padded_visibility, padded_centers, padded_ids


class GroundCourtDataset(Dataset[dict[str, object]]):
    """Deterministic procedural samples for one named split."""

    def __init__(
        self,
        config: GroundCourtDatasetConfig | None = None,
        *,
        split: GroundCourtSplit = "train",
        augmentation: GroundCourtAugmentation | None = None,
    ) -> None:
        self.config = GroundCourtDatasetConfig() if config is None else config
        if not isinstance(self.config, GroundCourtDatasetConfig):
            raise TypeError("config must be GroundCourtDatasetConfig.")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown ground-court split: {split!r}.")
        self.split: GroundCourtSplit = split
        resolved_augmentation: GroundCourtAugmentation
        if split != "train":
            if augmentation is not None and not isinstance(
                augmentation, IdentityAugmentation
            ):
                raise ValueError(
                    "Validation and test ground-court datasets only accept "
                    "IdentityAugmentation; augmentation is train-only."
                )
            resolved_augmentation = IdentityAugmentation()
        elif augmentation is not None:
            resolved_augmentation = augmentation
        elif self.config.augmentations:
            resolved_augmentation = build_augmentations(
                tuple(self.config.augmentations)
            )
        else:
            resolved_augmentation = build_augmentation(self.config.augmentation)
        self.augmentation = resolved_augmentation
        if not callable(self.augmentation):
            raise TypeError("augmentation must be callable.")

    def __len__(self) -> int:
        return int(self.config.split.size(self.split))

    def __getitem__(self, index: int) -> dict[str, object]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index must be an integer.")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        seed = stable_sample_seed(self.config.split.seed, self.split, index)
        rng = np.random.default_rng(seed)
        instances = _sample_instances(self.config, rng)
        image_size = _size_tuple(self.config.image_size)
        points = torch.stack([court_keypoints_for_instance(item) for item in instances])
        centers = torch.tensor(
            [item.center_xy_px for item in instances], dtype=torch.float32
        )
        instance_ids = torch.tensor(
            [item.instance_id for item in instances], dtype=torch.long
        )
        visibility = self._visibility(points)
        image = render_court_line_mask(
            image_size,
            instances,
            line_width_px=self.config.line_width_px,
        )
        rendered = AugmentableGroundCourtSample(
            image=image,
            keypoints=points,
            visibility=visibility,
            centers=centers,
            instance_ids=instance_ids,
        )
        augmented = self.augmentation(rendered, rng)
        if not isinstance(augmented, AugmentableGroundCourtSample):
            raise TypeError("augmentation must return AugmentableGroundCourtSample.")
        if augmented.keypoints.shape[0] != len(instances):
            raise ValueError(
                "Augmentation cannot change the number of court instances."
            )
        if augmented.keypoints.shape[0] > self.config.max_courts:
            raise ValueError("Augmentation returned more courts than max_courts.")
        if augmented.image.shape != (1, *image_size):
            raise ValueError("Augmentation must preserve the configured image shape.")
        if not augmented.image.is_floating_point() or not bool(
            torch.isfinite(augmented.image).all()
        ):
            raise ValueError("Augmented image must be finite floating point.")
        if bool(torch.any((augmented.image < 0.0) | (augmented.image > 1.0))):
            raise ValueError("Augmented image values must remain in [0,1].")
        padded_points, padded_visibility, padded_centers, padded_ids = _pad_geometry(
            augmented.keypoints,
            augmented.visibility,
            augmented.centers,
            augmented.instance_ids,
            max_courts=self.config.max_courts,
        )
        target_heatmaps = render_keypoint_heatmaps(
            image_size,
            padded_points,
            padded_visibility,
            sigma_px=self.config.sigma_px,
        )
        target_center_votes, target_center_vote_mask = render_center_vote_targets(
            image_size,
            padded_points,
            padded_centers,
            padded_visibility,
            sigma_px=self.config.sigma_px,
            vote_radius_px=self.config.vote_radius_px,
        )
        return {
            "image": augmented.image.float().contiguous(),
            "target_heatmaps": target_heatmaps.float().contiguous(),
            "target_center_votes": target_center_votes.float().contiguous(),
            "target_center_vote_mask": target_center_vote_mask.contiguous(),
            "keypoints": padded_points.contiguous(),
            "visibility": padded_visibility.contiguous(),
            "centers": padded_centers.contiguous(),
            "num_courts": torch.tensor(len(instances), dtype=torch.long),
            "instance_ids": padded_ids.contiguous(),
            "sample_id": f"{self.split}-{index:08d}",
        }

    def _visibility(self, points: Tensor) -> Tensor:
        height, width = _size_tuple(self.config.image_size)
        return (
            (points[..., 0] >= 0.0)
            & (points[..., 0] <= float(width - 1))
            & (points[..., 1] >= 0.0)
            & (points[..., 1] <= float(height - 1))
        )


def build_ground_court_datasets(
    config: GroundCourtDatasetConfig | None = None,
) -> dict[GroundCourtSplit, GroundCourtDataset]:
    """Construct all deterministic splits from one shared configuration."""

    resolved = GroundCourtDatasetConfig() if config is None else config
    datasets: dict[GroundCourtSplit, GroundCourtDataset] = {}
    for split in ("train", "val", "test"):
        if resolved.split.size(split) <= 0:
            continue
        datasets[split] = GroundCourtDataset(
            resolved,
            split=split,
        )
    return datasets


__all__ = [
    "GroundCourtDataset",
    "GroundCourtDatasetConfig",
    "build_ground_court_datasets",
]
