"""Lightning DataModule wrapper around the deterministic ground-court dataset."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.tasks.court_alignment.data.augmentation import (
    GroundCourtAugmentationConfig,
    build_augmentations,
)
from src.tasks.court_alignment.data.dataset import (
    GroundCourtDataset,
    GroundCourtDatasetConfig,
)
from src.tasks.court_alignment.data.splits import GroundCourtSplitConfig


def _size_tuple(image_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(image_size, bool):
        raise TypeError("image_size must be a positive integer or pair.")
    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        return image_size, image_size
    values = tuple(int(value) for value in image_size)
    if len(values) != 2 or any(value <= 0 for value in values):
        raise ValueError("image_size must be a positive integer or pair.")
    return values


def _configs(
    augmentations: Sequence[GroundCourtAugmentationConfig | Mapping[str, object] | str]
    | None,
) -> tuple[GroundCourtAugmentationConfig, ...]:
    if augmentations is None:
        return ()
    result: list[GroundCourtAugmentationConfig] = []
    for item in augmentations:
        if isinstance(item, GroundCourtAugmentationConfig):
            result.append(item)
        elif isinstance(item, Mapping):
            result.append(GroundCourtAugmentationConfig.from_mapping(item))
        elif isinstance(item, str):
            result.append(GroundCourtAugmentationConfig(name=item))
        else:
            raise TypeError(
                "Each augmentation must be a typed config, mapping, or name."
            )
    return tuple(result)


class GroundCourtDataModule(pl.LightningDataModule):
    """Build train/validation/test procedural datasets with fixed shapes.

    The constructor intentionally accepts flat Hydra-friendly primitives.  A
    single typed :class:`GroundCourtDatasetConfig` is then shared by all
    splits, guaranteeing that only split name and sample index affect the
    deterministic random seed.
    """

    def __init__(
        self,
        *,
        image_size: int | Sequence[int] = 256,
        train_samples: int = 10_000,
        val_samples: int = 1_000,
        test_samples: int = 1_000,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        min_courts: int = 1,
        max_courts: int = 2,
        sigma_px: float = 1.0,
        line_width_px: float = 1.0,
        vote_radius_px: float = 3.0,
        min_scale_px_per_metre: float = 3.0,
        max_scale_px_per_metre: float = 6.0,
        rotation_seam_margin_rad: float = 0.05,
        rotation_rad_range: Sequence[float] = (0.05, math.pi - 0.05),
        min_center_distance_px: float = 32.0,
        footprint_overlap_tolerance_px: float = 0.0,
        max_sampling_attempts: int = 64,
        court_margin_px: float = 0.0,
        seed: int = 42,
        augmentation: GroundCourtAugmentationConfig
        | Mapping[str, object]
        | str
        | None = None,
        augmentations: Sequence[
            GroundCourtAugmentationConfig | Mapping[str, object] | str
        ]
        | None = None,
    ) -> None:
        super().__init__()
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if type(num_workers) is not int or num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer.")
        if not isinstance(pin_memory, bool):
            raise TypeError("pin_memory must be boolean.")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        if augmentation is not None and augmentations is not None:
            raise ValueError("Specify either augmentation or augmentations, not both.")
        if augmentation is None:
            self._augmentation_configs = _configs(augmentations)
        else:
            self._augmentation_configs = _configs((augmentation,))
        # Resolve the registry while constructing the DataModule so a typo in
        # a Hydra transform cannot survive until the first training batch.
        self._train_augmentation = build_augmentations(self._augmentation_configs)
        rotation_range = tuple(float(value) for value in rotation_rad_range)
        if len(rotation_range) != 2:
            raise ValueError("rotation_rad_range must contain two values.")
        self.dataset_config = GroundCourtDatasetConfig(
            image_size=_size_tuple(image_size),
            max_courts=max_courts,
            min_courts=min_courts,
            split=GroundCourtSplitConfig(
                train_size=train_samples,
                val_size=val_samples,
                test_size=test_samples,
                seed=seed,
            ),
            sigma_px=sigma_px,
            line_width_px=line_width_px,
            vote_radius_px=vote_radius_px,
            scale_px_per_metre_range=(
                min_scale_px_per_metre,
                max_scale_px_per_metre,
            ),
            rotation_seam_margin_rad=rotation_seam_margin_rad,
            rotation_rad_range=rotation_range,
            min_center_distance_px=min_center_distance_px,
            footprint_overlap_tolerance_px=footprint_overlap_tolerance_px,
            max_sampling_attempts=max_sampling_attempts,
            court_margin_px=court_margin_px,
            augmentations=self._augmentation_configs,
        )
        if train_samples <= 0:
            raise ValueError("train_samples must be positive.")
        self.train_dataset: GroundCourtDataset | None = None
        self.val_dataset: GroundCourtDataset | None = None
        self.test_dataset: GroundCourtDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Materialize only requested splits; all are cheap procedural views."""

        if stage in (None, "fit"):
            self.train_dataset = GroundCourtDataset(
                self.dataset_config,
                split="train",
                augmentation=self._train_augmentation,
            )
            self.val_dataset = GroundCourtDataset(
                self.dataset_config,
                split="val",
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = GroundCourtDataset(
                self.dataset_config,
                split="test",
            )

    def _loader(self, dataset: GroundCourtDataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("fit")
        assert self.val_dataset is not None
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup("test")
        assert self.test_dataset is not None
        return self._loader(self.test_dataset, shuffle=False)


# Configuration-facing alias; the implementation and its contract are shared.
CourtAlignmentDataModule = GroundCourtDataModule


__all__ = ["CourtAlignmentDataModule", "GroundCourtDataModule"]
