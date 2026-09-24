"""Declared RGB normalization shared by checkpoint inference and training."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from src.tasks.ball_detection.model_io.contracts import BallModelIOError


def _triplet(value: object, *, name: str) -> tuple[float, float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
        or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value)
    ):
        raise BallModelIOError(f"{name} must contain three finite numbers.")
    result = tuple(float(x) for x in value)
    if not all(math.isfinite(x) for x in result):
        raise BallModelIOError(f"{name} must contain three finite numbers.")
    return result[0], result[1], result[2]


@dataclass(frozen=True, slots=True)
class BallImageNormalization:
    """The saved transform applied before RGB layout conversion or MDD."""

    enabled: bool = False
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise BallModelIOError("normalize_imagenet.enabled must be a boolean.")
        mean = _triplet(self.mean, name="normalize_imagenet.mean")
        std = _triplet(self.std, name="normalize_imagenet.std")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        if any(x <= 0 for x in std):
            raise BallModelIOError("normalize_imagenet.std must be positive.")

    @classmethod
    def from_config(cls, config: object) -> BallImageNormalization:
        """Read only the explicitly saved, inference-critical data transform."""
        node = config
        for key in ("data", "augmentation", "normalize_imagenet"):
            if not isinstance(node, Mapping) or key not in node:
                raise BallModelIOError(
                    "Checkpoint/config must explicitly declare "
                    "data.augmentation.normalize_imagenet.enabled."
                )
            node = node[key]
        if not isinstance(node, Mapping) or type(node.get("enabled")) is not bool:
            raise BallModelIOError("normalize_imagenet.enabled must be a boolean.")
        if node["enabled"]:
            return cls(
                enabled=True,
                mean=_triplet(node.get("mean"), name="normalize_imagenet.mean"),
                std=_triplet(node.get("std"), name="normalize_imagenet.std"),
            )
        # An explicit disabled transform is identity regardless of unused stats.
        return cls()

    def apply(self, images: Tensor) -> Tensor:
        """Transform validated raw BTCHW RGB, exactly once."""
        if not self.enabled:
            return images
        mean = images.new_tensor(self.mean).view(1, 1, 3, 1, 1)
        std = images.new_tensor(self.std).view(1, 1, 3, 1, 1)
        return (images - mean) / std

    def validate_preprocessed_range(self, images: Tensor) -> None:
        """Check dataset-preprocessed RGB against the declared channel bounds."""
        if not self.enabled:
            if bool(torch.any((images < 0.0) | (images > 1.0))):
                raise BallModelIOError("images values must be in [0, 1].")
            return
        mean = images.new_tensor(self.mean).view(1, 1, 3, 1, 1)
        std = images.new_tensor(self.std).view(1, 1, 3, 1, 1)
        lower, upper = -mean / std, (1.0 - mean) / std
        if bool(torch.any((images < lower - 1e-5) | (images > upper + 1e-5))):
            raise BallModelIOError(
                "Preprocessed images exceed the declared RGB normalization bounds."
            )


IDENTITY_NORMALIZATION = BallImageNormalization()
