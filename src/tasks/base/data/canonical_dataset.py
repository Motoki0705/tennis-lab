"""Shared lifecycle and contiguous-window utilities for canonical manifests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Generic, TypeVar, cast

import numpy as np
from torch.utils.data import Dataset

SampleT = TypeVar("SampleT")


class CanonicalDataset(Dataset[SampleT], Generic[SampleT]):
    """Task-neutral base that never interprets a legacy scene directory."""

    def __init__(
        self,
        *,
        config: object,
        augment: bool,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not isinstance(config, Mapping):
            raise TypeError("Canonical datasets require a mapping configuration.")
        data = config.get("data")
        if not isinstance(data, Mapping):
            raise TypeError("Canonical datasets require a data mapping.")
        values = data.get("seq_len_range")
        if (
            not isinstance(values, list | tuple)
            or len(values) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in values
            )
        ):
            raise ValueError("data.seq_len_range must contain two integers.")
        lower, upper = (int(value) for value in values)
        if lower <= 0 or upper < lower:
            raise ValueError("data.seq_len_range must be a positive ordered range.")
        self.config = config
        self.data_config = cast(Mapping[str, object], data)
        self.seq_len_range = (lower, upper)
        self.augment = augment
        self.rng = rng if rng is not None else np.random.default_rng()

    def contiguous_window(self, frame_count: int) -> slice:
        """Choose one contiguous crop while preserving source-frame order."""
        if frame_count <= 0:
            raise ValueError("Canonical samples require at least one frame.")
        lower, upper = self.seq_len_range
        if frame_count < lower:
            raise ValueError(
                f"Canonical sample has {frame_count} frames, below required {lower}."
            )
        length = min(frame_count, upper)
        if self.augment and frame_count > length:
            start = int(self.rng.integers(0, frame_count - length + 1))
        else:
            start = (frame_count - length) // 2
        return slice(start, start + length)


__all__ = ["CanonicalDataset"]
