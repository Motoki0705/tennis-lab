"""Stable split and per-example seed utilities for procedural court data."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

GroundCourtSplit = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class GroundCourtSplitConfig:
    """Immutable sizes and root seed for procedural train/val/test splits."""

    train_size: int = 10_000
    val_size: int = 1_000
    test_size: int = 1_000
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("train_size", "val_size", "test_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("split seed must be an integer.")

    def size(self, split: GroundCourtSplit) -> int:
        """Return the configured number of examples for ``split``."""

        if split == "train":
            return self.train_size
        if split == "val":
            return self.val_size
        if split == "test":
            return self.test_size
        raise ValueError(f"Unknown ground-court split: {split!r}.")


def stable_sample_seed(root_seed: int, split: GroundCourtSplit, index: int) -> int:
    """Derive a process/platform-independent uint64 seed for one sample.

    Hashing the split name and index prevents changing validation size or
    iterating another split from changing training examples.  The returned
    value is accepted directly by ``numpy.random.default_rng``.
    """

    if isinstance(root_seed, bool) or not isinstance(root_seed, int):
        raise TypeError("root_seed must be an integer.")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown ground-court split: {split!r}.")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("sample index must be a non-negative integer.")
    payload = f"ground-court-kp14:{root_seed}:{split}:{index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


__all__ = ["GroundCourtSplit", "GroundCourtSplitConfig", "stable_sample_seed"]
