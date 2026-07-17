"""Shared loading and collation helpers for canonical multi-object scenes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CanonicalTrackingDataset(Dataset[dict[str, Tensor]]):
    """Resolve immutable scene directories from an explicit split file."""

    def __init__(self, *, scene_dir: str | Path, split_file: str | Path) -> None:
        self.scene_dir = Path(scene_dir)
        split_path = Path(split_file)
        if not split_path.is_absolute():
            split_path = self.scene_dir / split_path
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        self.scenes = [
            self.scene_dir / "scenes" / name.strip()
            for name in split_path.read_text(encoding="utf-8").splitlines()
            if name.strip()
        ]
        if not self.scenes:
            raise RuntimeError(f"No scenes found from split file: {split_path}")
        missing = [str(path) for path in self.scenes if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"Scene directories not found: {missing}")

    def __len__(self) -> int:
        return len(self.scenes)


def pad_and_stack_tracking_batch(
    batch: Sequence[Mapping[str, Tensor]],
    *,
    time_dimensions: Mapping[str, int],
) -> dict[str, Tensor]:
    """Pad variable-duration tensor dictionaries and stack a batch."""
    if not batch:
        raise ValueError("Cannot collate an empty tracking batch.")
    max_frames = max(int(sample["frame_mask"].numel()) for sample in batch)
    collated: dict[str, Tensor] = {}
    for key in batch[0]:
        values: list[Tensor] = []
        for sample in batch:
            value = sample[key]
            if key in time_dimensions:
                dimension = time_dimensions[key]
                pad_frames = max_frames - int(value.shape[dimension])
                if pad_frames > 0:
                    shape = list(value.shape)
                    shape[dimension] = pad_frames
                    value = torch.cat(
                        [value, torch.zeros(shape, dtype=value.dtype)], dim=dimension
                    )
            values.append(value)
        collated[key] = torch.stack(values)
    return collated


__all__ = ["CanonicalTrackingDataset", "pad_and_stack_tracking_batch"]
