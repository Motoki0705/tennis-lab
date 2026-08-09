"""Shared config-aware loading and collation for canonical tracking scenes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.tasks.base.data.lifecycle_slots import (
    LifecycleSlotAssignment,
    pack_lifecycle_slots,
)
from src.tasks.base.data.scene_dataset import SceneDatasetBase, SceneDatasetConfig


def validate_lifecycle_capacity(
    *,
    timeline_config: Any,
    data_config: Any,
    num_queries: int,
) -> None:
    """Reject generation settings that cannot be packed by the dataset."""
    max_concurrent = int(timeline_config["max_concurrent"])
    if max_concurrent > num_queries:
        raise ValueError(
            "generation.timeline.max_concurrent must not exceed "
            f"model.num_queries ({max_concurrent} > {num_queries})."
        )
    generation_gap = int(timeline_config["min_reuse_gap_frames"])
    lifecycle_config = data_config["lifecycle"]
    packing_gap = int(lifecycle_config["min_reuse_gap_frames"])
    if generation_gap < packing_gap:
        raise ValueError(
            "generation.timeline.min_reuse_gap_frames must be at least "
            "data.lifecycle.min_reuse_gap_frames "
            f"({generation_gap} < {packing_gap})."
        )


class CanonicalTrackingDataset(SceneDatasetBase[dict[str, Tensor]]):
    """Config-aware base for fixed-directory BLCS/PLCS tracking datasets."""

    def __init__(
        self,
        *,
        scene_dir: str | Path,
        split_file: str | Path,
        config: Any | None = None,
        augment: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        if config is None:
            raise ValueError("CanonicalTrackingDataset requires a validated config.")
        self.hydra_cfg = config
        self.augment = augment
        data_cfg = self._resolve_data_cfg(self.hydra_cfg)
        seq_len_range = self._parse_int_range(data_cfg, "seq_len_range")
        num_views_range = self._parse_int_range(data_cfg, "num_views_range")
        lifecycle_cfg = data_cfg["lifecycle"]
        self.pack_to_query_slots = bool(lifecycle_cfg["pack_to_query_slots"])
        self.min_reuse_gap_frames = int(lifecycle_cfg["min_reuse_gap_frames"])
        self.randomize_slots_train = bool(lifecycle_cfg["randomize_slots_train"])
        model_cfg = self.hydra_cfg["model"]
        raw_num_queries = model_cfg["num_queries"]
        self.num_queries = int(raw_num_queries) if raw_num_queries is not None else None
        if self.min_reuse_gap_frames < 0:
            raise ValueError(
                "data.lifecycle.min_reuse_gap_frames must be non-negative."
            )

        super().__init__(
            config=SceneDatasetConfig(
                scene_dir=Path(scene_dir),
                split_file=Path(split_file),
                seq_len_range=seq_len_range,
                num_views_range=num_views_range,
                camera_mode=self._parse_camera_mode(data_cfg),
                crop_mode="random" if augment else "center",
                min_num_frames=1,
                min_num_cameras=1,
            ),
            rng=rng,
        )

    def pack_lifecycle(self, physical_presence: Tensor) -> LifecycleSlotAssignment:
        """Pack a clipped physical-track matrix into model query slots."""
        num_physical_tracks = int(physical_presence.shape[1])
        num_slots = self.num_queries or num_physical_tracks
        if not self.pack_to_query_slots:
            if num_physical_tracks > num_slots:
                raise ValueError(
                    "Physical targets exceed model.num_queries while lifecycle "
                    "packing is disabled."
                )
            track_to_slot = torch.arange(
                num_physical_tracks,
                dtype=torch.long,
                device=physical_presence.device,
            )
            target_presence = torch.zeros(
                physical_presence.shape[0],
                num_slots,
                dtype=torch.bool,
                device=physical_presence.device,
            )
            target_instance_id = torch.full(
                (physical_presence.shape[0], num_slots),
                -1,
                dtype=torch.long,
                device=physical_presence.device,
            )
            target_presence[:, :num_physical_tracks] = physical_presence
            physical_ids = torch.arange(
                num_physical_tracks,
                dtype=torch.long,
                device=physical_presence.device,
            ).view(1, -1)
            target_instance_id[:, :num_physical_tracks] = torch.where(
                physical_presence,
                physical_ids.expand_as(physical_presence),
                -1,
            )
            return LifecycleSlotAssignment(
                track_to_slot=track_to_slot,
                target_presence=target_presence,
                target_instance_id=target_instance_id,
            )
        return pack_lifecycle_slots(
            physical_presence,
            num_slots=num_slots,
            min_reuse_gap_frames=self.min_reuse_gap_frames,
            randomize_slots=self.augment and self.randomize_slots_train,
            rng=self.rng,
        )


def pad_and_stack_tracking_batch(
    batch: Sequence[Mapping[str, Tensor]],
    *,
    time_dimensions: Mapping[str, int] | None = None,
    padding_dimensions: Mapping[str, Sequence[int]] | None = None,
    pad_values: Mapping[str, float | int | bool] | None = None,
) -> dict[str, Tensor]:
    """Pad declared variable dimensions independently and stack a batch."""
    if not batch:
        raise ValueError("Cannot collate an empty tracking batch.")
    first_keys = tuple(batch[0])
    if any(tuple(sample) != first_keys for sample in batch[1:]):
        raise ValueError("All tracking samples must contain keys in the same order.")
    dimensions: dict[str, tuple[int, ...]] = {
        key: (dimension,) for key, dimension in (time_dimensions or {}).items()
    }
    for key, dimension_sequence in (padding_dimensions or {}).items():
        dimensions[key] = tuple(int(dimension) for dimension in dimension_sequence)
    fill_values = pad_values or {}

    collated: dict[str, Tensor] = {}
    for key in first_keys:
        values = [sample[key] for sample in batch]
        key_dimensions = dimensions.get(key, ())
        target_sizes = {
            dimension: max(int(value.shape[dimension]) for value in values)
            for dimension in key_dimensions
        }
        padded_values: list[Tensor] = []
        for tensor in values:
            padded = tensor
            for dimension in sorted(key_dimensions):
                pad_size = target_sizes[dimension] - int(padded.shape[dimension])
                if pad_size <= 0:
                    continue
                shape = list(padded.shape)
                shape[dimension] = pad_size
                padding = torch.full(
                    shape,
                    fill_values.get(key, 0),
                    dtype=padded.dtype,
                    device=padded.device,
                )
                padded = torch.cat([padded, padding], dim=dimension)
            padded_values.append(padded)
        collated[key] = torch.stack(padded_values)
    return collated


__all__ = ["CanonicalTrackingDataset", "pad_and_stack_tracking_batch"]
