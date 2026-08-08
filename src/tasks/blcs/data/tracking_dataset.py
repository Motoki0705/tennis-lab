"""Config-aware canonical-scene adapter for lifecycle BLCS tracking."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.synthetic_data_generation.dataset.blcs.assembler import (
    BLCSCompactDatasetReader,
    BLCSTrajectoryIndex,
)
from src.tasks.base.data.canonical_tracking import (
    CanonicalTrackingDataset,
    pad_and_stack_tracking_batch,
)
from src.tasks.blcs.data.tracking_augmentation import (
    BLCSTrackingCandidateAugmentation,
)
from src.utils.schema.court import COURT_COORD_SCALE_XYZ

BLCS_TRACKING_KEYS = (
    "scene_format_version",
    "ball_uv",
    "ball_visible",
    "court_kp",
    "court_vis",
    "frame_mask",
    "view_mask",
    "target_position",
    "target_velocity",
    "target_presence",
    "target_instance_id",
    "target_slot_mask",
    "clean_ball_uv",
    "clean_ball_visible",
    "candidate_gt_index",
)


class BLCSTrackingDataset(CanonicalTrackingDataset):
    """Load ID-ordered objects, pack lifecycle slots, and corrupt observations."""

    def __init__(
        self,
        *,
        dataset_dir: str | Path,
        split: str,
        config: Any,
        augment: bool = False,
    ) -> None:
        super().__init__(config=config, augment=augment)
        self.reader = BLCSCompactDatasetReader(Path(dataset_dir))
        self.index: tuple[BLCSTrajectoryIndex, ...] = self.reader.split_trajectories(
            split
        )
        if not self.index:
            raise ValueError(f"Canonical BLCS split {split!r} is empty.")
        augmentation = self.data_config["augmentation"]
        if not isinstance(augmentation, Mapping):
            raise TypeError("data.augmentation must be a mapping.")
        self.tracking_augmentation = BLCSTrackingCandidateAugmentation(augmentation)
        self._scale = torch.tensor(COURT_COORD_SCALE_XYZ, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        index = self.index[item]
        trajectory = self.reader.materialize_all_views(index.trajectory_id)
        window = self.contiguous_window(index.frame_count)
        position = torch.from_numpy(trajectory.positions_court_m[window]) / self._scale
        velocity = torch.from_numpy(trajectory.velocities_court_mps[window])
        physical_presence = torch.from_numpy(trajectory.present[window])
        num_frames, num_physical = physical_presence.shape
        packing = self.pack_lifecycle(physical_presence)
        uv = torch.from_numpy(trajectory.ball_uv[:, window]).clone()
        visible = torch.from_numpy(trajectory.ball_visible[:, window]).bool()
        uv[~visible] = 0.0
        ordered_object_ids = torch.arange(num_physical).expand(num_frames, -1)
        candidate_index = torch.where(
            visible, ordered_object_ids[None].expand(visible.shape[0], -1, -1), -1
        )
        court = torch.from_numpy(trajectory.court_kp[:, :14])[:, None].expand(
            -1, num_frames, -1, -1
        )
        court_visible = torch.from_numpy(trajectory.court_visible[:, :14])[
            :, None
        ].expand(-1, num_frames, -1)

        sample = {
            "scene_format_version": torch.tensor(3),
            "ball_uv": uv,
            "ball_visible": visible,
            "court_kp": court,
            "court_vis": court_visible,
            "frame_mask": torch.ones(num_frames, dtype=torch.bool),
            "view_mask": torch.ones(uv.shape[0], dtype=torch.bool),
            "target_position": packing.pack_tensor(position, physical_presence),
            "target_velocity": packing.pack_tensor(velocity, physical_presence),
            "target_presence": packing.target_presence,
            "target_instance_id": packing.target_instance_id,
            "target_slot_mask": packing.target_presence.any(0),
            "clean_ball_uv": uv.clone(),
            "clean_ball_visible": visible.clone(),
            "candidate_gt_index": candidate_index,
        }
        return self.augment_sample(sample)

    def augment_sample(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.augment:
            return sample
        augmented: dict[str, Tensor] = self.tracking_augmentation(sample)
        return augmented


def collate_blcs_tracking_batch(
    batch: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Pad variable camera/time/candidate dimensions and stack BLCS scenes."""
    collated: dict[str, Tensor] = pad_and_stack_tracking_batch(
        batch,
        padding_dimensions={
            "ball_uv": (0, 1, 2),
            "ball_visible": (0, 1, 2),
            "court_kp": (0, 1),
            "court_vis": (0, 1),
            "frame_mask": (0,),
            "view_mask": (0,),
            "target_position": (0, 1),
            "target_velocity": (0, 1),
            "target_presence": (0, 1),
            "target_instance_id": (0, 1),
            "target_slot_mask": (0,),
            "clean_ball_uv": (0, 1, 2),
            "clean_ball_visible": (0, 1, 2),
            "candidate_gt_index": (0, 1, 2),
        },
        pad_values={
            "target_instance_id": -1,
            "candidate_gt_index": -1,
        },
    )
    return collated


__all__ = [
    "BLCS_TRACKING_KEYS",
    "BLCSTrackingDataset",
    "collate_blcs_tracking_batch",
]
