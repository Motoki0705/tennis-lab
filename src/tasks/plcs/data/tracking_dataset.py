"""Config-aware canonical-scene adapter for lifecycle PLCS tracking."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.synthetic_data_generation.dataset.plcs.validation import (
    PLCSCompactDatasetReader,
    PLCSSceneIndex,
)
from src.tasks.base.data.canonical_tracking import (
    CanonicalTrackingDataset,
    pad_and_stack_tracking_batch,
)
from src.tasks.plcs.data.tracking_augmentation import (
    PLCSTrackingDetectionAugmentation,
)

PLCS_TRACKING_KEYS = (
    "scene_format_version",
    "human_kp",
    "human_vis",
    "detection_mask",
    "court_kp",
    "court_vis",
    "frame_mask",
    "view_mask",
    "target_position",
    "target_rotation",
    "target_canonical_pose_3d",
    "target_human_kp_3d",
    "target_presence",
    "target_instance_id",
    "target_slot_mask",
    "clean_human_kp",
    "clean_human_visible",
    "detection_gt_index",
)


class PLCSTrackingDataset(CanonicalTrackingDataset):
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
        self.reader = PLCSCompactDatasetReader(Path(dataset_dir))
        self.index: tuple[PLCSSceneIndex, ...] = self.reader.split_scenes(split)
        if not self.index:
            raise ValueError(f"Canonical PLCS split {split!r} is empty.")
        augmentation = self.data_config["augmentation"]
        if not isinstance(augmentation, Mapping):
            raise TypeError("data.augmentation must be a mapping.")
        self.tracking_augmentation = PLCSTrackingDetectionAugmentation(augmentation)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        index = self.index[item]
        scene = self.reader.materialize_all_views(index.scene_id).supervision
        window = self.contiguous_window(index.frame_count)
        position = torch.from_numpy(scene.position[window])
        rotation = torch.from_numpy(scene.rotation[window])
        canonical_pose = torch.from_numpy(scene.canonical_pose_3d[window])
        world_joints = torch.from_numpy(scene.human_kp_3d[window])
        physical_presence = torch.from_numpy(scene.present[window])
        num_frames, num_physical = physical_presence.shape
        packing = self.pack_lifecycle(physical_presence)
        human_kp = torch.from_numpy(scene.human_kp[window]).permute(1, 0, 2, 3, 4)
        human_vis = torch.from_numpy(scene.human_vis[window]).permute(1, 0, 2, 3)
        court_kp = torch.from_numpy(scene.court_kp[window, :, :14]).permute(1, 0, 2, 3)
        court_vis = torch.from_numpy(scene.court_vis[window, :, :14]).permute(1, 0, 2)
        ordered_object_ids = torch.arange(num_physical).expand(num_frames, -1)
        detection_mask = human_vis.any(-1)
        detection_index = torch.where(
            detection_mask,
            ordered_object_ids[None].expand(human_vis.shape[0], -1, -1),
            -1,
        )
        rotation_fill = torch.tensor([1.0, 0.0], dtype=rotation.dtype)
        sample = {
            "scene_format_version": torch.tensor(3),
            "human_kp": human_kp,
            "human_vis": human_vis,
            "detection_mask": detection_mask,
            "court_kp": court_kp,
            "court_vis": court_vis,
            "frame_mask": torch.ones(num_frames, dtype=torch.bool),
            "view_mask": torch.ones(human_kp.shape[0], dtype=torch.bool),
            "target_position": packing.pack_tensor(position, physical_presence),
            "target_rotation": packing.pack_tensor(
                rotation,
                physical_presence,
                fill_value=rotation_fill,
            ),
            "target_canonical_pose_3d": packing.pack_tensor(
                canonical_pose, physical_presence
            ),
            "target_human_kp_3d": packing.pack_tensor(world_joints, physical_presence),
            "target_presence": packing.target_presence,
            "target_instance_id": packing.target_instance_id,
            "target_slot_mask": packing.target_presence.any(0),
            "clean_human_kp": human_kp.clone(),
            "clean_human_visible": human_vis.clone(),
            "detection_gt_index": detection_index,
        }
        return self.augment_sample(sample)

    def augment_sample(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.augment:
            return sample
        return _tensor_mapping(self.tracking_augmentation(sample))


def _tensor_mapping(value: object) -> dict[str, Tensor]:
    """Validate the dynamically typed augmentation boundary."""
    if not isinstance(value, dict):
        raise TypeError("PLCS tracking augmentation must return a dictionary.")
    result: dict[str, Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, Tensor):
            raise TypeError("PLCS tracking augmentation returned invalid entries.")
        result[key] = tensor
    return result


def collate_plcs_tracking_batch(
    batch: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Pad variable camera/time/detection dimensions and stack PLCS scenes."""
    return pad_and_stack_tracking_batch(
        batch,
        padding_dimensions={
            "human_kp": (0, 1, 2),
            "human_vis": (0, 1, 2),
            "detection_mask": (0, 1, 2),
            "court_kp": (0, 1),
            "court_vis": (0, 1),
            "frame_mask": (0,),
            "view_mask": (0,),
            "target_position": (0, 1),
            "target_rotation": (0, 1),
            "target_canonical_pose_3d": (0, 1),
            "target_human_kp_3d": (0, 1),
            "target_presence": (0, 1),
            "target_instance_id": (0, 1),
            "target_slot_mask": (0,),
            "clean_human_kp": (0, 1, 2),
            "clean_human_visible": (0, 1, 2),
            "detection_gt_index": (0, 1, 2),
        },
        pad_values={
            "target_instance_id": -1,
            "detection_gt_index": -1,
        },
    )


__all__ = [
    "PLCS_TRACKING_KEYS",
    "PLCSTrackingDataset",
    "collate_plcs_tracking_batch",
]
