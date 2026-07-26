"""Config-aware canonical-scene adapter for lifecycle PLCS tracking."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from src.tasks.base.data.canonical_tracking import (
    CanonicalTrackingDataset,
    pad_and_stack_tracking_batch,
)
from src.tasks.base.data.scene_dataset import Scene
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        data_cfg = self._resolve_data_cfg(self.hydra_cfg)
        self.tracking_augmentation = PLCSTrackingDetectionAugmentation(
            data_cfg.get("augmentation", {})
        )

    def build_sample(self, scene: Scene) -> dict[str, Tensor]:
        position = torch.from_numpy(scene.get_array("position")).float()
        rotation = torch.from_numpy(scene.get_array("rotation")).float()
        canonical_pose = torch.from_numpy(scene.get_array("canonical_pose_3d")).float()
        world_joints = torch.from_numpy(scene.get_array("human_kp_3d")).float()
        if position.ndim == 2:
            position = position[:, None]
            rotation = rotation[:, None]
            canonical_pose = canonical_pose[:, None]
            world_joints = world_joints[:, None]
        num_frames, num_physical = position.shape[:2]
        if scene.has_key("person_present"):
            physical_presence = torch.from_numpy(
                scene.get_array("person_present")
            ).bool()
        else:
            physical_presence = torch.ones((num_frames, num_physical), dtype=torch.bool)
        window = self.select_window(scene, full_len=num_frames)
        cameras = self.select_cameras(scene)
        position = position[window.sl]
        rotation = rotation[window.sl]
        canonical_pose = canonical_pose[window.sl]
        world_joints = world_joints[window.sl]
        physical_presence = physical_presence[window.sl]
        packing = self.pack_lifecycle(physical_presence)

        kp_rows: list[Tensor] = []
        visible_rows: list[Tensor] = []
        index_rows: list[Tensor] = []
        clean_kp_rows: list[Tensor] = []
        clean_visible_rows: list[Tensor] = []
        court_rows: list[Tensor] = []
        court_vis_rows: list[Tensor] = []
        ordered_object_ids = torch.arange(num_physical).expand(window.seq_len, -1)
        for camera_index in cameras.indices:
            keypoints = torch.from_numpy(
                scene.get_camera_array(camera_index, "human_kp_uv", window=window)
            ).float()
            visible = torch.from_numpy(
                scene.get_camera_array(camera_index, "human_kp_visible", window=window)
            ).bool()
            if keypoints.ndim == 3:
                keypoints = keypoints[:, None]
                visible = visible[:, None]
            visible &= physical_presence[..., None]
            keypoints[~visible] = 0.0
            clean_kp_rows.append(keypoints.clone())
            clean_visible_rows.append(visible.clone())
            detection_mask = visible.any(-1)
            detection_index = torch.where(detection_mask, ordered_object_ids, -1)
            kp_rows.append(keypoints)
            visible_rows.append(visible)
            index_rows.append(detection_index)
            court_rows.append(
                torch.from_numpy(
                    scene.get_camera_array(camera_index, "court_kp_uv", window=window)[
                        :, :14
                    ]
                ).float()
            )
            court_vis_rows.append(
                torch.from_numpy(
                    scene.get_camera_array(
                        camera_index, "court_kp_visible", window=window
                    )[:, :14]
                ).bool()
            )

        human_kp = torch.stack(kp_rows)
        human_vis = torch.stack(visible_rows)
        rotation_fill = torch.tensor([1.0, 0.0], dtype=rotation.dtype)
        sample = {
            "scene_format_version": torch.tensor(3),
            "human_kp": human_kp,
            "human_vis": human_vis,
            "detection_mask": human_vis.any(-1),
            "court_kp": torch.stack(court_rows),
            "court_vis": torch.stack(court_vis_rows),
            "frame_mask": torch.ones(window.seq_len, dtype=torch.bool),
            "view_mask": torch.ones(len(cameras.indices), dtype=torch.bool),
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
            "clean_human_kp": torch.stack(clean_kp_rows),
            "clean_human_visible": torch.stack(clean_visible_rows),
            "detection_gt_index": torch.stack(index_rows),
        }
        return sample

    def augment_sample(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.augment:
            return sample
        return self.tracking_augmentation(sample)


def collate_plcs_tracking_batch(
    batch: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Pad variable camera/time/detection dimensions and stack PLCS scenes."""
    return cast(
        dict[str, Tensor],
        pad_and_stack_tracking_batch(
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
        ),
    )


__all__ = [
    "PLCS_TRACKING_KEYS",
    "PLCSTrackingDataset",
    "collate_plcs_tracking_batch",
]
