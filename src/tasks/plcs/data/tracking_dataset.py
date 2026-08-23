"""Config-aware canonical-scene adapter for lifecycle PLCS tracking."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.tasks.base.configuration import CourtCoordinateNormalizationConfig
from src.tasks.base.data.canonical_tracking import (
    CanonicalTrackingDataset,
    pad_and_stack_tracking_batch,
)
from src.tasks.base.data.court_coordinate_contract import (
    validate_dataset_court_coordinate_contract,
)
from src.tasks.base.data.lifecycle_slots import build_fixed_lifecycle_assignment
from src.tasks.base.data.scene_dataset import Scene
from src.tasks.plcs.data.tracking_augmentation import (
    PLCSTrackingDetectionAugmentation,
)
from src.utils.schema.court_normalization import (
    CourtCoordinateNormalization,
    resolve_court_coordinate_normalization,
)

PLCS_TRACKING_KEYS = (
    "scene_format_version",
    "human_kp",
    "human_vis",
    "court_kp",
    "court_vis",
    "padding_mask",
    "target_position",
    "target_rotation",
    "target_canonical_pose_3d",
    "target_human_kp_3d",
    "target_presence",
    "target_instance_id",
    "target_slot_mask",
    "clean_human_kp",
    "clean_human_vis",
    "detection_gt_index",
)


class PLCSTrackingDataset(CanonicalTrackingDataset):
    """Load ID-ordered objects, pack lifecycle slots, and corrupt observations."""

    def __init__(self, **kwargs: Any) -> None:
        config = kwargs.get("config")
        self.court_coordinate_normalization = _tracking_normalization_contract(
            config
        )
        super().__init__(**kwargs)
        validate_dataset_court_coordinate_contract(
            self.scene_dir,
            self.court_coordinate_normalization,
            scene_paths=self.scenes,
        )
        data_cfg = self._resolve_data_cfg(self.hydra_cfg)
        self.tracking_augmentation = PLCSTrackingDetectionAugmentation(
            data_cfg["augmentation"]
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
        if self.num_queries is None:
            raise ValueError("PLCS tracking requires model.num_queries.")
        packing = build_fixed_lifecycle_assignment(
            physical_presence,
            num_slots=self.num_queries,
            min_reuse_gap_frames=self.min_reuse_gap_frames,
            randomize_slots=self.augment and self.randomize_slots_train,
            generator=None,
        )

        kp_rows: list[Tensor] = []
        visible_rows: list[Tensor] = []
        index_rows: list[Tensor] = []
        clean_kp_rows: list[Tensor] = []
        clean_visible_rows: list[Tensor] = []
        court_rows: list[Tensor] = []
        court_vis_rows: list[Tensor] = []
        for camera_index in cameras.indices:
            keypoints = torch.from_numpy(
                scene.get_camera_array(camera_index, "human_kp_uv", window=window)
            ).float()
            visible = torch.from_numpy(
                scene.get_camera_array(camera_index, "human_kp_vis", window=window)
            ).bool()
            if keypoints.ndim == 3:
                keypoints = keypoints[:, None]
                visible = visible[:, None]
            visible &= physical_presence[..., None]
            keypoints[~visible] = 0.0
            packed_keypoints = packing.pack_tensor(keypoints, physical_presence)
            packed_visible = packing.pack_tensor(visible, physical_presence)
            clean_kp_rows.append(packed_keypoints.clone())
            clean_visible_rows.append(packed_visible.clone())
            detection_index = torch.where(
                packed_visible.any(-1), packing.target_instance_id, -1
            )
            kp_rows.append(packed_keypoints)
            visible_rows.append(packed_visible)
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
                        camera_index, "court_kp_vis", window=window
                    )[:, :14]
                ).bool()
            )

        human_kp = torch.stack(kp_rows)
        human_vis = torch.stack(visible_rows)
        rotation_fill = torch.tensor([1.0, 0.0], dtype=rotation.dtype)
        sample = {
            "scene_format_version": torch.tensor(4),
            "human_kp": human_kp,
            "human_vis": human_vis,
            "court_kp": torch.stack(court_rows),
            "court_vis": torch.stack(court_vis_rows),
            "padding_mask": torch.zeros(
                len(cameras.indices), window.seq_len, dtype=torch.bool
            ),
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
            "clean_human_vis": torch.stack(clean_visible_rows),
            "detection_gt_index": torch.stack(index_rows),
        }
        return sample

    def augment_sample(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.augment:
            return sample
        augmented: dict[str, Tensor] = self.tracking_augmentation(sample)
        return augmented


def _tracking_normalization_contract(
    config: object,
) -> CourtCoordinateNormalization:
    """Keep the pre-Hydra direct mapping constructor on the legacy v1 scale."""
    if isinstance(config, dict) and "court_coordinate_normalization" not in config:
        return resolve_court_coordinate_normalization("v1")
    return CourtCoordinateNormalizationConfig.from_config(config).contract


def collate_plcs_tracking_batch(
    batch: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Pad variable camera/time/detection dimensions and stack PLCS scenes."""
    collated: dict[str, Tensor] = pad_and_stack_tracking_batch(
        batch,
        padding_dimensions={
            "human_kp": (0, 1),
            "human_vis": (0, 1),
            "court_kp": (0, 1),
            "court_vis": (0, 1),
            "padding_mask": (0, 1),
            "target_position": (0,),
            "target_rotation": (0,),
            "target_canonical_pose_3d": (0,),
            "target_human_kp_3d": (0,),
            "target_presence": (0,),
            "target_instance_id": (0,),
            "clean_human_kp": (0, 1),
            "clean_human_vis": (0, 1),
            "detection_gt_index": (0, 1),
        },
        pad_values={
            "padding_mask": True,
            "target_instance_id": -1,
            "detection_gt_index": -1,
        },
    )
    return collated


__all__ = [
    "PLCS_TRACKING_KEYS",
    "PLCSTrackingDataset",
    "collate_plcs_tracking_batch",
]
