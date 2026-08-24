"""Config-aware canonical-scene adapter for lifecycle PLCS tracking."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from src.tasks.base.data.canonical_tracking import (
    CanonicalTrackingDataset,
    pad_and_stack_tracking_batch,
)
from src.tasks.base.data.lifecycle_slots import build_fixed_lifecycle_assignment
from src.tasks.base.data.scene_dataset import Scene
from src.tasks.base.generate_dataset import (
    camera_extrinsics_physical_to_target,
)
from src.tasks.plcs.court_keypoint_contract import (
    PLCSCourtKeypointRuntimeConfig,
    align_selected_court_array,
    choose_reference_provenance,
    court_keypoint_contract_document,
    normalized_headings_physical_to_target,
    normalized_points_physical_to_target,
    selected_court_views,
    validate_plcs_dataset_court_keypoints,
    world_joints_physical_to_target,
)
from src.tasks.plcs.data.tracking_augmentation import (
    PLCSTrackingDetectionAugmentation,
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
        self.court_keypoint_contract = PLCSCourtKeypointRuntimeConfig.from_config(
            config,
        ).contract
        self.court_keypoint_validation = validate_plcs_dataset_court_keypoints(
            kwargs["scene_dir"],
            kwargs["split_file"],
            self.court_keypoint_contract,
        )
        super().__init__(**kwargs)
        data_cfg = self._resolve_data_cfg(self.hydra_cfg)
        self.tracking_augmentation = PLCSTrackingDetectionAugmentation(
            data_cfg["augmentation"]
        )

    def build_sample(self, scene: Scene) -> dict[str, Any]:
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
        views = selected_court_views(
            self.court_keypoint_validation,
            scene.path,
            cameras.indices,
        )
        provenance, reference_view = choose_reference_provenance(
            self.court_keypoint_contract,
            views,
            rng=self.rng if self.augment else None,
        )
        position = normalized_points_physical_to_target(
            position,
            provenance,
        )
        rotation = normalized_headings_physical_to_target(rotation, provenance)
        world_joints = world_joints_physical_to_target(world_joints, provenance)
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
        camera_center_rows: list[Tensor] = []
        camera_rotation_rows: list[Tensor] = []
        for local_index, camera_index in enumerate(cameras.indices):
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
            source_view = views[local_index] if views else None
            court_rows.append(
                torch.from_numpy(
                    align_selected_court_array(
                        scene.get_camera_array(
                            camera_index, "court_kp_uv", window=window
                        ),
                        source_view,
                        reference_view,
                        keypoint_axis=-2,
                    )[:, :14]
                ).float()
            )
            court_vis_rows.append(
                torch.from_numpy(
                    align_selected_court_array(
                        scene.get_camera_array(
                            camera_index, "court_kp_vis", window=window
                        ),
                        source_view,
                        reference_view,
                        keypoint_axis=-1,
                    )[:, :14]
                ).bool()
            )
            params = scene.data.get(f"cam_{camera_index}_params")
            if not isinstance(params, Mapping):
                raise ValueError(
                    f"Scene {scene.path} camera {camera_index} has invalid params metadata."
                )
            center, camera_rotation = camera_extrinsics_physical_to_target(
                torch.as_tensor(params.get("C"), dtype=torch.float32),
                torch.as_tensor(params.get("R"), dtype=torch.float32),
                provenance,
            )
            camera_center_rows.append(center)
            camera_rotation_rows.append(camera_rotation)

        human_kp = torch.stack(kp_rows)
        human_vis = torch.stack(visible_rows)
        rotation_fill = torch.tensor([1.0, 0.0], dtype=rotation.dtype)
        sample: dict[str, Any] = {
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
            "camera_C": torch.stack(camera_center_rows),
            "camera_R": torch.stack(camera_rotation_rows),
        }
        sample["court_keypoint_metadata"] = court_keypoint_contract_document(
            self.court_keypoint_contract
        )
        sample["court_reference_provenance"] = provenance
        sample["selected_camera_ids"] = tuple(
            view.camera_id for view in views
        ) or tuple(f"camera_{index}" for index in cameras.indices)
        return sample

    def augment_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        if not self.augment:
            return sample
        metadata_keys = (
            "court_keypoint_metadata",
            "court_reference_provenance",
            "selected_camera_ids",
        )
        metadata = {key: sample[key] for key in metadata_keys}
        tensor_sample: dict[str, Tensor] = {
            key: value for key, value in sample.items() if isinstance(value, Tensor)
        }
        augmented = self.tracking_augmentation(tensor_sample)
        return {**augmented, **metadata}


def collate_plcs_tracking_batch(
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pad variable camera/time/detection dimensions and stack PLCS scenes."""
    metadata_keys = {
        "court_keypoint_metadata",
        "court_reference_provenance",
        "selected_camera_ids",
    }
    tensor_batch = [
        {key: value for key, value in sample.items() if key not in metadata_keys}
        for sample in batch
    ]
    result: dict[str, Any] = pad_and_stack_tracking_batch(
        tensor_batch,
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
            "camera_C": (0,),
            "camera_R": (0,),
        },
        pad_values={
            "padding_mask": True,
            "target_instance_id": -1,
            "detection_gt_index": -1,
        },
    )
    for key in metadata_keys:
        result[key] = tuple(sample[key] for sample in batch)
    return result


__all__ = [
    "PLCS_TRACKING_KEYS",
    "PLCSTrackingDataset",
    "collate_plcs_tracking_batch",
]
