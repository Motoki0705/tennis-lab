"""Config-aware canonical-scene adapter for lifecycle PLCS tracking."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.tasks.base.data.canonical_tracking import (
    CanonicalTrackingDataset,
    pad_and_stack_tracking_batch,
    permute_tracking_views,
)
from src.tasks.base.data.court_peaks import (
    ordered_court_to_semantic_peaks,
    parse_court_observation_profile,
)
from src.tasks.base.data.reference_orientation import (
    camera_centers_from_scene_payload,
    deterministic_sample_rng,
    reflect_court_vectors,
    reflect_heading,
    select_reference_view,
)
from src.tasks.base.data.scene_dataset import Scene
from src.tasks.plcs.data.targets import build_coco17_world_targets
from src.tasks.plcs.data.tracking_augmentation import (
    PLCSTrackingDetectionAugmentation,
)

PLCS_TRACKING_KEYS = (
    "scene_format_version",
    "human_kp",
    "human_vis",
    "joint_visibility",
    "detection_score",
    "detection_mask",
    "court_kp",
    "court_vis",
    "court_peak_uv",
    "court_peak_score",
    "court_peak_covariance",
    "court_peak_valid",
    "frame_mask",
    "view_mask",
    "reference_view_index",
    "orientation_sign",
    "camera_center",
    "target_position",
    "source_target_position",
    "target_rotation",
    "source_target_rotation",
    "target_canonical_pose_3d",
    "target_human_kp_3d",
    "source_target_human_kp_3d",
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
        model_cfg = self.hydra_cfg["model"]
        self.court_observation_profile = parse_court_observation_profile(
            model_cfg.get(
                "court_observation_profile", "kp14_reference_baseline"
            )
        )
        self.tracking_augmentation = PLCSTrackingDetectionAugmentation(
            data_cfg["augmentation"]
        )

    def build_sample(self, scene: Scene) -> dict[str, Tensor]:
        position = torch.from_numpy(scene.get_array("position")).float()
        rotation = torch.from_numpy(scene.get_array("rotation")).float()
        canonical_pose = torch.from_numpy(scene.get_array("canonical_pose_3d")).float()
        if scene.has_key("human_kp_3d"):
            world_joints = torch.from_numpy(scene.get_array("human_kp_3d")).float()
        else:
            payload_for_targets = {**scene.data, "meta": scene.meta}
            world_joints = torch.from_numpy(
                build_coco17_world_targets(payload_for_targets)
            ).float()
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
        camera_centers = camera_centers_from_scene_payload(
            scene.data, cameras.indices
        )
        reference_index, orientation_sign = select_reference_view(
            camera_centers,
            rng=(
                self.rng
                if self.augment
                else deterministic_sample_rng(
                    self.dataset_seed,
                    f"reference:{scene.path}:{cameras.indices}",
                )
            ),
        )

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
            court = torch.from_numpy(
                    scene.get_camera_array(camera_index, "court_kp_uv", window=window)[
                        :, :14
                    ]
                ).float()
            court_visible = torch.from_numpy(
                    scene.get_camera_array(
                        camera_index, "court_kp_visible", window=window
                    )[:, :14]
                ).bool()
            court_rows.append(court.masked_fill(~court_visible.unsqueeze(-1), 0.0))
            court_vis_rows.append(court_visible)

        human_kp = torch.stack(kp_rows)
        human_vis = torch.stack(visible_rows)
        rotation_fill = torch.tensor([1.0, 0.0], dtype=rotation.dtype)
        source_target_position = packing.pack_tensor(position, physical_presence)
        source_target_rotation = packing.pack_tensor(
            rotation,
            physical_presence,
            fill_value=rotation_fill,
        )
        source_target_human_kp_3d = packing.pack_tensor(
            world_joints, physical_presence
        )
        target_position = reflect_court_vectors(
            source_target_position, orientation_sign
        )
        target_rotation = reflect_heading(
            source_target_rotation, orientation_sign
        )
        target_human_kp_3d = reflect_court_vectors(
            source_target_human_kp_3d, orientation_sign
        )
        court_kp = torch.stack(court_rows)
        court_vis = torch.stack(court_vis_rows)
        sample = {
            "scene_format_version": torch.tensor(3),
            "human_kp": human_kp,
            "human_vis": human_vis,
            "joint_visibility": human_vis,
            "detection_score": human_vis.float().mean(-1),
            "detection_mask": human_vis.any(-1),
            "frame_mask": torch.ones(window.seq_len, dtype=torch.bool),
            "view_mask": torch.ones(len(cameras.indices), dtype=torch.bool),
            "reference_view_index": torch.tensor(reference_index, dtype=torch.long),
            "orientation_sign": torch.tensor(orientation_sign, dtype=torch.float32),
            "camera_center": camera_centers,
            "target_position": target_position,
            "source_target_position": source_target_position,
            "target_rotation": target_rotation,
            "source_target_rotation": source_target_rotation,
            "target_canonical_pose_3d": packing.pack_tensor(
                canonical_pose, physical_presence
            ),
            "target_human_kp_3d": target_human_kp_3d,
            "source_target_human_kp_3d": source_target_human_kp_3d,
            "target_presence": packing.target_presence,
            "target_instance_id": packing.target_instance_id,
            "target_slot_mask": packing.target_presence.any(0),
            "clean_human_kp": torch.stack(clean_kp_rows),
            "clean_human_visible": torch.stack(clean_visible_rows),
            "detection_gt_index": torch.stack(index_rows),
        }
        if self.court_observation_profile == "kp14_reference_baseline":
            sample["court_kp"] = court_kp
            sample["court_vis"] = court_vis
        else:
            peaks = ordered_court_to_semantic_peaks(
                court_kp.unsqueeze(0), court_vis.unsqueeze(0)
            )
            sample["court_peak_uv"] = peaks.uv[0]
            sample["court_peak_score"] = peaks.score[0]
            sample["court_peak_covariance"] = peaks.covariance[0]
            sample["court_peak_valid"] = peaks.valid[0]
        return sample

    def augment_sample(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.augment:
            return sample
        view_fields = [
            "human_kp",
            "human_vis",
            "joint_visibility",
            "detection_score",
            "detection_mask",
            "view_mask",
            "camera_center",
            "clean_human_kp",
            "clean_human_visible",
            "detection_gt_index",
        ]
        if self.court_observation_profile == "kp14_reference_baseline":
            view_fields.extend(("court_kp", "court_vis"))
        else:
            view_fields.extend(
                (
                    "court_peak_uv",
                    "court_peak_score",
                    "court_peak_covariance",
                    "court_peak_valid",
                )
            )
        permutation = torch.as_tensor(
            self.rng.permutation(sample["view_mask"].shape[0]),
            dtype=torch.long,
        )
        permuted = permute_tracking_views(
            sample,
            permutation,
            view_fields=view_fields,
        )
        augmented: dict[str, Tensor] = self.tracking_augmentation(permuted)
        return augmented


def collate_plcs_tracking_batch(
    batch: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Pad variable camera/time/detection dimensions and stack PLCS scenes."""
    collated: dict[str, Tensor] = pad_and_stack_tracking_batch(
        batch,
        padding_dimensions={
            "human_kp": (0, 1, 2),
            "human_vis": (0, 1, 2),
            "joint_visibility": (0, 1, 2),
            "detection_score": (0, 1, 2),
            "detection_mask": (0, 1, 2),
            "court_kp": (0, 1),
            "court_vis": (0, 1),
            "court_peak_uv": (0, 1, 3),
            "court_peak_score": (0, 1, 3),
            "court_peak_covariance": (0, 1, 3),
            "court_peak_valid": (0, 1, 3),
            "frame_mask": (0,),
            "view_mask": (0,),
            "target_position": (0, 1),
            "source_target_position": (0, 1),
            "target_rotation": (0, 1),
            "source_target_rotation": (0, 1),
            "target_canonical_pose_3d": (0, 1),
            "target_human_kp_3d": (0, 1),
            "source_target_human_kp_3d": (0, 1),
            "target_presence": (0, 1),
            "target_instance_id": (0, 1),
            "target_slot_mask": (0,),
            "clean_human_kp": (0, 1, 2),
            "clean_human_visible": (0, 1, 2),
            "detection_gt_index": (0, 1, 2),
            "camera_center": (0,),
        },
        pad_values={
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
