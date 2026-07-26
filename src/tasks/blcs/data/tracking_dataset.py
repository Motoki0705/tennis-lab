"""Config-aware canonical-scene adapter for lifecycle BLCS tracking."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from src.tasks.base.data.canonical_tracking import (
    CanonicalTrackingDataset,
    pad_and_stack_tracking_batch,
)
from src.tasks.base.data.scene_dataset import Scene
from src.tasks.blcs.data.tracking_augmentation import (
    BLCSTrackingCandidateAugmentation,
)

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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        data_cfg = self._resolve_data_cfg(self.hydra_cfg)
        self.tracking_augmentation = BLCSTrackingCandidateAugmentation(
            data_cfg.get("augmentation", {})
        )

    def build_sample(self, scene: Scene) -> dict[str, Tensor]:
        position = torch.from_numpy(scene.get_array("ball_pos_norm")).float()
        velocity = torch.from_numpy(scene.get_array("ball_vel_world")).float()
        if position.ndim == 2:
            position = position[:, None]
            velocity = velocity[:, None]
        num_frames, num_physical = position.shape[:2]
        if scene.has_key("ball_present"):
            physical_presence = torch.from_numpy(scene.get_array("ball_present")).bool()
        else:
            physical_presence = torch.ones((num_frames, num_physical), dtype=torch.bool)
        window = self.select_window(scene, full_len=num_frames)
        cameras = self.select_cameras(scene)
        position = position[window.sl]
        velocity = velocity[window.sl]
        physical_presence = physical_presence[window.sl]
        packing = self.pack_lifecycle(physical_presence)

        uv_rows: list[Tensor] = []
        visible_rows: list[Tensor] = []
        index_rows: list[Tensor] = []
        clean_uv_rows: list[Tensor] = []
        clean_visible_rows: list[Tensor] = []
        court_rows: list[Tensor] = []
        court_vis_rows: list[Tensor] = []
        ordered_object_ids = torch.arange(num_physical).expand(window.seq_len, -1)
        for camera_index in cameras.indices:
            uv = torch.from_numpy(
                scene.get_camera_array(camera_index, "ball_uv", window=window)
            ).float()
            visible = torch.from_numpy(
                scene.get_camera_array(camera_index, "ball_visible", window=window)
            ).bool()
            if uv.ndim == 2:
                uv = uv[:, None]
                visible = visible[:, None]
            visible &= physical_presence
            uv[~physical_presence] = 0.0
            clean_uv_rows.append(uv.clone())
            clean_visible_rows.append(visible.clone())
            candidate_index = torch.where(visible, ordered_object_ids, -1)
            uv_rows.append(uv)
            visible_rows.append(visible)
            index_rows.append(candidate_index)

            court_np = scene.get_camera_array(camera_index, "court_kp_uv")
            court_visible_np = scene.get_camera_array(camera_index, "court_kp_visible")
            if court_np.ndim == 2:
                court = (
                    torch.from_numpy(court_np[:14])
                    .float()[None]
                    .expand(window.seq_len, -1, -1)
                )
                court_visible = (
                    torch.from_numpy(court_visible_np[:14])
                    .bool()[None]
                    .expand(window.seq_len, -1)
                )
            else:
                court = torch.from_numpy(court_np[window.sl, :14]).float()
                court_visible = torch.from_numpy(
                    court_visible_np[window.sl, :14]
                ).bool()
            court_rows.append(court)
            court_vis_rows.append(court_visible)

        sample = {
            "scene_format_version": torch.tensor(3),
            "ball_uv": torch.stack(uv_rows),
            "ball_visible": torch.stack(visible_rows),
            "court_kp": torch.stack(court_rows),
            "court_vis": torch.stack(court_vis_rows),
            "frame_mask": torch.ones(window.seq_len, dtype=torch.bool),
            "view_mask": torch.ones(len(cameras.indices), dtype=torch.bool),
            "target_position": packing.pack_tensor(position, physical_presence),
            "target_velocity": packing.pack_tensor(velocity, physical_presence),
            "target_presence": packing.target_presence,
            "target_instance_id": packing.target_instance_id,
            "target_slot_mask": packing.target_presence.any(0),
            "clean_ball_uv": torch.stack(clean_uv_rows),
            "clean_ball_visible": torch.stack(clean_visible_rows),
            "candidate_gt_index": torch.stack(index_rows),
        }
        return sample

    def augment_sample(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.augment:
            return sample
        return self.tracking_augmentation(sample)


def collate_blcs_tracking_batch(
    batch: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Pad variable camera/time/candidate dimensions and stack BLCS scenes."""
    return cast(
        dict[str, Tensor],
        pad_and_stack_tracking_batch(
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
        ),
    )


__all__ = [
    "BLCS_TRACKING_KEYS",
    "BLCSTrackingDataset",
    "collate_blcs_tracking_batch",
]
