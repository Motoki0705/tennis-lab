"""Config-aware canonical-scene adapter for lifecycle BLCS tracking."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.tasks.base.data.canonical_tracking import (
    CanonicalTrackingDataset,
    pad_and_stack_tracking_batch,
)
from src.tasks.base.data.scene_dataset import Scene
from src.tasks.blcs.data.observation_candidates import (
    build_fixed_lifecycle_assignment,
    pack_observation_candidates,
)
from src.tasks.blcs.data.tracking_augmentation import (
    BLCSTrackingCandidateAugmentation,
)

BLCS_TRACKING_KEYS = (
    "scene_format_version",
    "ball_uv",
    "ball_visible",
    "candidate_mask",
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
            data_cfg["augmentation"]
        )
        if self.num_queries is None:
            raise ValueError("BLCS tracking requires model.num_queries.")
        if not self.pack_to_query_slots:
            raise ValueError(
                "BLCS tracking requires data.lifecycle.pack_to_query_slots=true."
            )

    def build_sample(self, scene: Scene) -> dict[str, Tensor]:
        if self.num_queries is None:
            raise RuntimeError("BLCS tracking dataset lost its fixed query width.")
        num_queries = self.num_queries
        position = torch.from_numpy(scene.get_array("ball_pos_norm")).float()
        velocity = torch.from_numpy(scene.get_array("ball_vel_world")).float()
        if position.ndim != 3 or velocity.shape != position.shape:
            raise ValueError(
                "Tracking scenes require explicit (T,P,3) position/velocity arrays."
            )
        num_frames, num_physical = position.shape[:2]
        if not scene.has_key("ball_present"):
            raise ValueError(
                "Tracking scene is incompatible: required ball_present is missing."
            )
        physical_presence = torch.from_numpy(scene.get_array("ball_present")).bool()
        if physical_presence.shape != (num_frames, num_physical):
            raise ValueError(
                "ball_present must match the explicit (T,P) physical object axes."
            )
        window = self.select_window(scene, full_len=num_frames)
        cameras = self.select_cameras(scene)
        position = position[window.sl]
        velocity = velocity[window.sl]
        physical_presence = physical_presence[window.sl]
        randomize_slots = self.augment and self.randomize_slots_train
        target_packing = build_fixed_lifecycle_assignment(
            physical_presence,
            num_slots=num_queries,
            min_reuse_gap_frames=self.min_reuse_gap_frames,
            randomize_slots=randomize_slots,
        )

        uv_rows: list[Tensor] = []
        visible_rows: list[Tensor] = []
        court_rows: list[Tensor] = []
        court_vis_rows: list[Tensor] = []
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
            uv_rows.append(uv)
            visible_rows.append(visible)

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

        physical_uv = torch.stack(uv_rows)
        physical_visible = torch.stack(visible_rows)
        observations = pack_observation_candidates(
            ball_uv=physical_uv,
            ball_visible=physical_visible,
            physical_presence=physical_presence,
            num_slots=num_queries,
            min_reuse_gap_frames=self.min_reuse_gap_frames,
            randomize_slots=randomize_slots,
        )
        sample = {
            "scene_format_version": torch.tensor(3),
            "ball_uv": observations.uv,
            "ball_visible": observations.visible,
            "candidate_mask": observations.candidate_mask,
            "court_kp": torch.stack(court_rows),
            "court_vis": torch.stack(court_vis_rows),
            "frame_mask": torch.ones(window.seq_len, dtype=torch.bool),
            "view_mask": torch.ones(len(cameras.indices), dtype=torch.bool),
            "target_position": target_packing.pack_tensor(position, physical_presence),
            "target_velocity": target_packing.pack_tensor(velocity, physical_presence),
            "target_presence": target_packing.target_presence,
            "target_instance_id": target_packing.target_instance_id,
            "target_slot_mask": target_packing.target_presence.any(0),
            "clean_ball_uv": observations.uv.clone(),
            "clean_ball_visible": observations.visible.clone(),
            "candidate_gt_index": observations.gt_index,
        }
        return sample

    def augment_sample(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.augment:
            return sample
        augmented: dict[str, Tensor] = self.tracking_augmentation(sample)
        return augmented


def collate_blcs_tracking_batch(
    batch: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Pad only camera/time dimensions and stack exact-width BLCS scenes."""
    candidate_width = int(batch[0]["ball_uv"].shape[2]) if batch else 0
    for sample in batch:
        if (
            sample["ball_uv"].shape[2] != candidate_width
            or sample["candidate_mask"].shape[2] != candidate_width
            or sample["target_position"].shape[1] != candidate_width
            or sample["target_slot_mask"].shape[0] != candidate_width
        ):
            raise ValueError(
                "Every BLCS tracking sample must already have the same exact "
                "candidate width before collation."
            )
    collated: dict[str, Tensor] = pad_and_stack_tracking_batch(
        batch,
        padding_dimensions={
            "ball_uv": (0, 1),
            "ball_visible": (0, 1),
            "candidate_mask": (0, 1),
            "court_kp": (0, 1),
            "court_vis": (0, 1),
            "frame_mask": (0,),
            "view_mask": (0,),
            "target_position": (0,),
            "target_velocity": (0,),
            "target_presence": (0,),
            "target_instance_id": (0,),
            "clean_ball_uv": (0, 1),
            "clean_ball_visible": (0, 1),
            "candidate_gt_index": (0, 1),
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
