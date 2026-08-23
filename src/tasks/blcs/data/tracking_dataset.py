"""Config-aware canonical-scene adapter for lifecycle BLCS tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.tasks.base.configuration import as_config_mapping
from src.tasks.base.data.canonical_tracking import (
    CanonicalTrackingDataset,
    pad_and_stack_tracking_batch,
)
from src.tasks.base.data.court_coordinate_contract import (
    validate_dataset_court_coordinate_contract,
)
from src.tasks.base.data.lifecycle_slots import build_fixed_lifecycle_assignment
from src.tasks.base.data.scene_dataset import Scene
from src.tasks.blcs.configuration import parse_court_coordinate_normalization
from src.tasks.blcs.data.observation_candidates import (
    pack_observation_candidates,
)
from src.tasks.blcs.data.tracking_augmentation import (
    BLCSTrackingCandidateAugmentation,
)
from src.tasks.blcs.data.visibility import zero_invisible_uv
from src.utils.schema.court_normalization import (
    resolve_court_coordinate_normalization,
)

BLCS_TRACKING_KEYS = (
    "scene_format_version",
    "ball_uv",
    "ball_vis",
    "court_kp",
    "court_vis",
    "padding_mask",
    "target_position",
    "target_velocity",
    "target_presence",
    "target_instance_id",
    "target_slot_mask",
    "clean_ball_uv",
    "clean_ball_vis",
    "candidate_gt_index",
)


class BLCSTrackingDataset(CanonicalTrackingDataset):
    """Load ID-ordered objects, pack lifecycle slots, and corrupt observations."""

    def __init__(self, **kwargs: Any) -> None:
        config = kwargs.get("config")
        config_root = as_config_mapping(config, path="configuration")
        # Preserve the pre-versioning direct-construction API as explicit v1.
        # Composed runtime configs carry this section and therefore continue to
        # inject and validate their selected v1/v2 contract.
        self.court_coordinate_normalization = (
            parse_court_coordinate_normalization(config)
            if "court_coordinate_normalization" in config_root
            else resolve_court_coordinate_normalization("v1")
        )
        scene_dir = Path(kwargs["scene_dir"])
        split_file = Path(kwargs["split_file"])
        scene_paths = self._resolve_scene_files(scene_dir, split_file)
        validate_dataset_court_coordinate_contract(
            scene_dir,
            self.court_coordinate_normalization,
            scene_paths=scene_paths,
        )
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
        normalized_velocity = self.court_coordinate_normalization.normalize_velocity(
            velocity
        )
        if not isinstance(normalized_velocity, Tensor):
            raise TypeError("BLCS tracking velocity normalization returned a non-tensor.")
        velocity = normalized_velocity
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
            generator=None,
        )

        uv_rows: list[Tensor] = []
        vis_rows: list[Tensor] = []
        court_rows: list[Tensor] = []
        court_vis_rows: list[Tensor] = []
        for camera_index in cameras.indices:
            uv = torch.from_numpy(
                scene.get_camera_array(camera_index, "ball_uv", window=window)
            ).float()
            ball_vis = torch.from_numpy(
                scene.get_camera_array(camera_index, "ball_vis", window=window)
            ).bool()
            if uv.ndim == 2:
                uv = uv[:, None]
                ball_vis = ball_vis[:, None]
            ball_vis &= physical_presence
            uv = zero_invisible_uv(uv, ball_vis)
            uv_rows.append(uv)
            vis_rows.append(ball_vis)

            court_np = scene.get_camera_array(camera_index, "court_kp_uv")
            court_vis_np = scene.get_camera_array(camera_index, "court_kp_vis")
            if court_np.ndim == 2:
                court = (
                    torch.from_numpy(court_np[:14])
                    .float()[None]
                    .expand(window.seq_len, -1, -1)
                )
                court_vis = (
                    torch.from_numpy(court_vis_np[:14])
                    .bool()[None]
                    .expand(window.seq_len, -1)
                )
            else:
                court = torch.from_numpy(court_np[window.sl, :14]).float()
                court_vis = torch.from_numpy(court_vis_np[window.sl, :14]).bool()
            court = zero_invisible_uv(court, court_vis)
            court_rows.append(court)
            court_vis_rows.append(court_vis)

        physical_uv = torch.stack(uv_rows)
        physical_vis = torch.stack(vis_rows)
        observations = pack_observation_candidates(
            ball_uv=physical_uv,
            ball_vis=physical_vis,
            physical_presence=physical_presence,
            num_slots=num_queries,
            min_reuse_gap_frames=self.min_reuse_gap_frames,
            randomize_slots=randomize_slots,
        )
        sample = {
            "scene_format_version": torch.tensor(4),
            "ball_uv": observations.uv,
            "ball_vis": observations.vis,
            "court_kp": torch.stack(court_rows),
            "court_vis": torch.stack(court_vis_rows),
            "padding_mask": torch.zeros(
                len(cameras.indices), window.seq_len, dtype=torch.bool
            ),
            "target_position": target_packing.pack_tensor(position, physical_presence),
            "target_velocity": target_packing.pack_tensor(velocity, physical_presence),
            "target_presence": target_packing.target_presence,
            "target_instance_id": target_packing.target_instance_id,
            "target_slot_mask": target_packing.target_presence.any(0),
            "clean_ball_uv": observations.uv.clone(),
            "clean_ball_vis": observations.vis.clone(),
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
            "ball_vis": (0, 1),
            "court_kp": (0, 1),
            "court_vis": (0, 1),
            "padding_mask": (0, 1),
            "target_position": (0,),
            "target_velocity": (0,),
            "target_presence": (0,),
            "target_instance_id": (0,),
            "clean_ball_uv": (0, 1),
            "clean_ball_vis": (0, 1),
            "candidate_gt_index": (0, 1),
        },
        pad_values={
            "target_instance_id": -1,
            "candidate_gt_index": -1,
            "padding_mask": True,
        },
    )
    return collated


__all__ = [
    "BLCS_TRACKING_KEYS",
    "BLCSTrackingDataset",
    "collate_blcs_tracking_batch",
]
