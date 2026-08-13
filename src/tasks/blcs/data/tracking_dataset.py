"""Config-aware canonical-scene adapter for lifecycle BLCS tracking."""

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
    select_reference_view,
)
from src.tasks.base.data.scene_dataset import Scene
from src.tasks.blcs.data.tracking_augmentation import (
    BLCSTrackingCandidateAugmentation,
)

BLCS_TRACKING_KEYS = (
    "scene_format_version",
    "ball_uv",
    "ball_score",
    "ball_visible",
    "candidate_mask",
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
    "target_velocity",
    "source_target_velocity",
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
        model_cfg = self.hydra_cfg["model"]
        self.court_observation_profile = parse_court_observation_profile(
            model_cfg.get(
                "court_observation_profile", "kp14_reference_baseline"
            )
        )
        self.tracking_augmentation = BLCSTrackingCandidateAugmentation(
            data_cfg["augmentation"]
        )

    def build_sample(self, scene: Scene) -> dict[str, Tensor]:
        position = torch.from_numpy(scene.get_array("ball_pos_norm")).float()
        velocity = torch.from_numpy(scene.get_array("ball_vel_world")).float()
        if position.ndim == 2:
            position = position[:, None]
            velocity = velocity[:, None]
        if position.ndim != 3 or velocity.shape != position.shape:
            raise ValueError(
                "Tracking scenes require explicit (T,P,3) position/velocity arrays."
            )
        num_frames, num_physical = position.shape[:2]
        physical_presence = (
            torch.from_numpy(scene.get_array("ball_present")).bool()
            if scene.has_key("ball_present")
            else torch.ones((num_frames, num_physical), dtype=torch.bool)
        )
        if physical_presence.shape != (num_frames, num_physical):
            raise ValueError(
                "ball_present must match the explicit (T,P) physical object axes."
            )
        window = self.select_window(scene, full_len=num_frames)
        cameras = self.select_cameras(scene)
        position = position[window.sl]
        velocity = velocity[window.sl]
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
            # Persisted projections may remain outside the normalized image for
            # invisible physical objects.  The observation boundary represents
            # every masked coordinate with an explicit in-domain value so the
            # model never receives unusable geometry through an invisible slot.
            uv = uv.masked_fill(~visible.unsqueeze(-1), 0.0)
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
            court = court.masked_fill(~court_visible.unsqueeze(-1), 0.0)
            court_rows.append(court)
            court_vis_rows.append(court_visible)

        source_target_position = packing.pack_tensor(position, physical_presence)
        source_target_velocity = packing.pack_tensor(velocity, physical_presence)
        target_position = reflect_court_vectors(
            source_target_position, orientation_sign
        )
        target_velocity = reflect_court_vectors(
            source_target_velocity, orientation_sign
        )
        court_kp = torch.stack(court_rows)
        court_vis = torch.stack(court_vis_rows)
        sample = {
            "scene_format_version": torch.tensor(3),
            "ball_uv": torch.stack(uv_rows),
            "ball_score": torch.stack(visible_rows).float(),
            "ball_visible": torch.stack(visible_rows),
            "candidate_mask": torch.ones_like(
                torch.stack(visible_rows), dtype=torch.bool
            ),
            "frame_mask": torch.ones(window.seq_len, dtype=torch.bool),
            "view_mask": torch.ones(len(cameras.indices), dtype=torch.bool),
            "reference_view_index": torch.tensor(reference_index, dtype=torch.long),
            "orientation_sign": torch.tensor(orientation_sign, dtype=torch.float32),
            "camera_center": camera_centers,
            "target_position": target_position,
            "source_target_position": source_target_position,
            "target_velocity": target_velocity,
            "source_target_velocity": source_target_velocity,
            "target_presence": packing.target_presence,
            "target_instance_id": packing.target_instance_id,
            "target_slot_mask": packing.target_presence.any(0),
            "clean_ball_uv": torch.stack(clean_uv_rows),
            "clean_ball_visible": torch.stack(clean_visible_rows),
            "candidate_gt_index": torch.stack(index_rows),
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
            "ball_uv",
            "ball_score",
            "ball_visible",
            "candidate_mask",
            "view_mask",
            "camera_center",
            "clean_ball_uv",
            "clean_ball_visible",
            "candidate_gt_index",
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
            "court_peak_uv": (0, 1, 3),
            "court_peak_score": (0, 1, 3),
            "court_peak_covariance": (0, 1, 3),
            "court_peak_valid": (0, 1, 3),
            "frame_mask": (0,),
            "view_mask": (0,),
            "target_position": (0, 1),
            "source_target_position": (0, 1),
            "target_velocity": (0, 1),
            "source_target_velocity": (0, 1),
            "target_presence": (0, 1),
            "target_instance_id": (0, 1),
            "target_slot_mask": (0,),
            "clean_ball_uv": (0, 1, 2),
            "clean_ball_visible": (0, 1, 2),
            "candidate_gt_index": (0, 1, 2),
            "ball_score": (0, 1, 2),
            "candidate_mask": (0, 1, 2),
            "camera_center": (0,),
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
