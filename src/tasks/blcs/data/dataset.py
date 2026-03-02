"""Dataset and collate/adaptation utilities for BLCS."""

from __future__ import annotations

import random as rng
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.tasks.blcs.data.types import BLCSBatch, BLCSMultiViewBatch, BLCSMultiViewSample
from src.utils.data.augmentation import (
    add_gaussian_noise,
    random_visibility_dropout,
    scale_uv_with_visibility,
)
from src.tasks.base.data.scene_dataset import NPZScene, NPZSceneDatasetBase, SceneDatasetConfig
if TYPE_CHECKING:
    from omegaconf import DictConfig


class BallTrajectoryDataset(NPZSceneDatasetBase[BLCSMultiViewSample]):
    """Unified BLCS dataset that always returns canonical multiview samples.

    The canonical sample format keeps camera and temporal dimensions:
    - ball_uv: (N, T, 2)
    - ball_vis: (N, T)
    - ball_mask: (N, T)
    - court_kp: (N, T, 20, 2)
    - court_vis: (N, T, 20)
    - position_3d: (T, 3)
    - velocity_3d: (T, 3)
    """

    def __init__(
        self,
        *,
        scene_dir: str | Path,
        split_file: str | Path,
        config: DictConfig | None = None,
        augment: bool = True,
    ) -> None:
        self.hydra_cfg = config or {}
        self.augment = augment
        data_cfg = self._resolve_data_cfg(self.hydra_cfg)
        self._configure_task(data_cfg)
        super().__init__(
            config=self._build_scene_dataset_config(
                scene_dir=scene_dir, split_file=split_file, data_cfg=data_cfg,
            )
        )

    # -- Composed-method hooks ------------------------------------------

    def _configure_task(self, data_cfg: dict) -> None:  # type: ignore[override]
        # Multiview ranges
        self.seq_len_range = self._parse_int_range(data_cfg, "seq_len_range")
        self.num_views_range = self._parse_int_range(data_cfg, "num_views_range")
        self.camera_mode = self._parse_camera_mode(data_cfg)

        # Augmentation parameters
        aug_cfg = data_cfg.get("augmentation", {})
        self.uv_noise_std = float(aug_cfg.get("uv_noise_std", 0.005))
        self.vis_drop_prob = float(aug_cfg.get("visibility_drop_prob", 0.1))
        scale_range_cfg = aug_cfg.get("scale_range", [1.0, 1.0])
        if (
            not isinstance(scale_range_cfg, Sequence)
            or isinstance(scale_range_cfg, (str, bytes))
            or len(scale_range_cfg) != 2
        ):
            raise ValueError(
                "augmentation.scale_range must be a list/tuple of two numbers: [min_scale, max_scale]."
            )
        self.scale_range = (float(scale_range_cfg[0]), float(scale_range_cfg[1]))
        if self.scale_range[0] <= 0 or self.scale_range[1] <= 0:
            raise ValueError(
                f"augmentation.scale_range must be positive, got {self.scale_range}."
            )
        if self.scale_range[0] > self.scale_range[1]:
            raise ValueError(
                f"augmentation.scale_range min must be <= max, got {self.scale_range}."
            )

    def _build_scene_dataset_config(  # type: ignore[override]
        self,
        *,
        scene_dir: str | Path,
        split_file: str | Path,
        data_cfg: dict,
    ) -> SceneDatasetConfig:
        return SceneDatasetConfig(
            scene_dir=Path(scene_dir),
            split_file=Path(split_file),
            seq_len_range=self.seq_len_range,
            num_views_range=self.num_views_range,
            cache_max_scenes=int(data_cfg.get("cache_max_scenes", 128)),
            camera_mode=self.camera_mode,
            crop_mode=("random" if self.augment else "center"),
        )

    def build_sample(self, scene: NPZScene) -> BLCSMultiViewSample:
        cams = self.select_cameras(scene, num_views_range=self.num_views_range, camera_mode=self.camera_mode)
        # Use camera trajectory length to guard against metadata drift.
        primary_len = int(scene.get_camera_array(cams.primary, "ball_uv").shape[0])
        pos_len = int(scene.data["ball_pos_norm"].shape[0])
        vel_len = int(scene.data["ball_vel_world"].shape[0])
        full_len = scene.effective_num_frames(primary_len, pos_len, vel_len)
        window = self.select_window(scene, full_len=full_len)
        ball_uv_list: list[Tensor] = []
        ball_vis_list: list[Tensor] = []
        court_kp_list: list[Tensor] = []
        court_vis_list: list[Tensor] = []

        for cam_idx in cams.indices:
            ball_uv = torch.from_numpy(scene.get_camera_array(cam_idx, "ball_uv", window=window)).float()
            ball_vis = torch.from_numpy(scene.get_camera_array(cam_idx, "ball_visible", window=window)).float()
            court_kp = torch.from_numpy(scene.get_camera_array(cam_idx, "court_kp_uv")).float()
            court_vis = torch.from_numpy(scene.get_camera_array(cam_idx, "court_kp_visible")).float()

            court_kp_expanded = court_kp.unsqueeze(0).expand(window.seq_len, -1, -1)
            court_vis_expanded = court_vis.unsqueeze(0).expand(window.seq_len, -1)

            ball_uv_list.append(ball_uv)
            ball_vis_list.append(ball_vis)
            court_kp_list.append(court_kp_expanded)
            court_vis_list.append(court_vis_expanded)

        sample: BLCSMultiViewSample = {
            "ball_uv": torch.stack(ball_uv_list, dim=0),
            "ball_vis": torch.stack(ball_vis_list, dim=0),
            "ball_mask": torch.ones(len(cams.indices), window.seq_len, dtype=torch.float32),
            "court_kp": torch.stack(court_kp_list, dim=0),
            "court_vis": torch.stack(court_vis_list, dim=0),
            "position_3d": torch.from_numpy(scene.get_array("ball_pos_norm", window=window)).float(),
            "velocity_3d": torch.from_numpy(scene.get_array("ball_vel_world", window=window)).float(),
            "seq_len": torch.tensor(window.seq_len, dtype=torch.long),
        }
        return sample

    def _apply_augmentation_multiview(
        self, sample: BLCSMultiViewSample
    ) -> BLCSMultiViewSample:
        sample = {k: (v.clone() if isinstance(v, Tensor) else v) for k, v in sample.items()}
        scale_min, scale_max = self.scale_range
        if not (scale_min == 1.0 and scale_max == 1.0):
            scale = rng.uniform(scale_min, scale_max)
            if abs(scale - 1.0) >= 1e-8:
                ball_uv = sample["ball_uv"]
                court_kp = sample["court_kp"]
                ball_vis = sample["ball_vis"]
                court_vis = sample["court_vis"]
                if not isinstance(ball_uv, Tensor) or not isinstance(court_kp, Tensor):
                    raise ValueError("ball_uv/court_kp must be tensors")
                if not isinstance(ball_vis, Tensor) or not isinstance(court_vis, Tensor):
                    raise ValueError("ball_vis/court_vis must be tensors")

                sample["ball_uv"], sample["ball_vis"] = scale_uv_with_visibility(
                    uv=ball_uv,
                    visibility=ball_vis,
                    scale=scale,
                )
                sample["court_kp"], sample["court_vis"] = scale_uv_with_visibility(
                    uv=court_kp,
                    visibility=court_vis,
                    scale=scale,
                )

        ball_uv = sample["ball_uv"]
        if isinstance(ball_uv, Tensor):
            sample["ball_uv"] = add_gaussian_noise(ball_uv, self.uv_noise_std).clamp(0, 1)

        court_kp = sample["court_kp"]
        if isinstance(court_kp, Tensor):
            sample["court_kp"] = add_gaussian_noise(court_kp, self.uv_noise_std).clamp(0, 1)

        ball_vis = sample["ball_vis"]
        if isinstance(ball_vis, Tensor):
            sample["ball_vis"] = random_visibility_dropout(ball_vis, self.vis_drop_prob)

        return sample

    def augment_sample(self, sample: BLCSMultiViewSample) -> BLCSMultiViewSample:
        if self.augment:
            return self._apply_augmentation_multiview(sample)
        return sample


def collate_multiview_trajectories(
    batch: list[BLCSMultiViewSample],
) -> BLCSMultiViewBatch:
    """Collate canonical BLCS samples into padded canonical batch tensors."""
    max_views = max(int(sample["ball_uv"].shape[0]) for sample in batch)
    max_seq_len = max(sample["seq_len"].item() for sample in batch)

    ball_uv_batch = []
    ball_vis_batch = []
    ball_mask_batch = []
    court_kp_batch = []
    court_vis_batch = []
    position_3d_batch = []
    velocity_3d_batch = []
    seq_len_batch = []
    for sample in batch:
        n_views = int(sample["ball_uv"].shape[0])
        seq_len = sample["seq_len"].item()
        pad_views = max_views - n_views
        pad_seq = max_seq_len - seq_len

        ball_uv = sample["ball_uv"]
        ball_vis = sample["ball_vis"]
        ball_mask = sample["ball_mask"]
        court_kp = sample["court_kp"]
        court_vis = sample["court_vis"]
        position_3d = sample["position_3d"]
        velocity_3d = sample["velocity_3d"]

        if pad_seq > 0:
            ball_uv = torch.cat([ball_uv, torch.zeros(n_views, pad_seq, 2)], dim=1)
            ball_vis = torch.cat([ball_vis, torch.zeros(n_views, pad_seq)], dim=1)
            ball_mask = torch.cat([ball_mask, torch.zeros(n_views, pad_seq)], dim=1)
            court_kp = torch.cat([court_kp, torch.zeros(n_views, pad_seq, 20, 2)], dim=1)
            court_vis = torch.cat([court_vis, torch.zeros(n_views, pad_seq, 20)], dim=1)
            position_3d = torch.cat([position_3d, torch.zeros(pad_seq, 3)], dim=0)
            velocity_3d = torch.cat([velocity_3d, torch.zeros(pad_seq, 3)], dim=0)

        if pad_views > 0:
            ball_uv = torch.cat([ball_uv, torch.zeros(pad_views, max_seq_len, 2)], dim=0)
            ball_vis = torch.cat([ball_vis, torch.zeros(pad_views, max_seq_len)], dim=0)
            ball_mask = torch.cat([ball_mask, torch.zeros(pad_views, max_seq_len)], dim=0)
            court_kp = torch.cat([court_kp, torch.zeros(pad_views, max_seq_len, 20, 2)], dim=0)
            court_vis = torch.cat([court_vis, torch.zeros(pad_views, max_seq_len, 20)], dim=0)

        ball_uv_batch.append(ball_uv)
        ball_vis_batch.append(ball_vis)
        ball_mask_batch.append(ball_mask)
        court_kp_batch.append(court_kp)
        court_vis_batch.append(court_vis)
        position_3d_batch.append(position_3d)
        velocity_3d_batch.append(velocity_3d)
        seq_len_batch.append(sample["seq_len"])
    return {
        "ball_uv": torch.stack(ball_uv_batch, dim=0),
        "ball_vis": torch.stack(ball_vis_batch, dim=0),
        "ball_mask": torch.stack(ball_mask_batch, dim=0),
        "court_kp": torch.stack(court_kp_batch, dim=0),
        "court_vis": torch.stack(court_vis_batch, dim=0),
        "position_3d": torch.stack(position_3d_batch, dim=0),
        "velocity_3d": torch.stack(velocity_3d_batch, dim=0),
        "seq_len": torch.stack(seq_len_batch, dim=0),
    }


def adapt_batch_for_model_profile(
    batch: BLCSMultiViewBatch,
    *,
    input_profile: str,
) -> BLCSBatch | BLCSMultiViewBatch:
    """Adapt canonical BLCS batch ``(B,N,T,...)`` to model input profile."""
    _, n, _ = batch["ball_uv"].shape[:3]
    if n <= 0:
        raise ValueError("Expected at least one camera view in batch.")

    if input_profile == "multiview":
        return batch
    if input_profile == "single":
        return {
            "ball_uv": batch["ball_uv"][:, 0],
            "ball_vis": batch["ball_vis"][:, 0],
            "ball_mask": batch["ball_mask"][:, 0],
            "court_kp": batch["court_kp"][:, 0, 0],
            "court_vis": batch["court_vis"][:, 0, 0],
            "position_3d": batch["position_3d"],
            "velocity_3d": batch["velocity_3d"],
            "seq_len": batch["seq_len"],
        }
    raise ValueError(
        "Unknown model input profile: "
        f"{input_profile}. Supported: ['single', 'multiview']"
    )


def collate_and_adapt_blcs_batch(
    batch: list[BLCSMultiViewSample],
    *,
    input_profile: str,
) -> BLCSBatch | BLCSMultiViewBatch:
    """Collate canonical BLCS samples and adapt to model input profile."""
    collated = collate_multiview_trajectories(batch)
    return adapt_batch_for_model_profile(collated, input_profile=input_profile)
