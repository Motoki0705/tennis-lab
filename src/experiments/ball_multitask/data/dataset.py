"""Dataset for unified UV completion + event + 3D trajectory training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from src.utils.data.soft_labels import extract_event_indices, gaussian_soft_labels
from src.utils.dataset.augmentation import add_gaussian_noise
from src.utils.dataset.npz_scene_dataset import NPZScene, NPZSceneDatasetBase, SceneDatasetConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass(frozen=True)
class CorruptionConfig:
    """Configuration for generating corrupted UV inputs."""

    enabled: bool = True
    noise_std: float = 0.01
    clamp_unit: bool = True
    point_dropout_prob: float = 0.05
    gap_dropout_prob: float = 0.25
    max_gap_len: int = 12
    max_masked_ratio: float = 0.6
    outlier_prob: float = 0.0


@dataclass(frozen=True)
class LabelConfig:
    """Configuration for event label generation."""

    sigma_frames: float = 2.5
    shot_time_key: str = "t_start"
    bounce_time_key: str = "t_bounce1"


def _apply_corruption(
    *,
    ball_uv_gt: Tensor,
    ball_gt_visible: Tensor,
    cfg: CorruptionConfig,
) -> tuple[Tensor, Tensor]:
    """Create (ball_uv_in, ball_obs_mask) from ground-truth UV and visibility."""
    T = ball_uv_gt.shape[0]
    ball_uv_in = ball_uv_gt.clone()
    ball_obs_mask = ball_gt_visible.clone()

    if not cfg.enabled:
        return ball_uv_in, ball_obs_mask

    if cfg.point_dropout_prob > 0:
        drop = (torch.rand(T, device=ball_uv_gt.device) < float(cfg.point_dropout_prob)).to(ball_obs_mask.dtype)
        ball_obs_mask = ball_obs_mask * (1.0 - drop)

    if cfg.gap_dropout_prob > 0 and cfg.max_gap_len > 0:
        if torch.rand(1).item() < float(cfg.gap_dropout_prob):
            gap_len = int(torch.randint(1, int(cfg.max_gap_len) + 1, (1,)).item())
            start = int(torch.randint(0, max(1, T - gap_len + 1), (1,)).item())
            ball_obs_mask[start : start + gap_len] = 0.0

    visible_idx = torch.where(ball_gt_visible > 0)[0]
    if visible_idx.numel() > 0 and cfg.max_masked_ratio < 1.0:
        num_visible = int(visible_idx.numel())
        max_masked = int(round(float(cfg.max_masked_ratio) * num_visible))
        masked_now = int((ball_obs_mask[visible_idx] <= 0).sum().item())
        if masked_now > max_masked:
            masked_idx = visible_idx[ball_obs_mask[visible_idx] <= 0]
            perm = masked_idx[torch.randperm(masked_idx.numel(), device=masked_idx.device)]
            enable = perm[: masked_now - max_masked]
            ball_obs_mask[enable] = 1.0

    if cfg.noise_std > 0:
        obs = ball_obs_mask > 0
        if obs.any():
            noisy = add_gaussian_noise(ball_uv_in[obs], float(cfg.noise_std))
            if cfg.clamp_unit:
                noisy = noisy.clamp(0.0, 1.0)
            ball_uv_in[obs] = noisy

    if cfg.outlier_prob > 0:
        obs = ball_obs_mask > 0
        if obs.any():
            outlier = (torch.rand(T, device=ball_uv_gt.device) < float(cfg.outlier_prob)) & obs
            if outlier.any():
                ball_uv_in[outlier] = torch.rand(int(outlier.sum().item()), 2, device=ball_uv_gt.device)

    miss = ball_obs_mask <= 0
    if miss.any():
        ball_uv_in[miss] = 0.0

    return ball_uv_in, ball_obs_mask


class BallMultitaskDataset(NPZSceneDatasetBase[dict[str, Tensor]]):
    """Dataset that yields UV inputs, 3D targets, and event labels per scene."""

    def __init__(
        self,
        *,
        scene_dir: str | Path,
        split_file: str | Path,
        config: DictConfig | None = None,
        augment: bool = True,
    ) -> None:
        self.hydra_cfg = config or {}
        data_cfg = self.hydra_cfg.get("data", {}) if hasattr(self.hydra_cfg, "get") else {}
        data_cfg = data_cfg or {}

        _scene_dir = Path(scene_dir)
        seq_len_range_cfg = data_cfg["seq_len_range"]
        seq_len_range = (int(seq_len_range_cfg[0]), int(seq_len_range_cfg[1]))
        self.supervise_visible_only = bool(data_cfg.get("supervise_visible_only", True))
        self.augment = bool(augment)

        corr_cfg = data_cfg.get("corruption", {}) or {}
        self.corruption = CorruptionConfig(
            enabled=bool(corr_cfg.get("enabled", True)),
            noise_std=float(corr_cfg.get("noise_std", 0.01)),
            clamp_unit=bool(corr_cfg.get("clamp_unit", True)),
            point_dropout_prob=float(corr_cfg.get("point_dropout_prob", 0.05)),
            gap_dropout_prob=float(corr_cfg.get("gap_dropout_prob", 0.25)),
            max_gap_len=int(corr_cfg.get("max_gap_len", 12)),
            max_masked_ratio=float(corr_cfg.get("max_masked_ratio", 0.6)),
            outlier_prob=float(corr_cfg.get("outlier_prob", 0.0)),
        )

        label_cfg = data_cfg.get("label", {}) or {}
        self.label_cfg = LabelConfig(
            sigma_frames=float(label_cfg.get("sigma_frames", 2.5)),
            shot_time_key=str(label_cfg.get("shot_time_key", "t_start")),
            bounce_time_key=str(label_cfg.get("bounce_time_key", "t_bounce1")),
        )
        super().__init__(
            config=SceneDatasetConfig(
                scene_dir=_scene_dir,
                split_file=Path(split_file),
                seq_len_range=seq_len_range,
                num_views_range=(1, 1),
                cache_max_scenes=int(data_cfg.get("cache_max_scenes", 128)),
                camera_mode=data_cfg.get("camera_mode", "random"),
                crop_mode=("random" if self.augment else "center"),
            )
        )

    def build_sample(self, scene: NPZScene) -> dict[str, Tensor]:
        cam_idx = self.select_camera(scene)
        ball_uv_full = scene.get_ball_uv(cam_idx)
        full_len = scene.effective_num_frames(int(ball_uv_full.shape[0]))
        window = self.select_window(scene, full_len=full_len)
        view = scene.get_camera_view(cam_idx, window=window)

        ball_uv_gt = torch.from_numpy(view.ball_uv).float()
        ball_visible = torch.from_numpy(view.ball_visible).to(torch.float32)
        court_kp = torch.from_numpy(view.court_kp_uv).float()
        court_vis = torch.from_numpy(view.court_kp_visible).to(torch.float32)
        position_3d = torch.from_numpy(scene.get_ball_pos_norm(window=window)).float()
        if scene.has_key("ball_pos_world"):
            ball_pos_world = torch.from_numpy(scene.get_ball_pos_world(window=window)).float()
        else:
            ball_pos_world = position_3d.clone()

        seq_len = int(window.seq_len)

        valid_t = torch.arange(ball_uv_gt.shape[0]) < seq_len
        ball_in_frame_gt = (ball_visible > 0).to(torch.float32) * valid_t.to(torch.float32)
        if self.supervise_visible_only:
            ball_gt_visible = (ball_visible > 0).to(torch.float32) * valid_t.to(torch.float32)
        else:
            ball_gt_visible = valid_t.to(torch.float32)

        # Build event labels on full length then slice (consistent with event_detection).
        device = torch.device("cpu")
        shot_indices = extract_event_indices(scene.meta, self.label_cfg.shot_time_key)
        bounce_indices = extract_event_indices(scene.meta, self.label_cfg.bounce_time_key)
        y_shot = gaussian_soft_labels(
            length=full_len,
            event_indices=shot_indices,
            sigma=self.label_cfg.sigma_frames,
            device=device,
        )
        y_bounce = gaussian_soft_labels(
            length=full_len,
            event_indices=bounce_indices,
            sigma=self.label_cfg.sigma_frames,
            device=device,
        )
        targets = torch.stack([y_shot, y_bounce], dim=-1)[window.sl]

        return {
            "ball_uv_in": ball_uv_gt.clone(),
            "ball_vis": ball_gt_visible.clone(),
            "ball_uv_gt": ball_uv_gt,
            "ball_in_frame_gt": ball_in_frame_gt,
            "court_kp": court_kp,
            "court_vis": court_vis,
            "position_3d": position_3d,
            "ball_pos_world": ball_pos_world,
            "event_targets": targets,
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
        }

    def augment_sample(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.augment:
            return sample
        ball_uv_in, ball_obs_mask = _apply_corruption(
            ball_uv_gt=sample["ball_uv_gt"],
            ball_gt_visible=sample["ball_vis"],
            cfg=self.corruption,
        )
        sample["ball_uv_in"] = ball_uv_in
        sample["ball_vis"] = ball_obs_mask
        return sample


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        scene = base / "scene_000.npz"
        split_file = base / "train.txt"
        T = 8
        meta = {"num_frames": T, "shots": [{"t_start": 2, "t_bounce1": 5}]}
        np.savez(
            scene,
            meta=json.dumps(meta),
            num_cameras=np.array(1),
            cam_0_ball_uv=np.zeros((T, 2), dtype=np.float32),
            cam_0_ball_visible=np.ones((T,), dtype=np.float32),
            cam_0_court_kp_uv=np.zeros((20, 2), dtype=np.float32),
            cam_0_court_kp_visible=np.ones((20,), dtype=np.float32),
            ball_pos_norm=np.zeros((T, 3), dtype=np.float32),
            ball_pos_world=np.zeros((T, 3), dtype=np.float32),
        )
        split_file.write_text("scene_000.npz\n")
        cfg = {"data": {"seq_len_range": [8, 8], "cache_max_scenes": 0}}
        ds = BallMultitaskDataset(scene_dir=base, split_file="train.txt", config=cfg, augment=False)
        sample = ds[0]
        assert sample["ball_uv_in"].shape == (T, 2)
        assert sample["event_targets"].shape == (T, 2)
        print("ball_multitask.dataset smoke ok")
