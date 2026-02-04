"""Datasets for UV trajectory completion.

Primary dataset reads BLCS rally scenes saved as NPZ files and creates
corrupted inputs (noise + masking) paired with the original UV trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.common.dataset.augmentation import add_gaussian_noise
from src.common.data.blcs_npz_adapter import load_camera_view
from src.common.dataset.npz_scene_dataset import NPZScene, NPZSceneDatasetBase, SceneDatasetConfig
from src.common.dataset.sequence import build_valid_mask, crop_to_max_len
from src.trajectory_completion.data.types import TrajectoryCompletionSample

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass(frozen=True)
class CorruptionConfig:
    """Configuration for generating corrupted trajectory inputs.

    Args:
        enabled: Enable corruption.
        noise_std: Stddev of Gaussian noise added to observed UV points.
        clamp_unit: Clamp noisy points to [0, 1].
        point_dropout_prob: Probability of dropping each visible frame.
        gap_dropout_prob: Probability of creating a contiguous missing gap.
        max_gap_len: Maximum missing gap length.
        max_masked_ratio: Upper bound of masked frames among visible frames.
        outlier_prob: Probability of replacing an observed point with a random UV.
    """

    enabled: bool = True
    noise_std: float = 0.01
    clamp_unit: bool = True
    point_dropout_prob: float = 0.05
    gap_dropout_prob: float = 0.25
    max_gap_len: int = 12
    max_masked_ratio: float = 0.6
    outlier_prob: float = 0.0


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


class BLCSUVTrajectoryCompletionDataset(NPZSceneDatasetBase[TrajectoryCompletionSample]):
    """Trajectory completion dataset backed by BLCS rally scenes (npz)."""

    def __init__(
        self,
        *,
        scene_dir: str | Path,
        split: str | None = None,
        split_file: str | Path | None = None,
        config: DictConfig | None = None,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or {}
        data_cfg = self.config.get("data", {}) if hasattr(self.config, "get") else {}
        data_cfg = data_cfg or {}

        self.scene_dir = Path(scene_dir)
        self.max_seq_len = int(data_cfg.get("max_seq_len", 256))
        self.min_seq_len = int(data_cfg.get("min_seq_len", 16))
        self.supervise_visible_only = bool(data_cfg.get("supervise_visible_only", True))
        self.augment = bool(augment)
        crop_mode = "random" if self.augment else "center"

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
        super().__init__(
            config=SceneDatasetConfig(
                scene_dir=self.scene_dir,
                split=split,
                split_file=Path(split_file) if split_file is not None else None,
                min_seq_len=self.min_seq_len,
                max_seq_len=self.max_seq_len,
                cache_max_scenes=int(data_cfg.get("cache_max_scenes", 128)),
                camera_mode=data_cfg.get("camera_mode", "random"),
                crop_mode=crop_mode,
            )
        )

    def build_sample(self, scene: NPZScene) -> TrajectoryCompletionSample:
        view = load_camera_view(scene.data, scene.camera_idx)
        ball_uv_gt = torch.from_numpy(view.ball_uv).float()
        ball_visible = torch.from_numpy(view.ball_vis).to(torch.float32)
        court_kp = torch.from_numpy(view.court_kp).float()
        court_vis = torch.from_numpy(view.court_vis).to(torch.float32)

        seq_len = min(scene.num_frames, int(ball_uv_gt.shape[0]))
        if seq_len < self.min_seq_len:
            seq_len = min(self.min_seq_len, int(ball_uv_gt.shape[0]))
        if ball_uv_gt.shape[0] > self.max_seq_len:
            cropped, seq_len = crop_to_max_len(
                {"ball_uv_gt": ball_uv_gt, "ball_visible": ball_visible},
                seq_len=seq_len,
                max_seq_len=self.max_seq_len,
                mode=self.config.crop_mode,
            )
            ball_uv_gt = cropped["ball_uv_gt"]
            ball_visible = cropped["ball_visible"]

        seq_len_t = torch.tensor(seq_len, dtype=torch.long)
        valid_t = build_valid_mask(ball_uv_gt.shape[0], seq_len_t).to(torch.float32)
        if self.supervise_visible_only:
            ball_vis = (ball_visible > 0).to(torch.float32) * valid_t
        else:
            ball_vis = valid_t

        if self.augment:
            ball_uv_in, ball_obs_mask = _apply_corruption(
                ball_uv_gt=ball_uv_gt,
                ball_gt_visible=ball_vis,
                cfg=self.corruption,
            )
        else:
            ball_uv_in = ball_uv_gt.clone()
            ball_obs_mask = ball_vis.clone()

        return {
            "ball_uv_in": ball_uv_in,
            "ball_obs_mask": ball_obs_mask,
            "ball_uv_gt": ball_uv_gt,
            "ball_vis": ball_vis,
            "court_kp": court_kp,
            "court_vis": court_vis,
            "seq_len": seq_len_t,
        }
