"""Datasets for UV trajectory completion.

Primary dataset reads BLCS rally scenes saved as NPZ files and creates
corrupted inputs (noise + masking) paired with the original UV trajectory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.base.data.augmentation import add_gaussian_noise
from src.common.data.scene_cache import get_scene_cache, load_npz_scene

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _load_split_file(scene_dir: Path, split_file: str | Path) -> list[Path]:
    split_path = Path(split_file)
    if not split_path.is_absolute():
        split_path = scene_dir / split_path
    if not split_path.exists():
        return []
    scenes_base = scene_dir / "scenes" if (scene_dir / "scenes").exists() else scene_dir
    files: list[Path] = []
    for line in split_path.read_text().splitlines():
        name = line.strip()
        if not name:
            continue
        files.append(scenes_base / name)
    return files


def _resolve_scenes(scene_dir: Path, split_file: str | Path | None) -> list[Path]:
    scenes_base = scene_dir / "scenes" if (scene_dir / "scenes").exists() else scene_dir
    if split_file is not None:
        scenes = _load_split_file(scene_dir, split_file)
        if scenes:
            return scenes
    return sorted(scenes_base.glob("*.npz"))


def _select_camera(camera_mode: Any, num_cameras: int) -> int:
    if camera_mode == "random":
        return int(np.random.randint(0, num_cameras))
    if isinstance(camera_mode, int):
        return min(int(camera_mode), num_cameras - 1)
    if isinstance(camera_mode, str) and camera_mode.isdigit():
        return min(int(camera_mode), num_cameras - 1)
    return 0


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


class BLCSUVTrajectoryCompletionDataset(Dataset):
    """Trajectory completion dataset backed by BLCS rally scenes (npz)."""

    def __init__(
        self,
        *,
        scene_dir: str | Path,
        split_file: str | Path | None,
        config: DictConfig | None = None,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or {}
        data_cfg = self.config.get("data", {}) if hasattr(self.config, "get") else {}
        data_cfg = data_cfg or {}

        self.scene_dir = Path(scene_dir)
        self.scenes = _resolve_scenes(self.scene_dir, split_file)
        if not self.scenes:
            raise RuntimeError(f"No scenes found under {self.scene_dir}")

        self.camera_mode = data_cfg.get("camera_mode", "random")
        self.max_seq_len = int(data_cfg.get("max_seq_len", 256))
        self.min_seq_len = int(data_cfg.get("min_seq_len", 16))
        self.supervise_visible_only = bool(data_cfg.get("supervise_visible_only", True))
        self.augment = bool(augment)
        self.cache_max_scenes = int(data_cfg.get("cache_max_scenes", 128))
        self._scene_cache = (
            get_scene_cache(load_fn=load_npz_scene, maxsize=self.cache_max_scenes)
            if self.cache_max_scenes > 0
            else None
        )

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

    def __len__(self) -> int:
        return len(self.scenes)

    def _load_scene(self, path: Path) -> dict[str, Tensor]:
        data = (
            self._scene_cache.get(path)
            if self._scene_cache is not None
            else load_npz_scene(path)
        )
        meta_raw: Any = data.get("meta", {})
        if isinstance(meta_raw, (bytes, bytearray)):
            meta_raw = meta_raw.decode("utf-8")
        if isinstance(meta_raw, str):
            meta = json.loads(meta_raw)
        else:
            meta = meta_raw if isinstance(meta_raw, dict) else {}

        num_cameras = int(data["num_cameras"])
        cam_idx = _select_camera(self.camera_mode, num_cameras)
        prefix = f"cam_{cam_idx}_"

        ball_uv = torch.from_numpy(data[f"{prefix}ball_uv"]).float()
        ball_visible = torch.from_numpy(data[f"{prefix}ball_visible"]).to(torch.float32)
        court_kp = torch.from_numpy(data[f"{prefix}court_kp_uv"]).float()
        court_vis = torch.from_numpy(data[f"{prefix}court_kp_visible"]).to(torch.float32)

        seq_len = int(meta.get("num_frames", ball_uv.shape[0]))
        seq_len = min(seq_len, int(ball_uv.shape[0]))

        return {
            "ball_uv_gt": ball_uv,
            "ball_visible": ball_visible,
            "court_kp": court_kp,
            "court_vis": court_vis,
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
        }

    def _crop(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        T = int(sample["ball_uv_gt"].shape[0])
        seq_len = int(sample["seq_len"].item())

        if seq_len < self.min_seq_len:
            seq_len = min(self.min_seq_len, T)
            sample["seq_len"] = torch.tensor(seq_len, dtype=torch.long)

        if T <= self.max_seq_len:
            return sample

        crop_len = self.max_seq_len
        max_start = max(0, seq_len - crop_len)
        if self.augment and max_start > 0:
            start = int(torch.randint(0, max_start + 1, (1,)).item())
        else:
            start = max_start // 2
        end = start + crop_len

        sample["ball_uv_gt"] = sample["ball_uv_gt"][start:end]
        sample["ball_visible"] = sample["ball_visible"][start:end]
        sample["seq_len"] = torch.clamp(sample["seq_len"] - start, min=0, max=crop_len)
        return sample

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        sample = self._load_scene(self.scenes[idx])
        sample = self._crop(sample)

        ball_uv_gt = sample["ball_uv_gt"]
        seq_len = int(sample["seq_len"].item())
        ball_visible = sample["ball_visible"]

        valid_t = torch.arange(ball_uv_gt.shape[0]) < seq_len
        if self.supervise_visible_only:
            ball_gt_visible = (ball_visible > 0).to(torch.float32) * valid_t.to(torch.float32)
        else:
            ball_gt_visible = valid_t.to(torch.float32)

        if self.augment:
            ball_uv_in, ball_obs_mask = _apply_corruption(
                ball_uv_gt=ball_uv_gt,
                ball_gt_visible=ball_gt_visible,
                cfg=self.corruption,
            )
        else:
            ball_uv_in = ball_uv_gt.clone()
            ball_obs_mask = ball_gt_visible.clone()

        return {
            "ball_uv_in": ball_uv_in,
            "ball_obs_mask": ball_obs_mask,
            "ball_uv_gt": ball_uv_gt,
            "ball_gt_mask": ball_gt_visible,
            "court_kp": sample["court_kp"],
            "court_vis": sample["court_vis"],
            "seq_len": sample["seq_len"],
        }


class DummyUVTrajectoryCompletionDataset(Dataset):
    """Small in-memory dataset for smoke tests and dry runs."""

    def __init__(
        self,
        *,
        num_samples: int = 16,
        max_seq_len: int = 64,
        corruption: CorruptionConfig | None = None,
    ) -> None:
        super().__init__()
        self.num_samples = int(num_samples)
        self.max_seq_len = int(max_seq_len)
        self.corruption = corruption or CorruptionConfig()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Tensor]:  # noqa: ARG002
        T = self.max_seq_len
        seq_len = int(torch.randint(low=max(8, T // 2), high=T + 1, size=(1,)).item())
        ball_uv_gt = torch.rand(T, 2)
        court_kp = torch.rand(20, 2)
        court_vis = torch.ones(20)
        ball_gt_visible = (torch.arange(T) < seq_len).to(torch.float32)

        ball_uv_in, ball_obs_mask = _apply_corruption(
            ball_uv_gt=ball_uv_gt,
            ball_gt_visible=ball_gt_visible,
            cfg=self.corruption,
        )

        return {
            "ball_uv_in": ball_uv_in,
            "ball_obs_mask": ball_obs_mask,
            "ball_uv_gt": ball_uv_gt,
            "ball_gt_mask": ball_gt_visible,
            "court_kp": court_kp,
            "court_vis": court_vis,
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
        }


if __name__ == "__main__":
    ds = DummyUVTrajectoryCompletionDataset(num_samples=2, max_seq_len=32)
    item = ds[0]
    assert item["ball_uv_in"].shape == (32, 2)
    assert item["ball_uv_gt"].shape == (32, 2)
    assert item["ball_obs_mask"].shape == (32,)
    assert item["court_kp"].shape == (20, 2)
    print("trajectory_completion.dataset smoke ok")
