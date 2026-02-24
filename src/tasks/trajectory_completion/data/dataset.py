"""Datasets for UV trajectory completion.

Primary dataset reads BLCS rally scenes saved as NPZ files and creates
corrupted inputs (noise + masking) paired with the original UV trajectory.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

from src.utils.data.blcs_npz_adapter import load_camera_view
from src.utils.dataset.npz_scene_dataset import NPZScene, NPZSceneDatasetBase, SceneDatasetConfig
from src.tasks.trajectory_completion.data.argument import TrajectoryArgumenter
from src.tasks.trajectory_completion.data.event_masking import extract_event_frames
from src.tasks.trajectory_completion.data.types import TrajectoryCompletionSample

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _build_valid_mask(T: int, seq_len: Tensor) -> Tensor:
    """Build a valid-length mask for a single sequence."""
    t = torch.arange(T, device=seq_len.device)
    return t < seq_len.to(torch.long)


def _crop_to_max_len(
    tensors: dict[str, Tensor],
    *,
    seq_len: int,
    max_seq_len: int,
    mode: Literal["random", "center"],
) -> tuple[dict[str, Tensor], int, int]:
    """Crop temporal tensors to max_seq_len with a shared offset."""
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")
    first = next(iter(tensors.values()))
    T = int(first.shape[0])
    if T <= max_seq_len:
        return tensors, min(seq_len, T), 0

    crop_len = max_seq_len
    max_start = max(0, seq_len - crop_len)
    if mode == "random" and max_start > 0:
        start = int(torch.randint(0, max_start + 1, (1,)).item())
    else:
        start = max_start // 2
    end = start + crop_len
    cropped = {k: v[start:end] for k, v in tensors.items()}
    new_seq_len = max(0, min(seq_len - start, crop_len))
    return cropped, new_seq_len, start


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
        self.config = config or {}
        data_cfg = self.config.get("data", {}) if hasattr(self.config, "get") else {}
        data_cfg = data_cfg or {}

        self.scene_dir = Path(scene_dir)
        self.max_seq_len = int(data_cfg.get("max_seq_len", 256))
        self.min_seq_len = int(data_cfg.get("min_seq_len", 16))
        self.supervise_visible_only = bool(data_cfg.get("supervise_visible_only", True))
        self.augment = bool(augment)
        crop_mode = "random" if self.augment else "center"

        argument_cfg = data_cfg.get("argument", {}) or {}
        self.argumenter = TrajectoryArgumenter(argument_cfg)
        ratio = argument_cfg.get("event_ratio", (2, 1))
        if isinstance(ratio, (list, tuple)) and len(ratio) == 2:
            self.event_ratio = (int(ratio[0]), int(ratio[1]))
        else:
            self.event_ratio = (2, 1)
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
        full_len = int(ball_uv_gt.shape[0])
        ball_visible = torch.from_numpy(view.ball_vis).to(torch.float32)
        court_kp = torch.from_numpy(view.court_kp).float()
        court_vis = torch.from_numpy(view.court_vis).to(torch.float32)

        seq_len = min(scene.num_frames, int(ball_uv_gt.shape[0]))
        if seq_len < self.min_seq_len:
            seq_len = min(self.min_seq_len, int(ball_uv_gt.shape[0]))
        crop_start = 0
        if ball_uv_gt.shape[0] > self.max_seq_len:
            cropped, seq_len, crop_start = _crop_to_max_len(
                {
                    "ball_uv_gt": ball_uv_gt,
                    "ball_visible": ball_visible,
                },
                seq_len=seq_len,
                max_seq_len=self.max_seq_len,
                mode=self.config.crop_mode,
            )
            ball_uv_gt = cropped["ball_uv_gt"]
            ball_visible = cropped["ball_visible"]

        seq_len_t = torch.tensor(seq_len, dtype=torch.long)
        valid_t = _build_valid_mask(ball_uv_gt.shape[0], seq_len_t).to(torch.float32)
        ball_in_frame_gt = (ball_visible > 0).to(torch.float32) * valid_t
        if self.supervise_visible_only:
            ball_gt_vis = (ball_visible > 0).to(torch.float32) * valid_t
        else:
            ball_gt_vis = valid_t

        if self.augment:
            event_frames = extract_event_frames(scene.meta, full_len)
            if crop_start > 0:
                event_frames = {
                    name: frames - int(crop_start) for name, frames in event_frames.items() if frames.numel() > 0
                }
            event_frames = {
                name: frames[(frames >= 0) & (frames < ball_uv_gt.shape[0])]
                for name, frames in event_frames.items()
            }
            ball_uv, ball_vis = self.argumenter(
                ball_uv_gt,
                ball_gt_vis,
                event_frames=event_frames,
                ratio=self.event_ratio,
            )
        else:
            ball_uv = ball_uv_gt.clone()
            ball_vis = ball_gt_vis.clone()

        return {
            "ball_uv": ball_uv,
            "ball_vis": ball_vis,
            "ball_uv_gt": ball_uv_gt,
            "ball_gt_vis": ball_gt_vis,
            "ball_in_frame_gt": ball_in_frame_gt,
            "court_kp": court_kp,
            "court_vis": court_vis,
            "seq_len": seq_len_t,
        }
