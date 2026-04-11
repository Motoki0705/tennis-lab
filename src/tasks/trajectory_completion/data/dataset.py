"""Datasets for UV trajectory completion.

Primary dataset reads BLCS rally scenes saved as NPZ files and creates
corrupted inputs (noise + masking) paired with the original UV trajectory.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.tasks.base.data.scene_dataset import NPZScene, NPZSceneDatasetBase
from src.tasks.trajectory_completion.data.argument import TrajectoryArgumenter
from src.utils.data.event_utils import extract_event_frames
from src.tasks.trajectory_completion.data.types import TrajectoryCompletionSample

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _validate_num_court_kp(num_court_kp: int) -> int:
    """Validate configured court keypoint count."""
    if not 1 <= int(num_court_kp) <= 20:
        raise ValueError(
            f"data.num_court_kp must be in [1, 20], got {int(num_court_kp)}."
        )
    return int(num_court_kp)


def _build_valid_mask(T: int, seq_len: Tensor) -> Tensor:
    """Build a valid-length mask for a single sequence."""
    t = torch.arange(T, device=seq_len.device)
    return t < seq_len.to(torch.long)


class BLCSUVTrajectoryCompletionDataset(NPZSceneDatasetBase[TrajectoryCompletionSample]):
    """Trajectory completion dataset backed by BLCS rally scenes (npz)."""

    def __init__(
        self,
        *,
        scene_dir: str | Path,
        split_file: str | Path,
        config: DictConfig | None = None,
        augment: bool = True,
    ) -> None:
        self.hydra_cfg = config or {}
        self.augment = bool(augment)
        data_cfg = self._resolve_data_cfg(self.hydra_cfg)
        self._configure_task(data_cfg)
        super().__init__(
            config=self._build_scene_dataset_config(
                scene_dir=scene_dir, split_file=split_file, data_cfg=data_cfg,
            )
        )

    # -- Composed-method hooks ------------------------------------------

    def _configure_task(self, data_cfg: dict) -> None:  # type: ignore[override]
        self.supervise_visible_only = bool(data_cfg.get("supervise_visible_only", True))

        argument_cfg = data_cfg.get("argument", {}) or {}
        self.argumenter = TrajectoryArgumenter(argument_cfg)
        ratio = argument_cfg.get("event_ratio", (2, 1))
        if (
            not isinstance(ratio, Sequence)
            or isinstance(ratio, (str, bytes))
            or len(ratio) != 2
        ):
            raise ValueError(
                "data.argument.event_ratio must be a list/tuple of length 2."
            )
        self.event_ratio = (int(ratio[0]), int(ratio[1]))
        # Number of court keypoints to use (first N from the canonical order)
        self.num_court_kp = _validate_num_court_kp(data_cfg.get("num_court_kp", 20))

    def build_sample(self, scene: NPZScene) -> TrajectoryCompletionSample:
        cam_idx = self.select_camera(scene)
        ball_uv_full = scene.get_camera_array(cam_idx, "ball_uv")
        full_len = scene.effective_num_frames(int(ball_uv_full.shape[0]))
        window = self.select_window(scene, full_len=full_len)

        ball_uv_gt = torch.from_numpy(scene.get_camera_array(cam_idx, "ball_uv", window=window)).float()
        ball_visible = torch.from_numpy(scene.get_camera_array(cam_idx, "ball_visible", window=window)).to(torch.float32)
        court_kp = torch.from_numpy(scene.get_camera_array(cam_idx, "court_kp_uv")).float()
        court_vis = torch.from_numpy(scene.get_camera_array(cam_idx, "court_kp_visible")).to(torch.float32)
        court_kp = court_kp[: self.num_court_kp]
        court_vis = court_vis[: self.num_court_kp]

        seq_len_t = torch.tensor(window.seq_len, dtype=torch.long)
        valid_t = _build_valid_mask(ball_uv_gt.shape[0], seq_len_t).to(torch.float32)
        ball_in_frame_gt = (ball_visible > 0).to(torch.float32) * valid_t
        if self.supervise_visible_only:
            ball_gt_vis = (ball_visible > 0).to(torch.float32) * valid_t
        else:
            ball_gt_vis = valid_t

        sample: TrajectoryCompletionSample = {
            "ball_uv": ball_uv_gt.clone(),
            "ball_vis": ball_gt_vis.clone(),
            "ball_uv_gt": ball_uv_gt,
            "ball_gt_vis": ball_gt_vis,
            "ball_in_frame_gt": ball_in_frame_gt,
            "court_kp": court_kp,
            "court_vis": court_vis,
            "seq_len": seq_len_t,
        }
        # Store transient event info for augment_sample (popped after use).
        event_frames = extract_event_frames(scene.meta, full_len, offset=window.start)
        sample["_event_frames"] = event_frames  # type: ignore[typeddict-unknown-key]
        return sample

    def augment_sample(
        self, sample: TrajectoryCompletionSample
    ) -> TrajectoryCompletionSample:
        event_frames: dict[str, Tensor] | None = sample.pop("_event_frames", None)  # type: ignore[misc]
        if not self.augment:
            return sample
        if event_frames is None:
            event_frames = {"bounce": torch.empty(0, dtype=torch.long), "shot": torch.empty(0, dtype=torch.long)}
        ball_uv, ball_vis = self.argumenter(
            sample["ball_uv_gt"],
            sample["ball_gt_vis"],
            event_frames=event_frames,
            ratio=self.event_ratio,
        )
        sample["ball_uv"] = ball_uv
        sample["ball_vis"] = ball_vis
        return sample
