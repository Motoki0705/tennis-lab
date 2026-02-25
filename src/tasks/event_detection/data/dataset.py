"""Dataset classes for event detection using BLCS rally NPZ files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import Tensor

from src.tasks.event_detection.data.types import Event3DSample, EventUVSample
from src.utils.data.soft_labels import extract_event_indices, gaussian_soft_labels
from src.utils.dataset.npz_scene_dataset import NPZScene, NPZSceneDatasetBase, SceneDatasetConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _gaussian_soft_labels(
    length: int,
    event_indices: list[int],
    sigma: float,
    device: torch.device,
) -> Tensor:
    """Create soft labels with Gaussian peaks at given indices.

    .. deprecated::
        Use :func:`src.utils.data.soft_labels.gaussian_soft_labels` instead.
    """
    return gaussian_soft_labels(length, event_indices, sigma, device)


@dataclass(frozen=True)
class LabelConfig:
    """Configuration for event label generation."""

    sigma_frames: float = 2.5
    shot_time_key: str = "t_start"
    bounce_time_key: str = "t_bounce1"


class BLCSRallyEventDataset(NPZSceneDatasetBase[EventUVSample | Event3DSample]):
    """Event detection dataset from BLCS rally NPZ files.

    Supports two input modes:
    - UV: ball_uv + court_kp
    - 3D: ball_pos_world only
    """

    def __init__(
        self,
        *,
        scene_dir: str | Path,
        split_file: str | Path,
        input_type: Literal["uv", "3d"] = "uv",
        config: DictConfig | None = None,
        augment: bool = False,
    ) -> None:
        _scene_dir = Path(scene_dir)
        self.input_type = input_type
        self.hydra_cfg = config or {}
        self.augment = augment

        data_cfg = self.hydra_cfg.get("data", {}) or {}
        seq_len_range_cfg = data_cfg["seq_len_range"]
        seq_len_range = (int(seq_len_range_cfg[0]), int(seq_len_range_cfg[1]))

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

    def _make_targets(self, meta: dict, T: int, device: torch.device) -> Tensor:
        shot_times = extract_event_indices(meta, self.label_cfg.shot_time_key)
        bounce_times = extract_event_indices(meta, self.label_cfg.bounce_time_key)

        y_shot = gaussian_soft_labels(
            length=T,
            event_indices=shot_times,
            sigma=self.label_cfg.sigma_frames,
            device=device,
        )
        y_bounce = gaussian_soft_labels(
            length=T,
            event_indices=bounce_times,
            sigma=self.label_cfg.sigma_frames,
            device=device,
        )
        return torch.stack([y_shot, y_bounce], dim=-1)  # (T, 2)

    def build_sample(self, scene: NPZScene) -> EventUVSample | Event3DSample:
        meta = scene.meta
        device = torch.device("cpu")

        if self.input_type == "3d":
            ball_pos_world_full = scene.get_ball_pos_world()
            T_full = scene.effective_num_frames(int(ball_pos_world_full.shape[0]))
            window = self.select_window(scene, full_len=T_full)
            ball_pos_world = torch.from_numpy(scene.get_ball_pos_world(window=window)).float()
            targets = self._make_targets(meta=meta, T=T_full, device=device)
            targets = targets[window.sl]
            seq_len = torch.tensor(window.seq_len, dtype=torch.long)
            return {
                "ball_pos_world": ball_pos_world,
                "targets": targets,
                "seq_len": seq_len,
            }

        if self.input_type == "uv":
            cam_idx = self.select_camera(scene)
            ball_uv_full = scene.get_ball_uv(cam_idx)
            T_full = scene.effective_num_frames(int(ball_uv_full.shape[0]))
            window = self.select_window(scene, full_len=T_full)
            view = scene.get_camera_view(cam_idx, window=window)
            ball_uv = torch.from_numpy(view.ball_uv).float()
            ball_vis = torch.from_numpy(view.ball_visible).float()
            court_kp = torch.from_numpy(view.court_kp_uv).float()
            court_vis = torch.from_numpy(view.court_kp_visible).float()
            targets = self._make_targets(meta=meta, T=T_full, device=device)
            targets = targets[window.sl]
            seq_len = torch.tensor(window.seq_len, dtype=torch.long)

            return {
                "ball_uv": ball_uv,
                "ball_vis": ball_vis,
                "court_kp": court_kp,
                "court_vis": court_vis,
                "targets": targets,
                "seq_len": seq_len,
            }

        raise ValueError(
            f"Unsupported input_type={self.input_type!r}. Expected 'uv' or '3d'."
        )


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        scene_dir = Path(tmp_dir)
        scene_path = scene_dir / "scene_000.npz"
        split_path = scene_dir / "train.txt"
        T = 8
        meta = {"num_frames": T, "shots": [{"t_start": 2, "t_bounce1": 5}]}
        np.savez(
            scene_path,
            ball_pos_world=np.zeros((T, 3), dtype=np.float32),
            num_cameras=np.array(1),
            cam_0_ball_uv=np.zeros((T, 2), dtype=np.float32),
            cam_0_ball_visible=np.ones((T,), dtype=np.float32),
            cam_0_court_kp_uv=np.zeros((20, 2), dtype=np.float32),
            cam_0_court_kp_visible=np.ones((20,), dtype=np.float32),
            meta=json.dumps(meta),
        )
        split_path.write_text("scene_000.npz\n")
        cfg = {"data": {"seq_len_range": [T, T], "cache_max_scenes": 0}}
        blcs_uv = BLCSRallyEventDataset(scene_dir=scene_dir, split_file="train.txt", input_type="uv", config=cfg)
        sample_uv = blcs_uv[0]
        assert sample_uv["ball_uv"].shape == (T, 2)
        assert sample_uv["court_kp"].shape == (20, 2)

        blcs_3d = BLCSRallyEventDataset(scene_dir=scene_dir, split_file="train.txt", input_type="3d", config=cfg)
        sample_3d = blcs_3d[0]
        assert sample_3d["ball_pos_world"].shape == (T, 3)
    print("dataset smoke ok")
