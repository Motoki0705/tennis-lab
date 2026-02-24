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
from src.utils.data.blcs_npz_adapter import load_3d_arrays, load_camera_view
from src.utils.dataset.npz_scene_dataset import NPZScene, NPZSceneDatasetBase, SceneDatasetConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _crop_to_max_len(
    tensors: dict[str, Tensor],
    *,
    seq_len: int,
    max_seq_len: int,
    mode: Literal["random", "center"] = "random",
) -> tuple[dict[str, Tensor], int]:
    """Crop temporal tensors to max_seq_len with a shared offset."""
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")
    first = next(iter(tensors.values()))
    T = int(first.shape[0])
    if T <= max_seq_len:
        return tensors, min(seq_len, T)

    crop_len = max_seq_len
    max_start = max(0, seq_len - crop_len)
    if mode == "random" and max_start > 0:
        start = int(torch.randint(0, max_start + 1, (1,)).item())
    else:
        start = max_start // 2
    end = start + crop_len

    cropped = {k: v[start:end] for k, v in tensors.items()}
    new_seq_len = max(0, min(seq_len - start, crop_len))
    return cropped, new_seq_len


def _gaussian_soft_labels(
    length: int,
    event_indices: list[int],
    sigma: float,
    device: torch.device,
) -> Tensor:
    """Create soft labels with Gaussian peaks at given indices.

    Args:
        length: Sequence length T.
        event_indices: List of event frame indices (0-based).
        sigma: Standard deviation in frames.
        device: Output device.

    Returns:
        Soft label tensor of shape (T,).
    """
    if length <= 0:
        return torch.zeros((0,), device=device)
    if not event_indices:
        return torch.zeros((length,), device=device)

    t = torch.arange(length, device=device, dtype=torch.float32)
    out = torch.zeros((length,), device=device, dtype=torch.float32)
    denom = 2.0 * float(sigma) * float(sigma)
    for idx in event_indices:
        if 0 <= idx < length:
            out = torch.maximum(out, torch.exp(-((t - float(idx)) ** 2) / denom))
    return out


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
        scene_dir: str | Path,
        split: Literal["train", "val", "test"] = "train",
        input_type: Literal["uv", "3d"] = "uv",
        config: DictConfig | None = None,
        augment: bool = False,
    ) -> None:
        self.scene_dir = Path(scene_dir)
        self.input_type = input_type
        self.config = config or {}
        self.augment = augment

        data_cfg = self.config.get("data", {}) or {}
        crop_mode = str(data_cfg.get("crop_mode", "center"))

        label_cfg = data_cfg.get("label", {}) or {}
        self.label_cfg = LabelConfig(
            sigma_frames=float(label_cfg.get("sigma_frames", 2.5)),
            shot_time_key=str(label_cfg.get("shot_time_key", "t_start")),
            bounce_time_key=str(label_cfg.get("bounce_time_key", "t_bounce1")),
        )

        super().__init__(
            config=SceneDatasetConfig(
                scene_dir=self.scene_dir,
                split=split,
                split_file=None,
                max_seq_len=int(data_cfg.get("max_seq_len", 256)),
                cache_max_scenes=int(data_cfg.get("cache_max_scenes", 128)),
                camera_mode=data_cfg.get("camera_mode", "random"),
                crop_mode=crop_mode,
            )
        )

    def _make_targets(self, meta: dict, T: int, device: torch.device) -> Tensor:
        shots = meta.get("shots", []) or []
        shot_times: list[int] = []
        bounce_times: list[int] = []
        for s in shots:
            if not isinstance(s, dict):
                continue
            t_shot = int(s.get(self.label_cfg.shot_time_key, -1))
            t_bounce = int(s.get(self.label_cfg.bounce_time_key, -1))
            if t_shot >= 0:
                shot_times.append(t_shot)
            if t_bounce >= 0:
                bounce_times.append(t_bounce)

        y_shot = _gaussian_soft_labels(
            length=T,
            event_indices=shot_times,
            sigma=self.label_cfg.sigma_frames,
            device=device,
        )
        y_bounce = _gaussian_soft_labels(
            length=T,
            event_indices=bounce_times,
            sigma=self.label_cfg.sigma_frames,
            device=device,
        )
        return torch.stack([y_shot, y_bounce], dim=-1)  # (T, 2)

    def build_sample(self, scene: NPZScene) -> EventUVSample | Event3DSample:
        data = scene.data
        meta = scene.meta
        max_seq_len = self.config.max_seq_len
        device = torch.device("cpu")

        if self.input_type == "3d":
            arrays = load_3d_arrays(data, position_world_key="ball_pos_world")
            ball_pos_world = torch.from_numpy(arrays["ball_pos_world"]).float()
            T_full = min(scene.num_frames, int(ball_pos_world.shape[0]))
            targets = self._make_targets(meta=meta, T=T_full, device=device)
            seq_len = torch.tensor(T_full, dtype=torch.long)
            if ball_pos_world.shape[0] > max_seq_len:
                cropped, T = _crop_to_max_len(
                    {"ball_pos_world": ball_pos_world},
                    seq_len=int(seq_len.item()),
                    max_seq_len=max_seq_len,
                    mode=self.config.crop_mode,
                )
                ball_pos_world = cropped["ball_pos_world"]
                seq_len = torch.tensor(T, dtype=torch.long)
                targets = targets[:T]
            return {
                "ball_pos_world": ball_pos_world,
                "targets": targets,
                "seq_len": seq_len,
            }

        view = load_camera_view(data, scene.camera_idx)
        ball_uv = torch.from_numpy(view.ball_uv).float()
        ball_vis = torch.from_numpy(view.ball_vis).float()
        court_kp = torch.from_numpy(view.court_kp).float()
        court_vis = torch.from_numpy(view.court_vis).float()
        T_full = min(scene.num_frames, int(ball_uv.shape[0]))
        targets = self._make_targets(meta=meta, T=T_full, device=device)
        seq_len = torch.tensor(T_full, dtype=torch.long)

        if ball_uv.shape[0] > max_seq_len:
            cropped, T = _crop_to_max_len(
                {
                    "ball_uv": ball_uv,
                    "ball_vis": ball_vis,
                },
                seq_len=int(seq_len.item()),
                max_seq_len=max_seq_len,
                mode=self.config.crop_mode,
            )
            ball_uv = cropped["ball_uv"]
            ball_vis = cropped["ball_vis"]
            seq_len = torch.tensor(T, dtype=torch.long)
            targets = targets[:T]

        return {
            "ball_uv": ball_uv,
            "ball_vis": ball_vis,
            "court_kp": court_kp,
            "court_vis": court_vis,
            "targets": targets,
            "seq_len": seq_len,
        }


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        scene_dir = Path(tmp_dir)
        scene_path = scene_dir / "scene_000.npz"
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
        cfg = {"data": {"max_seq_len": T, "cache_max_scenes": 0}}
        blcs_uv = BLCSRallyEventDataset(scene_dir, split="train", input_type="uv", config=cfg)
        sample_uv = blcs_uv[0]
        assert sample_uv["ball_uv"].shape == (T, 2)
        assert sample_uv["court_kp"].shape == (20, 2)

        blcs_3d = BLCSRallyEventDataset(scene_dir, split="train", input_type="3d", config=cfg)
        sample_3d = blcs_3d[0]
        assert sample_3d["ball_pos_world"].shape == (T, 3)
    print("dataset smoke ok")
