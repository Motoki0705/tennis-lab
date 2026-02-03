"""Dataset classes for event detection using BLCS rally NPZ files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.event_detection.data.types import Event3DSample, EventUVSample
from src.common.data.scene_cache import get_scene_cache, load_npz_scene

if TYPE_CHECKING:
    from omegaconf import DictConfig


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


def _load_meta(data: dict[str, Any] | np.lib.npyio.NpzFile) -> dict:
    """Load and decode metadata from a scene payload."""
    if isinstance(data, dict):
        meta_raw = data.get("meta", {})
    else:
        meta_raw = data["meta"].item() if hasattr(data["meta"], "item") else data["meta"]
    if isinstance(meta_raw, (bytes, bytearray)):
        meta_raw = meta_raw.decode("utf-8")
    if isinstance(meta_raw, str):
        return json.loads(meta_raw)
    return meta_raw if isinstance(meta_raw, dict) else {}


def _resolve_scenes_base(scene_dir: Path) -> Path:
    scenes_subdir = scene_dir / "scenes"
    return scenes_subdir if scenes_subdir.exists() else scene_dir


def _load_split(scene_dir: Path, split: str) -> list[Path]:
    split_path = scene_dir / f"{split}.txt"
    if not split_path.exists():
        return sorted(_resolve_scenes_base(scene_dir).glob("*.npz"))
    base = _resolve_scenes_base(scene_dir)
    paths: list[Path] = []
    with open(split_path) as f:
        for line in f:
            name = line.strip()
            if name:
                paths.append(base / name)
    return paths


@dataclass(frozen=True)
class LabelConfig:
    """Configuration for event label generation."""

    sigma_frames: float = 2.5
    shot_time_key: str = "t_start"
    bounce_time_key: str = "t_bounce1"


class BLCSRallyEventDataset(Dataset):
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
        super().__init__()
        self.scene_dir = Path(scene_dir)
        self.split = split
        self.input_type = input_type
        self.config = config or {}
        self.augment = augment

        data_cfg = self.config.get("data", {}) or {}
        self.max_seq_len = int(data_cfg.get("max_seq_len", 256))
        self.camera_mode = data_cfg.get("camera_mode", "random")
        self.cache_max_scenes = int(data_cfg.get("cache_max_scenes", 128))
        self._scene_cache = (
            get_scene_cache(load_fn=load_npz_scene, maxsize=self.cache_max_scenes)
            if self.cache_max_scenes > 0
            else None
        )

        label_cfg = data_cfg.get("label", {}) or {}
        self.label_cfg = LabelConfig(
            sigma_frames=float(label_cfg.get("sigma_frames", 2.5)),
            shot_time_key=str(label_cfg.get("shot_time_key", "t_start")),
            bounce_time_key=str(label_cfg.get("bounce_time_key", "t_bounce1")),
        )

        self.scenes = _load_split(self.scene_dir, split)

    def __len__(self) -> int:
        return len(self.scenes)

    def _select_camera(self, num_cameras: int) -> int:
        if num_cameras <= 0:
            return 0
        if self.camera_mode == "random":
            return int(np.random.randint(0, num_cameras))
        if isinstance(self.camera_mode, int):
            return min(int(self.camera_mode), num_cameras - 1)
        return 0

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

    def __getitem__(self, idx: int) -> EventUVSample | Event3DSample:
        path = self.scenes[idx]
        data = (
            self._scene_cache.get(path)
            if self._scene_cache is not None
            else load_npz_scene(path)
        )
        meta = _load_meta(data)

        T_full = int(meta.get("num_frames", int(data["ball_pos_world"].shape[0])))
        T = min(T_full, self.max_seq_len)
        device = torch.device("cpu")

        targets = self._make_targets(meta=meta, T=T, device=device)
        seq_len = torch.tensor(T, dtype=torch.long)

        if self.input_type == "3d":
            ball_pos_world = torch.from_numpy(data["ball_pos_world"][:T]).float()
            return {
                "ball_pos_world": ball_pos_world,
                "targets": targets,
                "seq_len": seq_len,
            }

        # UV input (single camera selected)
        num_cameras = int(data["num_cameras"])
        cam_idx = self._select_camera(num_cameras)
        prefix = f"cam_{cam_idx}_"

        ball_uv = torch.from_numpy(data[f"{prefix}ball_uv"][:T]).float()
        ball_vis = torch.from_numpy(data[f"{prefix}ball_visible"][:T]).float()
        court_kp = torch.from_numpy(data[f"{prefix}court_kp_uv"]).float()
        court_vis = torch.from_numpy(data[f"{prefix}court_kp_visible"]).float()

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
