"""Scene IO helpers for event detection visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch import Tensor

from src.utils.data.scene_cache import load_npz_scene
from src.tasks.event_detection.visualization.types import (
    RuntimeConfig,
    Traj3DEventInputs,
    UVEventInputs,
)


@dataclass(frozen=True)
class EventLabelConfig:
    """Configuration for event label generation (shot/bounce)."""

    sigma_frames: float = 2.5
    shot_time_key: str = "t_start"
    bounce_time_key: str = "t_bounce1"


def select_camera(camera: Any, num_cameras: int) -> int:
    """Select a camera index from config values."""
    if num_cameras <= 0:
        return 0
    if camera is None:
        return 0
    if camera == "random":
        return int(torch.randint(0, num_cameras, (1,)).item())
    if isinstance(camera, int):
        return min(max(int(camera), 0), num_cameras - 1)
    if isinstance(camera, str) and camera.isdigit():
        return min(max(int(camera), 0), num_cameras - 1)
    return 0


def extract_event_indices(
    meta: dict[str, Any], *, cfg: EventLabelConfig
) -> tuple[list[int], list[int]]:
    """Extract shot/bounce frame indices from rally metadata."""
    shots = meta.get("shots", []) or []
    shot_times: list[int] = []
    bounce_times: list[int] = []

    for shot in shots:
        if not isinstance(shot, dict):
            continue
        t_shot = int(shot.get(cfg.shot_time_key, -1))
        t_bounce = int(shot.get(cfg.bounce_time_key, -1))
        if t_shot >= 0:
            shot_times.append(t_shot)
        if t_bounce >= 0:
            bounce_times.append(t_bounce)

    return sorted(set(shot_times)), sorted(set(bounce_times))


def gaussian_soft_labels(
    length: int,
    event_indices: list[int],
    *,
    sigma: float,
    device: torch.device,
) -> Tensor:
    """Create soft labels with Gaussian peaks at the given frame indices."""
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


def build_targets(
    length: int,
    *,
    shot_indices: list[int],
    bounce_indices: list[int],
    cfg: EventLabelConfig,
    device: torch.device,
) -> Tensor:
    """Build stacked targets of shape ``(T, 2)`` for (shot, bounce)."""
    y_shot = gaussian_soft_labels(
        length,
        shot_indices,
        sigma=float(cfg.sigma_frames),
        device=device,
    )
    y_bounce = gaussian_soft_labels(
        length,
        bounce_indices,
        sigma=float(cfg.sigma_frames),
        device=device,
    )
    return torch.stack([y_shot, y_bounce], dim=-1)


def resolve_device(device: str) -> str:
    """Resolve auto device selection."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def set_seed(seed: int) -> None:
    """Set deterministic random seeds for visualization sampling."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    """Build a runtime config from composed Hydra config."""
    vis = cfg.visualization
    run = cfg.run

    top_k = vis.get("top_k")
    top_k_value = int(top_k) if top_k is not None else None
    task = str(vis.get("task", "uv")).strip().lower()

    return RuntimeConfig(
        task=task,
        mode=str(vis.mode),
        scene_path=Path(to_absolute_path(str(vis.scene_path))),
        camera=vis.get("camera"),
        fps=float(vis.fps),
        save=Path(to_absolute_path(str(vis.save))) if vis.save else None,
        info=bool(vis.info),
        checkpoint=to_absolute_path(str(vis.checkpoint)) if vis.checkpoint else None,
        device=resolve_device(str(run.device)),
        output=to_absolute_path(str(vis.output)) if vis.output else None,
        seed=int(run.seed),
        threshold=float(vis.threshold),
        min_distance=max(1, int(vis.min_distance)),
        top_k=top_k_value,
        event_radius_frames=max(0, int(vis.get("event_radius_frames", 6))),
        event_sigma_frames=max(1e-6, float(vis.get("event_sigma_frames", 2.5))),
        show_court_lines=bool(vis.get("show_court_lines", True)),
        hydra_cfg=cfg,
    )


def _label_cfg_from_hydra(cfg: DictConfig) -> EventLabelConfig:
    data_cfg = cfg.get("data", {}) or {}
    label_cfg = data_cfg.get("label", {}) or {}
    return EventLabelConfig(
        sigma_frames=float(label_cfg.get("sigma_frames", 2.5)),
        shot_time_key=str(label_cfg.get("shot_time_key", "t_start")),
        bounce_time_key=str(label_cfg.get("bounce_time_key", "t_bounce1")),
    )


def _resolve_num_court_kp(cfg: DictConfig) -> int:
    """Read and validate the configured court keypoint count."""
    data_cfg = cfg.get("data", {}) or {}
    num_court_kp = int(data_cfg.get("num_court_kp", 20))
    if not 1 <= num_court_kp <= 20:
        raise ValueError(f"data.num_court_kp must be in [1, 20], got {num_court_kp}.")
    return num_court_kp


def _load_uv_arrays(
    payload: dict[str, Any], camera_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prefix = f"cam_{camera_idx}_"

    ball_uv_key = f"{prefix}ball_uv" if f"{prefix}ball_uv" in payload else "ball_uv"
    ball_vis_key = (
        f"{prefix}ball_visible"
        if f"{prefix}ball_visible" in payload
        else "ball_visible"
    )
    court_kp_key = (
        f"{prefix}court_kp_uv" if f"{prefix}court_kp_uv" in payload else "court_kp_uv"
    )
    court_vis_key = (
        f"{prefix}court_kp_visible"
        if f"{prefix}court_kp_visible" in payload
        else "court_kp_visible"
    )

    missing = [
        k
        for k in (ball_uv_key, ball_vis_key, court_kp_key, court_vis_key)
        if k not in payload
    ]
    if missing:
        raise KeyError(f"Missing keys in scene NPZ: {missing}")

    ball_uv = np.asarray(payload[ball_uv_key], dtype=np.float32)
    ball_vis = np.asarray(payload[ball_vis_key], dtype=np.float32)
    court_kp = np.asarray(payload[court_kp_key], dtype=np.float32)
    court_vis = np.asarray(payload[court_vis_key], dtype=np.float32)
    return ball_uv, ball_vis, court_kp, court_vis


def load_uv_inputs(cfg: RuntimeConfig) -> UVEventInputs:
    """Load a BLCS scene and build UV event visualization inputs."""
    set_seed(cfg.seed)

    payload = load_npz_scene(cfg.scene_path)
    meta = payload.get("meta", {})

    num_cameras = int(payload.get("num_cameras", 1))
    cam_idx = select_camera(cfg.camera, num_cameras)
    ball_uv_full, ball_vis_full, court_kp, court_vis = _load_uv_arrays(payload, cam_idx)
    num_court_kp = _resolve_num_court_kp(cfg.hydra_cfg)
    court_kp = court_kp[:num_court_kp]
    court_vis = court_vis[:num_court_kp]

    t_full = int(ball_uv_full.shape[0])
    num_frames_meta = int(meta.get("num_frames", t_full))
    max_seq_len = int((cfg.hydra_cfg.get("data", {}) or {}).get("max_seq_len", t_full))
    t_len = min(t_full, max(0, num_frames_meta), max_seq_len)

    ball_uv = ball_uv_full[:t_len]
    ball_vis = ball_vis_full[:t_len] > 0

    label_cfg = _label_cfg_from_hydra(cfg.hydra_cfg)
    shot_idx, bounce_idx = extract_event_indices(meta, cfg=label_cfg)
    targets_t = build_targets(
        t_len,
        shot_indices=shot_idx,
        bounce_indices=bounce_idx,
        cfg=label_cfg,
        device=torch.device("cpu"),
    )

    return UVEventInputs(
        ball_uv=ball_uv,
        ball_vis=ball_vis,
        court_kp=court_kp,
        court_vis=(court_vis > 0),
        targets=targets_t.numpy(),
        shot_indices=[i for i in shot_idx if 0 <= i < t_len],
        bounce_indices=[i for i in bounce_idx if 0 <= i < t_len],
        meta=meta,
        camera_idx=cam_idx,
    )


def load_traj3d_inputs(cfg: RuntimeConfig) -> Traj3DEventInputs:
    """Load a BLCS scene and build 3D event visualization inputs."""
    set_seed(cfg.seed)

    payload = load_npz_scene(cfg.scene_path)
    meta = payload.get("meta", {})

    if "ball_pos_world" not in payload:
        raise KeyError("Missing key in scene NPZ: ball_pos_world")

    pos_full = np.asarray(payload["ball_pos_world"], dtype=np.float32)

    t_full = int(pos_full.shape[0])
    num_frames_meta = int(meta.get("num_frames", t_full))
    max_seq_len = int((cfg.hydra_cfg.get("data", {}) or {}).get("max_seq_len", t_full))
    t_len = min(t_full, max(0, num_frames_meta), max_seq_len)

    pos = pos_full[:t_len]

    label_cfg = _label_cfg_from_hydra(cfg.hydra_cfg)
    shot_idx, bounce_idx = extract_event_indices(meta, cfg=label_cfg)
    targets_t = build_targets(
        t_len,
        shot_indices=shot_idx,
        bounce_indices=bounce_idx,
        cfg=label_cfg,
        device=torch.device("cpu"),
    )

    return Traj3DEventInputs(
        ball_pos_world=pos,
        targets=targets_t.numpy(),
        shot_indices=[i for i in shot_idx if 0 <= i < t_len],
        bounce_indices=[i for i in bounce_idx if 0 <= i < t_len],
        meta=meta,
    )
