"""Scene IO helpers for ball multitask visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.ball_multitask.visualization.types import RuntimeConfig, SceneInputs
from src.common.data.npz_meta import decode_meta
from src.common.data.scene_cache import load_npz_scene


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _get_top_k(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


def _choose_camera(camera: Any, num_cameras: int) -> int:
    if num_cameras <= 0:
        return 0
    if camera is None:
        return 0
    if isinstance(camera, int):
        return min(max(camera, 0), num_cameras - 1)
    if isinstance(camera, str) and camera.isdigit():
        return min(max(int(camera), 0), num_cameras - 1)
    return 0


def _select_value(cfg: DictConfig, key: str, default: Any) -> Any:
    vis = cfg.get("visualization", {}) or {}
    inf = cfg.get("inference", {}) or {}
    if key in vis and vis.get(key) is not None:
        return vis.get(key)
    if key in inf and inf.get(key) is not None:
        return inf.get(key)
    return default


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    """Build runtime config from composed Hydra config."""
    run = cfg.get("run", {}) or {}

    scene_path = _select_value(cfg, "scene_path", "data/blcs/scenes/rally_000000.npz")
    checkpoint_raw = _select_value(cfg, "checkpoint", None)

    output_raw = _select_value(cfg, "output", None)
    save_raw = _select_value(cfg, "save", None)

    mode = str(_select_value(cfg, "mode", "predict"))
    view = str(_select_value(cfg, "view", "summary"))

    run_device = str(run.get("device", _select_value(cfg, "device", "cpu")))

    return RuntimeConfig(
        mode=mode,
        view=view,
        scene_path=Path(to_absolute_path(str(scene_path))),
        camera=_select_value(cfg, "camera", 0),
        checkpoint=Path(to_absolute_path(str(checkpoint_raw))) if checkpoint_raw else None,
        device=_resolve_device(run_device),
        output=Path(to_absolute_path(str(output_raw))) if output_raw else None,
        save=Path(to_absolute_path(str(save_raw))) if save_raw else None,
        info=bool(_select_value(cfg, "info", False)),
        threshold=float(_select_value(cfg, "threshold", 0.5)),
        min_distance=max(1, int(_select_value(cfg, "min_distance", 1))),
        top_k=_get_top_k(_select_value(cfg, "top_k", None)),
        denormalize=bool(_select_value(cfg, "denormalize", True)),
        hydra_cfg=cfg,
    )


def load_scene_inputs(cfg: RuntimeConfig) -> SceneInputs:
    """Load a single NPZ scene and extract per-camera arrays."""
    scene = load_npz_scene(cfg.scene_path)
    meta = decode_meta(scene.get("meta", {}))

    num_cameras = int(scene.get("num_cameras", 1))
    cam_idx = _choose_camera(cfg.camera, num_cameras)
    prefix = f"cam_{cam_idx}_"

    ball_uv_key = f"{prefix}ball_uv" if f"{prefix}ball_uv" in scene else "ball_uv"
    ball_vis_key = f"{prefix}ball_visible" if f"{prefix}ball_visible" in scene else "ball_visible"
    court_kp_key = f"{prefix}court_kp_uv" if f"{prefix}court_kp_uv" in scene else "court_kp_uv"
    court_vis_key = f"{prefix}court_kp_visible" if f"{prefix}court_kp_visible" in scene else "court_kp_visible"

    missing = [
        key
        for key in (ball_uv_key, ball_vis_key, court_kp_key, court_vis_key)
        if key not in scene
    ]
    if missing:
        raise KeyError(f"Missing keys in scene NPZ: {missing}")

    ball_uv = np.asarray(scene[ball_uv_key], dtype=np.float32)
    ball_vis = np.asarray(scene[ball_vis_key], dtype=np.float32)
    court_kp = np.asarray(scene[court_kp_key], dtype=np.float32)
    court_vis = np.asarray(scene[court_vis_key], dtype=np.float32)

    seq_len = int(meta.get("num_frames", ball_uv.shape[0]))
    seq_len = min(seq_len, int(ball_uv.shape[0]))

    return SceneInputs(
        ball_uv=ball_uv[:seq_len],
        ball_vis=ball_vis[:seq_len],
        court_kp=court_kp,
        court_vis=court_vis,
        seq_len=seq_len,
        meta=meta,
        camera_idx=cam_idx,
    )
