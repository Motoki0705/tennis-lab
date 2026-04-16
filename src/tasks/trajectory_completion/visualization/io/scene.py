"""Scene IO helpers for trajectory completion visualization."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.tasks.trajectory_completion.data.argument import TrajectoryArgumenter
from src.utils.data.event_utils import extract_event_frames
from src.utils.data.scene_io import load_npz_scene
from src.tasks.trajectory_completion.visualization.types import RuntimeConfig, TrajectoryInputs

TMP_LOG_PATH = Path("data/tmp/trajectory_completion_visualize.log")


def _resolve_num_court_kp(cfg: DictConfig) -> int:
    """Read and validate the configured court keypoint count."""
    data_cfg = cfg.get("data", {}) or {}
    num_court_kp = int(data_cfg.get("num_court_kp", 20))
    if not 1 <= num_court_kp <= 20:
        raise ValueError(f"data.num_court_kp must be in [1, 20], got {num_court_kp}.")
    return num_court_kp


def resolve_device(device: str) -> str:
    """Resolve auto device setting."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def set_seed(seed: int) -> None:
    """Set deterministic seeds for sampling/camera selection."""
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    """Build runtime config from composed Hydra config."""
    vis = cfg.visualization
    run = cfg.run

    return RuntimeConfig(
        mode=str(vis.mode),
        scene_path=Path(to_absolute_path(str(vis.scene_path))),
        camera=vis.camera,
        fps=float(vis.fps) if vis.fps is not None else None,
        save=Path(to_absolute_path(str(vis.save))) if vis.save else None,
        info=bool(vis.info),
        checkpoint=to_absolute_path(str(vis.checkpoint)) if vis.checkpoint else None,
        merge_observed=bool(vis.get("merge_observed", True)),
        in_frame_threshold=float(vis.get("in_frame_threshold", 0.5)),
        cut_out_of_frame=bool(vis.get("cut_out_of_frame", False)),
        device=resolve_device(str(run.device)),
        output=to_absolute_path(str(vis.output)) if vis.output else None,
        seed=int(run.seed),
        apply_corruption=bool(vis.apply_corruption),
        use_scene_visibility=bool(vis.use_scene_visibility),
        start=max(0, int(vis.start)),
        max_frames=int(vis.max_frames) if vis.max_frames is not None else None,
        show_court_lines=bool(vis.show_court_lines),
        hydra_cfg=cfg,
    )


def _select_camera(camera: Any, num_cameras: int) -> int:
    if num_cameras <= 0:
        return 0
    if camera is None:
        return 0
    if camera == "random":
        return int(np.random.randint(0, num_cameras))
    if isinstance(camera, int):
        return min(max(int(camera), 0), num_cameras - 1)
    if isinstance(camera, str) and camera.isdigit():
        return min(max(int(camera), 0), num_cameras - 1)
    return 0


def _slice_sequence(arr: np.ndarray, *, start: int, end: int) -> np.ndarray:
    if arr.ndim == 1:
        return arr[start:end]
    return arr[start:end, ...]


def _build_argumenter(cfg: DictConfig) -> TrajectoryArgumenter:
    data_cfg = cfg.get("data", {}) or {}
    arg_cfg = data_cfg.get("argument", {}) or {}
    return TrajectoryArgumenter(arg_cfg)


def _append_tmp_log(lines: list[str]) -> None:
    try:
        TMP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TMP_LOG_PATH.open("a", encoding="utf-8") as file:
            for line in lines:
                file.write(f"{line}\n")
    except OSError:
        return


def _format_tensor_indices(tensor: torch.Tensor, *, limit: int = 20) -> str:
    if tensor.numel() == 0:
        return "[]"
    total = int(tensor.numel())
    sample = tensor[:limit].detach().cpu().tolist()
    if total > limit:
        return f"{sample} ... (total={total})"
    return str(sample)


def _load_uv_from_scene(
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
        key
        for key in (ball_uv_key, ball_vis_key, court_kp_key, court_vis_key)
        if key not in payload
    ]
    if missing:
        raise KeyError(f"Missing keys in scene NPZ: {missing}")

    ball_uv = np.asarray(payload[ball_uv_key], dtype=np.float32)
    ball_vis = np.asarray(payload[ball_vis_key], dtype=np.float32)
    court_kp = np.asarray(payload[court_kp_key], dtype=np.float32)
    court_vis = np.asarray(payload[court_vis_key], dtype=np.float32)
    return ball_uv, ball_vis, court_kp, court_vis


def load_trajectory_inputs(cfg: RuntimeConfig) -> TrajectoryInputs:
    """Load a BLCS scene and prepare trajectory completion inputs."""
    set_seed(cfg.seed)

    payload = load_npz_scene(cfg.scene_path)
    meta = payload.get("meta", {})

    num_cameras = int(payload.get("num_cameras", 1))
    cam_idx = _select_camera(cfg.camera, num_cameras)

    ball_uv_full, ball_vis_full, court_kp, court_vis = _load_uv_from_scene(payload, cam_idx)
    num_court_kp = _resolve_num_court_kp(cfg.hydra_cfg)
    court_kp = court_kp[:num_court_kp]
    court_vis = court_vis[:num_court_kp]

    num_frames_meta = int(meta.get("num_frames", int(ball_uv_full.shape[0])))
    seq_len = min(int(ball_uv_full.shape[0]), max(0, num_frames_meta))

    max_seq_len_cfg = int((cfg.hydra_cfg.get("data", {}) or {}).get("max_seq_len", seq_len))
    seq_len = min(seq_len, max_seq_len_cfg)

    start = min(cfg.start, max(0, seq_len - 1)) if seq_len > 0 else 0
    end = seq_len
    if cfg.max_frames is not None:
        end = min(end, start + int(cfg.max_frames))

    if end <= start:
        raise ValueError(f"Invalid slice: start={start}, end={end}, seq_len={seq_len}")

    ball_uv_gt = _slice_sequence(ball_uv_full, start=start, end=end)
    ball_vis = _slice_sequence(ball_vis_full, start=start, end=end)

    if cfg.use_scene_visibility:
        ball_gt_visible = ball_vis > 0
    else:
        ball_gt_visible = np.ones((ball_uv_gt.shape[0],), dtype=bool)

    ball_uv_gt_t = torch.from_numpy(ball_uv_gt).float()
    ball_gt_visible_t = torch.from_numpy(ball_gt_visible.astype(np.float32))

    event_frames = extract_event_frames(meta, ball_uv_gt.shape[0], offset=start)
    argumenter: TrajectoryArgumenter | None = None

    if cfg.apply_corruption:
        argumenter = _build_argumenter(cfg.hydra_cfg)
        ball_uv_in_t, ball_obs_mask_t = argumenter(
            ball_uv_gt_t,
            ball_gt_visible_t,
            event_frames=event_frames,
        )
    else:
        ball_uv_in_t = ball_uv_gt_t.clone()
        ball_obs_mask_t = ball_gt_visible_t.clone()
        miss = ball_obs_mask_t <= 0
        if miss.any():
            ball_uv_in_t[miss] = 0.0

    log_lines = [
        "=" * 60,
        f"time={datetime.now().isoformat(timespec='seconds')}",
        f"scene={cfg.scene_path}",
        f"camera_idx={cam_idx}",
        f"slice=start:{start} end:{end} length:{ball_uv_gt.shape[0]}",
        f"event_frames(bounce)={_format_tensor_indices(event_frames.get('bounce', torch.empty(0)))}",
        f"event_frames(shot)={_format_tensor_indices(event_frames.get('shot', torch.empty(0)))}",
    ]
    if argumenter is not None:
        event_candidates = TrajectoryArgumenter._expand_event_candidates(
            event_frames=event_frames,
            length=ball_obs_mask_t.shape[0],
            window=argumenter.config.event_window,
            device=ball_obs_mask_t.device,
        )
        orig_vis = ball_gt_visible_t > 0
        newly_masked = (ball_obs_mask_t <= 0) & orig_vis
        masked_event = newly_masked & event_candidates
        log_lines.extend(
            [
                f"event_dropout_prob={argumenter.config.event_dropout_prob}",
                f"event_window={argumenter.config.event_window}",
                f"event_ratio={argumenter.config.event_ratio}",
                f"event_candidates_count={int(event_candidates.sum().item())}",
                f"newly_masked_count={int(newly_masked.sum().item())}",
                f"newly_masked_event_count={int(masked_event.sum().item())}",
                f"masked_indices_sample={_format_tensor_indices(torch.where(newly_masked)[0])}",
                f"masked_event_indices_sample={_format_tensor_indices(torch.where(masked_event)[0])}",
            ]
        )
    else:
        log_lines.append("event_dropout=disabled")
    _append_tmp_log(log_lines)

    return TrajectoryInputs(
        ball_uv_gt=ball_uv_gt_t.cpu().numpy(),
        ball_uv_in=ball_uv_in_t.cpu().numpy(),
        ball_gt_visible=ball_gt_visible.astype(bool),
        ball_obs_mask=(ball_obs_mask_t.cpu().numpy() > 0),
        court_kp=court_kp,
        court_vis=(court_vis > 0),
        meta=meta,
        camera_idx=cam_idx,
        start=start,
    )
