"""Visualize trajectory completion inputs and predictions (Hydra-based).

This script mirrors the UX of BLCS visualization utilities, but focuses on UV
trajectory completion:
- Dataset-side: how much GT is hidden (masked) and how much observed points
  jitter due to noise.
- Inference-side: model predictions are shown separately for observed vs masked
  frames for easier debugging.

Example commands:
    `uv run python -m src.trajectory_completion.scripts.visualize`
    `uv run python -m src.trajectory_completion.scripts.visualize visualization.scene_path=data/blcs/scenes/rally_000000.npz`
    `uv run python -m src.trajectory_completion.scripts.visualize run.seed=0 data.argument.noise_std=0.02`
    `uv run python -m src.trajectory_completion.scripts.visualize visualization.mode=predict visualization.checkpoint=outputs/trajectory_completion/.../last.ckpt`

Config entry point: `src/trajectory_completion/configs/visualize.yaml`
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.common.data.npz_meta import decode_meta
from src.common.data.scene_cache import load_npz_scene
from src.trajectory_completion.data.argument import TrajectoryArgumenter
from src.trajectory_completion.data.event_masking import extract_event_frames
from src.trajectory_completion.inference.uv_predictor import (
    UVTrajectoryCompletionPredictor,
)
from src.utils.geometry.constants import COURT_LINE_CONNECTIONS

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

F = TypeVar("F", bound=Callable[..., Any])

TMP_LOG_PATH = Path("data/tmp/trajectory_completion_visualize.log")


def hydra_main(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper for hydra.main to keep mypy satisfied."""

    return cast(Callable[[F], F], hydra.main(*args, **kwargs))


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime configuration for visualization/prediction."""

    mode: str
    scene_path: Path
    camera: Any
    frame: int
    view: str
    fps: float | None
    save: Path | None
    info: bool
    checkpoint: str | None
    device: str
    output: str | None
    seed: int
    apply_corruption: bool
    use_scene_visibility: bool
    connector_stride: int
    start: int
    max_frames: int | None
    error_threshold: float
    show_court_lines: bool


@dataclass(frozen=True)
class TrajectoryInputs:
    """Prepared UV trajectory inputs for visualization/inference."""

    ball_uv_gt: np.ndarray  # (T, 2)
    ball_uv_in: np.ndarray  # (T, 2)
    ball_gt_visible: np.ndarray  # (T,) bool  (visibility from the scene)
    ball_obs_mask: np.ndarray  # (T,) bool  (after augmentation)
    court_kp: np.ndarray  # (20, 2)
    court_vis: np.ndarray  # (20,) bool
    meta: dict[str, Any]
    camera_idx: int
    start: int


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _set_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    """Build a runtime config from the composed Hydra config."""

    vis = cfg.visualization
    run = cfg.run

    return RuntimeConfig(
        mode=str(vis.mode),
        scene_path=Path(to_absolute_path(str(vis.scene_path))),
        camera=vis.camera,
        frame=int(vis.frame),
        view=str(vis.view),
        fps=float(vis.fps) if vis.fps is not None else None,
        save=Path(to_absolute_path(str(vis.save))) if vis.save else None,
        info=bool(vis.info),
        checkpoint=to_absolute_path(str(vis.checkpoint)) if vis.checkpoint else None,
        device=_resolve_device(str(run.device)),
        output=to_absolute_path(str(vis.output)) if vis.output else None,
        seed=int(run.seed),
        apply_corruption=bool(vis.apply_corruption),
        use_scene_visibility=bool(vis.use_scene_visibility),
        connector_stride=max(1, int(vis.connector_stride)),
        start=max(0, int(vis.start)),
        max_frames=int(vis.max_frames) if vis.max_frames is not None else None,
        error_threshold=float(vis.error_threshold),
        show_court_lines=bool(vis.show_court_lines),
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
        with TMP_LOG_PATH.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(f"{line}\n")
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

    # Prefer multi-camera keys, but allow single-camera fallbacks.
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


def prepare_inputs(cfg: RuntimeConfig, hydra_cfg: DictConfig) -> TrajectoryInputs:
    """Load a BLCS NPZ and prepare (GT, input, masks) for visualization."""

    _set_seed(cfg.seed)

    payload = load_npz_scene(cfg.scene_path)
    meta = decode_meta(payload.get("meta", {}))

    num_cameras = int(payload.get("num_cameras", 1))
    cam_idx = _select_camera(cfg.camera, num_cameras)

    ball_uv_full, ball_vis_full, court_kp, court_vis = _load_uv_from_scene(
        payload, cam_idx
    )

    num_frames_meta = int(meta.get("num_frames", int(ball_uv_full.shape[0])))
    seq_len = min(int(ball_uv_full.shape[0]), max(0, num_frames_meta))

    max_seq_len_cfg = int((hydra_cfg.get("data", {}) or {}).get("max_seq_len", seq_len))
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
        argumenter = _build_argumenter(hydra_cfg)
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


def _draw_court_uv(
    ax: Axes,
    *,
    court_kp: np.ndarray,
    court_vis: np.ndarray,
    show_lines: bool,
) -> None:
    ax.set_facecolor("#1a1a1a")

    if show_lines:
        for i, j in COURT_LINE_CONNECTIONS:
            if bool(court_vis[i]) and bool(court_vis[j]):
                ax.plot(
                    [court_kp[i, 0], court_kp[j, 0]],
                    [court_kp[i, 1], court_kp[j, 1]],
                    c="lime",
                    linewidth=1.5,
                    alpha=0.8,
                )

    # Keypoints
    for i in range(int(court_kp.shape[0])):
        if bool(court_vis[i]):
            ax.scatter(
                court_kp[i, 0],
                court_kp[i, 1],
                c="lime",
                s=25,
                marker="s",
                alpha=0.7,
                zorder=2,
            )


def _summarize_inputs(inputs: TrajectoryInputs) -> dict[str, float]:
    T = int(inputs.ball_uv_gt.shape[0])
    orig_vis = inputs.ball_gt_visible
    obs = inputs.ball_obs_mask

    orig_vis_count = int(orig_vis.sum())
    obs_count = int(obs.sum())

    newly_masked = orig_vis & (~obs)
    newly_masked_count = int(newly_masked.sum())

    # Jitter only makes sense on frames that are observed after augmentation.
    err_in = np.linalg.norm(inputs.ball_uv_in - inputs.ball_uv_gt, axis=-1)
    jitter = err_in[obs]

    stats: dict[str, float] = {
        "frames": float(T),
        "orig_visible_ratio": float(orig_vis.mean()) if T > 0 else 0.0,
        "observed_ratio": float(obs.mean()) if T > 0 else 0.0,
        "orig_visible_count": float(orig_vis_count),
        "observed_count": float(obs_count),
        "newly_masked_count": float(newly_masked_count),
        "jitter_mean": float(jitter.mean()) if jitter.size > 0 else 0.0,
        "jitter_p95": float(np.quantile(jitter, 0.95)) if jitter.size > 0 else 0.0,
        "jitter_max": float(jitter.max()) if jitter.size > 0 else 0.0,
    }
    return stats


def print_info(cfg: RuntimeConfig, inputs: TrajectoryInputs) -> None:
    meta = inputs.meta
    scene_id = meta.get("scene_id", "Unknown")
    print("=" * 60)
    print("TRAJECTORY COMPLETION VISUALIZATION")
    print("=" * 60)
    print(f"Scene:      {scene_id}")
    print(f"Path:       {cfg.scene_path}")
    print(f"Camera:     {inputs.camera_idx}")
    print(f"Start:      {inputs.start}")
    print(f"Frames:     {inputs.ball_uv_gt.shape[0]}")

    stats = _summarize_inputs(inputs)
    print("\nVisibility / masking:")
    print(
        f"  Original visible: {stats['orig_visible_count']:.0f}"
        f" ({stats['orig_visible_ratio']:.1%})"
    )
    print(
        f"  Observed (input):  {stats['observed_count']:.0f}"
        f" ({stats['observed_ratio']:.1%})"
    )
    print(f"  Newly masked by augmentation: {stats['newly_masked_count']:.0f}")

    print("\nObserved-point jitter (|input - GT|):")
    print(
        f"  mean={stats['jitter_mean']:.4f}  p95={stats['jitter_p95']:.4f}  max={stats['jitter_max']:.4f}"
    )


def render_uv_panel(
    ax: Axes,
    *,
    cfg: RuntimeConfig,
    inputs: TrajectoryInputs,
    pred_uv: np.ndarray | None = None,
    completed_uv: np.ndarray | None = None,
) -> None:
    """Render UV comparison panel."""

    _draw_court_uv(
        ax,
        court_kp=inputs.court_kp,
        court_vis=inputs.court_vis,
        show_lines=cfg.show_court_lines,
    )

    gt = inputs.ball_uv_gt
    uv_in = inputs.ball_uv_in
    orig_vis = inputs.ball_gt_visible
    obs = inputs.ball_obs_mask

    # Categories
    orig_missing = ~orig_vis
    newly_masked = orig_vis & (~obs)

    ax.plot(gt[:, 0], gt[:, 1], color="white", alpha=0.25, linewidth=1.0, label="GT")

    if orig_missing.any():
        ax.scatter(
            gt[orig_missing, 0],
            gt[orig_missing, 1],
            c="gray",
            s=18,
            marker="x",
            alpha=0.5,
            label="GT (missing in scene)",
            zorder=4,
        )

    if newly_masked.any():
        ax.scatter(
            gt[newly_masked, 0],
            gt[newly_masked, 1],
            c="#FF4444",
            s=22,
            marker="x",
            alpha=0.9,
            label="GT (masked by augmentation)",
            zorder=5,
        )

    if obs.any():
        ax.scatter(
            uv_in[obs, 0],
            uv_in[obs, 1],
            c="lime",
            s=28,
            marker="o",
            alpha=0.8,
            label="Input (observed)",
            zorder=6,
        )

    if obs.any() and cfg.connector_stride > 0:
        idx = np.where(obs)[0]
        idx = idx[:: cfg.connector_stride]
        for i in idx:
            ax.plot(
                [gt[i, 0], uv_in[i, 0]],
                [gt[i, 1], uv_in[i, 1]],
                color="orange",
                alpha=0.15,
                linewidth=0.6,
                zorder=3,
            )

    if pred_uv is not None:
        # Distinguish predicted points at observed vs masked frames.
        ax.scatter(
            pred_uv[obs, 0],
            pred_uv[obs, 1],
            facecolors="none",
            edgecolors="#00D1FF",
            s=60,
            marker="o",
            linewidths=1.8,
            alpha=0.9,
            label="Pred @ observed",
            zorder=7,
        )
        ax.scatter(
            pred_uv[~obs, 0],
            pred_uv[~obs, 1],
            c="#FF00FF",
            s=55,
            marker="^",
            alpha=0.8,
            label="Pred @ masked",
            zorder=7,
        )

    if completed_uv is not None:
        ax.plot(
            completed_uv[:, 0],
            completed_uv[:, 1],
            color="#00D1FF",
            alpha=0.7,
            linewidth=1.6,
            label="Completed (merge_observed)",
            zorder=6,
        )

    if 0 <= cfg.frame < gt.shape[0]:
        ax.scatter(
            [gt[cfg.frame, 0]],
            [gt[cfg.frame, 1]],
            c="yellow",
            s=120,
            marker="*",
            edgecolors="black",
            linewidths=1.0,
            zorder=10,
            label=f"Frame {cfg.frame}",
        )

    scene_id = inputs.meta.get("scene_id", "Unknown")
    ax.set_title(f"UV View | scene={scene_id} cam={inputs.camera_idx}")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xlabel("U (normalized)")
    ax.set_ylabel("V (normalized)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)


def render_timeline_panel(
    ax_mask: Axes,
    ax_err: Axes,
    *,
    cfg: RuntimeConfig,
    inputs: TrajectoryInputs,
    pred_uv: np.ndarray | None = None,
) -> None:
    """Render masking and error timeline panels."""

    T = int(inputs.ball_uv_gt.shape[0])
    t = np.arange(T)

    orig_vis = inputs.ball_gt_visible
    obs = inputs.ball_obs_mask

    newly_masked = orig_vis & (~obs)
    orig_missing = ~orig_vis

    # Mask panel
    ax_mask.plot(
        t, orig_vis.astype(np.float32), color="white", alpha=0.4, label="GT visible"
    )
    ax_mask.plot(
        t, obs.astype(np.float32), color="lime", alpha=0.8, label="Observed (input)"
    )

    if orig_missing.any():
        ax_mask.scatter(
            t[orig_missing],
            np.zeros_like(t[orig_missing]),
            s=14,
            c="gray",
            marker="x",
            alpha=0.7,
            label="Missing in scene",
        )
    if newly_masked.any():
        ax_mask.scatter(
            t[newly_masked],
            np.zeros_like(t[newly_masked]),
            s=16,
            c="#FF4444",
            marker="x",
            alpha=0.9,
            label="Masked by augmentation",
        )

    ax_mask.set_ylim(-0.2, 1.2)
    ax_mask.set_yticks([0.0, 1.0])
    ax_mask.set_title("Visibility / Observation Mask")
    ax_mask.set_xlabel("Frame")
    ax_mask.grid(True, alpha=0.25)
    ax_mask.legend(loc="upper right", fontsize=8)

    # Error panel
    err_in = np.linalg.norm(inputs.ball_uv_in - inputs.ball_uv_gt, axis=-1)
    ax_err.scatter(
        t[obs],
        err_in[obs],
        s=10,
        c="lime",
        alpha=0.8,
        label="|Input - GT| (observed)",
    )

    if pred_uv is not None:
        err_pred = np.linalg.norm(pred_uv - inputs.ball_uv_gt, axis=-1)
        ax_err.scatter(
            t[obs],
            err_pred[obs],
            s=10,
            c="#00D1FF",
            alpha=0.8,
            label="|Pred - GT| @ observed",
        )
        ax_err.scatter(
            t[~obs],
            err_pred[~obs],
            s=10,
            c="#FF00FF",
            alpha=0.8,
            label="|Pred - GT| @ masked",
        )

    if cfg.error_threshold > 0:
        ax_err.axhline(
            float(cfg.error_threshold),
            color="yellow",
            alpha=0.5,
            linewidth=1.2,
            linestyle="--",
            label=f"threshold={cfg.error_threshold:g}",
        )

    ax_err.set_title("Per-frame L2 Error (UV units)")
    ax_err.set_xlabel("Frame")
    ax_err.set_ylabel("L2")
    ax_err.grid(True, alpha=0.25)
    ax_err.legend(loc="upper right", fontsize=8)


def create_multi_figure(
    *,
    cfg: RuntimeConfig,
    inputs: TrajectoryInputs,
    pred_uv: np.ndarray | None = None,
    completed_uv: np.ndarray | None = None,
) -> Figure:
    """Create a multi-panel figure (UV + mask + error)."""

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0])

    ax_uv = fig.add_subplot(gs[:, 0])
    ax_mask = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, 1])

    render_uv_panel(
        ax_uv,
        cfg=cfg,
        inputs=inputs,
        pred_uv=pred_uv,
        completed_uv=completed_uv,
    )
    render_timeline_panel(
        ax_mask,
        ax_err,
        cfg=cfg,
        inputs=inputs,
        pred_uv=pred_uv,
    )

    plt.tight_layout()
    return fig


def _save_figure(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")


def _save_outputs(outputs: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".pt":
        torch.save(outputs, output_path)
        return
    if output_path.suffix == ".json":
        json_data = {}
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                json_data[k] = v.squeeze(0).cpu().tolist()
            else:
                json_data[k] = v
        output_path.write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return
    print(
        f"Warning: Unknown output format '{output_path.suffix}', only .pt and .json are supported. Skipping save."
    )


def main_visualize(cfg: RuntimeConfig, hydra_cfg: DictConfig) -> int:
    """Visualize dataset-side inputs (GT vs corrupted input)."""

    inputs = prepare_inputs(cfg, hydra_cfg)

    if cfg.info:
        print_info(cfg, inputs)
        return 0

    if cfg.frame < 0 or cfg.frame >= inputs.ball_uv_gt.shape[0]:
        print(
            f"Error: Frame {cfg.frame} out of range (0-{inputs.ball_uv_gt.shape[0] - 1})"
        )
        return 1

    if cfg.view == "uv":
        fig, ax = plt.subplots(figsize=(10, 8))
        render_uv_panel(ax, cfg=cfg, inputs=inputs)
    elif cfg.view == "timeline":
        fig, (ax_mask, ax_err) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        render_timeline_panel(ax_mask, ax_err, cfg=cfg, inputs=inputs)
        plt.tight_layout()
    elif cfg.view == "multi":
        fig = create_multi_figure(cfg=cfg, inputs=inputs)
    else:
        print(f"Error: unknown visualization.view '{cfg.view}'")
        return 1

    if cfg.save is not None:
        _save_figure(fig, cfg.save)
        plt.close(fig)
        print(f"Saved to {cfg.save}")
    else:
        plt.show()

    return 0


def main_predict(cfg: RuntimeConfig, hydra_cfg: DictConfig) -> int:
    """Run the trajectory completion predictor and visualize its UV outputs."""

    if cfg.checkpoint is None:
        print("Error: visualization.checkpoint must be set for predict mode.")
        return 1

    inputs = prepare_inputs(cfg, hydra_cfg)

    if cfg.info:
        print_info(cfg, inputs)
        return 0

    if cfg.frame < 0 or cfg.frame >= inputs.ball_uv_gt.shape[0]:
        print(
            f"Error: Frame {cfg.frame} out of range (0-{inputs.ball_uv_gt.shape[0] - 1})"
        )
        return 1

    print(f"Loading checkpoint from {cfg.checkpoint}...")
    predictor = UVTrajectoryCompletionPredictor.load_from_checkpoint(
        checkpoint_path=cfg.checkpoint,
        device=cfg.device,
    )

    ball_uv_in_t = torch.from_numpy(inputs.ball_uv_in).float()
    ball_obs_mask_t = torch.from_numpy(inputs.ball_obs_mask.astype(np.float32))
    court_kp_t = torch.from_numpy(inputs.court_kp).float()
    court_vis_t = torch.from_numpy(inputs.court_vis.astype(np.float32))

    print("Running trajectory completion prediction...")
    outputs = predictor.predict(
        ball_uv_in=ball_uv_in_t,
        ball_obs_mask=ball_obs_mask_t,
        court_kp=court_kp_t,
        court_vis=court_vis_t,
        merge_observed=True,
    )

    pred_uv = outputs["ball_uv_pred"].squeeze(0).cpu().numpy()
    completed_uv = (
        outputs.get("ball_uv_completed", outputs["ball_uv_pred"])
        .squeeze(0)
        .cpu()
        .numpy()
    )

    if cfg.output is not None:
        _save_outputs(outputs, Path(cfg.output))
        print(f"Saved prediction outputs to {cfg.output}")

    if cfg.view == "uv":
        fig, ax = plt.subplots(figsize=(10, 8))
        render_uv_panel(
            ax, cfg=cfg, inputs=inputs, pred_uv=pred_uv, completed_uv=completed_uv
        )
    elif cfg.view == "timeline":
        fig, (ax_mask, ax_err) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        render_timeline_panel(ax_mask, ax_err, cfg=cfg, inputs=inputs, pred_uv=pred_uv)
        plt.tight_layout()
    elif cfg.view == "multi":
        fig = create_multi_figure(
            cfg=cfg, inputs=inputs, pred_uv=pred_uv, completed_uv=completed_uv
        )
    else:
        print(f"Error: unknown visualization.view '{cfg.view}'")
        return 1

    if cfg.save is not None:
        _save_figure(fig, cfg.save)
        plt.close(fig)
        print(f"Saved to {cfg.save}")
    else:
        plt.show()

    return 0


@hydra_main(config_path="../configs", config_name="visualize", version_base="1.3")
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Hydra entry point."""

    runtime = build_runtime_config(cfg)
    if runtime.mode == "visualize":
        return main_visualize(runtime, cfg)
    if runtime.mode == "predict":
        return main_predict(runtime, cfg)
    print(
        f"Error: unknown visualization.mode '{runtime.mode}' (expected visualize|predict)"
    )
    return 1


if __name__ == "__main__":
    sys.exit(cast(Callable[[], int], main)())
