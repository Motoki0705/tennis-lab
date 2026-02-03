"""Visualize 3D trajectory event detection inputs and predictions (Hydra-based).

This script focuses on the 3D-input event detector:
- Dataset-side: visualize soft targets and GT event timings.
- Inference-side: visualize event probabilities and extracted peaks.
- Animation: the ball color changes on event frames (GT and prediction).

Example commands:
    `uv run python -m src.evnet_detection.scripts.visualize_3d`
    `uv run python -m src.evnet_detection.scripts.visualize_3d visualization.scene_path=data/blcs/scenes/rally_000000.npz`
    `uv run python -m src.evnet_detection.scripts.visualize_3d visualization.mode=predict visualization.checkpoint=outputs/evnet_detection/.../last.ckpt`

Config entry point: `src/evnet_detection/configs/visualize_3d.yaml`
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import to_absolute_path
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from omegaconf import DictConfig

from src.common.data.scene_cache import load_npz_scene
from src.evnet_detection.inference.traj3d_predictor import Traj3DEventPredictor
from src.evnet_detection.utils.visualization import (
    EventLabelConfig,
    build_targets,
    decode_meta,
    extract_event_indices,
    save_outputs,
)
from src.utils.geometry.court import HALF_DOUBLES_WIDTH, HALF_LENGTH
from src.utils.rendering.court_renderer import CourtRenderer

F = TypeVar("F", bound=Callable[..., Any])

# Colors
DEFAULT_BALL_COLOR: str = "#CCFF00"  # tennis ball
GT_SHOT_COLOR: str = "#00FF00"
GT_BOUNCE_COLOR: str = "#FFD700"
PRED_SHOT_COLOR: str = "#FF00FF"
PRED_BOUNCE_COLOR: str = "#00D1FF"


def hydra_main(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper for hydra.main to keep mypy satisfied."""

    return cast(Callable[[F], F], hydra.main(*args, **kwargs))


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime configuration for visualization/prediction."""

    mode: str
    scene_path: Path
    view: str
    frame: int
    fps: float
    save: Path | None
    info: bool
    checkpoint: str | None
    device: str
    output: str | None
    seed: int
    threshold: float
    min_distance: int
    top_k: int | None


@dataclass(frozen=True)
class Traj3DEventInputs:
    """Loaded 3D inputs + GT labels for a single scene."""

    ball_pos_world: np.ndarray  # (T, 3)
    targets: np.ndarray  # (T, 2)
    shot_indices: list[int]
    bounce_indices: list[int]
    meta: dict[str, Any]


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _set_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_runtime_config(cfg: DictConfig) -> RuntimeConfig:
    vis = cfg.visualization
    run = cfg.run

    top_k = vis.top_k
    top_k_value = int(top_k) if top_k is not None else None

    return RuntimeConfig(
        mode=str(vis.mode),
        scene_path=Path(to_absolute_path(str(vis.scene_path))),
        view=str(vis.view),
        frame=int(vis.frame),
        fps=float(vis.fps),
        save=Path(to_absolute_path(str(vis.save))) if vis.save else None,
        info=bool(vis.info),
        checkpoint=to_absolute_path(str(vis.checkpoint)) if vis.checkpoint else None,
        device=_resolve_device(str(run.device)),
        output=to_absolute_path(str(vis.output)) if vis.output else None,
        seed=int(run.seed),
        threshold=float(vis.threshold),
        min_distance=max(1, int(vis.min_distance)),
        top_k=top_k_value,
    )


def _label_cfg_from_hydra(cfg: DictConfig) -> EventLabelConfig:
    data_cfg = cfg.get("data", {}) or {}
    label_cfg = data_cfg.get("label", {}) or {}
    return EventLabelConfig(
        sigma_frames=float(label_cfg.get("sigma_frames", 2.5)),
        shot_time_key=str(label_cfg.get("shot_time_key", "t_start")),
        bounce_time_key=str(label_cfg.get("bounce_time_key", "t_bounce1")),
    )


def prepare_inputs(cfg: RuntimeConfig, hydra_cfg: DictConfig) -> Traj3DEventInputs:
    """Load a BLCS NPZ and prepare 3D inputs + GT targets."""

    _set_seed(cfg.seed)

    payload = load_npz_scene(cfg.scene_path)
    meta = decode_meta(payload.get("meta", {}))

    if "ball_pos_world" not in payload:
        raise KeyError("Missing key in scene NPZ: ball_pos_world")

    pos_full = np.asarray(payload["ball_pos_world"], dtype=np.float32)

    T_full = int(pos_full.shape[0])
    num_frames_meta = int(meta.get("num_frames", T_full))
    max_seq_len = int((hydra_cfg.get("data", {}) or {}).get("max_seq_len", T_full))
    T = min(T_full, max(0, num_frames_meta), max_seq_len)

    pos = pos_full[:T]

    label_cfg = _label_cfg_from_hydra(hydra_cfg)
    shot_idx, bounce_idx = extract_event_indices(meta, cfg=label_cfg)

    targets_t = build_targets(
        T,
        shot_indices=shot_idx,
        bounce_indices=bounce_idx,
        cfg=label_cfg,
        device=torch.device("cpu"),
    )

    return Traj3DEventInputs(
        ball_pos_world=pos,
        targets=targets_t.numpy(),
        shot_indices=[i for i in shot_idx if 0 <= i < T],
        bounce_indices=[i for i in bounce_idx if 0 <= i < T],
        meta=meta,
    )


def _event_sets(
    *,
    inputs: Traj3DEventInputs,
    pred_peaks: list[list[int]] | None,
) -> tuple[set[int], set[int], set[int], set[int]]:
    gt_shot = set(inputs.shot_indices)
    gt_bounce = set(inputs.bounce_indices)

    pred_shot: set[int] = set()
    pred_bounce: set[int] = set()
    if pred_peaks is not None:
        if len(pred_peaks) > 0:
            pred_shot = set(int(i) for i in pred_peaks[0])
        if len(pred_peaks) > 1:
            pred_bounce = set(int(i) for i in pred_peaks[1])

    return gt_shot, gt_bounce, pred_shot, pred_bounce


def _colors_for_frame(
    frame: int,
    *,
    gt_shot: set[int],
    gt_bounce: set[int],
    pred_shot: set[int],
    pred_bounce: set[int],
) -> tuple[str, str]:
    gt_event = "bounce" if frame in gt_bounce else "shot" if frame in gt_shot else None
    pred_event = (
        "bounce" if frame in pred_bounce else "shot" if frame in pred_shot else None
    )

    face = DEFAULT_BALL_COLOR
    edge = "black"

    if gt_event == "bounce":
        face = GT_BOUNCE_COLOR
    elif gt_event == "shot":
        face = GT_SHOT_COLOR
    elif pred_event == "bounce":
        face = PRED_BOUNCE_COLOR
    elif pred_event == "shot":
        face = PRED_SHOT_COLOR

    if gt_event is not None and pred_event is not None:
        edge = PRED_BOUNCE_COLOR if pred_event == "bounce" else PRED_SHOT_COLOR

    return face, edge


def render_topdown_trajectory(
    ax: Axes,
    *,
    inputs: Traj3DEventInputs,
    pred_peaks: list[list[int]] | None = None,
) -> None:
    """Render top-down 2D trajectory (XY) with event markers."""

    court = CourtRenderer()
    court.render_2d(ax, show_fence=True, set_limits=False)

    pos = inputs.ball_pos_world
    ax.plot(
        pos[:, 0],
        pos[:, 1],
        color="#FF6B6B",
        alpha=0.5,
        linewidth=1.2,
        label="Trajectory",
    )

    if inputs.shot_indices:
        idx = np.asarray(inputs.shot_indices, dtype=int)
        ax.scatter(
            pos[idx, 0],
            pos[idx, 1],
            c=GT_SHOT_COLOR,
            s=120,
            marker="*",
            edgecolors="black",
            linewidths=1.0,
            label="GT shot",
            zorder=6,
        )
    if inputs.bounce_indices:
        idx = np.asarray(inputs.bounce_indices, dtype=int)
        ax.scatter(
            pos[idx, 0],
            pos[idx, 1],
            c=GT_BOUNCE_COLOR,
            s=90,
            marker="o",
            edgecolors="black",
            linewidths=1.0,
            label="GT bounce",
            zorder=6,
        )

    if pred_peaks is not None:
        gt_shot, gt_bounce, pred_shot, pred_bounce = _event_sets(
            inputs=inputs, pred_peaks=pred_peaks
        )
        _ = gt_shot, gt_bounce
        if pred_shot:
            idx = np.asarray(sorted(pred_shot), dtype=int)
            ax.scatter(
                pos[idx, 0],
                pos[idx, 1],
                facecolors="none",
                edgecolors=PRED_SHOT_COLOR,
                s=110,
                marker="*",
                linewidths=1.5,
                label="Pred shot",
                zorder=7,
            )
        if pred_bounce:
            idx = np.asarray(sorted(pred_bounce), dtype=int)
            ax.scatter(
                pos[idx, 0],
                pos[idx, 1],
                facecolors="none",
                edgecolors=PRED_BOUNCE_COLOR,
                s=90,
                marker="o",
                linewidths=1.5,
                label="Pred bounce",
                zorder=7,
            )

    ax.set_xlim(-HALF_DOUBLES_WIDTH - 2, HALF_DOUBLES_WIDTH + 2)
    ax.set_ylim(-HALF_LENGTH - 2, HALF_LENGTH + 2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    scene_id = inputs.meta.get("scene_id", "Unknown")
    ax.set_title(f"3D top-down | scene={scene_id}")
    ax.legend(loc="upper right", fontsize=8)


def _annotate_peaks(
    ax: Axes,
    *,
    peaks: list[int],
    scores: list[float],
    y: np.ndarray,
    color: str,
) -> None:
    for idx, score in zip(peaks, scores, strict=False):
        if 0 <= idx < y.shape[0]:
            ax.text(
                idx,
                float(y[idx]) + 0.03,
                f"{score:.2f}",
                fontsize=7,
                color=color,
                ha="center",
                va="bottom",
            )


def render_timeline_axes(
    axes: list[Axes],
    *,
    cfg: RuntimeConfig,
    inputs: Traj3DEventInputs,
    probs: np.ndarray | None = None,  # (T, E)
    pred_peaks: list[list[int]] | None = None,
    pred_scores: list[list[float]] | None = None,
    event_names: list[str] | None = None,
) -> list[str]:
    """Render per-event timeline plots onto pre-created axes."""

    targets = inputs.targets
    T, E = targets.shape
    if len(axes) != E:
        raise ValueError(f"Expected {E} axes, got {len(axes)}")

    if event_names is None:
        event_names = ["shot", "bounce"][:E]
    if len(event_names) < E:
        event_names = event_names + [f"event_{i}" for i in range(len(event_names), E)]

    x = np.arange(T)
    for e in range(E):
        ax = axes[e]
        ax.plot(
            x,
            targets[:, e],
            color="white",
            alpha=0.6,
            linestyle="--",
            label="GT target",
        )

        if probs is not None:
            ax.plot(x, probs[:, e], color="#00D1FF", alpha=0.9, label="Pred prob")

        ax.axhline(
            cfg.threshold,
            color="yellow",
            alpha=0.5,
            linestyle="--",
            linewidth=1.2,
        )

        gt_idx = (
            inputs.shot_indices if e == 0 else inputs.bounce_indices if e == 1 else []
        )
        for t_idx in gt_idx:
            ax.axvline(t_idx, color="lime", alpha=0.25)

        if pred_peaks is not None and e < len(pred_peaks):
            for t_idx in pred_peaks[e]:
                ax.axvline(t_idx, color="magenta", alpha=0.25)

        if (
            probs is not None
            and pred_peaks is not None
            and pred_scores is not None
            and e < len(pred_peaks)
        ):
            _annotate_peaks(
                ax,
                peaks=pred_peaks[e],
                scores=pred_scores[e] if e < len(pred_scores) else [],
                y=probs[:, e],
                color="magenta",
            )

        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(event_names[e])
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Frame")
    return list(event_names)


def create_timeline_figure(
    *,
    cfg: RuntimeConfig,
    inputs: Traj3DEventInputs,
    probs: np.ndarray | None = None,  # (T, E)
    pred_peaks: list[list[int]] | None = None,
    pred_scores: list[list[float]] | None = None,
    event_names: list[str] | None = None,
) -> Figure:
    _, E = inputs.targets.shape

    fig, axes_raw = plt.subplots(E, 1, figsize=(14, 2.6 * E), sharex=True)
    axes: list[Axes] = (
        [cast(Axes, axes_raw)] if E == 1 else [cast(Axes, a) for a in axes_raw]
    )

    names = render_timeline_axes(
        axes,
        cfg=cfg,
        inputs=inputs,
        probs=probs,
        pred_peaks=pred_peaks,
        pred_scores=pred_scores,
        event_names=event_names,
    )
    _ = names

    scene_id = inputs.meta.get("scene_id", "Unknown")
    fig.suptitle(f"Event timeline | scene={scene_id}")
    plt.tight_layout()
    return fig


def create_multi_figure(
    *,
    cfg: RuntimeConfig,
    inputs: Traj3DEventInputs,
    probs: np.ndarray | None = None,
    pred_peaks: list[list[int]] | None = None,
    pred_scores: list[list[float]] | None = None,
    event_names: list[str] | None = None,
) -> Figure:
    """Create a combined figure (top-down trajectory + timeline)."""

    _, E = inputs.targets.shape

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(E, 2, width_ratios=[1.0, 1.2])

    ax_traj = fig.add_subplot(gs[:, 0])
    render_topdown_trajectory(ax_traj, inputs=inputs, pred_peaks=pred_peaks)

    axes: list[Axes] = []
    for e in range(E):
        if e == 0:
            ax = fig.add_subplot(gs[e, 1])
        else:
            ax = fig.add_subplot(gs[e, 1], sharex=axes[0])
        axes.append(ax)

    render_timeline_axes(
        axes,
        cfg=cfg,
        inputs=inputs,
        probs=probs,
        pred_peaks=pred_peaks,
        pred_scores=pred_scores,
        event_names=event_names,
    )

    scene_id = inputs.meta.get("scene_id", "Unknown")
    fig.suptitle(f"3D top-down + timeline | scene={scene_id}")
    plt.tight_layout()
    return fig


def create_animation(
    *,
    cfg: RuntimeConfig,
    inputs: Traj3DEventInputs,
    pred_peaks: list[list[int]] | None = None,
) -> FuncAnimation:
    pos = inputs.ball_pos_world
    T = int(pos.shape[0])

    fig, ax = plt.subplots(figsize=(10, 8))
    court = CourtRenderer()
    court.render_2d(ax, show_fence=True, set_limits=False)

    (line,) = ax.plot([], [], color="#FF6B6B", alpha=0.5, linewidth=1.2)
    point = ax.scatter(
        [],
        [],
        c=DEFAULT_BALL_COLOR,
        s=120,
        edgecolors="black",
        linewidths=2.0,
        zorder=10,
    )

    gt_shot, gt_bounce, pred_shot, pred_bounce = _event_sets(
        inputs=inputs, pred_peaks=pred_peaks
    )

    ax.set_xlim(-HALF_DOUBLES_WIDTH - 2, HALF_DOUBLES_WIDTH + 2)
    ax.set_ylim(-HALF_LENGTH - 2, HALF_LENGTH + 2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    def update(frame: int) -> tuple[Line2D, PathCollection]:
        line.set_data(pos[: frame + 1, 0], pos[: frame + 1, 1])
        point.set_offsets([[pos[frame, 0], pos[frame, 1]]])
        face, edge = _colors_for_frame(
            frame,
            gt_shot=gt_shot,
            gt_bounce=gt_bounce,
            pred_shot=pred_shot,
            pred_bounce=pred_bounce,
        )
        point.set_facecolor([face])
        point.set_edgecolor([edge])
        ax.set_title(f"3D top-down animation | frame {frame}/{T - 1}")
        return line, point

    return FuncAnimation(
        fig, update, frames=T, interval=1000.0 / float(cfg.fps), blit=False
    )


def print_info(cfg: RuntimeConfig, inputs: Traj3DEventInputs) -> None:
    scene_id = inputs.meta.get("scene_id", "Unknown")
    print("=" * 60)
    print("EVNET_DETECTION 3D VISUALIZATION")
    print("=" * 60)
    print(f"Scene:   {scene_id}")
    print(f"Path:    {cfg.scene_path}")
    print(f"Frames:  {inputs.ball_pos_world.shape[0]}")
    print(f"GT shot indices:   {inputs.shot_indices}")
    print(f"GT bounce indices: {inputs.bounce_indices}")


def _save_figure(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")


def _save_animation(anim: FuncAnimation, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), fps=float(fps))


def main_visualize(cfg: RuntimeConfig, hydra_cfg: DictConfig) -> int:
    inputs = prepare_inputs(cfg, hydra_cfg)

    if cfg.info:
        print_info(cfg, inputs)
        return 0

    if cfg.frame < 0 or cfg.frame >= inputs.ball_pos_world.shape[0]:
        print(
            f"Error: Frame {cfg.frame} out of range (0-{inputs.ball_pos_world.shape[0] - 1})"
        )
        return 1

    if cfg.view == "trajectory":
        fig, ax = plt.subplots(figsize=(10, 8))
        render_topdown_trajectory(ax, inputs=inputs)
    elif cfg.view == "timeline":
        fig = create_timeline_figure(cfg=cfg, inputs=inputs)
    elif cfg.view == "multi":
        fig = create_multi_figure(cfg=cfg, inputs=inputs)
    elif cfg.view == "animation":
        anim = create_animation(cfg=cfg, inputs=inputs)
        if cfg.save is not None:
            _save_animation(anim, cfg.save, cfg.fps)
            plt.close()
            print(f"Saved animation to {cfg.save}")
            return 0
        plt.show()
        return 0
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
    if cfg.checkpoint is None:
        print("Error: visualization.checkpoint must be set for predict mode.")
        return 1

    inputs = prepare_inputs(cfg, hydra_cfg)

    if cfg.info:
        print_info(cfg, inputs)
        return 0

    print(f"Loading checkpoint from {cfg.checkpoint}...")
    predictor = Traj3DEventPredictor.load_from_checkpoint(
        checkpoint_path=cfg.checkpoint,
        device=cfg.device,
    )

    pos_t = torch.from_numpy(inputs.ball_pos_world).float()

    outputs = predictor.predict(
        ball_pos_world=pos_t,
        threshold=float(cfg.threshold),
        min_distance=int(cfg.min_distance),
        top_k=cfg.top_k,
    )

    probs = outputs["event_probs"].squeeze(0).cpu().numpy()  # (T, E)
    peaks = outputs["event_peaks"][0]
    scores = outputs["event_peak_scores"][0]
    names = outputs.get("event_names")

    if cfg.output is not None:
        save_outputs(outputs, Path(cfg.output))
        print(f"Saved prediction outputs to {cfg.output}")

    if cfg.view == "trajectory":
        fig, ax = plt.subplots(figsize=(10, 8))
        render_topdown_trajectory(ax, inputs=inputs, pred_peaks=peaks)
    elif cfg.view == "timeline":
        fig = create_timeline_figure(
            cfg=cfg,
            inputs=inputs,
            probs=probs,
            pred_peaks=peaks,
            pred_scores=scores,
            event_names=names,
        )
    elif cfg.view == "multi":
        fig = create_multi_figure(
            cfg=cfg,
            inputs=inputs,
            probs=probs,
            pred_peaks=peaks,
            pred_scores=scores,
            event_names=names,
        )
    elif cfg.view == "animation":
        anim = create_animation(cfg=cfg, inputs=inputs, pred_peaks=peaks)
        if cfg.save is not None:
            _save_animation(anim, cfg.save, cfg.fps)
            plt.close()
            print(f"Saved animation to {cfg.save}")
            return 0
        plt.show()
        return 0
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


@hydra_main(config_path="../configs", config_name="visualize_3d", version_base="1.3")
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
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
