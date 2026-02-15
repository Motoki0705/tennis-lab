"""Orchestrate event detection visualization for UV and 3D trajectories."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt

from pathlib import Path

from src.event_detection.visualization.analysis.report import (
    print_traj3d_info,
    print_uv_info,
    save_animation,
    save_figure,
    save_outputs,
)
from src.event_detection.visualization.adapters.predict_inputs import (
    build_traj3d_predict_inputs,
    build_uv_predict_inputs,
)
from src.event_detection.visualization.api.predict import (
    predict_traj3d_events,
    predict_uv_events,
)
from src.event_detection.visualization.io.scene import (
    build_runtime_config,
    load_traj3d_inputs,
    load_uv_inputs,
)
from src.event_detection.visualization.rendering.timeline import create_timeline_figure
from src.event_detection.visualization.rendering.traj3d_renderer import (
    create_traj3d_animation,
    create_traj3d_multi_figure,
    render_topdown_trajectory,
)
from src.event_detection.visualization.rendering.uv_renderer import (
    create_uv_animation,
    create_uv_multi_figure,
    render_uv_trajectory,
)
from src.event_detection.visualization.types import RuntimeConfig

logger = logging.getLogger(__name__)


def _render_uv(
    *,
    cfg: RuntimeConfig,
    probs=None,
    peaks=None,
    scores=None,
    names=None,
) -> int:
    inputs = load_uv_inputs(cfg)

    if cfg.info:
        print_uv_info(cfg.scene_path, inputs)
        return 0

    if cfg.frame < 0 or cfg.frame >= inputs.ball_uv.shape[0]:
        logger.error(
            "Error: Frame %s out of range (0-%s)",
            cfg.frame,
            inputs.ball_uv.shape[0] - 1,
        )
        return 1

    if cfg.view == "trajectory":
        fig, ax = plt.subplots(figsize=(10, 8))
        render_uv_trajectory(ax, cfg=cfg, inputs=inputs, pred_peaks=peaks)
    elif cfg.view == "timeline":
        scene_id = str(inputs.meta.get("scene_id", "Unknown"))
        fig = create_timeline_figure(
            threshold=cfg.threshold,
            targets=inputs.targets,
            shot_indices=inputs.shot_indices,
            bounce_indices=inputs.bounce_indices,
            scene_id=scene_id,
            title_suffix=f" cam={inputs.camera_idx}",
            probs=probs,
            pred_peaks=peaks,
            pred_scores=scores,
            event_names=names,
        )
    elif cfg.view == "multi":
        fig = create_uv_multi_figure(
            cfg=cfg,
            inputs=inputs,
            probs=probs,
            pred_peaks=peaks,
            pred_scores=scores,
            event_names=names,
        )
    elif cfg.view == "animation":
        anim = create_uv_animation(cfg=cfg, inputs=inputs, pred_peaks=peaks)
        if cfg.save is not None:
            save_animation(anim, cfg.save, cfg.fps)
            plt.close()
            print(f"Saved animation to {cfg.save}")
            return 0
        plt.show()
        return 0
    else:
        logger.error("Error: unknown visualization.view '%s'", cfg.view)
        return 1

    if cfg.save is not None:
        save_figure(fig, cfg.save)
        plt.close(fig)
        print(f"Saved to {cfg.save}")
    else:
        plt.show()

    return 0


def _render_traj3d(
    *,
    cfg: RuntimeConfig,
    probs=None,
    peaks=None,
    scores=None,
    names=None,
) -> int:
    inputs = load_traj3d_inputs(cfg)

    if cfg.info:
        print_traj3d_info(cfg.scene_path, inputs)
        return 0

    if cfg.frame < 0 or cfg.frame >= inputs.ball_pos_world.shape[0]:
        logger.error(
            "Error: Frame %s out of range (0-%s)",
            cfg.frame,
            inputs.ball_pos_world.shape[0] - 1,
        )
        return 1

    if cfg.view == "trajectory":
        fig, ax = plt.subplots(figsize=(10, 8))
        render_topdown_trajectory(ax, inputs=inputs, pred_peaks=peaks)
    elif cfg.view == "timeline":
        scene_id = str(inputs.meta.get("scene_id", "Unknown"))
        fig = create_timeline_figure(
            threshold=cfg.threshold,
            targets=inputs.targets,
            shot_indices=inputs.shot_indices,
            bounce_indices=inputs.bounce_indices,
            scene_id=scene_id,
            title_suffix="",
            probs=probs,
            pred_peaks=peaks,
            pred_scores=scores,
            event_names=names,
        )
    elif cfg.view == "multi":
        fig = create_traj3d_multi_figure(
            cfg=cfg,
            inputs=inputs,
            probs=probs,
            pred_peaks=peaks,
            pred_scores=scores,
            event_names=names,
        )
    elif cfg.view == "animation":
        anim = create_traj3d_animation(cfg=cfg, inputs=inputs, pred_peaks=peaks)
        if cfg.save is not None:
            save_animation(anim, cfg.save, cfg.fps)
            plt.close()
            print(f"Saved animation to {cfg.save}")
            return 0
        plt.show()
        return 0
    else:
        logger.error("Error: unknown visualization.view '%s'", cfg.view)
        return 1

    if cfg.save is not None:
        save_figure(fig, cfg.save)
        plt.close(fig)
        print(f"Saved to {cfg.save}")
    else:
        plt.show()

    return 0


def run_visualization(cfg: RuntimeConfig) -> int:
    """Run event detection visualization for selected task and mode."""
    task = cfg.task
    mode = cfg.mode.strip().lower()

    if task not in {"uv", "traj3d", "3d"}:
        logger.error("Error: unknown visualization.task '%s' (expected uv|traj3d)", task)
        return 1
    if task == "3d":
        task = "traj3d"

    if mode == "visualize":
        return _render_uv(cfg=cfg) if task == "uv" else _render_traj3d(cfg=cfg)

    if mode != "predict":
        logger.error(
            "Error: unknown visualization.mode '%s' (expected visualize|predict)",
            cfg.mode,
        )
        return 1

    if cfg.checkpoint is None:
        logger.error("Error: visualization.checkpoint must be set for predict mode.")
        return 1

    print(f"Loading checkpoint from {cfg.checkpoint}...")

    if task == "uv":
        inputs = load_uv_inputs(cfg)
        predict_inputs = build_uv_predict_inputs(inputs)
        pred = predict_uv_events(
            checkpoint_path=cfg.checkpoint,
            device=cfg.device,
            inputs=predict_inputs,
            threshold=cfg.threshold,
            min_distance=cfg.min_distance,
            top_k=cfg.top_k,
        )
        if cfg.output is not None:
            save_outputs(pred["raw"], Path(cfg.output))
            print(f"Saved prediction outputs to {cfg.output}")
        return _render_uv(
            cfg=cfg,
            probs=pred["probs"],
            peaks=pred["peaks"],
            scores=pred["scores"],
            names=pred["names"],
        )

    inputs3d = load_traj3d_inputs(cfg)
    predict_inputs3d = build_traj3d_predict_inputs(inputs3d)
    pred3d = predict_traj3d_events(
        checkpoint_path=cfg.checkpoint,
        device=cfg.device,
        inputs=predict_inputs3d,
        threshold=cfg.threshold,
        min_distance=cfg.min_distance,
        top_k=cfg.top_k,
    )
    if cfg.output is not None:
        save_outputs(pred3d["raw"], Path(cfg.output))
        print(f"Saved prediction outputs to {cfg.output}")
    return _render_traj3d(
        cfg=cfg,
        probs=pred3d["probs"],
        peaks=pred3d["peaks"],
        scores=pred3d["scores"],
        names=pred3d["names"],
    )


__all__ = ["RuntimeConfig", "build_runtime_config", "run_visualization"]
