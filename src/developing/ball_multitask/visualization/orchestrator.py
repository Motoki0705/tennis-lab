"""Orchestrate ball multitask predict/visualize workflow."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt

from src.developing.ball_multitask.visualization.adapters.predict_inputs import (
    build_ball_multitask_predict_inputs,
)
from src.developing.ball_multitask.visualization.analysis.report import (
    print_info,
    print_predict_summary,
    save_animation,
    save_outputs,
)
from src.developing.ball_multitask.visualization.api.predict import load_predictor, predict_scene
from src.developing.ball_multitask.visualization.io.scene import build_runtime_config, load_scene_inputs
from src.developing.ball_multitask.visualization.rendering.animations import build_animations
from src.developing.ball_multitask.visualization.types import RuntimeConfig

logger = logging.getLogger(__name__)


def run_visualization(cfg: RuntimeConfig) -> int:
    """Run ball multitask orchestration."""
    mode = cfg.mode.strip().lower()
    if mode not in {"predict", "visualize", "info"}:
        logger.error("Error: unknown visualization.mode '%s' (expected predict|visualize|info)", cfg.mode)
        return 1

    scene_inputs = load_scene_inputs(cfg)
    if cfg.info or mode == "info":
        print_info(scene_inputs)
        return 0

    if cfg.checkpoint is None:
        logger.error("Error: checkpoint is required for predict/visualize mode.")
        return 1

    predictor = load_predictor(checkpoint_path=cfg.checkpoint, device=cfg.device)
    predict_inputs = build_ball_multitask_predict_inputs(scene_inputs)
    outputs = predict_scene(
        predictor=predictor,
        inputs=predict_inputs,
        threshold=cfg.threshold,
        min_distance=cfg.min_distance,
        top_k=cfg.top_k,
        denormalize=cfg.denormalize,
        in_frame_threshold=cfg.in_frame_threshold,
        cut_out_of_frame=cfg.cut_out_of_frame,
    )

    if cfg.output is not None:
        save_outputs(cfg.output, outputs)
        print(f"Saved outputs to {cfg.output}")

    if mode == "predict":
        print_predict_summary(outputs)
        return 0

    animations = build_animations(cfg=cfg, inputs=scene_inputs, outputs=outputs)
    if cfg.save_dir is not None:
        cfg.save_dir.mkdir(parents=True, exist_ok=True)
        suffix = cfg.save_format if cfg.save_format.startswith(".") else f".{cfg.save_format}"
        for artifact in animations:
            path = cfg.save_dir / f"{artifact.default_filename}{suffix}"
            save_animation(path, artifact.animation, fps=cfg.fps)
            plt.close(artifact.animation._fig)
            print(f"Saved animation to {path}")
    else:
        plt.show()

    return 0


__all__ = ["RuntimeConfig", "build_runtime_config", "run_visualization"]
