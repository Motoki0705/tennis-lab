"""Orchestrate ball multitask predict/visualize workflow."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt

from src.ball_multitask.visualization.adapters.predict_inputs import (
    build_ball_multitask_predict_inputs,
)
from src.ball_multitask.visualization.analysis.report import (
    print_info,
    print_predict_summary,
    save_figure,
    save_outputs,
)
from src.ball_multitask.visualization.api.predict import load_predictor, predict_scene
from src.ball_multitask.visualization.io.scene import build_runtime_config, load_scene_inputs
from src.ball_multitask.visualization.rendering.summary import create_summary_figure
from src.ball_multitask.visualization.types import RuntimeConfig

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
    )

    if cfg.output is not None:
        save_outputs(cfg.output, outputs)
        print(f"Saved outputs to {cfg.output}")

    if mode == "predict":
        print_predict_summary(outputs)
        return 0

    if cfg.view != "summary":
        logger.error("Error: unknown visualization.view '%s' (expected summary)", cfg.view)
        return 1

    fig = create_summary_figure(inputs=scene_inputs, outputs=outputs)
    if cfg.save is not None:
        save_figure(cfg.save, fig)
        plt.close(fig)
        print(f"Saved figure to {cfg.save}")
    else:
        plt.show()

    return 0


__all__ = ["RuntimeConfig", "build_runtime_config", "run_visualization"]
