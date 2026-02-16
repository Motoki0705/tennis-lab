"""Orchestrate trajectory completion animation and prediction inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from src.trajectory_completion.data.event_masking import extract_event_frames
from src.trajectory_completion.visualization.adapters.predict_inputs import (
    build_uv_completion_predict_inputs,
)
from src.trajectory_completion.visualization.analysis.metrics import (
    print_info,
    print_predict_info,
)
from src.trajectory_completion.visualization.api.predict import predict_uv_completion
from src.trajectory_completion.visualization.io.scene import (
    build_runtime_config,
    load_trajectory_inputs,
)
from src.trajectory_completion.visualization.rendering.uv_animation import (
    create_uv_completion_animation,
)
from src.trajectory_completion.visualization.types import RuntimeConfig


def _save_animation(anim: FuncAnimation, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), fps=float(fps))


def _save_outputs(outputs: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".pt":
        torch.save(outputs, output_path)
        return
    if output_path.suffix == ".json":
        json_data = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                json_data[key] = value.squeeze(0).cpu().tolist()
            else:
                json_data[key] = value
        output_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    print(
        f"Warning: Unknown output format '{output_path.suffix}', only .pt and .json are supported. Skipping save."
    )


def _render(
    *,
    cfg: RuntimeConfig,
    pred_uv: Any | None = None,
    completed_uv: Any | None = None,
) -> int:
    inputs = load_trajectory_inputs(cfg)

    if cfg.info:
        if pred_uv is None or completed_uv is None:
            print_info(cfg, inputs)
        else:
            event_window = int(((cfg.hydra_cfg.get("data", {}) or {}).get("argument", {}) or {}).get("event_window", 2))
            event_frames = extract_event_frames(inputs.meta, inputs.ball_uv_gt.shape[0], offset=inputs.start)
            print_predict_info(
                cfg,
                inputs,
                pred_uv,
                completed_uv,
                event_frames=event_frames,
                event_window=event_window,
            )
        return 0

    fps = float(cfg.fps) if cfg.fps is not None else 30.0
    anim = create_uv_completion_animation(
        cfg=cfg,
        inputs=inputs,
        pred_uv=pred_uv,
    )

    if cfg.save is not None:
        _save_animation(anim, cfg.save, fps=fps)
        plt.close()
        print(f"Saved animation to {cfg.save}")
    else:
        plt.show()

    return 0


def run_visualization(runtime: RuntimeConfig) -> int:
    """Run trajectory completion visualization orchestration."""
    mode = runtime.mode.strip().lower()

    if mode == "visualize":
        return _render(cfg=runtime)

    if mode != "predict":
        print(f"Error: unknown visualization.mode '{runtime.mode}' (expected visualize|predict)")
        return 1

    if runtime.checkpoint is None:
        print("Error: visualization.checkpoint must be set for predict mode.")
        return 1

    inputs = load_trajectory_inputs(runtime)
    predict_inputs = build_uv_completion_predict_inputs(inputs)
    print(f"Loading checkpoint from {runtime.checkpoint}...")
    pred = predict_uv_completion(
        checkpoint_path=runtime.checkpoint,
        device=runtime.device,
        inputs=predict_inputs,
        merge_observed=runtime.merge_observed,
        in_frame_threshold=runtime.in_frame_threshold,
        cut_out_of_frame=runtime.cut_out_of_frame,
    )

    pred_uv = pred["pred_uv"]
    completed_uv = pred["completed_uv"]

    if runtime.output is not None:
        _save_outputs(pred["raw"], Path(runtime.output))
        print(f"Saved prediction outputs to {runtime.output}")

    return _render(cfg=runtime, pred_uv=pred_uv, completed_uv=completed_uv)


__all__ = ["RuntimeConfig", "build_runtime_config", "run_visualization"]
