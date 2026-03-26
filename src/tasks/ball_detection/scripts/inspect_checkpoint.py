"""Inspect a ball-detection checkpoint and print a minimal summary.

Example commands:
    `uv run python -m src.tasks.ball_detection.scripts.inspect_checkpoint`
    `uv run python -m src.tasks.ball_detection.scripts.inspect_checkpoint checkpoint_path=checkpoints/ball_detection/best.pt`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


def _resolve_checkpoint_path(cfg: DictConfig) -> Path:
    path = Path(to_absolute_path(str(cfg.checkpoint_path))).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _extract_optimizer_step(payload: dict[str, Any]) -> float | None:
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state, dict):
        return None
    states = optimizer_state.get("state")
    if not isinstance(states, dict):
        return None
    for state in states.values():
        if not isinstance(state, dict) or "step" not in state:
            continue
        step = state["step"]
        if isinstance(step, torch.Tensor):
            return float(step.detach().cpu().item())
        if isinstance(step, (int, float)):
            return float(step)
    return None


def _extract_history_tail(payload: dict[str, Any]) -> dict[str, Any] | None:
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        return None
    tail = history[-1]
    if not isinstance(tail, dict):
        return None
    return tail


def _build_summary(payload: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    history_tail = _extract_history_tail(payload)
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "top_level_keys": sorted(payload.keys()),
        "epoch": payload.get("epoch"),
        "phase": payload.get("phase"),
        "phase_epoch": payload.get("phase_epoch"),
        "best_monitor": payload.get("best_monitor"),
        "monitor_value": payload.get("monitor_value"),
        "optimizer_step": _extract_optimizer_step(payload),
        "best_metrics": payload.get("best_metrics"),
        "history_length": len(payload["history"]) if isinstance(payload.get("history"), list) else 0,
        "last_history_entry": history_tail,
    }
    return summary


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="inspect_checkpoint",
)
def main(cfg: DictConfig) -> None:
    """Load the configured checkpoint and emit a compact JSON summary."""
    checkpoint_path = _resolve_checkpoint_path(cfg)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(
            "Expected the checkpoint payload to be a dict, "
            f"got {type(payload).__name__}."
        )

    summary = _build_summary(payload, checkpoint_path)
    summary_text = json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    print(summary_text)

    output_path = str(cfg.get("output_path", "") or "").strip()
    if output_path:
        resolved_output_path = Path(to_absolute_path(output_path)).resolve()
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(summary_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
