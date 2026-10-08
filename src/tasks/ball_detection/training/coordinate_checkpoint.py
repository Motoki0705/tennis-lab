"""Epoch-boundary restart for an unchanged coordinate training recipe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.training.coordinate_evaluation import CoordinateModel
from src.utils.checksum import dual_sha256


def save_coordinate_checkpoint(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_best(output: Path, epoch: int) -> None:
    path = output / f"epoch-{epoch:03d}.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    record = dict(epoch=epoch, checkpoint=path.name, checkpoint_sha256=dual_sha256(path),
                  selection_scope=saved["selection_scope"], selection_error_px=saved["selection_error_px"],
                  **saved["validation"])
    temporary = output / "best.json.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(output / "best.json")


def resume_coordinate_training(path: Path, output: Path, model: CoordinateModel,
                               optimizer: torch.optim.Optimizer, *, recipe: dict[str, Any],
                               manifest_sha256: str, code: dict[str, Any],
                               device: torch.device) -> tuple[int, int, float, int]:
    """Return next epoch, optimizer steps, best score and best epoch.

    Only the total epoch budget may change. Keeping the run directory and all
    completed checkpoints makes the selected best model unambiguous after resume.
    A partially processed epoch is rerun from its last completed predecessor.
    """
    if path.parent != output or not (output / "config.json").is_file():
        raise ValueError("Resume requires a checkpoint inside the original output directory")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved.get("training_state")
    if saved.get("schema") != "mdd_coordinates.v2" or not isinstance(state, dict):
        raise ValueError("Checkpoint lacks the epoch-restart training state")
    if saved["manifest_sha256"] != manifest_sha256 or state["recipe"] != recipe:
        raise ValueError("Resume data or training recipe changed")
    if saved["code"]["source_sha256"] != code["source_sha256"]:
        raise ValueError("Resume implementation changed; use a new experiment")
    epoch = int(saved["epoch"])
    if path.name != f"epoch-{epoch:03d}.pt":
        raise ValueError("Resume checkpoint filename does not match its epoch")
    later = [p for p in output.glob("epoch-*.pt") if int(p.stem.removeprefix("epoch-")) > epoch]
    if later:
        raise ValueError("Resume must use the latest completed epoch; refusing to overwrite later work")
    best_epoch = int(state["best_epoch"])
    best_path = output / f"epoch-{best_epoch:03d}.pt"
    best_saved = torch.load(best_path, map_location="cpu", weights_only=True)
    if best_saved["selection_error_px"] != state["best_error_px"]:
        raise ValueError("Saved best checkpoint differs from restart state")
    model.load_state_dict(saved["state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"])
    if device.type == "cuda":
        if state["cuda_rng"] is None:
            raise ValueError("CUDA restart requires saved CUDA RNG state")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    write_best(output, best_epoch)
    return epoch + 1, int(state["global_step"]), float(state["best_error_px"]), best_epoch
