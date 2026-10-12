from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.models.mdd import DeepMDDQueryDetector
from src.utils.checksum import dual_sha256

SCHEMA = "mdd_query_posttraining.v1"


def resume_posttraining(path: Path, output: Path, model: DeepMDDQueryDetector,
                        optimizer: torch.optim.Optimizer, *, recipe: dict[str, Any],
                        code: dict[str, Any], device: torch.device) -> tuple[int, int, float, int]:
    """Resume the original cosine budget and restore best metadata from its checkpoint."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if path.parent != output or saved.get("schema") != SCHEMA or saved["recipe"] != recipe:
        raise ValueError("Resume requires identical posttraining recipe and output")
    if saved["code"]["source_sha256"] != code["source_sha256"]:
        raise ValueError("Resume source code changed; use the recorded commit for a historical run")
    first_epoch = saved["epoch"] + 1
    if path.name != f"epoch-{saved['epoch']:03d}.pt":
        raise ValueError("Resume checkpoint filename differs from its epoch")
    if first_epoch >= recipe["epochs"] or any(int(p.stem.split("-")[1]) >= first_epoch for p in output.glob("epoch-*.pt")):
        raise ValueError("Resume requires latest completed epoch and remaining budget")
    best_epoch = int(saved["best_epoch"])
    if not 0 <= best_epoch < first_epoch:
        raise ValueError("Resume best epoch lies outside the completed history")
    best_path = output / f"epoch-{best_epoch:03d}.pt"
    selected = torch.load(best_path, map_location="cpu", weights_only=True)
    if selected["validation"]["scopes"]["common"]["macro_mean_error_px"] != saved["best_score"]:
        raise ValueError("Resume best checkpoint disagrees with its selection score")
    model.load_state_dict(saved["state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    record = dict(checkpoint=best_path.name, sha256=dual_sha256(best_path), epoch=best_epoch,
                  global_step=selected["global_step"], selection_error_px=saved["best_score"], **selected["validation"])
    temporary = output / "best.json.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(output / "best.json")
    return first_epoch, int(saved["global_step"]), float(saved["best_score"]), best_epoch
