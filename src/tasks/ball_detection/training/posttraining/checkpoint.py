from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.training.heatmap_pretraining.checkpoint import (
    pretraining_config,
)
from src.utils.checksum import dual_sha256

SCHEMA = "mdd_query_posttraining.v1"


def completed_pretraining(run: Path, manifest: Path) -> tuple[Path, dict[str, Any]]:
    completed = json.loads((run / "COMPLETED.json").read_text())
    selected = json.loads((run / "best.json").read_text())
    if completed.get("stage") != "heatmap_pretraining":
        raise ValueError("Pretraining must be completed before decoder training")
    filename = selected["checkpoint"]
    if Path(filename).name != filename or not filename.startswith("epoch-"):
        raise ValueError("Invalid selected checkpoint path")
    path = run / filename
    if dual_sha256(path) != selected["sha256"]:
        raise ValueError("Selected pretraining checkpoint checksum mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("stage") != "heatmap_pretraining":
        raise ValueError("Selected checkpoint is not a pretraining model")
    pretraining_config(saved)
    if saved["recipe"]["manifest_sha256"] != dual_sha256(manifest):
        raise ValueError("Posttraining must use the pretrained CNN's frozen manifest")
    if selected["epoch"] != completed["best_epoch"] or selected["epoch"] != saved["epoch"]:
        raise ValueError("Pretraining completion and selected checkpoint disagree")
    scope = saved["recipe"]["selection_scope"]
    score = saved["validation"]["scopes"][scope]["macro_mean_error_px"]
    if selected["selection_error_px"] != score or completed["best_error_px"] != score:
        raise ValueError("Saved pretraining selection scores disagree")
    return path, saved
