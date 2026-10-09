from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.models.mdd_pretrain import (
    DeepMDDQueryDetector,
    MDDDPTDetector,
)
from src.tasks.ball_detection.training.coordinate_checkpoint import (
    save_coordinate_checkpoint,
)
from src.utils.checksum import dual_sha256

SCHEMA = "mdd_dpt_pretraining.v1"


def save_pretraining(path: Path, model: MDDDPTDetector, optimizer: torch.optim.Optimizer,
                     *, epoch: int, step: int, recipe: dict[str, Any], code: dict[str, Any],
                     report: dict[str, Any], best: float, best_epoch: int) -> None:
    save_coordinate_checkpoint(dict(schema=SCHEMA, stage="heatmap_pretraining", epoch=epoch, global_step=step,
        model_config=asdict(model.config), state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
        recipe=recipe, code=code, validation=report, best_score=best, best_epoch=best_epoch,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None), path)


def transfer_encoder(path: Path, model: DeepMDDQueryDetector) -> dict[str, Any]:
    """Weights-only transfer; validate encoder topology and fixed MDD identity before loading."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema") != SCHEMA or saved.get("stage") != "heatmap_pretraining":
        raise ValueError("Expected a DPT pretraining checkpoint")
    for key in ("stem_channels", "mixed_channels", "residual_blocks"):
        if tuple(saved["model_config"][key]) != tuple(getattr(model.config, key)):
            raise ValueError(f"Pretrained encoder topology mismatch: {key}")
    if saved["recipe"]["input_contract"] != model.mdd.input_contract():
        raise ValueError("Pretrained MDD input contract mismatch")
    weights = {k.removeprefix("encoder."): v for k, v in saved["state_dict"].items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(weights, strict=True)
    model.freeze_encoder(True)
    return dict(checkpoint=str(path), sha256=dual_sha256(path), global_step=saved["global_step"],
                input_contract=saved["recipe"]["input_contract"], image_decode=saved["recipe"]["image_decode"])
