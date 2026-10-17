from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.checkpoints.configuration import checkpoint_model_config
from src.tasks.ball_detection.checkpoints.coordinates import (
    save_coordinate_checkpoint,
)
from src.tasks.ball_detection.checkpoints.provenance import parent_provenance
from src.tasks.ball_detection.models.mdd import (
    DeepMDDQueryDetector,
    MDDDPTDetector,
    MDDPretrainConfig,
)
from src.utils.checksum import dual_sha256

SCHEMA = "mdd_dpt_pretraining.v2"


def pretraining_config(saved: dict[str, Any]) -> MDDPretrainConfig:
    """Identify historical CNNs by schema and their exact saved input contract."""
    if saved.get("schema") not in {"mdd_dpt_pretraining.v1", SCHEMA}:
        raise ValueError("Unsupported pretraining checkpoint schema")
    return checkpoint_model_config(saved)


def save_pretraining(path: Path, model: MDDDPTDetector, optimizer: torch.optim.Optimizer,
                     *, epoch: int, step: int, recipe: dict[str, Any], code: dict[str, Any],
                     report: dict[str, Any], best: float, best_epoch: int) -> None:
    if recipe["input_contract"] != model.mdd.input_contract():
        raise ValueError("Pretraining recipe input contract differs from the model")
    save_coordinate_checkpoint(dict(schema=SCHEMA, stage="heatmap_pretraining", epoch=epoch, global_step=step,
        model_config=asdict(model.config), state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
        recipe=recipe, code=code, validation=report, best_score=best, best_epoch=best_epoch,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None), path)


def transfer_encoder(path: Path, model: DeepMDDQueryDetector) -> dict[str, Any]:
    """Weights-only transfer; validate encoder topology and fixed MDD identity before loading."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("stage") != "heatmap_pretraining":
        raise ValueError("Expected a DPT pretraining checkpoint")
    previous = pretraining_config(saved)
    for key in ("input_channels", "encoder_variant", "temporal_mixing"):
        if getattr(previous, key) != getattr(model.config, key):
            raise ValueError(f"Pretrained encoder topology mismatch: {key}")
    for key in ("stem_channels", "mixed_channels", "residual_blocks"):
        if tuple(saved["model_config"][key]) != tuple(getattr(model.config, key)):
            raise ValueError(f"Pretrained encoder topology mismatch: {key}")
    if saved["recipe"]["input_contract"] != model.mdd.input_contract():
        raise ValueError("Pretrained MDD input contract mismatch")
    weights = {k.removeprefix("encoder."): v for k, v in saved["state_dict"].items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(weights, strict=True)
    model.freeze_encoder(True)
    return dict(mode="encoder", checkpoint=str(path), sha256=dual_sha256(path), global_step=saved["global_step"],
                input_contract=saved["recipe"]["input_contract"], image_decode=saved["recipe"]["image_decode"],
                transferred_modules=["encoder"],
                transferred_parameters=[k for k in model.state_dict() if k.startswith("encoder.")],
                initialized_parameters=[k for k in model.state_dict() if not k.startswith("encoder.")],
                discarded_modules=["decoder", "head"], ancestry=parent_provenance(saved))


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
