"""Portable hybrid checkpoints, exact epoch resume and weights-only continuation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.checkpoints.coordinates import save_coordinate_checkpoint
from src.tasks.ball_detection.checkpoints.hybrid import initialize_hybrid_from_query
from src.tasks.ball_detection.checkpoints.provenance import parent_provenance
from src.tasks.ball_detection.models.mdd import DeepMDDHybridDetector, MDDHybridConfig
from src.tasks.ball_detection.preprocessing.rgb_mdd import input_transform_from_contract
from src.tasks.ball_detection.training.hybrid.configuration import (
    validate_loss,
    validate_threshold,
)
from src.tasks.ball_detection.training.runtime.images import jpeg_decoder_contract
from src.utils.checksum import dual_sha256

HYBRID_CHECKPOINT_SCHEMA = "mdd_hybrid_training.v1"
HYBRID_MODEL_KIND = "mdd_hybrid"


def restore_hybrid_model(saved: dict[str, Any]) -> tuple[DeepMDDHybridDetector, dict[str, Any]]:
    """Restore without opening training data, previous runs or ancestor weights."""
    if (saved.get("schema") != HYBRID_CHECKPOINT_SCHEMA or saved.get("stage") != "hybrid"
            or saved.get("model_kind") != HYBRID_MODEL_KIND):
        raise ValueError("Expected an explicit mdd_hybrid_training.v1 checkpoint")
    recipe = saved["recipe"]
    if recipe["model"] != saved["model_config"]:
        raise ValueError("Hybrid model configuration differs from its recipe")
    config = MDDHybridConfig(**saved["model_config"])
    preprocessing = input_transform_from_contract(recipe["input_contract"], input_channels=config.input_channels)
    validate_threshold(recipe["presence_threshold"])
    validate_loss(recipe["loss"])
    if recipe["parent"]["input_contract"] != preprocessing.input_contract() or recipe["parent"]["image_decode"] != recipe["image_decode"]:
        raise ValueError("Hybrid preprocessing differs from its initialization receipt")
    if recipe["image_decode"] != jpeg_decoder_contract(recipe["runtime"]["jpeg_decoder"]):
        raise ValueError("Hybrid JPEG decoder/version differs from its saved contract")
    model = DeepMDDHybridDetector(config, mdd_a=preprocessing.a, mdd_b=preprocessing.b)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model, recipe


def initialize_hybrid(path: Path, config: MDDHybridConfig, *, mode: str,
                      data: dict[str, Any], image_decode: dict[str, Any]) -> tuple[DeepMDDHybridDetector, dict[str, Any]]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    previous = saved["recipe"]
    if mode == "query":
        if previous["manifest_sha256"] != data["manifest_sha256"]:
            raise ValueError("Stage three must use stage two's frozen snapshot/manifest")
        model, receipt = initialize_hybrid_from_query(path, config)
    elif mode == "continual":
        model, previous = restore_hybrid_model(saved)
        if model.config != config:
            raise ValueError("Continual learning requires the adopted model's exact configuration")
        old = previous["data_identity"]
        if old["snapshot"]["hashes"] == data["snapshot"]["hashes"]:
            raise ValueError("Continual learning requires a new snapshot; use resume for the same run")
        if any(data["splits"].get(clip) != split for clip, split in old["splits"].items()):
            raise ValueError("Cumulative snapshot must retain old clips and their splits")
        receipt = dict(checkpoint=str(path), sha256=dual_sha256(path), source_stage="hybrid",
                       global_step=saved["global_step"], input_contract=model.mdd.input_contract(),
                       image_decode=previous["image_decode"], transferred_parameters=list(model.state_dict()),
                       transferred_modules=["encoder", "projection", "spatial_position", "ball_query", "blocks",
                                            "decoder", "heatmap_head", "presence_head"],
                       initialized_modules=[], initialized_parameters=[], ancestry=parent_provenance(saved))
    else:
        raise ValueError("Initialization mode must be query or continual")
    if receipt["image_decode"] != image_decode:
        raise ValueError("Initialization JPEG decoder must match the previous stage exactly")
    receipt.update(mode=mode, optimizer="new", global_step_policy="reset", freeze_policy="new recipe")
    return model, receipt


def save_hybrid(path: Path, model: DeepMDDHybridDetector, optimizer: torch.optim.Optimizer, *,
                recipe: dict[str, Any], code: dict[str, Any], epoch: int, step: int,
                report: dict[str, Any], best: float, best_epoch: int) -> None:
    if recipe["model"] != asdict(model.config) or recipe["input_contract"] != model.mdd.input_contract():
        raise ValueError("Hybrid save model/recipe mismatch")
    save_coordinate_checkpoint(dict(schema=HYBRID_CHECKPOINT_SCHEMA, model_kind=HYBRID_MODEL_KIND, stage="hybrid",
        model_config=asdict(model.config), state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
        recipe=recipe, code=code, epoch=epoch, global_step=step, validation=report,
        best_score=best, best_epoch=best_epoch, torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None), path)


def write_hybrid_best(output: Path, saved: dict[str, Any]) -> None:
    path = output / f"epoch-{saved['epoch']:03d}.pt"
    scope = saved["recipe"]["selection_scope"]
    record = dict(checkpoint=path.name, sha256=dual_sha256(path), epoch=saved["epoch"],
                  global_step=saved["global_step"], selection_error_px=saved["validation"]["scopes"][scope]["macro_mean_error_px"],
                  **saved["validation"])
    temporary = output / "best.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "best.json")


def resume_hybrid(path: Path, output: Path, model: DeepMDDHybridDetector, optimizer: torch.optim.Optimizer, *,
                  recipe: dict[str, Any], code: dict[str, Any], device: torch.device) -> tuple[int, int, float, int]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if path.parent != output or not (output / "config.json").is_file() or saved["recipe"] != recipe:
        raise ValueError("Resume requires the original output and identical recipe/snapshot/initialization")
    if saved["code"]["source_sha256"] != code["source_sha256"]:
        raise ValueError("Resume source code changed")
    restored, _ = restore_hybrid_model(saved)
    if restored.config != model.config:
        raise ValueError("Resume model configuration changed")
    epoch = int(saved["epoch"])
    if path.name != f"epoch-{epoch:03d}.pt" or epoch + 1 >= recipe["epochs"]:
        raise ValueError("Resume requires an epoch checkpoint with remaining original budget")
    if any(int(p.stem.split("-")[1]) > epoch for p in output.glob("epoch-*.pt")):
        raise ValueError("Resume requires the latest completed epoch")
    best_epoch = int(saved["best_epoch"])
    if not 0 <= best_epoch <= epoch:
        raise ValueError("Resume best epoch is outside completed history")
    best = torch.load(output / f"epoch-{best_epoch:03d}.pt", map_location="cpu", weights_only=True)
    scope = recipe["selection_scope"]
    if best["recipe"] != recipe or best["validation"]["scopes"][scope]["macro_mean_error_px"] != saved["best_score"]:
        raise ValueError("Resume best checkpoint/selection provenance differs")
    model.load_state_dict(saved["state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"])
    if device.type == "cuda":
        if saved["cuda_rng"] is None:
            raise ValueError("CUDA resume requires CUDA RNG states")
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    write_hybrid_best(output, best)
    for filename in ("train.jsonl", "metrics.jsonl"):
        log = output / filename
        if log.is_file():
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            log.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows if row["epoch"] <= epoch))
    return epoch + 1, int(saved["global_step"]), float(saved["best_score"]), best_epoch
