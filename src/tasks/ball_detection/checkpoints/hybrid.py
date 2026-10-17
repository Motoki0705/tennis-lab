"""Stage-two weights into a newly initialized hybrid; no stage-one DPT is read."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.checkpoints.coordinates import query_checkpoint_config
from src.tasks.ball_detection.models.mdd import DeepMDDHybridDetector, MDDHybridConfig
from src.utils.checksum import dual_sha256

QUERY_TRANSFER_MODULES = ("encoder", "projection", "spatial_position", "ball_query", "blocks")
HYBRID_NEW_MODULES = ("decoder", "heatmap_head", "presence_head")


def initialize_hybrid_from_query(path: Path, config: MDDHybridConfig) -> tuple[DeepMDDHybridDetector, dict[str, Any]]:
    """Return a fresh stage-three model and an explicit weights-only provenance receipt.

    Only the supplied query checkpoint is opened. Its parent checkpoint path is
    provenance, never a source of weights. Optimizer/freeze state is not carried
    over; stage three owns its schedule.
    """
    saved = torch.load(path, map_location="cpu", weights_only=True)
    previous, preprocessing = query_checkpoint_config(saved)
    # The DPT width and activation checkpointing do not change transferred weights.
    for key, value in asdict(previous).items():
        if key not in {"decoder_channels", "activation_checkpointing"} and value != getattr(config, key):
            raise ValueError(f"Query transfer configuration mismatch: {key}")
    model = DeepMDDHybridDetector(config, mdd_a=preprocessing.a, mdd_b=preprocessing.b)
    if model.mdd.input_contract() != preprocessing.input_contract():
        raise ValueError("Query transfer input contract mismatch")
    target = model.state_dict()
    transferred = [key for key in target if key.split(".")[0] in QUERY_TRANSFER_MODULES]
    initialized = [key for key in target if key.split(".")[0] in HYBRID_NEW_MODULES]
    discarded = {"head.0.weight", "head.0.bias", "head.1.weight", "head.1.bias"}
    weights = saved["state_dict"]
    if set(weights) != set(transferred) | discarded:
        raise ValueError("Query checkpoint state must contain exactly the encoder, query decoder and coordinate head")
    if set(target) != set(transferred) | set(initialized):
        raise ValueError("Hybrid parameters lack an explicit transfer/initialization policy")
    for key in transferred:
        if not isinstance(weights[key], torch.Tensor) or weights[key].shape != target[key].shape:
            raise ValueError(f"Query transfer weight shape mismatch: {key}")
    # Keep the freshly initialized tensors for every new module; strict=True also
    # enforces that future architecture changes declare their transfer policy.
    model.load_state_dict({key: weights[key] if key in transferred else value for key, value in target.items()}, strict=True)
    receipt = dict(
        checkpoint=str(path), sha256=dual_sha256(path), source_stage="query_posttraining",
        global_step=saved["global_step"], input_contract=model.mdd.input_contract(),
        image_decode=saved["recipe"]["image_decode"], source_model_config=asdict(previous),
        target_model_config=asdict(config), transferred_modules=list(QUERY_TRANSFER_MODULES),
        initialized_modules=list(HYBRID_NEW_MODULES), discarded_modules=["head"],
        transferred_parameters=transferred, initialized_parameters=initialized,
    )
    return model, receipt
