"""Explicit, weight-preserving export of legacy Re-ID checkpoints without the head."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from src.tasks.plcs.model_io.person_association import MODEL_CONTRACTS, REID_MODEL
from src.utils.checksum import dual_sha256

_LEGACY_CONTRACT = "plcs_fixed_track_reid_v1"
_REMOVED_PARAMETERS = ("model.player_head.weight", "model.player_head.bias")


def export_headless_reid_checkpoint(source: str | Path, destination: str | Path) -> Path:
    """Export v1 as v2 for inference or run.init_weights, never optimizer resume.

    Every retained tensor and the validation-selected matching threshold stay
    unchanged. Loading a legacy checkpoint never invokes this conversion.
    """
    source_path, destination_path = Path(source), Path(destination)
    if destination_path.exists() or source_path.resolve() == destination_path.resolve():
        raise FileExistsError(f"Checkpoint export destination must be new: {destination_path}")
    source_sha256 = dual_sha256(source_path)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("person_association_model") != REID_MODEL or checkpoint.get("person_association_contract") != _LEGACY_CONTRACT:
        raise ValueError("Explicit export requires a PLCS Re-ID v1 checkpoint")
    hyperparameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyperparameters, Mapping) or not isinstance(hyperparameters.get("config"), DictConfig):
        raise ValueError("Legacy Re-ID checkpoint must contain its DictConfig")
    source_config = hyperparameters["config"]
    config = deepcopy(source_config)
    for section, key in ((config.loss, "player_weight"), (config.metrics, "player_threshold"), (config.data.augmentation, "false_track_probability")):
        if key not in section:
            raise ValueError(f"Legacy Re-ID checkpoint is missing {key}")
        with open_dict(section):
            del section[key]
    with open_dict(config.run):
        config.run.resume = None
        config.run.init_weights = None
        # Test-only consumers of the exported module must not overwrite the source run.
        config.run.output_dir = f"{config.run.output_dir}/headless_v2"
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping) or any(key not in state for key in _REMOVED_PARAMETERS):
        raise ValueError("Legacy Re-ID checkpoint must contain both auxiliary-head parameters")
    weight, bias = (state[key] for key in _REMOVED_PARAMETERS)
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor) or weight.shape != (1, int(config.model.hidden_dim)) or bias.shape != (1,):
        raise ValueError("Legacy auxiliary-head dimensions disagree with model config")
    retained = {key: value for key, value in state.items() if key not in _REMOVED_PARAMETERS}
    if any(not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()) for value in retained.values()):
        raise ValueError("Re-ID checkpoint tensors must be finite")
    exported: dict[str, Any] = {
        "state_dict": retained,
        "hyper_parameters": {"config": config},
        "pytorch-lightning_version": checkpoint["pytorch-lightning_version"],
        "epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"],
        "person_association_model": REID_MODEL,
        "person_association_contract": MODEL_CONTRACTS[REID_MODEL],
        "weights_only_export": True,
        "reid_export_provenance": {
            "operation": "remove_auxiliary_head_without_retraining",
            "source_path": str(source_path.resolve()),
            "source_sha256": source_sha256,
            "source_contract": _LEGACY_CONTRACT,
            "removed_parameters": list(_REMOVED_PARAMETERS),
            "source_training_config": OmegaConf.to_container(source_config, resolve=True),
        },
    }
    from src.tasks.plcs.training.association_lightning_module import (
        PLCSAssociationLightningModule,
    )

    module = PLCSAssociationLightningModule(config)
    module.on_load_checkpoint(exported)
    module.load_state_dict(retained, strict=True)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("xb") as stream:
        torch.save(exported, stream)
    return destination_path
