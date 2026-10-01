"""Convert a fine-tuning checkpoint into the upstream DINO checkpoint format.

The exported ``{"model", "args"}`` payload is exactly what
``DinoPersonDetector`` loads, so the tennis-scene pipeline uses it through
``people_models.dino_checkpoint`` without code changes. Class id 1 means
"tennis player" in exported checkpoints. Provenance is stored under
``"tennis_lab"``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.submodules.models.dino.architecture import (
    COCO_PERSON_CLASS_ID,
    build_dino,
    dino_4scale_swin_args,
)
from src.tasks.player_detection.configuration import ExportConfig

LIGHTNING_PREFIX = "model.dino."
EXPORT_FORMAT = "tennis_lab_player_dino.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_dino_state_dict(lightning_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip the Lightning/``PlayerDinoModel`` prefix from the upstream weights."""
    state = {
        key[len(LIGHTNING_PREFIX) :]: value
        for key, value in lightning_state.items()
        if key.startswith(LIGHTNING_PREFIX)
    }
    if not state:
        raise ValueError(f"Checkpoint has no '{LIGHTNING_PREFIX}*' weights")
    return state


def export_player_checkpoint(config: ExportConfig) -> Path:
    if config.destination.exists():
        raise FileExistsError(f"Refusing to overwrite exported checkpoint: {config.destination}")
    payload: Any = torch.load(config.lightning_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "state_dict" not in payload:
        raise ValueError(f"Not a Lightning checkpoint: {config.lightning_checkpoint}")
    state = extract_dino_state_dict(payload["state_dict"])
    args = dino_4scale_swin_args("cpu", use_checkpoint=False)
    model, _ = build_dino(config.repository, args)
    model.load_state_dict(state, strict=True)  # Architecture check before writing.
    train_config = payload["hyper_parameters"]["config"]
    if isinstance(train_config, DictConfig):
        train_config = OmegaConf.to_container(train_config, resolve=True)
    exported = {
        "model": state,
        # Plain Namespace like the released checkpoint (no repository types pickled).
        "args": argparse.Namespace(**{**vars(args), "device": "cuda"}),
        "tennis_lab": {
            "format": EXPORT_FORMAT,
            "task": "player_detection",
            "class_id": COCO_PERSON_CLASS_ID,
            "class_meaning": "tennis player on the main court (chat-annotation definition)",
            "exported_at": datetime.now(UTC).isoformat(),
            "source_checkpoint": str(config.lightning_checkpoint),
            "source_checkpoint_sha256": _sha256(config.lightning_checkpoint),
            "epoch": int(payload["epoch"]),
            "global_step": int(payload["global_step"]),
            "train_config": train_config,
        },
    }
    config.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.destination.with_suffix(".pth.partial")
    torch.save(exported, temporary)
    os.replace(temporary, config.destination)
    return config.destination
