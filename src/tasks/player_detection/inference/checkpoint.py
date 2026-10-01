"""Validate the provenance of a deployable player DINO checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.submodules.models.dino.architecture import COCO_PERSON_CLASS_ID
from src.tasks.player_detection.model_export import EXPORT_FORMAT


@dataclass(frozen=True, slots=True)
class PlayerCheckpointInfo:
    """Identity of a fine-tuned checkpoint; the weights are loaded separately."""

    path: Path
    source_checkpoint_sha256: str
    epoch: int
    global_step: int


def inspect_player_checkpoint(checkpoint: Path) -> PlayerCheckpointInfo:
    """Reject COCO and Lightning checkpoints before player inference.

    ``mmap`` avoids reading all model tensors merely to inspect provenance.
    ``DinoPersonDetector`` subsequently validates the architecture and loads
    the full state dict strictly.
    """
    if not checkpoint.is_absolute():
        raise ValueError(f"Player checkpoint must be absolute: {checkpoint}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Player checkpoint not found: {checkpoint}")
    payload: Any = torch.load(
        checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("tennis_lab"), Mapping
    ):
        raise ValueError(
            "Expected an exported player DINO checkpoint with tennis_lab provenance; "
            "export a Lightning .ckpt with scripts.export_checkpoint first."
        )
    metadata: Mapping[str, Any] = payload["tennis_lab"]
    expected = {
        "format": EXPORT_FORMAT,
        "task": "player_detection",
        "class_id": COCO_PERSON_CLASS_ID,
    }
    mismatches = {
        name: (value, metadata.get(name))
        for name, value in expected.items()
        if type(metadata.get(name)) is not type(value) or metadata.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint is not a supported tennis-player model: {mismatches}")
    source_hash = metadata.get("source_checkpoint_sha256")
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(char not in "0123456789abcdef" for char in source_hash)
    ):
        raise ValueError("Checkpoint has no valid source_checkpoint_sha256")
    epoch = metadata.get("epoch")
    global_step = metadata.get("global_step")
    if type(epoch) is not int or epoch < 0:
        raise ValueError(f"Checkpoint has invalid epoch: {epoch!r}")
    if type(global_step) is not int or global_step < 0:
        raise ValueError(f"Checkpoint has invalid global_step: {global_step!r}")
    return PlayerCheckpointInfo(
        path=checkpoint,
        source_checkpoint_sha256=source_hash,
        epoch=epoch,
        global_step=global_step,
    )
