"""Start a new query training run from all adopted weights, never optimizer state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.tasks.ball_detection.checkpoints.coordinates import restore_coordinate_model
from src.tasks.ball_detection.checkpoints.posttraining import SCHEMA
from src.tasks.ball_detection.models.mdd import DeepMDDQueryDetector
from src.tasks.ball_detection.training.runtime.images import jpeg_decoder_contract
from src.utils.checksum import dual_sha256
from src.utils.models.state import state_dict_sha256


def initialize_full_model(
    path: Path, *, expected_sha256: str, jpeg_decoder: str,
) -> tuple[DeepMDDQueryDetector, dict[str, Any]]:
    if dual_sha256(path) != expected_sha256:
        raise ValueError("Initial full-model checkpoint hash changed")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if dual_sha256(path) != expected_sha256:
        raise ValueError("Initial checkpoint changed while loading")
    if not isinstance(saved, dict) or saved.get("schema") != SCHEMA:
        raise ValueError("Full-model initialization requires query posttraining v1")
    model, recipe, _ = restore_coordinate_model(saved)
    if not isinstance(model, DeepMDDQueryDetector):
        raise ValueError("Full-model initialization requires the deep MDD query model")
    if recipe["image_decode"] != jpeg_decoder_contract(jpeg_decoder):
        raise ValueError("Full-model JPEG decoder must match the adopted model")
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Initial full-model weights contain nonfinite values")
    initial_digest = state_dict_sha256(saved["state_dict"])
    if state_dict_sha256(model.state_dict()) != initial_digest:
        raise ValueError("Full-model loading changed tensor names, dtype or values")
    provenance = dict(
        mode="full_model", path=str(path), sha256=expected_sha256,
        schema=SCHEMA, state_dict_sha256=initial_digest,
        input_contract=recipe["input_contract"], image_decode=recipe["image_decode"],
        parent_manifest_sha256=recipe["manifest_sha256"],
        optimizer_initialization="fresh", schedule_initialization="step_zero",
    )
    return model, provenance
