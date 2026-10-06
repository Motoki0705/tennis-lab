"""Evaluate a checkpoint against its recorded fixed held-out input profile."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from src.tasks.ball_refiner_3d.configuration.core import parse_section
from src.tasks.ball_refiner_3d.configuration.data import CorruptionConfig
from src.tasks.ball_refiner_3d.configuration.evaluation import (
    evaluation_profile,
    training_profile_sections,
)
from src.tasks.ball_refiner_3d.data.dataset import SharedDataset
from src.tasks.ball_refiner_3d.data.preprocessing import prepare
from src.tasks.ball_refiner_3d.evaluation.artifacts import evaluate_checkpoint
from src.tasks.ball_refiner_3d.model_io.checkpoint import checkpoint_model_config
from src.utils.device import resolve_device


def evaluate_run(
    *,
    checkpoint: Path,
    run_config: Path,
    dataset: Path,
    output: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Evaluation output must be a new directory")
    if output.resolve().is_relative_to(dataset.resolve()):
        raise ValueError("Evaluation output must be outside the dataset")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    raw = yaml.safe_load(run_config.read_text())
    profile = evaluation_profile(raw)
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = checkpoint_model_config(metadata)
    _, event = training_profile_sections(raw)
    if event["sigma_frames"] != metadata["event_sigma_frames"]:
        raise ValueError("Recorded Gaussian target width does not match the checkpoint")
    if raw["model"] != metadata["model_config"]:
        raise ValueError("Recorded training model does not match the checkpoint")
    shared = SharedDataset(dataset)
    if (
        shared.manifest_hash != metadata["manifest_sha256"]
        or shared.fps != metadata["fps"]
    ):
        raise ValueError("Dataset manifest/FPS does not match the training checkpoint")
    augmentation = parse_section(CorruptionConfig, profile["augmentation"])
    test = prepare(
        shared.split("test"),
        model.dimensions,
        augmentation,
        profile["augmentation_seed"],
        event_sigma_frames=metadata["event_sigma_frames"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    return evaluate_checkpoint(
        checkpoint,
        output,
        test,
        resolve_device(device),
        seed=profile["flow_seed"],
        batch_size=batch_size,
        common_metadata={
            "dataset_manifest_sha256": shared.manifest_hash,
            "evaluation_profile": profile,
        },
    )
