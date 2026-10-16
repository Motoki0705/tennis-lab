"""Versioned training-record profiles for reproducible held-out evaluation."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from src.tasks.ball_refiner_3d.configuration.core import parse_section
from src.tasks.ball_refiner_3d.configuration.data import CorruptionConfig


def evaluation_profile(config: dict[str, Any]) -> dict[str, Any]:
    augmentation, _ = training_profile_sections(config)
    corruption = parse_section(CorruptionConfig, augmentation)
    corruption = replace(
        corruption,
        event_probability=float(config["data"]["evaluation_event_probability"]),
    )
    seed = int(config["data"]["evaluation_seed"])
    return {
        "augmentation": asdict(corruption),
        "augmentation_seed": seed,
        "flow_seed": seed,
        "missing_enabled": True,
        "noise_enabled": True,
    }


def training_profile_sections(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read either recorded v1 training config or modular v2 config explicitly."""
    if "corruption" in config and "augmentation" not in config and "loss" not in config:
        return config["corruption"], config["training"]["event"]
    if "augmentation" in config and "loss" in config and "corruption" not in config:
        return config["augmentation"], config["loss"]["event"]
    raise ValueError("Unsupported or ambiguous refiner training configuration format")
