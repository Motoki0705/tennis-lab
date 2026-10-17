"""Resolve historical CNN configurations from their saved, versioned input contract."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.tasks.ball_detection.models.mdd import MDDPretrainConfig
from src.tasks.ball_detection.preprocessing.rgb_mdd import input_transform_from_contract


def checkpoint_model_config(saved: dict[str, Any]) -> MDDPretrainConfig:
    """Accept missing channels only for a fully identified historical MDD input.

    The checkpoint schema alone does not identify two channels: the current
    pretraining v2 and query v1 schemas also support five-channel models.
    Never infer an input contract from tensor shapes or mutate saved metadata.
    """
    schema = saved.get("schema")
    if schema not in {
        "mdd_dpt_pretraining.v1", "mdd_dpt_pretraining.v2", "mdd_query_posttraining.v1",
    }:
        raise ValueError("Unsupported CNN checkpoint schema")
    raw = dict(saved["model_config"])
    contract = saved["recipe"].get("input_contract")
    if "input_channels" not in raw:
        # This exact versioned contract denotes the original two-channel MDD.
        # Missing, partial, unknown and RGB+MDD contracts are rejected here.
        input_transform_from_contract(contract, input_channels=2)
        raw["input_channels"] = 2
    expected = set(MDDPretrainConfig.__dataclass_fields__)
    if schema == "mdd_dpt_pretraining.v1":
        if set(raw) != expected - {"encoder_variant", "temporal_mixing"} or raw["input_channels"] != 2:
            raise ValueError("Invalid historical v1 pretraining configuration")
        raw.update(encoder_variant="residual", temporal_mixing="dense3d")
    if set(raw) != expected:
        raise ValueError("Incomplete pretraining CNN configuration")
    for key in ("stem_channels", "mixed_channels", "residual_blocks"):
        raw[key] = tuple(raw[key])
    config = MDDPretrainConfig(**raw)
    input_transform_from_contract(contract, input_channels=config.input_channels)
    return config


def checkpoint_training_recipe(
    saved: dict[str, Any], config: MDDPretrainConfig,
) -> dict[str, Any]:
    """Compare resolved model fields while keeping all other restart guards exact."""
    recipe = saved["recipe"]
    if recipe["model"] != saved["model_config"]:
        raise ValueError("Checkpoint model configuration differs from its recipe")
    return dict(recipe, model=asdict(config))
