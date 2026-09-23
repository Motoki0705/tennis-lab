"""Strict PLCS residual checkpoint contract; historical conversion is explicit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.tasks.plcs.configuration import validate_residual_config
from src.tasks.plcs.configuration_contracts import ResidualConfig
from src.tasks.plcs.model_io.residual_contracts import feature_dimension
from src.utils.schema.court_normalization import validate_court_coordinate_normalization


def checkpoint_contract(config: ResidualConfig) -> dict[str, Any]:
    return {
        "family": config.model.name,
        "schema_version": 3,
        "ffn_type": config.model.ffn_type,
        "features": {"residual_encoding": "raw", "residual_scale": 1.0},
        "task": "plcs",
        "joints": 17,
        "feature_dim": feature_dimension(17),
        "coordinate_frame": "physical_court",
        "output_unit": "metre",
        "root_definition": "coco_hip_midpoint",
        "relative_axes": "court",
        "root_indices": [11, 12],
        "missing_seed": "linear_interpolation_edge_hold_else_root_with_original_valid_mask",
        "camera_estimate_is_input_only": True,
    }


def validate_residual_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Reject incompatible artifacts before consuming weights or configuration."""
    validate_court_coordinate_normalization(
        checkpoint, artifact="PLCS residual checkpoint"
    )
    marker = checkpoint.get("geometric_residual_contract")
    if not isinstance(marker, dict) or marker.get("schema_version") != 3:
        raise ValueError(
            "Expected PLCS residual checkpoint schema 3; convert the selected historical "
            "checkpoint explicitly with scripts.migrate_residual_checkpoint"
        )
    parameters = checkpoint.get("hyper_parameters")
    if not isinstance(parameters, Mapping) or "config" not in parameters:
        raise ValueError("Residual checkpoint requires its embedded config")
    embedded = parameters["config"]
    if isinstance(embedded, DictConfig):
        embedded = OmegaConf.to_container(embedded, resolve=True)
    config = OmegaConf.create(embedded)
    if not isinstance(config, DictConfig):
        raise ValueError("Residual checkpoint config must be a mapping")
    if marker != checkpoint_contract(validate_residual_config(config)):
        raise ValueError("Incompatible PLCS residual checkpoint semantics")
    if not isinstance(checkpoint.get("state_dict"), Mapping):
        raise ValueError("Residual checkpoint requires a complete state_dict")
    return dict(checkpoint)
