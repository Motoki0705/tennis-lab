"""Configuration utilities using OmegaConf."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        DictConfig: Loaded configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return OmegaConf.load(path)  # type: ignore[return-value]


def merge_configs(*configs: DictConfig | dict[str, Any]) -> DictConfig:
    """Merge multiple configurations.

    Later configs override earlier ones.

    Args:
        *configs: Configuration dictionaries to merge.

    Returns:
        DictConfig: Merged configuration.

    """
    return OmegaConf.merge(*configs)  # type: ignore[return-value]


def get_default_config() -> DictConfig:
    """Get the default PLCS configuration.

    Returns:
        DictConfig: Default configuration with all parameters.

    """
    default = {
        "model": {
            "hidden_dim": 256,
            "num_layers": 4,
            "num_heads": 8,
            "dropout": 0.1,
            "use_court_context": True,
        },
        "data": {
            "batch_size": 64,
            "num_workers": 4,
            "num_cameras": 4,
            "num_scenes_per_epoch": 10000,
            "sequence_length": 1,
        },
        "training": {
            "max_epochs": 100,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "warmup_steps": 1000,
            "position_loss_weight": 1.0,
            "rotation_loss_weight": 1.0,
        },
        "camera": {
            "z_min": 3.0,
            "z_max": 5.0,
            "r_in": 1.0,
            "r_out": 2.0,
            "hfov_deg": 60.0,
            "image_size": [1280, 720],
        },
    }
    return OmegaConf.create(default)
