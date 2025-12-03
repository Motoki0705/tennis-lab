"""Configuration utilities for BLCS.

Re-exports config utilities from plcs and adds BLCS-specific helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    pass

# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def load_config(config_path: str | Path | None = None) -> DictConfig:
    """Load BLCS configuration from YAML file.

    Args:
        config_path: Path to config file. If None, loads default config.

    Returns:
        DictConfig: Loaded configuration.

    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return OmegaConf.load(config_path)


def merge_configs(base: DictConfig, override: DictConfig) -> DictConfig:
    """Merge two configurations, with override taking precedence.

    Args:
        base: Base configuration.
        override: Override configuration.

    Returns:
        DictConfig: Merged configuration.

    """
    return OmegaConf.merge(base, override)


def save_config(config: DictConfig, path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration to save.
        path: Output path.

    """
    OmegaConf.save(config, path)
