"""Configuration utilities for WASB tennis training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _apply_defaults(cfg: DictConfig, base_dir: Path) -> DictConfig:
    """Resolve Hydra-like defaults by loading referenced config files."""
    defaults = cfg.pop("defaults", [])
    merged: DictConfig = OmegaConf.create()

    for item in defaults:
        if isinstance(item, (dict, DictConfig)):
            for group, name in item.items():
                cfg_path = base_dir / group / f"{name}.yaml"
                if not cfg_path.exists():
                    raise FileNotFoundError(f"Default config file not found: {cfg_path}")
                part = OmegaConf.load(cfg_path)
                # Merge each part under its group key, e.g. merged[group] <- merged[group] + part
                existing_group_cfg = merged.get(group, OmegaConf.create())
                merged[group] = OmegaConf.merge(existing_group_cfg, part)
        else:
            raise ValueError(f"Unsupported defaults entry: {item}")

    return OmegaConf.merge(merged, cfg)


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML configuration file, resolving `defaults` includes."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)  # type: ignore[assignment]
    if cfg is None:
        raise ValueError(f"Config file is empty or invalid: {config_path}")

    if "defaults" in cfg:
        cfg = _apply_defaults(cfg, config_path.parent)

    return cfg


def merge_configs(*configs: DictConfig | dict[str, Any]) -> DictConfig:
    """Merge multiple configurations; later entries override earlier ones."""
    if not configs:
        raise ValueError("At least one config must be provided to merge_configs.")
    return OmegaConf.merge(*configs)  # type: ignore[return-value]


def resolve_model_name(config: DictConfig, config_path: str | Path) -> str:
    model_cfg = None
    if hasattr(config, "get"):
        model_cfg = config.get("model")
    if model_cfg is None:
        model_cfg = getattr(config, "model", None)

    name = None
    if model_cfg is not None:
        if hasattr(model_cfg, "get"):
            name = model_cfg.get("name")
        else:
            name = getattr(model_cfg, "name", None)

    if name is not None and str(name).strip() != "":
        return str(name).strip()

    return Path(config_path).stem
