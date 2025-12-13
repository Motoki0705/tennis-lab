"""Configuration utilities leveraging Hydra composition."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a Hydra config file with optional CLI-style overrides."""

    cfg_path = Path(path)
    overrides = overrides or []
    with initialize_config_dir(
        config_dir=str(cfg_path.parent), job_name="load_config", version_base="1.3"
    ):
        return compose(config_name=cfg_path.stem, overrides=overrides)


def resolve_model_name(config: DictConfig, config_path: str | Path | None) -> str:
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

    if config_path is None:
        return "wasb"
    return Path(config_path).stem

