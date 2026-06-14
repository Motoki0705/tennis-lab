"""Configuration helpers shared across PLCS generation entry points."""

from __future__ import annotations

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


def prepare_generation_config(
    config: DictConfig,
    *,
    resolve: bool = True,
) -> DictConfig:
    """Prepare PLCS generation config by absolutizing required data paths."""
    prepared: DictConfig
    if resolve:
        prepared = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        if not isinstance(prepared, DictConfig):
            raise TypeError("PLCS generation config must resolve to DictConfig.")
    else:
        prepared = config

    paths_cfg = prepared.get("paths")
    if paths_cfg is not None and paths_cfg.get("smplh_model_path") is not None:
        prepared.paths.smplh_model_path = to_absolute_path(str(prepared.paths.smplh_model_path))

    motion_sources = prepared.get("motion_sources")
    if motion_sources is not None:
        for _category, source in motion_sources.items():
            if source is None or source.get("paths") is None:
                continue
            source.paths = [to_absolute_path(str(path)) for path in source.paths]

    return prepared
